"""Dataset statistics and checks of the assumptions the pipeline relies on.

    python explore.py --tokenizer Qwen/Qwen3-8B --output outputs/explore/report.json

Prints story/question counts, answer distributions, and prompt token lengths
(branch F prompt + answer, Qwen3 chat template), then checks:
  1. FR label 7 (DK) never appears with another label.
  2. FR never contains 0&1, 2&3, or 4&5.
  3. Whether Human YN has DK.
  4. Which Human/Auto files are labeled.
  5. Whether q_id is unique per file or only per story.
  6. Whether ``indifinite`` matches DK exactly (YN DK / FR [7]).
The postprocess defaults in spartqa/postprocess.py assume these results.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from spartqa import tracking
from spartqa.data import TASKS, read_json, write_json
from spartqa.prompting import build_messages, format_answer, render_prompt, story_text
from spartqa.predictions import payload_of

FILES = ("human_train", "human_public_test", "auto_train", "auto_public_test")


def quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    pick = lambda share: values[min(len(values) - 1, int(share * len(values)))]  # noqa: E731
    return {"min": values[0], "median": pick(0.5), "p95": pick(0.95), "p99": pick(0.99), "max": values[-1],
            "mean": round(sum(values) / len(values), 1)}


def shares(counter: Counter) -> dict[str, Any]:
    total = sum(counter.values()) or 1
    return {str(key): {"count": value, "share": round(value / total, 4)} for key, value in counter.most_common()}


def describe(data: dict[str, Any]) -> dict[str, Any]:
    questions = [question for item in data["data"] for question in item["questions"]]
    labeled = bool(questions) and all("answer" in question for question in questions)
    report: dict[str, Any] = {
        "stories": len(data["data"]),
        "questions": len(questions),
        "labeled": labeled,
        "questions_per_story": quantiles([len(item["questions"]) for item in data["data"]]),
        "story_chars": quantiles([len(story_text(item["story"])) for item in data["data"]]),
        "story_words": quantiles([len(story_text(item["story"]).split()) for item in data["data"]]),
        "by_task": {task: sum(q["q_type"] == task for q in questions) for task in TASKS},
        "candidate_formats": {task: shares(Counter(
            len(q["candidate_answers"]) for q in questions if q["q_type"] == task)) for task in TASKS},
        "indifinite_true": {task: sum(bool(q.get("indifinite")) for q in questions if q["q_type"] == task)
                            for task in TASKS},
    }
    if not labeled:
        return report
    by = {task: [q["answer"] for q in questions if q["q_type"] == task] for task in TASKS}
    report["answers"] = {
        "YN": shares(Counter(answer[0] for answer in by["YN"])),
        "CO": shares(Counter(answer[0] for answer in by["CO"])),
        "FR_size": shares(Counter(len(answer) for answer in by["FR"])),
        "FR_label": shares(Counter(label for answer in by["FR"] for label in answer)),
        "FR_top_sets": dict(list(shares(Counter(tuple(sorted(a)) for a in by["FR"])).items())[:10]),
        "FB_size": shares(Counter(len(answer) for answer in by["FB"])),
        "FB_empty_share": round(sum(not answer for answer in by["FB"]) / max(len(by["FB"]), 1), 4),
    }
    return report


def check_assumptions(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, data in datasets.items():
        questions = [q for item in data["data"] for q in item["questions"]]
        q_ids = Counter(q["q_id"] for q in questions)
        per_story = all(len({q["q_id"] for q in item["questions"]}) == len(item["questions"]) for item in data["data"])
        entry = {"q_id_unique_in_file": max(q_ids.values(), default=0) == 1, "q_id_unique_in_story": per_story}
        if questions and all("answer" in q for q in questions):
            fr = [q for q in questions if q["q_type"] == "FR"]
            yn = [q for q in questions if q["q_type"] == "YN"]
            entry.update({
                "fr_dk_with_other_labels": sum(7 in q["answer"] and len(q["answer"]) > 1 for q in fr),
                "fr_conflicting_pairs": sum(any({a, b} <= set(q["answer"]) for a, b in ((0, 1), (2, 3), (4, 5)))
                                            for q in fr),
                "yn_dk_count": sum(q["answer"] == ["DK"] for q in yn),
                "indifinite_vs_dk_mismatches": sum(
                    bool(q.get("indifinite")) != (q["answer"] in (["DK"], [7]))
                    for q in yn + fr),
            })
        checks[name] = entry
    return checks


def token_lengths(datasets: dict[str, dict[str, Any]], tokenizer_name: str, sample: int) -> dict[str, Any]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    result = {}
    for name, data in datasets.items():
        dataset = name.split("_")[0]
        lengths = []
        for item in data["data"][:sample]:
            for question in item["questions"]:
                payload = payload_of(item, question)
                prompt = render_prompt(tokenizer, build_messages(payload, dataset))
                target = format_answer(question.get("answer") or [], payload)
                lengths.append(len(tokenizer(prompt + target, add_special_tokens=False)["input_ids"]) + 1)
        result[name] = quantiles(lengths)
    longest = max(values["max"] for values in result.values())
    result["recommended_max_len"] = int(-(-(longest + 16) // 64) * 64)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="'none' skips token statistics.")
    parser.add_argument("--token-sample-stories", type=int, default=3000,
                        help="Stories per file used for token lengths (all Human stories fit).")
    parser.add_argument("--output", type=Path, default=Path("outputs/explore/report.json"))
    args = parser.parse_args(argv)

    datasets = {name: read_json(args.data_dir / f"{name}.json") for name in FILES
                if (args.data_dir / f"{name}.json").exists()}
    if not datasets:
        parser.error(f"No data files found in {args.data_dir}")
    tracking.init_run("explore", "explore", tracking.run_name("explore", "stats", "all", "all"), ["explore"],
                      {"files": list(datasets)})
    report = {"files": {name: describe(data) for name, data in datasets.items()},
              "assumptions": check_assumptions(datasets)}
    if args.tokenizer.lower() != "none":
        report["prompt_tokens_F"] = token_lengths(datasets, args.tokenizer, args.token_sample_stories)

    for name, info in report["files"].items():
        print(f"== {name}: {info['stories']} stories, {info['questions']} questions, labeled={info['labeled']}")
        print(f"   by task {info['by_task']}  story words {info['story_words']}")
        print(f"   indifinite=True {info['indifinite_true']}")
        for key, value in info.get("answers", {}).items():
            print(f"   {key}: {json.dumps(value, ensure_ascii=False)[:300]}")
    print("== assumptions")
    for name, values in report["assumptions"].items():
        print(f"   {name}: {values}")
    if "prompt_tokens_F" in report:
        print("== F prompt+answer tokens")
        for name, values in report["prompt_tokens_F"].items():
            print(f"   {name}: {values}")

    write_json(args.output, report)
    print(f"Wrote {args.output}")
    values = {}
    for name, info in report["files"].items():
        values.update({f"explore/{name}/{key}": info[key] for key in ("stories", "questions")})
    tracking.log(values)
    tracking.update_config({"explore_report": report})
    tracking.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
