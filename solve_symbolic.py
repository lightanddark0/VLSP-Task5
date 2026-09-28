"""Branch S: rule-based answers for the generated data; abstains when unsure (CPU only).

    python solve_symbolic.py \\
        --job auto,Data/splits/auto_dev.json,outputs/predictions/S/auto_dev.jsonl \\
        --job auto,Data/auto_public_test.json,outputs/predictions/S/auto_test.jsonl \\
        --job human,Data/splits/human_dev.json,outputs/predictions/S/human_dev.jsonl

Writes the shared JSONL prediction format (answer null when abstaining, one-hot
scores). On labeled inputs it prints coverage and accuracy on the covered
questions per type, and --errors writes wrong answers for debugging. Story
sentences the parser does not understand are counted; stories with such a
sentence are abstained entirely. Solver switches: --set yn_no_rule=false etc.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path

from spartqa import tracking
from spartqa.data import TASKS, read_json
from spartqa.inference import parse_job
from spartqa.metrics import answer_matches
from spartqa.predictions import iter_questions, one_hot_scores
from spartqa.symbolic import ParseError, Solver, SolverConfig, read_story


def parse_setting(text: str) -> tuple[str, object]:
    key, _, value = text.partition("=")
    known = {f.name: f.type for f in fields(SolverConfig)}
    if key not in known:
        raise argparse.ArgumentTypeError(f"unknown setting {key}; known: {sorted(known)}")
    if value.lower() in ("true", "false"):
        return key, value.lower() == "true"
    return key, value


def solve_file(data: dict, dataset: str, config: SolverConfig, limit: int | None = None):
    """Predictions by key plus per-story parse diagnostics."""
    records, failures = {}, Counter()
    stories = data["data"][:limit] if limit else data["data"]
    for story_index, item in enumerate(stories):
        try:
            world = read_story(item["story"])
            solver = Solver(world, config)
            error = None
        except (ParseError, IndexError, KeyError, AttributeError) as exc:
            solver, error = None, str(exc)
            failures[error[:120]] += 1
        for key, _, question in iter_questions({"data": [item]}):
            key = f"{story_index}_{question['q_id']}"
            answer = solver.answer(question) if solver is not None else None
            records[key] = {"key": key, "q_type": question["q_type"], "answer": answer,
                            "scores": one_hot_scores(question["q_type"], answer), "source": "S",
                            "story_parsed": error is None}
    return records, failures


def report(data: dict, records: dict, limit: int | None = None) -> dict:
    stats = {task: Counter() for task in TASKS}
    wrong = []
    stories = data["data"][:limit] if limit else data["data"]
    for story_index, item in enumerate(stories):
        for question in item["questions"]:
            key = f"{story_index}_{question['q_id']}"
            record = records[key]
            task_stats = stats[question["q_type"]]
            task_stats["total"] += 1
            if record["answer"] is None:
                continue
            task_stats["covered"] += 1
            if "answer" in question:
                task_stats["labeled"] += 1
                correct = answer_matches(question["q_type"], record["answer"], question["answer"])
                task_stats["correct"] += correct
                if not correct:
                    wrong.append({"key": key, "q_type": question["q_type"], "question": question["question"],
                                  "predicted": record["answer"], "gold": question["answer"],
                                  "story": " ".join(item["story"])})
    summary = {}
    for task, counts in stats.items():
        total, covered = counts["total"], counts["covered"]
        summary[task] = {"total": total, "coverage": covered / total if total else 0.0,
                         "accuracy_on_covered": counts["correct"] / counts["labeled"] if counts["labeled"] else None}
    return {"by_task": summary, "wrong": wrong}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", type=parse_job, action="append", required=True,
                        help="DATASET,INPUT.json,OUTPUT.jsonl; repeat for several files.")
    parser.add_argument("--set", type=parse_setting, action="append", default=[], help="Solver switch key=value.")
    parser.add_argument("--limit", type=int, help="Only the first N stories of each file.")
    parser.add_argument("--errors", type=Path, help="Write wrong answers (labeled inputs) to this JSONL.")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    config = SolverConfig(**dict(args.set))

    datasets = sorted({job.dataset for job in args.job})
    splits = sorted({job.split for job in args.job})
    tracking.init_run("S", "infer", args.run_name or tracking.run_name("S", "infer", "+".join(datasets),
                                                                        "+".join(splits)),
                      datasets + splits, {"method": "symbolic", "solver": vars(config)})
    all_wrong = []
    for job in args.job:
        started = time.time()
        data = read_json(job.input)
        records, failures = solve_file(data, job.dataset, config, args.limit)
        job.output.parent.mkdir(parents=True, exist_ok=True)
        with job.output.open("w", encoding="utf-8") as stream:
            for record in records.values():
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stories = len(data["data"][:args.limit] if args.limit else data["data"])
        parsed = stories - sum(failures.values())
        result = report(data, records, args.limit)
        print(f"[S {job.dataset}/{job.split}] stories parsed {parsed}/{stories} "
              f"({parsed / max(stories, 1):.1%}), {time.time() - started:.1f}s -> {job.output}")
        values = {f"S/{job.dataset}/{job.split}/stories_parsed": parsed / max(stories, 1)}
        for task, row in result["by_task"].items():
            accuracy = row["accuracy_on_covered"]
            print(f"  {task}: coverage {row['coverage']:.1%}" +
                  (f", accuracy on covered {accuracy:.2%}" if accuracy is not None else ""))
            values[f"S/{job.dataset}/{job.split}/{task}/coverage"] = row["coverage"]
            if accuracy is not None:
                values[f"S/{job.dataset}/{job.split}/{task}/accuracy_on_covered"] = accuracy
        tracking.log(values)
        for message, count in failures.most_common(8):
            print(f"  unparsed ({count}): {message}")
        all_wrong += [{**row, "file": str(job.input)} for row in result["wrong"]]
    if args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        with args.errors.open("w", encoding="utf-8") as stream:
            for row in all_wrong:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_wrong)} wrong answers to {args.errors}")
    tracking.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
