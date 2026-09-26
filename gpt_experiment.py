"""Zero-shot API evaluation of SPARTQA; no training or gold labels in prompts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from spartqa.api import api_error_details, request_answer, response_format
from spartqa.data import TASKS, examples_from_data, read_json, validate_answer, write_json
from spartqa.metrics import answer_matches, set_jaccard

ROOT = Path(__file__).resolve().parent


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def export_results(
    data: dict[str, Any], examples: list[dict[str, Any]], records: list[dict[str, Any]],
    output_dir: Path, input_stem: str,
) -> dict[str, Any]:
    selected = {example["key"] for example in examples}
    successful = {record["key"]: record for record in records if "answer" in record and record["key"] in selected}
    output = copy.deepcopy(data)
    for item in output["data"]:
        for question in item["questions"]:
            question.pop("answer", None)
    state: dict[str, dict[str, Any]] = {}
    for example in examples:
        record = successful.get(example["key"])
        if record is not None:
            output["data"][example["story_index"]]["questions"][example["question_index"]]["answer"] = record["answer"]
        gold = example["gold"]
        if gold is None:
            continue
        task = example["payload"]["q_type"]
        values = state.setdefault(task, {"count": 0, "answered": 0, "correct": 0, "jaccard_sum": 0.0})
        values["count"] += 1
        if record is None:
            continue
        values["answered"] += 1
        prediction = record["answer"]
        values["correct"] += int(answer_matches(task, prediction, gold))
        if task in {"FR", "FB"}:
            values["jaccard_sum"] += set_jaccard(prediction, gold)
    metrics = {}
    for task, values in state.items():
        task_metrics = {"count": values["count"], "answered": values["answered"]}
        primary = "accuracy" if task in {"YN", "CO"} else "exact_match"
        task_metrics[primary] = values["correct"] / values["count"]
        if task in {"FR", "FB"}:
            task_metrics["jaccard"] = values["jaccard_sum"] / values["count"]
        metrics[task] = task_metrics
    usage = {
        field: sum((record.get("usage") or {}).get(field, 0) for record in records)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    report = {
        "evaluation": "zero-shot on supplied file; no model training",
        "selected_questions": len(examples),
        "answered_questions": len(successful),
        "coverage": len(successful) / len(examples) if examples else 0.0,
        "status": "complete" if len(successful) == len(examples) else "partial",
        "missing_predictions_count_as_incorrect": True,
        "by_task": metrics,
        "usage_all_attempts": usage,
    }
    write_json(output_dir / f"{input_stem}_predictions.json", output)
    write_json(output_dir / "metrics.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Data/human_train.json")
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "Docs/spartqa_cot.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/gpt41mini_cot_human_train")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", help="Overrides OPENAI_MODEL or BASE_MODEL_ID from the environment.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.env_file.exists():
        try:
            from dotenv import load_dotenv
        except ImportError:
            parser.exit(2, "Install API dependencies: python -m pip install -r requirements-api.txt\n")
        load_dotenv(args.env_file, override=False)
    args.model = args.model or os.environ.get("OPENAI_MODEL") or os.environ.get("BASE_MODEL_ID") or "gpt-4.1-mini-2025-04-14"
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/") + "/"
    endpoint = urlsplit(base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        parser.error("OPENAI_BASE_URL must be an HTTP(S) endpoint without credentials, query, or fragment")
    if args.max_questions is not None and args.max_questions < 1:
        parser.error("--max-questions must be positive")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    data = read_json(args.input)
    examples = examples_from_data(data)[:args.max_questions]
    if not examples:
        parser.error("No questions found")
    config = {
        "format_version": 2,
        "base_url": base_url,
        "model": args.model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "prompt": args.prompt_file.read_text(encoding="utf-8"),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    print(json.dumps({"model": args.model, "questions": len(examples), "by_task": dict(Counter(example["payload"]["q_type"] for example in examples))}, indent=2))
    if args.dry_run:
        print("Dry run passed. No API calls or prediction files created.")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        parser.exit(2, "Missing OPENAI_API_KEY. Configure it privately in your terminal; do not paste it into chat.\n")
    try:
        from openai import OpenAI
    except ImportError:
        parser.exit(2, "Install API dependencies: python -m pip install -r requirements-api.txt\n")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    log_path = output_dir / "responses.jsonl"
    if config_path.exists():
        if read_json(config_path) != config:
            parser.exit(2, "Input, prompt, or settings changed. Choose a new --output-dir.\n")
    else:
        if any(output_dir.iterdir()):
            parser.exit(2, "Output directory is not empty and has no matching config. Choose a new --output-dir.\n")
        write_json(config_path, config)
    records = load_records(log_path)
    completed = {record["key"] for record in records if "answer" in record}
    fatal = False
    try:
        with OpenAI(base_url=base_url, timeout=60.0, max_retries=2) as client, log_path.open("a", encoding="utf-8") as log:
            for position, example in enumerate(examples, start=1):
                if example["key"] in completed:
                    continue
                try:
                    record = request_answer(client, example, config)
                except Exception as error:
                    record = {"key": example["key"], "error": type(error).__name__, "error_details": api_error_details(error)}
                    fatal = True
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                records.append(record)
                status = record.get("error", "OK")
                print(f"[{position}/{len(examples)}] {example['key']} {example['payload']['q_type']}: {status}", flush=True)
                if fatal:
                    print(json.dumps(record["error_details"], ensure_ascii=True))
                    print("API request failed. Check credentials, quota, network, and model access; rerun to resume.")
                    break
    except KeyboardInterrupt:
        print("Interrupted. Completed responses are saved; rerun the same command to resume.")
    finally:
        report = export_results(data, examples, records, output_dir, args.input.stem)
        print(json.dumps(report, indent=2))
        print(f"Results: {output_dir}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())