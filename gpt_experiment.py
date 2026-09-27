"""CoT and Path-of-Thoughts API evaluation of SPARTQA; no gold labels in prompts."""

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
from spartqa.pot import REASONING_INSTRUCTIONS, request_pot_answer, validate_graph

ROOT = Path(__file__).resolve().parent


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_graph_cache(directory: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_config = read_json(directory / "config.json")
    fields = ("input_sha256", "base_url", "model", "temperature", "extraction_prompt",
              "max_extraction_tokens", "pot_version")
    if any(source_config.get(field) != config.get(field) for field in fields):
        raise ValueError("Graph cache extraction settings or input do not match this run")
    cached = {}
    for event in load_records(directory / "responses.jsonl"):
        if event.get("stage") == "extract" and "graph" in event.get("result", {}):
            result = event["result"]
            cached[event["key"]] = {
                **result, "graph": validate_graph(result["graph"]),
                "usage": {}, "reused_extraction": True,
            }
    if not cached:
        raise ValueError("No successful graph extractions in cache")
    return cached


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
        values = state.setdefault(task, {"count": 0, "answered": 0, "correct": 0, "jaccard_sum": 0.0, "hits": 0})
        values["count"] += 1
        if record is None:
            continue
        values["answered"] += 1
        prediction = record["answer"]
        values["correct"] += int(answer_matches(task, prediction, gold))
        if task in {"FR", "FB"}:
            values["jaccard_sum"] += set_jaccard(prediction, gold)
        if task == "FR":
            values["hits"] += int(bool(set(prediction) & set(gold)))
    metrics = {}
    for task, values in state.items():
        task_metrics = {"count": values["count"], "answered": values["answered"]}
        primary = "accuracy" if task in {"YN", "CO"} else "exact_match"
        task_metrics[primary] = values["correct"] / values["count"]
        if task in {"FR", "FB"}:
            task_metrics["jaccard"] = values["jaccard_sum"] / values["count"]
        if task == "FR":
            task_metrics["hit_accuracy"] = values["hits"] / values["count"]
        metrics[task] = task_metrics
    usage = {
        field: sum((record.get("usage") or {}).get(field, 0) for record in records)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    report = {
        "evaluation": "API inference on supplied file; no model training; see config for method",
        "selected_questions": len(examples),
        "answered_questions": len(successful),
        "coverage": len(successful) / len(examples) if examples else 0.0,
        "status": "complete" if len(successful) == len(examples) else "partial",
        "missing_predictions_count_as_incorrect": True,
        "by_task": metrics,
        "usage_all_attempts": usage,
    }
    pot_records = [record for record in successful.values() if record.get("method", "").startswith("pot")]
    if pot_records:
        report["pot_diagnostics"] = {
            "answered": len(pot_records),
            "fallback_questions": sum(record["fallback"] for record in pot_records),
            "search_limited_questions": sum(bool(record["path_search"]["limits_hit"]) for record in pot_records),
            "path_disagreement_questions": (
                sum(bool(record.get("path_disagreement")) for record in pot_records)
                if any(record.get("path_disagreement") is not None for record in pot_records) else None
            ),
            "reasoner_calls_for_answers": sum(len(record["path_results"]) for record in pot_records),
            "stage_attempts_all": sum("stage" in record and not record["result"].get("reused_extraction", False) for record in records),
            "reused_extractions": sum(record.get("result", {}).get("reused_extraction", False) for record in records),
        }
    write_json(output_dir / f"{input_stem}_predictions.json", output)
    write_json(output_dir / "metrics.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Data/human_train.json")
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "Docs/spartqa_cot.txt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=("cot", "pot", "pot-no-path"), default="cot")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, help="Default: all four tasks for every method.")
    parser.add_argument("--extraction-prompt-file", type=Path, default=ROOT / "Docs/spartqa_pot_extract.txt")
    parser.add_argument("--graph-cache-dir", type=Path, help="Reuse saved PoT extractions for a controlled ablation.")
    parser.add_argument("--max-extraction-tokens", type=int, default=8192)
    parser.add_argument("--max-paths", type=int, default=16, help="PoT path cap per queried pair; 0 means all paths.")
    parser.add_argument("--max-hops", type=int, default=0, help="PoT hop cap; 0 means no hop cap.")
    parser.add_argument("--max-expansions", type=int, default=10000, help="PoT DFS work budget shared across queried pairs.")
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
    if args.max_extraction_tokens < 1 or args.max_paths < 0 or args.max_hops < 0 or args.max_expansions < 1:
        parser.error("Extraction tokens and expansions must be positive; path and hop caps must be nonnegative")
    tasks = sorted(set(args.tasks or TASKS))
    if args.graph_cache_dir is not None and args.method == "cot":
        parser.error("--graph-cache-dir requires a PoT method")
    if args.output_dir is None:
        suffix = "" if args.method == "cot" else "_v2"
        args.output_dir = ROOT / f"outputs/gpt41mini_{args.method}{suffix}_human_train"
    data = read_json(args.input)
    examples = [example for example in examples_from_data(data) if example["payload"]["q_type"] in tasks][:args.max_questions]
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
    if args.tasks is not None or args.method != "cot":
        config["tasks"] = tasks
    if args.method != "cot":
        config.update({
            "format_version": 3, "method": args.method, "pot_version": 2,
            "paper": "https://arxiv.org/abs/2412.17963v2",
            "extraction_prompt": args.extraction_prompt_file.read_text(encoding="utf-8"),
            "reasoning_instructions": REASONING_INSTRUCTIONS,
            "max_extraction_tokens": args.max_extraction_tokens,
            "max_paths": args.max_paths, "max_hops": args.max_hops,
            "max_expansions": args.max_expansions,
            "aggregation": "joint_evidence_by_question_type",
        })
    graph_cache = {}
    if args.graph_cache_dir is not None:
        try:
            graph_cache = load_graph_cache(args.graph_cache_dir, config)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"Cannot reuse graph cache: {error}")
        if any(example["key"] not in graph_cache for example in examples):
            parser.error("Graph cache lacks selected questions; complete the source PoT run first")
        config["graph_cache_sha256"] = hashlib.sha256(
            json.dumps(graph_cache, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    print(json.dumps({"model": args.model, "method": args.method, "questions": len(examples), "by_task": dict(Counter(example["payload"]["q_type"] for example in examples))}, indent=2))
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
    stage_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if "stage" in record and "error" not in record["result"]:
            stage_cache.setdefault(record["key"], {})[record["stage"]] = record["result"]
    fatal = False
    consecutive_invalid_graphs = 0
    try:
        with OpenAI(base_url=base_url, timeout=60.0, max_retries=2) as client, log_path.open("a", encoding="utf-8") as log:
            for position, example in enumerate(examples, start=1):
                if example["key"] in completed:
                    continue
                try:
                    if args.method == "cot":
                        record = request_answer(client, example, config)
                    else:
                        def checkpoint(stage, result):
                            event = {"key": example["key"], "stage": stage, "result": result,
                                     "usage": result.get("usage", {})}
                            log.write(json.dumps(event, ensure_ascii=False) + "\n")
                            log.flush()
                            records.append(event)

                        cached = stage_cache.setdefault(example["key"], {})
                        if example["key"] in graph_cache and "extract" not in cached:
                            cached["extract"] = graph_cache[example["key"]]
                            checkpoint("extract", cached["extract"])
                        record = request_pot_answer(client, example, config, cached, checkpoint)
                        fatal = "error_details" in record
                except Exception as error:
                    record = {"key": example["key"], "error": type(error).__name__, "error_details": api_error_details(error)}
                    fatal = True
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                records.append(record)
                status = record.get("error", "OK")
                print(f"[{position}/{len(examples)}] {example['key']} {example['payload']['q_type']}: {status}", flush=True)
                if "validation_error" in record:
                    print(json.dumps(record["validation_error"], ensure_ascii=True), flush=True)
                consecutive_invalid_graphs = consecutive_invalid_graphs + 1 if status == "InvalidGraph" else 0
                if consecutive_invalid_graphs >= 3:
                    print("Stopped after 3 consecutive invalid graphs. Inspect validation_error and raw_graph_response in responses.jsonl; rerun to resume.")
                    break
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