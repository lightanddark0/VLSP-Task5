"""Shared plumbing for inference scripts: jobs, resume, fallback, dev scoring, uploads."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spartqa import hub, tracking
from spartqa.data import read_json
from spartqa.metrics import detailed_metrics, score_records
from spartqa.postprocess import PostprocessOptions, finalize_answer
from spartqa.predictions import iter_questions, read_predictions
from spartqa.submission import majority_answers


@dataclass
class Job:
    dataset: str
    input: Path
    output: Path

    @property
    def split(self) -> str:
        stem = self.input.stem
        return "dev" if stem.endswith("_dev") else "test" if "test" in stem else "train" if "train" in stem else stem


def parse_job(text: str) -> Job:
    """'DATASET,INPUT.json,OUTPUT.jsonl' where DATASET is human or auto."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3 or parts[0] not in {"human", "auto"}:
        raise argparse.ArgumentTypeError(f"Expected human|auto,INPUT,OUTPUT but got {text!r}")
    return Job(parts[0], Path(parts[1]), Path(parts[2]))


def pending(data: dict[str, Any], output: Path, limit: int | None) -> tuple[list[tuple[str, Any, Any]], int]:
    """Questions still to predict (file order, first ``limit`` only) and the number already done."""
    done = read_predictions(output)
    selected = list(iter_questions(data))
    if limit:
        selected = selected[:limit]
    todo = [entry for entry in selected if entry[0] not in done]
    return todo, len(selected) - len(todo)


def fallback_answers(splits_dir: Path, dataset: str) -> dict[str, list[Any]]:
    path = splits_dir / f"{dataset}_train.json"
    return majority_answers(read_json(path)) if path.exists() else {"YN": ["Yes"], "FR": [7], "FB": [], "CO": [3]}


def score_predictions(data: dict[str, Any], records: dict[str, dict[str, Any]], dataset: str,
                      options: PostprocessOptions, fallback: dict[str, list[Any]]) -> dict[str, Any] | None:
    """Score predicted questions against gold (None if the file is unlabeled); post-processing applied."""
    rows = []
    for key, _, question in iter_questions(data):
        if key not in records or "answer" not in question:
            continue
        record = records[key]
        answer = finalize_answer(question, record.get("answer"), record.get("scores"), dataset, options, fallback)
        rows.append({"key": key, "story_index": int(key.split("_")[0]), "task": question["q_type"],
                     "reasoning_types": list(question.get("reasoning_type") or []), "prediction": answer,
                     "gold": question["answer"], "question": question["question"], "source": record.get("source")})
    if not rows:
        return None
    report = score_records(rows)
    report["details"] = detailed_metrics(rows)
    report["rows"] = rows
    return report


def log_report(report: dict[str, Any], dataset: str, split: str, story_of: dict[str, str] | None = None) -> None:
    """Metrics, YN/CO confusion matrices, and up to 200 wrong answers to W&B."""
    prefix = f"eval/{dataset}/{split}"
    tracking.log_eval(report, prefix)
    rows = report["rows"]
    for task, classes in (("YN", ["Yes", "No", "DK"]), ("CO", [0, 1, 2, 3])):
        group = [row for row in rows if row["task"] == task]
        tracking.log_confusion(f"{prefix}/{task}/confusion", [row["gold"][0] for row in group],
                               [row["prediction"][0] for row in group], classes)
    wrong = [row for row in rows if set(map(str, row["prediction"])) != set(map(str, row["gold"]))]
    tracking.log_table(f"{prefix}/errors", [
        [row["key"], row["task"], row["question"], str(row["gold"]), str(row["prediction"]), row["source"],
         (story_of or {}).get(row["key"], "")[:400]] for row in wrong],
        ["key", "q_type", "question", "gold", "predicted", "source", "story"])


def print_report(report: dict[str, Any], title: str) -> None:
    print(title)
    for task, values in report["by_task"].items():
        text = ", ".join(f"{name} {value:.4f}" for name, value in values.items() if name != "count")
        print(f"  {task}: {text} (n={int(values['count'])})")
    print(f"  mean primary (unofficial): {report['primary_macro_unofficial']:.4f}")


def upload_predictions(path: Path, results_repo: str | None, prefix: str) -> str | None:
    repo = hub.resolve_repo(results_repo) if results_repo and hub.hub_enabled() else None
    if not repo:
        return None
    return hub.upload_file(path, repo, f"{prefix}/{path.name}", "dataset", f"Predictions {prefix}/{path.name}")


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job", type=parse_job, action="append", required=True,
                        help="DATASET,INPUT.json,OUTPUT.jsonl; repeat for several files (one model load).")
    parser.add_argument("--splits-dir", type=Path, default=Path("Data/splits"),
                        help="Train splits used for majority fallback answers.")
    parser.add_argument("--limit", type=int, help="Only the first N questions of each file (smoke test).")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Questions per saved chunk.")
    parser.add_argument("--results-repo", help="Private Hub dataset repo for prediction files.")
    parser.add_argument("--results-prefix", default="predictions", help="Folder inside the results repo.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
