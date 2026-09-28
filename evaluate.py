"""Score prediction files against labeled files with the official task metrics.

    python evaluate.py --human PRED GOLD --auto PRED GOLD --output outputs/run/eval.json

Human and Auto are scored separately; with both, the Final Score is the mean
of the two dataset-level values for each metric. Missing or invalid answers
count as wrong. The prediction file must keep the gold file's structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spartqa.data import TASKS, read_json, write_json
from spartqa.metrics import PRIMARY_METRIC, bootstrap_intervals, final_scores, question_records, score_records
from spartqa.submission import compare_structure


def evaluate_file(prediction_path: Path, gold_path: Path, bootstrap: int, seed: int) -> dict:
    prediction, gold = read_json(prediction_path), read_json(gold_path)
    problems = compare_structure(prediction, gold)
    if problems:
        details = "\n  ".join(problems[:20])
        raise ValueError(f"{prediction_path} does not match {gold_path} ({len(problems)} differences):\n  {details}")
    records = question_records(prediction, gold)
    report = {"prediction": prediction_path.as_posix(), "gold": gold_path.as_posix(), **score_records(records)}
    if bootstrap:
        report["bootstrap_95ci_by_story"] = bootstrap_intervals(records, bootstrap, seed)
    return report


def format_table(title: str, by_task: dict, macro: float) -> str:
    lines = [title]
    for task in TASKS:
        if task in by_task:
            values = by_task[task]
            text = f"  {task}  {PRIMARY_METRIC[task]:<11} {values[PRIMARY_METRIC[task]]:.4f}"
            if "jaccard" in values:
                text += f"   jaccard {values['jaccard']:.4f}"
            if "count" in values:
                text += f"   (n={int(values['count'])})"
            lines.append(text)
    lines.append(f"  mean primary (unofficial) {macro:.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", nargs=2, type=Path, metavar=("PRED", "GOLD"))
    parser.add_argument("--auto", nargs=2, type=Path, metavar=("PRED", "GOLD"))
    parser.add_argument("--bootstrap", type=int, default=0, help="Story-level bootstrap samples (0 disables).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args(argv)
    if not args.human and not args.auto:
        parser.error("Pass --human PRED GOLD and/or --auto PRED GOLD")

    report = {}
    try:
        for name in ("human", "auto"):
            if getattr(args, name):
                report[name] = evaluate_file(*getattr(args, name), args.bootstrap, args.seed)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    for name, result in report.items():
        print(format_table(f"{name} ({result['answered_valid']}/{result['questions']} valid answers)",
                           result["by_task"], result["primary_macro_unofficial"]))
        for metric, (low, high) in result.get("bootstrap_95ci_by_story", {}).items():
            print(f"  95% CI {metric}: [{low:.4f}, {high:.4f}]")
    if len(report) == 2:
        report["final"] = final_scores(report["human"], report["auto"])
        print(format_table("final = (human + auto) / 2", report["final"]["by_task"],
                           report["final"]["primary_macro_unofficial"]))
    if args.output:
        write_json(args.output, report)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
