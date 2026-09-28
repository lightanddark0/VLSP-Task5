"""Compare branches alone, in pairs, and all together, from existing predictions only.

    python compare_branches.py --pred-dir outputs/predictions --output-dir outputs/combos

No model is run: every branch must already have written its dev and test JSONL
files (outputs/predictions/<source>/<dataset>_{dev,test}.jsonl). Branches:
    F = F2, F1 (fine-tuned Qwen3-8B)   S = S (rule-based)   L = LCOT (Qwen3-32B)
For each dataset and each combination, the per-type configuration (sources,
weights, FR/FB threshold) is tuned on dev as in ensemble.py, then applied to
test. Two dev numbers are reported: "tuned" (tuned and scored on the same dev
questions, optimistic) and "cv" (5-fold cross-validation over dev stories:
tuned on four folds, scored on the fifth), which is the fairer comparison. A
branch with no predictions for a dataset is left out of the combination
there (noted in the table), so every combination has a Final score.
Each combination's submission files are written and validated.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from ensemble import candidate_configs, full_report, predict
from spartqa import tracking
from spartqa.data import TASKS, read_json, write_json
from spartqa.inference import fallback_answers
from spartqa.metrics import PRIMARY_METRIC, answer_matches, score_records
from spartqa.postprocess import PostprocessOptions
from spartqa.predictions import fill_answers, iter_questions, read_predictions, restore_field_order
from spartqa.submission import answer_problems, compare_structure

BRANCHES = {"F": ["F2", "F1"], "S": ["S"], "L": ["LCOT"]}
PRIORITY = ["S", "F2", "F1", "LCOT"]   # fallback order when every weighted source abstains


def tune(questions: list[tuple[str, dict]], sources: dict, names: list[str], dataset: str,
         options: PostprocessOptions, fallback: dict, grid: list[float], thresholds: list[float]) -> dict:
    """Best configuration per question type on the given questions (ties: simplest)."""
    chosen = {}
    for task in TASKS:
        subset = [(key, q) for key, q in questions if q["q_type"] == task]
        best, best_score = None, -1.0
        for config in candidate_configs(names, task, grid, thresholds):
            correct = sum(answer_matches(task, predict(key, q, sources, config, dataset, options, fallback),
                                         q["answer"]) for key, q in subset)
            score = correct / len(subset) if subset else 0.0
            if score > best_score + 1e-12:
                best, best_score = config, score
        chosen[task] = best
    return chosen


def rows_for(questions: list[tuple[str, dict]], sources: dict, configs: dict, dataset: str,
             options: PostprocessOptions, fallback: dict) -> list[dict]:
    rows = []
    for key, q in questions:
        answer = predict(key, q, sources, configs[q["q_type"]], dataset, options, fallback)
        rows.append({"story_index": int(key.split("_")[0]), "task": q["q_type"], "reasoning_types": [],
                     "prediction": answer, "gold": q["answer"]})
    return rows


def cross_validated(questions: list[tuple[str, dict]], sources: dict, names: list[str], dataset: str,
                    options: PostprocessOptions, fallback: dict, grid: list[float], thresholds: list[float],
                    folds: int) -> dict:
    stories = sorted({int(key.split("_")[0]) for key, _ in questions})
    folds = max(2, min(folds, len(stories)))
    fold_of = {story: index % folds for index, story in enumerate(stories)}
    rows = []
    for fold in range(folds):
        train = [(k, q) for k, q in questions if fold_of[int(k.split("_")[0])] != fold]
        held = [(k, q) for k, q in questions if fold_of[int(k.split("_")[0])] == fold]
        configs = tune(train, sources, names, dataset, options, fallback, grid, thresholds)
        rows += rows_for(held, sources, configs, dataset, options, fallback)
    return score_records(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--splits-dir", type=Path, default=Path("Data/splits"))
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/combos"))
    parser.add_argument("--datasets", nargs="+", default=["human", "auto"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--weights", default="0,1,2")
    parser.add_argument("--thresholds", default="0.4,0.5,0.6")
    parser.add_argument("--use-indifinite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--human-yn-dk", choices=("keep", "no"), default="keep")
    parser.add_argument("--submission-names", default="human=human_submission.json,auto=auto_submission.json")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)

    options = PostprocessOptions(use_indifinite=args.use_indifinite, human_yn_dk=args.human_yn_dk)
    grid = [float(v) for v in args.weights.split(",")]
    thresholds = [float(v) for v in args.thresholds.split(",")]
    submission_names = dict(item.split("=", 1) for item in args.submission_names.split(","))
    combos = [combo for size in (1, 2, 3) for combo in itertools.combinations(BRANCHES, size)]
    tracking.init_run("ensemble", "eval", args.run_name or tracking.run_name("compare", "eval", "all", "dev"),
                      ["compare"], {"postprocess": options.as_dict(), "branches": BRANCHES, "cv_folds": args.cv_folds})

    results: dict[str, dict[str, Any]] = {"+".join(c): {} for c in combos}
    for dataset in args.datasets:
        dev_data = read_json(args.splits_dir / f"{dataset}_dev.json")
        test_data = read_json(args.data_dir / f"{dataset}_public_test.json")
        fallback = fallback_answers(args.splits_dir, dataset)
        available = {}
        for source in PRIORITY:
            dev_path = args.pred_dir / source / f"{dataset}_dev.jsonl"
            test_path = args.pred_dir / source / f"{dataset}_test.jsonl"
            if dev_path.exists() and test_path.exists():
                available[source] = (read_predictions(dev_path), read_predictions(test_path))
        print(f"== {dataset}: sources with dev+test predictions: {list(available) or 'none'}")
        questions = [(key, q) for key, _, q in iter_questions(dev_data) if "answer" in q]
        for combo in combos:
            name = "+".join(combo)
            used = [b for b in combo if any(s in available for s in BRANCHES[b])]
            names = [s for s in PRIORITY if s in available and any(s in BRANCHES[b] for b in used)]
            if not names:
                results[name][dataset] = None
                continue
            dev_sources = {s: available[s][0] for s in names}
            test_sources = {s: available[s][1] for s in names}
            configs = tune(questions, dev_sources, names, dataset, options, fallback, grid, thresholds)
            tuned = score_records(rows_for(questions, dev_sources, configs, dataset, options, fallback))
            cv = cross_validated(questions, dev_sources, names, dataset, options, fallback, grid, thresholds,
                                 args.cv_folds)
            _, test_answers = full_report(test_data, test_sources, configs, dataset, options, fallback)
            submission = restore_field_order(fill_answers(test_data, test_answers), test_data)
            problems = compare_structure(submission, test_data) + [str(p) for p in answer_problems(submission)]
            out_dir = args.output_dir / name
            write_json(out_dir / submission_names[dataset], submission)
            write_json(out_dir / f"{dataset}_config.json", {"sources": names, "per_type": configs,
                                                            "postprocess": options.as_dict()})
            results[name][dataset] = {"used_branches": used, "sources": names, "tuned": tuned["by_task"],
                                      "tuned_mean": tuned["primary_macro_unofficial"], "cv": cv["by_task"],
                                      "cv_mean": cv["primary_macro_unofficial"], "valid": not problems}
            print(f"  {name:<6} sources {names}: tuned {tuned['primary_macro_unofficial']:.4f}  "
                  f"cv {cv['primary_macro_unofficial']:.4f}  submission {'OK' if not problems else 'INVALID'}")

    # Summary table: per dataset (cv) and Final = (Human + Auto) / 2.
    lines = ["| Combination | Human cv | Auto cv | **Final cv** | Human tuned | Auto tuned | Final tuned | Note |",
             "|---|---|---|---|---|---|---|---|"]
    table = []
    for name, per in results.items():
        cells, note = [], []
        for kind in ("cv_mean", "tuned_mean"):
            values = [per.get(d)[kind] if per.get(d) else None for d in ("human", "auto")]
            final = sum(values) / 2 if all(v is not None for v in values) else None
            cells.append((values, final))
        for dataset in ("human", "auto"):
            entry = per.get(dataset)
            if dataset in per and entry is None:
                note.append(f"{dataset}: no predictions")
            if entry and set(entry["used_branches"]) != set(name.split("+")):
                note.append(f"{dataset}: {'+'.join(entry['used_branches'])}")
            if entry and not entry["valid"]:
                note.append(f"{dataset}: INVALID")
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"  # noqa: E731
        (cv_values, cv_final), (tuned_values, tuned_final) = cells
        lines.append(f"| {name} | {fmt(cv_values[0])} | {fmt(cv_values[1])} | **{fmt(cv_final)}** | "
                     f"{fmt(tuned_values[0])} | {fmt(tuned_values[1])} | {fmt(tuned_final)} | {'; '.join(note)} |")
        table.append([name, cv_values[0], cv_values[1], cv_final, tuned_values[0], tuned_values[1], tuned_final,
                      "; ".join(note)])
    per_type = ["", "Per type (cv, primary metric):"]
    for name, per in results.items():
        for dataset in ("human", "auto"):
            entry = per.get(dataset)
            if entry:
                text = "  ".join(f"{t} {entry['cv'][t][PRIMARY_METRIC[t]]:.3f}" for t in TASKS if t in entry["cv"])
                per_type.append(f"  {name:<6} {dataset:<5} {text}")
    report = "\n".join(lines + per_type)
    print("\n" + report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.md").write_text(report + "\n", encoding="utf-8")
    write_json(args.output_dir / "comparison.json", results)
    best = max((row for row in table if row[3] is not None), key=lambda row: row[3], default=None)
    if best:
        print(f"\nBest by Final cv: {best[0]} ({best[3]:.4f}) -> {args.output_dir / best[0]}")
        write_json(args.output_dir / "best.json", {"combination": best[0], "final_cv": best[3]})
    tracking.log_table("compare/combinations", table,
                       ["combination", "human_cv", "auto_cv", "final_cv", "human_tuned", "auto_tuned",
                        "final_tuned", "note"])
    tracking.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
