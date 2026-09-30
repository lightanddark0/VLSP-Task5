"""Step 2: stacking — learn how to combine sources from their confidence, instead of fixed weights.

    python stack_ensemble.py --pred-dir outputs/predictions_e4 --split trdev \\
        --branches "F=F2C,F1 S=S L=LCOT P=LPX" \\
        --auto-submission outputs/e4/trdev_LPX/F+S+L+P/auto_submission.json --output-dir outputs/stacking

For each question type, a logistic regression scores every candidate label
from each source's vote share for it, whether the source answered, whether it
chose that label, how many sources agree, and the label itself. YN/CO take the
best label, FR/FB every label with probability >= 0.5 (FR falls back to the
best one). Scored with the same story folds as the weighted ensemble's "cv",
which is recomputed here for a side-by-side comparison; C is picked by cv.
The final models are refit on the whole split and applied to the public test.
"hybrid" uses, per type, whichever of weighted / stacked had the better cv
(slightly optimistic: the choice is made on the same folds).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from compare_branches import tune
from ensemble import predict
from spartqa.analysis import GRID, THRESHOLDS, group_of, load_setup, oof_ensemble, story_folds, story_of
from spartqa.data import TASKS, read_json, write_json
from spartqa.metrics import PRIMARY_METRIC, answer_matches
from spartqa.postprocess import EMPTY, PostprocessOptions, finalize_answer
from spartqa.predictions import fill_answers, iter_questions, restore_field_order
from spartqa.submission import answer_problems, compare_structure

LABELS = {"YN": ["Yes", "No", "DK"], "CO": ["0", "1", "2", "3"], "FR": [str(i) for i in range(8)]}
MAX_BLOCKS = 4


def candidates(question: dict[str, Any]) -> list[str]:
    if question["q_type"] == "FB":
        return [str(block) for block in question["candidate_answers"]][:MAX_BLOCKS]
    return LABELS[question["q_type"]]


def features(key: str, question: dict[str, Any], sources: dict, names: list[str]) -> list[list[float]]:
    """One row per candidate label of the question."""
    task = question["q_type"]
    records = [sources.get(name, {}).get(key) for name in names]
    answered = [r is not None and r.get("answer") is not None for r in records]
    chosen = [{str(label) for label in r["answer"]} if ok else set() for r, ok in zip(records, answered)]
    labels = candidates(question)
    rows = []
    for position, label in enumerate(labels):
        row: list[float] = []
        for record, ok, picked in zip(records, answered, chosen):
            score = float((record.get("scores") or {}).get(label, 0.0)) if ok else 0.0
            row += [score, float(ok), float(label in picked)]
            if task == "FB":
                row.append(float((record.get("scores") or {}).get(EMPTY, 0.0)) if ok else 0.0)
        agree = sum(label in picked for picked in chosen)
        row += [agree / max(sum(answered), 1), float(sum(answered) == 0)]
        width = MAX_BLOCKS if task == "FB" else len(LABELS[task])
        row += [float(position == i) for i in range(width)]
        rows.append(row)
    return rows


def labels_of(question: dict[str, Any]) -> list[int]:
    gold = {str(label) for label in question["answer"]}
    return [int(label in gold) for label in candidates(question)]


def decide(question: dict[str, Any], probabilities: list[float]) -> list[Any]:
    task, labels = question["q_type"], candidates(question)
    best = max(range(len(labels)), key=lambda i: probabilities[i])
    if task in ("YN", "CO"):
        return [labels[best] if task == "YN" else int(labels[best])]
    if task == "FB":
        return [label for label, p in zip(labels, probabilities) if p >= 0.5]
    chosen = [int(label) for label, p in zip(labels, probabilities) if p >= 0.5 and label != "7"]
    return sorted(chosen) or [int(labels[best])]


class TypeModel:
    def __init__(self, c: float) -> None:
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=c, max_iter=5000)
        self.constant: float | None = None

    def fit(self, rows: list[list[float]], targets: list[int]) -> "TypeModel":
        if len(set(targets)) < 2:
            self.constant = float(targets[0]) if targets else 0.0
        else:
            self.model.fit(rows, targets)
        return self

    def probabilities(self, rows: list[list[float]]) -> list[float]:
        if self.constant is not None:
            return [self.constant] * len(rows)
        return [float(p) for p in self.model.predict_proba(rows)[:, 1]]


def fit_models(questions: list[tuple[str, dict]], sources: dict, names: list[str], c: float) -> dict[str, TypeModel]:
    models = {}
    for task in TASKS:
        rows, targets = [], []
        for key, q in questions:
            if q["q_type"] == task:
                rows += features(key, q, sources, names)
                targets += labels_of(q)
        if rows:
            models[task] = TypeModel(c).fit(rows, targets)
    return models


def stacked_answer(models: dict, key: str, question: dict, sources: dict, names: list[str], dataset: str,
                   options: PostprocessOptions, fallback: dict) -> list[Any]:
    model = models.get(question["q_type"])
    if model is None:
        return finalize_answer(question, None, None, dataset, options, fallback)
    probabilities = model.probabilities(features(key, question, sources, names))
    scores = {label: p for label, p in zip(candidates(question), probabilities)}
    return finalize_answer(question, decide(question, probabilities), scores, dataset, options, fallback)


def oof_stacked(questions, sources, names, dataset, options, fallback, c: float) -> dict[str, list[Any]]:
    fold_of = story_folds(questions)
    answers = {}
    for fold in sorted(set(fold_of.values())):
        train = [(k, q) for k, q in questions if fold_of[story_of(k)] != fold]
        models = fit_models(train, sources, names, c)
        for key, q in questions:
            if fold_of[story_of(key)] == fold:
                answers[key] = stacked_answer(models, key, q, sources, names, dataset, options, fallback)
    return answers


def nested_hybrid(questions, sources, names, dataset, options, fallback, c_grid: list[float]) -> dict[str, list[Any]]:
    """Per type, weighted or stacked (and C) chosen by an inner cv on each outer training part only."""
    fold_of = story_folds(questions)
    answers = {}
    for fold in sorted(set(fold_of.values())):
        train = [(k, q) for k, q in questions if fold_of[story_of(k)] != fold]
        held = [(k, q) for k, q in questions if fold_of[story_of(k)] == fold]
        candidates_ = {"weighted": per_type(train, oof_ensemble(train, sources, names, dataset, options, fallback))}
        for c in c_grid:
            candidates_[c] = per_type(train, oof_stacked(train, sources, names, dataset, options, fallback, c))
        choice = {t: max(candidates_, key=lambda name: (candidates_[name].get(t, 0), name == "weighted"))
                  for t in TASKS}
        configs = tune(train, sources, names, dataset, options, fallback, GRID, THRESHOLDS)
        models = {c: fit_models(train, sources, names, c) for c in set(choice.values()) if c != "weighted"}
        for key, q in held:
            method = choice[q["q_type"]]
            answers[key] = (predict(key, q, sources, configs[q["q_type"]], dataset, options, fallback)
                            if method == "weighted" else
                            stacked_answer(models[method], key, q, sources, names, dataset, options, fallback))
    return answers


def group_means(questions, answers) -> dict[str, float]:
    """Mean primary metric over the four types, within each question batch (A: reasoning_type given, B: not)."""
    means = {}
    for group in ("A", "B"):
        subset = [(k, q) for k, q in questions if group_of(q) == group]
        if subset:
            means[group] = mean(per_type(subset, answers))
    return means


def per_type(questions, answers) -> dict[str, float]:
    scores = {}
    for task in TASKS:
        subset = [(k, q) for k, q in questions if q["q_type"] == task]
        if subset:
            scores[task] = sum(answer_matches(task, answers[k], q["answer"]) for k, q in subset) / len(subset)
    return scores


def mean(scores: dict[str, float]) -> float:
    return sum(scores.values()) / max(len(scores), 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("Data/splits"))
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--split", default="trdev")
    parser.add_argument("--branches", default="F=F2C,F1 S=S L=LCOT P=LPX")
    parser.add_argument("--c-grid", default="0.1,1,10")
    parser.add_argument("--human-yn-dk", choices=("keep", "no"), default="keep")
    parser.add_argument("--human-fr-dk", choices=("keep", "avoid"), default="keep",
                        help="avoid: Human FR [7] becomes the relations some source scored.")
    parser.add_argument("--auto-submission", type=Path, help="Copied next to the new Human file for a complete pair.")
    parser.add_argument("--auto-cv", type=float, default=0.9873, help="Auto cv, only to print Final = (Human + Auto) / 2.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stacking"))
    args = parser.parse_args(argv)

    options = PostprocessOptions(human_yn_dk=args.human_yn_dk, human_fr_dk=args.human_fr_dk)
    setup = load_setup(args.pred_dir, args.splits_dir, args.split, args.branches)
    questions, sources, names = setup["questions"], setup["sources"], setup["names"]
    dataset, fallback = setup["dataset"], setup["fallback"]
    print(f"{len(questions)} labeled questions; sources {names}")

    c_grid = [float(v) for v in args.c_grid.split(",")]
    all_answers = {"weighted": oof_ensemble(questions, sources, names, dataset, options, fallback)}
    for c in c_grid:
        all_answers[f"stack C={c:g}"] = oof_stacked(questions, sources, names, dataset, options, fallback, c)
    all_answers["hybrid nested"] = nested_hybrid(questions, sources, names, dataset, options, fallback, c_grid)
    results = {name: per_type(questions, answers) for name, answers in all_answers.items()}
    groups = {name: group_means(questions, answers) for name, answers in all_answers.items()}
    weighted = results["weighted"]
    best_c_name = max((n for n in results if n.startswith("stack C")), key=lambda n: mean(results[n]))
    best_c = float(best_c_name.split("=")[1])
    hybrid_choice = {t: "stack" if results[best_c_name][t] > weighted[t] else "weighted" for t in weighted}
    results["hybrid (chọn trên cùng fold, lạc quan)"] = {t: max(results[best_c_name][t], weighted[t]) for t in weighted}

    lines = ["| Cách gộp | " + " | ".join(TASKS) + " | **Human cv** | Nhóm A | Nhóm B | Final cv |",
             "|---|" + "---|" * (len(TASKS) + 4)]
    for name, scores in results.items():
        group = groups.get(name, {})
        lines.append(f"| {name} | " + " | ".join(f"{scores.get(t, 0):.3f}" for t in TASKS)
                     + f" | **{mean(scores):.4f}** | {group.get('A', float('nan')):.3f} | {group.get('B', float('nan')):.3f}"
                     + f" | {(mean(scores) + args.auto_cv) / 2:.4f} |")
    report = "\n".join(lines) + (f"\n\nC tốt nhất: {best_c:g}. Hybrid chọn theo loại câu: {hybrid_choice} "
                                  "(lạc quan hơn một chút vì chọn trên cùng các fold).\n")
    print(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.md").write_text("# Stacking so với ensemble trọng số (cv theo story)\n\n" + report,
                                              encoding="utf-8")
    write_json(args.output_dir / "results.json", {"results": results, "groups": groups, "best_c": best_c,
                                                  "hybrid": hybrid_choice,
                                                  "sources": names, "metric": PRIMARY_METRIC})

    # Refit on the whole split and answer the public test
    test_data = read_json(args.data_dir / f"{dataset}_public_test.json")
    tests = setup["tests"]
    models = fit_models(questions, sources, names, best_c)
    configs = tune(questions, sources, names, dataset, options, fallback, GRID, THRESHOLDS)
    for variant in ("stack", "hybrid"):
        answers = {}
        for key, _, q in iter_questions(test_data):
            use_stack = variant == "stack" or hybrid_choice.get(q["q_type"]) == "stack"
            answers[key] = (stacked_answer(models, key, q, tests, names, dataset, options, fallback) if use_stack
                            else predict(key, q, tests, configs[q["q_type"]], dataset, options, fallback))
        submission = restore_field_order(fill_answers(test_data, answers), test_data)
        problems = compare_structure(submission, test_data) + [str(p) for p in answer_problems(submission)]
        out_dir = args.output_dir / variant
        write_json(out_dir / "human_submission.json", submission)
        if args.auto_submission and args.auto_submission.exists():
            shutil.copy2(args.auto_submission, out_dir / "auto_submission.json")
        print(f"{variant}: {out_dir / 'human_submission.json'} {'OK' if not problems else f'INVALID {problems[:3]}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
