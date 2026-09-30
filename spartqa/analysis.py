"""Shared pieces of the Human error analysis and the stacking ensemble (steps 1 and 2 after E4).

Both work on a labeled tuning split (normally human_trdev: 613 questions, made
by crossfit.py combine) where every source has out-of-sample predictions.
Folds are the same story folds as compare_branches.cross_validated, so the
numbers are directly comparable with its "cv" column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compare_branches import parse_branches, priority_of, tune
from ensemble import predict
from spartqa.data import read_json
from spartqa.inference import fallback_answers
from spartqa.postprocess import PostprocessOptions, finalize_answer
from spartqa.predictions import iter_questions, read_predictions

GRID = [0.0, 1.0, 2.0]
THRESHOLDS = [0.4, 0.5, 0.6]


def group_of(question: dict[str, Any]) -> str:
    """Human questions come in two batches: "A" with reasoning_type annotated, "B" without (only for reporting)."""
    kinds = question.get("reasoning_type") or []
    return "B" if not kinds or kinds == ["None"] else "A"


def story_of(key: str) -> int:
    return int(key.split("_")[0])


def story_folds(questions: list[tuple[str, dict]], folds: int = 5) -> dict[int, int]:
    """Story -> fold, exactly as compare_branches.cross_validated assigns them."""
    stories = sorted({story_of(key) for key, _ in questions})
    folds = max(2, min(folds, len(stories)))
    return {story: index % folds for index, story in enumerate(stories)}


def load_setup(pred_dir: Path, splits_dir: Path, split: str, branches: str, dataset: str = "human",
               test_split: str = "test") -> dict[str, Any]:
    """Labeled questions, gold data, and every available source's records for ``split`` (and test)."""
    data = read_json(splits_dir / f"{dataset}_{split}.json")
    names = [s for s in priority_of(parse_branches(branches))
             if (pred_dir / s / f"{dataset}_{split}.jsonl").exists()]
    sources = {s: read_predictions(pred_dir / s / f"{dataset}_{split}.jsonl") for s in names}
    tests = {s: read_predictions(pred_dir / s / f"{dataset}_{test_split}.jsonl") for s in names
             if (pred_dir / s / f"{dataset}_{test_split}.jsonl").exists()}
    questions = [(key, q) for key, _, q in iter_questions(data) if "answer" in q]
    stories = {story_of(key): item for key, item, _ in iter_questions(data)}
    return {"data": data, "questions": questions, "names": names, "sources": sources, "tests": tests,
            "stories": stories, "dataset": dataset, "fallback": fallback_answers(splits_dir, dataset)}


def oof_ensemble(questions: list[tuple[str, dict]], sources: dict, names: list[str], dataset: str,
                 options: PostprocessOptions, fallback: dict, folds: int = 5) -> dict[str, list[Any]]:
    """Answer of the weighted ensemble for every question, tuned on the other story folds."""
    fold_of = story_folds(questions, folds)
    answers = {}
    for fold in sorted(set(fold_of.values())):
        train = [(k, q) for k, q in questions if fold_of[story_of(k)] != fold]
        held = [(k, q) for k, q in questions if fold_of[story_of(k)] == fold]
        configs = tune(train, sources, names, dataset, options, fallback, GRID, THRESHOLDS)
        for key, question in held:
            answers[key] = predict(key, question, sources, configs[question["q_type"]], dataset, options, fallback)
    return answers


def source_answer(record: dict[str, Any] | None, question: dict[str, Any], dataset: str,
                  options: PostprocessOptions, fallback: dict) -> list[Any] | None:
    """One source's post-processed answer, or None when it abstained."""
    if record is None or record.get("answer") is None:
        return None
    return finalize_answer(question, record["answer"], record.get("scores"), dataset, options, fallback)
