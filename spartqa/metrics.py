"""Per-question scoring shared by API and encoder experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

from spartqa.data import TASKS, validate_answer

PRIMARY_METRIC = {"YN": "accuracy", "CO": "accuracy", "FR": "exact_match", "FB": "exact_match"}


def set_jaccard(prediction: Iterable[Any], gold: Iterable[Any]) -> float:
    pred_set = set(prediction)
    gold_set = set(gold)
    union = pred_set | gold_set
    return len(pred_set & gold_set) / len(union) if union else 1.0


def answer_matches(task: str, prediction: list[Any], gold: list[Any]) -> bool:
    if task in {"YN", "CO"}:
        return prediction == gold
    return set(prediction) == set(gold)


def update_metric_state(
    state: dict[str, dict[str, float]], task: str, prediction: list[Any] | None, gold: list[Any],
) -> None:
    """Add one question; a None prediction (missing or invalid) scores zero."""
    task_state = state.setdefault(task, {"count": 0.0, "correct": 0.0, "jaccard": 0.0})
    task_state["count"] += 1.0
    if prediction is None:
        return
    task_state["correct"] += float(answer_matches(task, prediction, gold))
    if task in {"FR", "FB"}:
        task_state["jaccard"] += set_jaccard(prediction, gold)


def finalize_metrics(state: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for task, values in state.items():
        count = max(values["count"], 1.0)
        if task in {"YN", "CO"}:
            metrics[task] = {"accuracy": values["correct"] / count, "count": values["count"]}
        else:
            metrics[task] = {
                "exact_match": values["correct"] / count,
                "jaccard": values["jaccard"] / count,
                "count": values["count"],
            }
    return metrics


def primary_macro(metrics: dict[str, dict[str, float]]) -> float:
    """Unofficial single number for model selection: mean primary metric over tasks."""
    scores = [metrics[task][PRIMARY_METRIC[task]] for task in TASKS if task in metrics]
    return sum(scores) / len(scores) if scores else 0.0


def question_records(prediction: dict[str, Any], gold: dict[str, Any]) -> list[dict[str, Any]]:
    """Pair questions of structurally identical files; invalid predictions become None."""
    records = []
    for story_index, (story, gold_story) in enumerate(zip(prediction["data"], gold["data"])):
        for question, gold_question in zip(story["questions"], gold_story["questions"]):
            if "answer" not in gold_question:
                raise ValueError(f"Gold file has no answer in story {story_index}")
            answer = question.get("answer")
            try:
                validate_answer(answer, gold_question)
            except (ValueError, KeyError, TypeError):
                answer = None
            records.append({
                "story_index": story_index,
                "task": gold_question["q_type"],
                "reasoning_types": list(gold_question.get("reasoning_type") or []),
                "prediction": answer,
                "gold": gold_question["answer"],
            })
    return records


def _task_metrics(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    state: dict[str, dict[str, float]] = {}
    for record in records:
        update_metric_state(state, record["task"], record["prediction"], record["gold"])
    return finalize_metrics(state)


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for reasoning_type in record["reasoning_types"] or ["(none)"]:
            by_type[reasoning_type].append(record)
    metrics = _task_metrics(records)
    return {
        "questions": len(records),
        "answered_valid": sum(record["prediction"] is not None for record in records),
        "by_task": metrics,
        "primary_macro_unofficial": primary_macro(metrics),
        "by_reasoning_type": {name: _task_metrics(group) for name, group in sorted(by_type.items())},
    }


def bootstrap_intervals(
    records: list[dict[str, Any]], samples: int = 1000, seed: int = 42, level: float = 0.95,
) -> dict[str, list[float]]:
    """Percentile intervals from resampling whole stories with replacement."""
    stories: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        stories[record["story_index"]].append(record)
    groups = list(stories.values())
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sample = [record for _ in groups for record in rng.choice(groups)]
        metrics = _task_metrics(sample)
        for task, values in metrics.items():
            draws[task].append(values[PRIMARY_METRIC[task]])
        draws["primary_macro_unofficial"].append(primary_macro(metrics))
    tail = (1.0 - level) / 2
    intervals = {}
    for name, values in draws.items():
        values.sort()
        intervals[name] = [values[int(tail * (len(values) - 1))], values[int((1 - tail) * (len(values) - 1))]]
    return intervals


def final_scores(human: dict[str, Any], auto: dict[str, Any]) -> dict[str, Any]:
    """Official averaging: each metric is (Human + Auto) / 2, per question type."""
    by_task = {}
    for task in TASKS:
        if task in human["by_task"] and task in auto["by_task"]:
            by_task[task] = {
                metric: (value + auto["by_task"][task][metric]) / 2
                for metric, value in human["by_task"][task].items() if metric != "count"
            }
    return {
        "by_task": by_task,
        "primary_macro_unofficial": (human["primary_macro_unofficial"] + auto["primary_macro_unofficial"]) / 2,
    }
