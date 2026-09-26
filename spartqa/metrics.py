"""Per-question scoring shared by API and encoder experiments."""

from __future__ import annotations

from typing import Any, Iterable


def set_jaccard(prediction: Iterable[Any], gold: Iterable[Any]) -> float:
    pred_set = set(prediction)
    gold_set = set(gold)
    union = pred_set | gold_set
    return len(pred_set & gold_set) / len(union) if union else 1.0


def answer_matches(task: str, prediction: list[Any], gold: list[Any]) -> bool:
    if task in {"YN", "CO"}:
        return prediction == gold
    return set(prediction) == set(gold)


def update_metric_state(state: dict[str, dict[str, float]], task: str, prediction: list[Any], gold: list[Any]) -> None:
    task_state = state.setdefault(task, {"count": 0.0, "correct": 0.0, "jaccard": 0.0})
    task_state["count"] += 1.0
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