"""Label voting for self-consistency samples and weighted source ensembles."""

from __future__ import annotations

from collections import Counter
from typing import Any

from spartqa.postprocess import EMPTY


def _label(task: str, key: str) -> Any:
    return int(key) if task in {"FR", "CO"} else key


def vote_samples(task: str, samples: list[list[Any] | None]) -> tuple[list[Any] | None, dict[str, float]]:
    """Majority vote over sampled answers; unparseable (None) samples are ignored.

    YN/CO keep the most frequent answer. FR/FB keep every label present in at
    least half of the valid samples. Scores are vote shares.
    """
    valid = [sample for sample in samples if sample is not None]
    if not valid:
        return None, {}
    if task in {"YN", "CO"}:
        counts = Counter(sample[0] for sample in valid)
        scores = {str(label): count / len(valid) for label, count in counts.items()}
        return [counts.most_common(1)[0][0]], scores
    counts = Counter(label for sample in valid for label in set(sample))
    scores = {str(label): count / len(valid) for label, count in counts.items()}
    if task == "FB":
        scores[EMPTY] = sum(not sample for sample in valid) / len(valid)
    return decide(task, scores, 0.5), scores


def combine_scores(task: str, sources: list[tuple[float, dict[str, Any] | None]]) -> dict[str, float] | None:
    """Weighted mean of per-label scores over sources that answered; None if none did."""
    total, weight_sum = Counter(), 0.0
    for weight, record in sources:
        if weight <= 0 or record is None or record.get("answer") is None:
            continue
        weight_sum += weight
        for label, score in (record.get("scores") or {}).items():
            total[label] += weight * float(score)
    if weight_sum == 0:
        return None
    return {label: value / weight_sum for label, value in total.items()}


def decide(task: str, scores: dict[str, float], threshold: float) -> list[Any] | None:
    """Answer from label scores; None for YN/CO without any scored label."""
    labels = [label for label in scores if label != EMPTY]
    if task in {"YN", "CO"}:
        if not labels:
            return None
        best = max(labels, key=lambda label: scores[label])
        return [_label(task, best)]
    chosen = [_label(task, label) for label in labels if scores[label] >= threshold]
    if task == "FR" and not chosen and labels:
        chosen = [_label(task, max(labels, key=lambda label: scores[label]))]
    return sorted(chosen, key=str) if task == "FB" else sorted(chosen)
