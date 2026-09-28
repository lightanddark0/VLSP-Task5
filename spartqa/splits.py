"""Story-level train/dev splits that keep every question of a story together."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any

from spartqa.data import TASKS


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def story_features(story: dict[str, Any]) -> Counter:
    """Question-type and coarse answer-label counts used to balance a split."""
    features: Counter = Counter()
    for question in story["questions"]:
        task = question["q_type"]
        features[task] += 1
        answer = question.get("answer")
        if answer is None:
            continue
        if task in {"YN", "CO"}:
            features[f"{task}={answer[0]}"] += 1
        else:
            features[f"{task}/size={len(answer)}"] += 1
    return features


def split_stories(
    data: dict[str, Any], dev_ratio: float = 0.2, dev_stories: int | None = None,
    seed: int = 42, trials: int = 100,
) -> tuple[list[int], list[int]]:
    """Return sorted (train, dev) story indices.

    Each trial is a seeded random shuffle; the dev set whose question-type and
    label distribution is closest to the whole file is kept. The result depends
    only on the data, seed, size, and number of trials.
    """
    count = len(data["data"])
    if count < 2:
        raise ValueError("At least two stories are needed for a train/dev split")
    if trials < 1:
        raise ValueError("trials must be positive")
    size = dev_stories if dev_stories is not None else round(count * dev_ratio)
    size = min(max(size, 1), count - 1)
    features = [story_features(story) for story in data["data"]]
    total: Counter = Counter()
    for story in features:
        total.update(story)
    total_questions = sum(total[task] for task in TASKS) or 1
    target = {key: value / total_questions for key, value in total.items()}

    rng = random.Random(seed)
    order = list(range(count))
    best: tuple[float, list[int]] | None = None
    for _ in range(trials):
        rng.shuffle(order)
        counts: Counter = Counter()
        for index in order[:size]:
            counts.update(features[index])
        dev_questions = sum(counts[task] for task in TASKS) or 1
        distance = sum(abs(counts[key] / dev_questions - share) for key, share in target.items())
        if best is None or distance < best[0]:
            best = (distance, sorted(order[:size]))
    dev = best[1]
    dev_set = set(dev)
    return [index for index in range(count) if index not in dev_set], dev


def subset_data(data: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    return {**{key: value for key, value in data.items() if key != "data"},
            "data": [data["data"][index] for index in indices]}


def split_summary(data: dict[str, Any]) -> dict[str, Any]:
    tasks = Counter(question["q_type"] for story in data["data"] for question in story["questions"])
    return {"stories": len(data["data"]), "questions": sum(tasks.values()),
            "by_task": {task: tasks[task] for task in TASKS}}
