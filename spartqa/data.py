"""SPARTQA JSON input and answer contracts, independent of model libraries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TASKS = ("YN", "FR", "FB", "CO")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def answer_choices(payload: dict[str, Any]) -> list[Any]:
    return {
        "YN": ["Yes", "No", "DK"],
        "FR": list(range(8)),
        "FB": payload["candidate_answers"],
        "CO": list(range(4)),
    }[payload["q_type"]]


def validate_answer(answer: Any, payload: dict[str, Any]) -> None:
    task = payload["q_type"]
    choices = answer_choices(payload)
    expected_type = int if task in {"FR", "CO"} else str
    if not isinstance(answer, list):
        raise ValueError("answer must be a list")
    if any(type(value) is not expected_type or value not in choices for value in answer):
        raise ValueError(f"Invalid answer labels for {task}")
    if len(answer) != len(set(answer)):
        raise ValueError("Duplicate answer labels")
    if task in {"YN", "CO"} and len(answer) != 1:
        raise ValueError(f"{task} requires exactly one label")
    if task == "FR" and (not answer or (7 in answer and len(answer) != 1)):
        raise ValueError("FR requires known relations or [7] alone")


def examples_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    examples = []
    for story_index, item in enumerate(data["data"]):
        for question_index, question in enumerate(item["questions"]):
            if question["q_type"] not in TASKS:
                raise ValueError(f"Unsupported task at {story_index}:{question_index}")
            example = {
                "key": f"{story_index}:{question_index}",
                "story_index": story_index,
                "question_index": question_index,
                "payload": {
                    "story": item["story"],
                    "question": question["question"],
                    "q_type": question["q_type"],
                    "candidate_answers": question["candidate_answers"],
                },
                "gold": question.get("answer"),
            }
            if example["gold"] is not None:
                validate_answer(example["gold"], example["payload"])
            examples.append(example)
    return examples