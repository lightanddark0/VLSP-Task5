"""Intermediate prediction files shared by every branch (F, L-CoT, ensemble).

One JSONL line per question:
    {"key": "12_3", "q_type": "FR", "answer": [2, 5], "scores": {"2": 1.0, "5": 1.0}, "source": "F1"}

``key`` is ``<story index in the file>_<q_id>`` because q_id is unique only
within a story. ``answer`` is null when a branch abstains. ``scores`` holds a
0..1 score per label (FB also has "__EMPTY__"). Files are appended in chunks,
so an interrupted run resumes by skipping keys already written.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterator

from spartqa.postprocess import EMPTY


def question_key(story_index: int, q_id: Any) -> str:
    return f"{story_index}_{q_id}"


def iter_questions(data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Yield (key, story item, question) in file order."""
    for story_index, item in enumerate(data["data"]):
        seen = set()
        for question in item["questions"]:
            key = question_key(story_index, question["q_id"])
            if key in seen:
                raise ValueError(f"Duplicate q_id {question['q_id']} in story {story_index}")
            seen.add(key)
            yield key, item, question


def payload_of(item: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    return {"story": item["story"], "question": question["question"], "q_type": question["q_type"],
            "candidate_answers": question["candidate_answers"]}


def one_hot_scores(task: str, answer: list[Any] | None) -> dict[str, float]:
    if answer is None:
        return {}
    scores = {str(label): 1.0 for label in answer}
    if task == "FB" and not answer:
        scores[EMPTY] = 1.0
    return scores


def read_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Records by key; a later line for the same key replaces an earlier one."""
    records: dict[str, dict[str, Any]] = {}
    path = Path(path)
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave a partial last line; everything before it is kept.
                print(f"Warning: skipping unreadable line {line_number} of {path}")
                continue
            records[record["key"]] = record
    return records


class PredictionWriter:
    """Append-only JSONL writer; every ``write_many`` call is flushed to disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_many(self, records: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


def fill_answers(reference: dict[str, Any], answers: dict[str, list[Any]]) -> dict[str, Any]:
    """Copy of ``reference`` whose answer fields are replaced by ``answers`` (by key).

    Gold answers present in the reference are removed first, so a question
    without a prediction ends up without an answer and is reported invalid.
    """
    output = copy.deepcopy(reference)
    for key, _, question in iter_questions(output):
        question.pop("answer", None)
        if key in answers and answers[key] is not None:
            question["answer"] = answers[key]
    return output


def restore_field_order(output: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Put ``answer`` back where the reference file had it, for readable diffs."""
    for (_, _, question), (_, _, original) in zip(iter_questions(output), iter_questions(reference)):
        if "answer" in original and "answer" in question:
            ordered = {key: question[key] for key in original if key in question}
            ordered.update({key: value for key, value in question.items() if key not in ordered})
            question.clear()
            question.update(ordered)
    return output
