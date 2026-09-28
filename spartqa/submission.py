"""Submission-file checks: structure preservation, answer formats, and fallbacks."""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any

from spartqa.data import TASKS, validate_answer

_MISSING = object()


def compare_structure(prediction: Any, reference: dict[str, Any]) -> list[str]:
    """List every difference from the reference file other than answer fields."""
    if not isinstance(prediction, dict) or not isinstance(prediction.get("data"), list):
        return ["prediction must be a JSON object with a data list"]
    problems = [
        f"top-level field {key!r} differs"
        for key in sorted(set(prediction) | set(reference))
        if key != "data" and prediction.get(key, _MISSING) != reference.get(key, _MISSING)
    ]
    if len(prediction["data"]) != len(reference["data"]):
        return problems + [f"expected {len(reference['data'])} stories, found {len(prediction['data'])}"]
    for story_index, (story, expected) in enumerate(zip(prediction["data"], reference["data"])):
        if not isinstance(story, dict) or not isinstance(story.get("questions"), list):
            problems.append(f"story {story_index}: missing questions list")
            continue
        for key in sorted((set(story) | set(expected)) - {"questions"}):
            if story.get(key, _MISSING) != expected.get(key, _MISSING):
                problems.append(f"story {story_index}: field {key!r} differs")
        if len(story["questions"]) != len(expected["questions"]):
            problems.append(f"story {story_index}: expected {len(expected['questions'])} questions, "
                            f"found {len(story['questions'])}")
            continue
        for question_index, (question, gold) in enumerate(zip(story["questions"], expected["questions"])):
            if not isinstance(question, dict):
                problems.append(f"story {story_index} question {question_index}: not an object")
                continue
            changed = sorted(
                key for key in (set(question) | set(gold)) - {"answer"}
                if question.get(key, _MISSING) != gold.get(key, _MISSING)
            )
            if changed:
                problems.append(f"story {story_index} question {question_index}: fields {changed} differ")
    return problems


def answer_problem(question: dict[str, Any]) -> str | None:
    if "answer" not in question:
        return "missing answer"
    try:
        validate_answer(question["answer"], question)
    except (ValueError, KeyError, TypeError) as error:
        return str(error) or type(error).__name__
    return None


def answer_problems(prediction: dict[str, Any]) -> list[tuple[int, int, str]]:
    return [
        (story_index, question_index, problem)
        for story_index, story in enumerate(prediction["data"])
        for question_index, question in enumerate(story["questions"])
        if (problem := answer_problem(question)) is not None
    ]


def majority_answers(train: dict[str, Any]) -> dict[str, list[Any]]:
    """Most frequent gold answer per question type, used only to fill gaps."""
    counts: dict[str, Counter] = {task: Counter() for task in TASKS}
    for story in train["data"]:
        for question in story["questions"]:
            if "answer" in question:
                counts[question["q_type"]][json.dumps(question["answer"])] += 1
    defaults = {"YN": ["Yes"], "FR": [7], "FB": [], "CO": [3]}
    return {task: json.loads(counts[task].most_common(1)[0][0]) if counts[task] else defaults[task]
            for task in TASKS}


def fill_invalid_answers(
    prediction: dict[str, Any], fallback: dict[str, list[Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Replace missing or invalid answers; valid predictions are left unchanged."""
    output = copy.deepcopy(prediction)
    filled = []
    for story_index, question_index, _ in answer_problems(output):
        question = output["data"][story_index]["questions"][question_index]
        answer = list(fallback[question["q_type"]])
        if question["q_type"] == "FB":
            answer = [label for label in answer if label in question["candidate_answers"]]
        question["answer"] = answer
        filled.append(f"{story_index}:{question_index}")
    return output, filled
