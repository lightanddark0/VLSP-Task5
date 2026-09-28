"""Prompt text and answer parsing for generative (LLM) fine-tuning.

The target is the gold answer serialized exactly as it appears in the task
JSON, so a parsed generation can be written into a submission unchanged.
Metadata fields such as reasoning_type and indifinite are never included.
"""

from __future__ import annotations

import json
from typing import Any

from spartqa.data import validate_answer

ANSWER_FORMATS = {
    "YN": '["Yes"], ["No"] or ["DK"]',
    "FR": "JSON list of all option indices that hold; [7] alone if none can be determined",
    "FB": "JSON list of all qualifying block names; [] if none",
    "CO": "[0] first object, [1] second object, [2] both, [3] neither",
}


def story_text(story: str | list[str]) -> str:
    if isinstance(story, list):
        return " ".join(part.strip() for part in story if part.strip())
    return story.strip()


def build_prompt(payload: dict[str, Any], dataset: str) -> str:
    task = payload["q_type"]
    candidates = payload["candidate_answers"]
    lines = [
        f"### Dataset: {dataset}",
        "### Story:",
        story_text(payload["story"]),
        f"### Question type: {task}",
        "### Question:",
        payload["question"].strip(),
    ]
    if task in {"FR", "CO"}:
        lines += ["### Options:", *(f"{index}: {text}" for index, text in enumerate(candidates))]
    elif task == "FB":
        lines += ["### Options:", ", ".join(str(block) for block in candidates)]
    lines += [f"### Answer format: {ANSWER_FORMATS[task]}", "### Answer:"]
    return "\n".join(lines)


def canonical_answer(answer: list[Any], payload: dict[str, Any]) -> list[Any]:
    if payload["q_type"] == "FR":
        return sorted(answer)
    if payload["q_type"] == "FB":
        order = {block: index for index, block in enumerate(payload["candidate_answers"])}
        return sorted(answer, key=lambda block: order.get(block, len(order)))
    return list(answer)


def format_answer(answer: list[Any], payload: dict[str, Any]) -> str:
    return json.dumps(canonical_answer(answer, payload), ensure_ascii=False)


def parse_answer(text: str, payload: dict[str, Any]) -> list[Any] | None:
    """Return a valid answer from the first line of a generation, or None."""
    lines = text.strip().splitlines()
    if not lines:
        return None
    try:
        answer = json.loads(lines[0].strip())
        validate_answer(answer, payload)
    except (ValueError, KeyError, TypeError):
        return None
    return canonical_answer(answer, payload)
