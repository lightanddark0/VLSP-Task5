"""Converse data augmentation for the human-written training questions (experiment E2).

FR  "Mối quan hệ giữa X và Y là gì?" -> "... giữa Y và X ..." with converse labels
    (left <-> right, above <-> below; near, far, touch, DK unchanged).
YN  "X có (ở|nằm) <rel> Y không?" -> "Y có (ở|nằm) <converse rel> X không?", same answer.
    Only when Y is a definite description (no "một", no quantifier) and the
    relation is a single phrase, because swapping "một Y" changes the meaning.
Questions that do not match these shapes are not augmented.
"""

from __future__ import annotations

import copy
import re
from typing import Any

FR_CONVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7}
YN_CONVERSE = {"bên trái": "bên phải", "bên phải": "bên trái", "phía trên": "phía dưới", "phía dưới": "phía trên",
               "bên trên": "bên dưới", "bên dưới": "bên trên", "trên": "dưới", "dưới": "trên",
               "gần": "gần", "xa": "xa", "chạm vào": "chạm vào"}
_FR = re.compile(r"^(?P<head>(?:Có\s+)?[Mm]ối\s+(?:quan|liên)\s+hệ\s+(?:nào\s+)?giữa)\s+(?P<x>.+?)\s+và\s+(?P<y>.+?)"
                 r"(?P<tail>\s*(?:là\s+gì)?\s*\??)$")
_YN_RELATIONS = "|".join(sorted(map(re.escape, YN_CONVERSE), key=len, reverse=True))
_YN = re.compile(rf"^(?P<x>.+?)\s+có\s+(?P<verb>(?:ở|nằm)\s+)?(?P<rel>{_YN_RELATIONS})\s+(?P<y>.+?)\s+không\s*\?$")
_BLOCKED = re.compile(r"\b(tất cả|mọi|nào|bất kỳ|đều|có phải|những|các)\b", re.IGNORECASE)
_RELATION_START = re.compile(r"^(phía|bên|trên|dưới|gần|xa|chạm)\b")


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _lower_first(text: str) -> str:
    # Keep block names and similar capitalised tokens ("A", "B") as they are.
    return text if len(text) > 1 and text[1:2].isupper() else text[:1].lower() + text[1:]


def converse_fr(question: dict[str, Any]) -> dict[str, Any] | None:
    match = _FR.match(re.sub(r"\s+", " ", question["question"].strip()))
    if match is None or " và " in match["x"] or " và " in match["y"]:
        return None
    new = copy.deepcopy(question)
    new["question"] = f"{match['head']} {match['y']} và {match['x']}{match['tail']}"
    new["answer"] = sorted(FR_CONVERSE[label] for label in question["answer"])
    return new


def converse_yn(question: dict[str, Any]) -> dict[str, Any] | None:
    text = re.sub(r"\s+", " ", question["question"].strip())
    if _BLOCKED.search(text):
        return None
    match = _YN.match(text)
    if match is None:
        return None
    x, y = match["x"], match["y"]
    if (y.lower().startswith(("một ", "và ", "của ", "với ")) or _RELATION_START.match(y.lower())
            or " và " in x or " và " in y):
        return None
    new = copy.deepcopy(question)
    verb = match["verb"] or ""
    new["question"] = f"{_capitalize(y)} có {verb}{YN_CONVERSE[match['rel']]} {_lower_first(x)} không?"
    return new


def converse_questions(questions: list[dict[str, Any]], kinds: set[str]) -> list[dict[str, Any]]:
    """Converse copies of the questions whose type is in ``kinds`` ({"FR", "YN"}) and whose shape is safe."""
    made = []
    for question in questions:
        if question["q_type"] == "FR" and "FR" in kinds:
            new = converse_fr(question)
        elif question["q_type"] == "YN" and "YN" in kinds:
            new = converse_yn(question)
        else:
            new = None
        if new is not None:
            made.append(new)
    return made
