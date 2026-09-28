"""Vocabulary of the generated Vietnamese stories and questions, as token sequences."""

from __future__ import annotations

import re
import unicodedata

# Longer phrases first: matching takes the first entry that fits.
SHAPES = [
    (("hình", "vuông"), "square"), (("hình", "tròn"), "circle"), (("hình", "tam", "giác"), "triangle"),
    (("vật", "thể"), None), (("hình",), None), (("vật",), None),
]
SIZES = [(("trung", "bình"), "medium"), (("nhỏ",), "small"), (("vừa",), "medium"), (("lớn",), "large")]
COLORS = [
    (("màu", "xanh", "lam"), "blue"), (("màu", "xanh", "dương"), "blue"), (("màu", "xanh"), "blue"),
    (("màu", "vàng"), "yellow"), (("màu", "đen"), "black"),
]
RELATIONS = [
    (("phía", "bên", "trái"), "LEFT"), (("phía", "bên", "phải"), "RIGHT"),
    (("bên", "trái"), "LEFT"), (("bên", "phải"), "RIGHT"),
    (("phía", "trên"), "ABOVE"), (("bên", "trên"), "ABOVE"),
    (("phía", "dưới"), "BELOW"), (("bên", "dưới"), "BELOW"),
    (("gần",), "NEAR"), (("xa",), "FAR"),
]
TOUCH = ("chạm", "vào")
EDGES = {"trên": "top", "dưới": "bottom", "trái": "left", "phải": "right"}
QUANTIFIERS = [(("tất", "cả"), "all"), (("bất", "kỳ"), "any"), (("một",), "one"), (("các",), "all")]
COUNTS = {"hai": 2, "ba": 3, "bốn": 4}
PRONOUNS = [("cái", "này"), ("cái", "đó"), ("nó",)]

# FR answer indices.
FR_INDEX = {"LEFT": 0, "RIGHT": 1, "ABOVE": 2, "BELOW": 3, "NEAR": 4, "FAR": 5, "TOUCH": 6}
CONVERSE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "ABOVE": "BELOW", "BELOW": "ABOVE",
            "NEAR": "NEAR", "FAR": "FAR", "TOUCH": "TOUCH"}
OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "ABOVE": "BELOW", "BELOW": "ABOVE", "NEAR": "FAR", "FAR": "NEAR"}
DIRECTIONS = ("LEFT", "RIGHT", "ABOVE", "BELOW")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercased words; commas and colons become their own tokens, final punctuation is dropped."""
    text = normalize(text)
    text = re.sub(r"[.?!]+$", "", text)
    text = text.replace(",", " , ").replace(":", " : ")
    return text.split()


def split_sentences(story: str) -> list[str]:
    story = unicodedata.normalize("NFC", story)
    parts = re.split(r"(?<=[.!?])\s+", story.strip())
    return [part.strip() for part in parts if part.strip()]


def block_names(sentence: str) -> list[str]:
    """Single capital letters used as block names in the original-case sentence."""
    return re.findall(r"\b([A-Z])\b", unicodedata.normalize("NFC", sentence))
