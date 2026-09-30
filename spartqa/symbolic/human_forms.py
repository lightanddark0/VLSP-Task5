"""B1: rule-based logical forms for the common Human question shapes (no LLM).

Human questions are written freely but follow a few shapes: "Mối quan hệ giữa
X và Y là gì?", "Khối nào (không) có X?", "Vật nào (không) ở <rel> Y, A hay B?",
"X có <rel> Y không?", "Có phải tất cả X đều <rel> Y?". Each shape is parsed
into the JSON form of spartqa.symbolic.structured; anything unrecognised
returns None and the caller falls back to the LLM forms. Every noun phrase must
be consumed completely, so a parse is either exact or refused.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Phrase tables: longer phrases first.
DETERMINERS = [("tất", "cả", "các"), ("tất", "cả"), ("bất", "kỳ"), ("những",), ("các",), ("một",), ("cái",),
               ("hai",)]
HEADS = [
    (("hình", "tam", "giác"), "triangle"), (("hình", "vuông"), "square"), (("hình", "tròn"), "circle"),
    (("hình", "chữ", "nhật"), "rectangle"), (("tam", "giác"), "triangle"),
    (("vật", "thể"), None), (("đồ", "vật"), None), (("vật",), None), (("thứ",), None), (("hình",), None),
]
SIZES = [(("có", "kích", "thước", "trung", "bình"), "medium"), (("kích", "thước", "trung", "bình"), "medium"),
         (("cỡ", "trung", "bình"), "medium"), (("cỡ", "nhỏ"), "small"), (("cỡ", "lớn"), "large"),
         (("trung", "bình"), "medium"), (("nhỏ",), "small"), (("vừa",), "medium"), (("lớn",), "large"), (("to",), "large")]
COLOR_WORDS = [
    (("xanh", "lá", "cây"), "green"), (("xanh", "lá"), "green"), (("xanh", "lam"), "blue"), (("xanh", "dương"), "blue"),
    (("xanh",), "blue"), (("vàng",), "yellow"), (("đen",), "black"), (("đỏ",), "red"), (("trắng",), "white"),
]
BLOCK_PREPS = [("nằm", "trong"), ("ở", "trong"), ("thuộc",), ("trong",), ("ở",)]
RELATION_WORDS = {"trái": "left", "phải": "right", "trên": "above", "dưới": "below", "gần": "near", "xa": "far"}
CONTACT_SIDES = {"đỉnh": ("touch", "above"), "đầu": ("touch", "above"), "trên": ("touch", "above"),
                 "đáy": ("touch", "below"), "dưới": ("touch", "below"),
                 "trái": ("touch", "left"), "phải": ("touch", "right")}
FILLERS = {"ở", "nằm", "về", "phía", "bên", "của", "và", ",", "có", "đều", "khỏi", "tới", "so", "với"}


def tokens_of(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[?.!]+$", "", text)
    return text.replace(",", " , ").replace(":", " : ").split()


def _match(tokens: list[str], i: int, table: list[tuple[tuple[str, ...], Any]]) -> tuple[Any, int] | None:
    for words, value in table:
        if tuple(tokens[i:i + len(words)]) == words:
            return value, i + len(words)
    return None


def noun_phrase(tokens: list[str], i: int) -> tuple[dict[str, Any], str, int] | None:
    """(description, quantifier "a"|"all", next index) for a noun phrase starting at i."""
    quant = "a"
    for det in DETERMINERS:
        if tuple(tokens[i:i + len(det)]) == det:
            quant = "all" if det[0] in ("tất", "những", "các") else "a"   # "hai X" (between) is "a"
            i += len(det)
            break
    head = _match(tokens, i, HEADS)
    if head is None:
        return None
    desc: dict[str, Any] = {}
    if head[0] is not None:
        desc["shape"] = head[0]
    i = head[1]
    while True:
        size = _match(tokens, i, SIZES)
        if size is not None and "size" not in desc:
            desc["size"], i = size
            continue
        start = i + 1 if tokens[i:i + 1] == ["màu"] else i
        color = _match(tokens, start, COLOR_WORDS)
        if color is not None and "color" not in desc:
            desc["color"], i = color
            continue
        break
    for prep in BLOCK_PREPS:
        j = i + len(prep)
        if tuple(tokens[i:j]) == prep:
            if tokens[j:j + 1] == ["khối"]:
                j += 1
            if j < len(tokens) and re.fullmatch(r"[a-h]", tokens[j]):
                desc["block"] = tokens[j].upper()
                i = j + 1
            break
    return desc, quant, i


def whole_phrase(tokens: list[str], nested: bool = True) -> tuple[dict[str, Any], str] | None:
    """A complete noun phrase, optionally with one relative clause: "X phía dưới (một) Y"."""
    parsed = noun_phrase(tokens, 0)
    if parsed is None:
        return None
    desc, quant, i = parsed
    if i == len(tokens):
        return desc, quant
    if not nested:
        return None
    rel = relation(tokens, i)
    if rel is None or rel[1]:
        return None
    target = whole_phrase(tokens[rel[2]:], nested=False)
    if target is None:
        return None
    return {**desc, "rels": [{"rel": rel[0], "quant": target[1], "target": target[0]}]}, quant


def relation(tokens: list[str], i: int) -> tuple[list[str], bool, int] | None:
    """(relations, negated, next index); stops where the target noun phrase starts."""
    rels: list[str] = []
    negated = False
    while i < len(tokens):
        word = tokens[i]
        if word == "không":
            negated, i = True, i + 1
        elif tuple(tokens[i:i + 2]) == ("chạm", "vào") or word == "chạm":
            i += 2 if tokens[i + 1:i + 2] == ["vào"] else 1
            side = None
            if tokens[i:i + 1] == ["cạnh"] and tokens[i + 1:i + 2] and tokens[i + 1] in CONTACT_SIDES:
                side, i = tokens[i + 1], i + 2
            elif tokens[i:i + 1] and tokens[i] in ("đỉnh", "đầu", "đáy"):
                side, i = tokens[i], i + 1
            rels += list(CONTACT_SIDES[side]) if side else ["touch"]
        elif word == "giữa":
            rels.append("between")
            i += 1
        elif word in RELATION_WORDS and not (word in ("trên", "dưới", "trái", "phải") and rels and rels[-1] == "touch"):
            rels.append(RELATION_WORDS[word])
            i += 1
        elif word in FILLERS:
            i += 1
        else:
            break
    rels = list(dict.fromkeys(rels))
    return (rels, negated, i) if rels else None


def _target(tokens: list[str]) -> dict[str, Any] | None:
    """A relation target "một X" / "tất cả các X" / "X khác"."""
    if tokens[-1:] in (["khác"], ["kia"]):
        tokens = tokens[:-1]
    if tokens[:2] == ["bất", "kỳ"]:
        return None
    parsed = whole_phrase(tokens)
    if parsed is None:
        return None
    desc, quant = parsed
    return {"desc": desc, "quant": quant}


def fr_form(tokens: list[str]) -> dict[str, Any] | None:
    text = " ".join(tokens)
    match = re.fullmatch(r"mối (?:quan|liên) hệ giữa (.+?)(?: là gì)?", text)
    if match:
        body = match.group(1).split()
        splits = []
        for index, word in enumerate(body):
            if word == "và":
                first, second = whole_phrase(body[:index]), whole_phrase(body[index + 1:])
                if first and second:
                    splits.append((first[0], second[0]))
        return {"first": splits[0][0], "second": splits[0][1]} if len(splits) == 1 else None
    match = re.fullmatch(r"(.+) ở đâu so với (.+)", text)
    if match:
        first, second = whole_phrase(match.group(1).split()), whole_phrase(match.group(2).split())
        if first and second:
            return {"first": first[0], "second": second[0]}
    return None


def fb_form(tokens: list[str]) -> dict[str, Any] | None:
    if tokens[:2] != ["khối", "nào"]:
        return None
    rest = tokens[2:]
    negated = rest[:1] == ["không"]
    rest = rest[1:] if negated else rest
    if not rest or rest[0] not in ("có", "chứa"):
        return None
    rest = rest[1:]
    while rest and rest[-1] in ("nó", "trong", "bên", "ở", "nào"):
        rest = rest[:-1]
    every = rest[:2] == ["tất", "cả"]
    parsed = noun_phrase(rest, 0)
    if parsed is None:
        return None
    desc, _, i = parsed
    if rest[i:i + 1] == ["nào"]:
        i += 1
    if i < len(rest):
        if rest[i:] in (["chạm", "vào", "cạnh"], ["chạm", "vào", "cạnh", "của"], ["chạm", "vào", "các", "cạnh"]):
            desc["edge"] = "any"
        else:
            rel = relation(rest, i)
            if rel is None or rel[1]:
                return None
            target = _target(rest[rel[2]:])
            if target is None:
                return None
            desc["rels"] = [{"rel": rel[0], "quant": target["quant"], "target": target["desc"]}]
    if every and not negated and not desc.get("rels"):
        return {"condition": "has_all", "object": desc}
    return {"condition": "has_not" if negated else "has", "object": desc}


def yn_form(tokens: list[str]) -> dict[str, Any] | None:
    while tokens and tokens[-1] in ("không", "phải"):
        tokens = tokens[:-1]
    quant = "the"
    if tokens[:4] == ["có", "phải", "tất", "cả"]:
        quant, tokens = "all", tokens[2:]
    elif tokens[:2] == ["có", "phải"]:
        tokens = tokens[2:]
    elif tokens[:1] == ["có"]:
        quant, tokens = "some", tokens[1:]
    parsed = noun_phrase(tokens, 0)
    if parsed is None:
        return None
    subject, _, i = parsed
    if tokens[i:i + 1] == ["nào"]:
        i += 1
    rel = relation(tokens, i)
    if rel is None:
        return None
    target = _target(tokens[rel[2]:])
    if target is None:
        return None
    return {"subject": subject, "quant": quant, "negated": rel[1],
            "predicate": {"rels": [{"rel": rel[0], "quant": target["quant"], "target": target["desc"]}]}}


def co_form(tokens: list[str], options: list[str]) -> dict[str, Any] | None:
    """"Vật (thể) nào/gì (không) <rel> Y, A hay B?" or "<rel> Y là gì: A hay B?"."""
    cut = next((i for i, word in enumerate(tokens) if word in (",", ":")), None)
    if cut is None:
        return None
    head = tokens[:cut]
    if head[:2] == ["vật", "thể"] or head[:2] == ["cái", "gì"]:
        head = ["vật"] + head[2:] if head[0] == "vật" else ["vật", "nào"] + head[2:]
    if head[:2] in (["vật", "nào"], ["vật", "gì"]):
        start = 2
    elif head[-2:] == ["là", "gì"]:
        head, start = head[:-2], 0
    else:
        return None
    tokens, cut = head, len(head)
    rel = relation(tokens[:cut], start)
    if rel is None:
        return None
    target = _target(tokens[rel[2]:cut])
    parsed_options = [whole_phrase(tokens_of(option)) for option in options[:2]]
    if target is None or not all(parsed_options):
        return None
    return {"options": [option[0] for option in parsed_options], "negated": rel[1],
            "predicate": {"rels": [{"rel": rel[0], "quant": target["quant"], "target": target["desc"]}]}}


def rule_form(question: dict[str, Any]) -> dict[str, Any] | None:
    """Logical form for a Human question, or None if its shape is not recognised."""
    tokens = tokens_of(question["question"])
    if not tokens:
        return None
    task = question["q_type"]
    if task == "FR":
        return fr_form(tokens)
    if task == "FB":
        return fb_form(tokens)
    if task == "YN":
        return yn_form(tokens)
    if task == "CO":
        return co_form(tokens, list(question.get("candidate_answers") or []))
    return None
