"""Story-disjoint split, prompt/answer contract and audit helpers for Qwen QLoRA SFT.

Independent of torch/transformers/PEFT so it stays covered by the offline test suite.
"""

from __future__ import annotations

import hashlib
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any

from spartqa.data import TASKS, examples_from_data, read_json, validate_answer

SOURCES = ("human", "auto")

SYSTEM_INSTRUCTIONS = """Bạn là một hệ thống hỏi đáp không gian bằng tiếng Việt. Đọc câu chuyện và trả lời đúng một câu hỏi.
Chỉ trả về một đối tượng JSON có duy nhất khóa "answer" với giá trị là một danh sách, không giải thích thêm.

Định dạng câu trả lời theo loại câu hỏi:
- YN (Có/Không): danh sách có đúng một chuỗi, một trong "Yes", "No", "DK" ("DK" nghĩa là không thể xác định từ câu chuyện).
- FR (Tìm quan hệ): danh sách một hoặc nhiều số nguyên từ candidate_answers theo chỉ số (0: bên trái, 1: bên phải, 2: bên trên, 3: bên dưới, 4: gần, 5: xa, 6: chạm vào, 7: DK). Nếu dùng 7 thì danh sách chỉ có [7].
- FB (Tìm khối): danh sách gồm không, một hoặc nhiều mã khối trong candidate_answers; danh sách rỗng nếu không khối nào thỏa mãn.
- CO (Chọn đối tượng): danh sách có đúng một số nguyên: 0 là đối tượng thứ nhất, 1 là đối tượng thứ hai, 2 là cả hai, 3 là không đối tượng nào."""


def normalize_story(story: list[str]) -> str:
    return " ".join(" ".join(story).split())


def story_fingerprint(story: list[str]) -> str:
    return hashlib.sha256(normalize_story(story).encode("utf-8")).hexdigest()


def load_source_examples(source: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}")
    examples = []
    for example in examples_from_data(data):
        qualified = dict(example)
        qualified["source"] = source
        qualified["key"] = f"{source}:{example['key']}"
        qualified["story_fingerprint"] = story_fingerprint(example["payload"]["story"])
        examples.append(qualified)
    return examples


def group_by_fingerprint(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(example["story_fingerprint"], []).append(example)
    return groups


def assign_splits(examples: list[dict[str, Any]], seed: int = 42, val_ratio: float = 0.2) -> dict[str, str]:
    """Story-fingerprint groups (shared across sources) are assigned atomically to one split."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    groups = group_by_fingerprint(examples)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    running = {source: {"train": 0, "val": 0} for source in SOURCES}
    assignment: dict[str, str] = {}
    for group_id in group_ids:
        members = groups[group_id]
        sources_here = sorted({member["source"] for member in members})
        need_val = 0.0
        for source in sources_here:
            total = running[source]["train"] + running[source]["val"]
            current_ratio = (running[source]["val"] / total) if total else 0.0
            need_val += val_ratio - current_ratio
        split = "val" if need_val > 0 else "train"
        for member in members:
            assignment[member["key"]] = split
            running[member["source"]][split] += 1
    return assignment


def repair_zero_coverage(examples: list[dict[str, Any]], assignment: dict[str, str]) -> list[str]:
    """Move whole groups into val for any (source, task) cell missing validation support.

    A group is only moved when it does not drop any (source, task) cell it contains to zero
    in train. Mutates assignment in place; returns cells that could not be repaired safely.
    """
    groups = group_by_fingerprint(examples)
    counts: dict[str, dict[tuple[str, str], int]] = {"train": {}, "val": {}}
    for example in examples:
        split = assignment[example["key"]]
        cell = (example["source"], example["payload"]["q_type"])
        counts[split][cell] = counts[split].get(cell, 0) + 1
    warnings: list[str] = []
    for source in SOURCES:
        for task in TASKS:
            cell = (source, task)
            if counts["val"].get(cell, 0) > 0:
                continue
            moved = False
            for group_id in sorted(groups):
                members = groups[group_id]
                if any(assignment[member["key"]] != "train" for member in members):
                    continue
                if not any(member["source"] == source and member["payload"]["q_type"] == task for member in members):
                    continue
                group_cells: dict[tuple[str, str], int] = {}
                for member in members:
                    member_cell = (member["source"], member["payload"]["q_type"])
                    group_cells[member_cell] = group_cells.get(member_cell, 0) + 1
                if any(counts["train"].get(member_cell, 0) - amount <= 0 for member_cell, amount in group_cells.items()):
                    continue
                for member in members:
                    assignment[member["key"]] = "val"
                for member_cell, amount in group_cells.items():
                    counts["train"][member_cell] -= amount
                    counts["val"][member_cell] = counts["val"].get(member_cell, 0) + amount
                moved = True
                break
            if not moved:
                warnings.append(f"No validation coverage available for source={source} task={task}")
    return warnings


def split_counts(examples: list[dict[str, Any]], assignment: dict[str, str]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for example in examples:
        split = assignment[example["key"]]
        bucket = counts.setdefault(example["source"], {}).setdefault(example["payload"]["q_type"], {"train": 0, "val": 0})
        bucket[split] += 1
    return counts


def label_distribution(examples: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    distribution: dict[str, dict[str, dict[str, Any]]] = {}
    for example in examples:
        gold = example.get("gold")
        if gold is None:
            continue
        bucket = distribution.setdefault(example["source"], {}).setdefault(
            example["payload"]["q_type"], {"count": 0, "cardinality": {}, "labels": {}}
        )
        bucket["count"] += 1
        cardinality = str(len(gold))
        bucket["cardinality"][cardinality] = bucket["cardinality"].get(cardinality, 0) + 1
        for label in gold:
            bucket["labels"][str(label)] = bucket["labels"].get(str(label), 0) + 1
    return distribution


def detect_label_conflicts(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag identical (story, question, q_type) pairs whose gold answers disagree."""
    seen: dict[tuple[str, str, str], tuple[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for example in examples:
        gold = example.get("gold")
        if gold is None:
            continue
        signature = (example["story_fingerprint"], example["payload"]["q_type"], example["payload"]["question"])
        previous = seen.get(signature)
        if previous is not None and previous[1] != gold:
            conflicts.append({
                "story_fingerprint": signature[0], "q_type": signature[1], "question": signature[2],
                "keys": [previous[0], example["key"]], "answers": [previous[1], gold],
            })
        else:
            seen[signature] = (example["key"], gold)
    return conflicts


def build_manifest(paths: dict[str, str | Path], seed: int = 42, val_ratio: float = 0.2) -> dict[str, Any]:
    all_examples: list[dict[str, Any]] = []
    sources_info: dict[str, Any] = {}
    for source, path in paths.items():
        resolved = Path(path)
        data = read_json(resolved)
        source_examples = load_source_examples(source, data)
        all_examples.extend(source_examples)
        sources_info[source] = {
            "path": str(resolved),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "count": len(source_examples),
        }
    assignment = assign_splits(all_examples, seed=seed, val_ratio=val_ratio)
    warnings = repair_zero_coverage(all_examples, assignment)
    return {
        "seed": seed,
        "val_ratio": val_ratio,
        "sources": sources_info,
        "assignment": assignment,
        "counts": split_counts(all_examples, assignment),
        "label_conflicts": detect_label_conflicts(all_examples),
        "warnings": warnings,
    }


def build_user_content(payload: dict[str, Any]) -> str:
    lines = [
        f"Loại câu hỏi: {payload['q_type']}",
        f"Câu chuyện: {normalize_story(payload['story'])}",
        f"Câu hỏi: {payload['question']}",
    ]
    candidates = payload["candidate_answers"]
    if candidates:
        lines.append("Các lựa chọn: " + " | ".join(str(candidate) for candidate in candidates))
    return "\n".join(lines)


def build_prompt_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTIONS}]},
        {"role": "user", "content": [{"type": "text", "text": build_user_content(payload)}]},
    ]


def canonical_answer(answer: list[Any], payload: dict[str, Any]) -> list[Any]:
    validate_answer(answer, payload)
    task = payload["q_type"]
    if task == "FR":
        return sorted(answer)
    if task == "FB":
        return [candidate for candidate in payload["candidate_answers"] if candidate in answer]
    return list(answer)


def completion_text(answer: list[Any], payload: dict[str, Any]) -> str:
    return json.dumps({"answer": canonical_answer(answer, payload)}, ensure_ascii=False)


def legal_answers(payload: dict[str, Any]) -> list[list[Any]]:
    task = payload["q_type"]
    if task == "YN":
        return [["Yes"], ["No"], ["DK"]]
    if task == "CO":
        return [[value] for value in range(4)]
    if task == "FR":
        relations = list(range(7))
        answers = [list(combo) for size in range(1, len(relations) + 1) for combo in combinations(relations, size)]
        answers.append([7])
        return answers
    if task == "FB":
        candidates = list(payload["candidate_answers"])
        if len(candidates) > 20:
            raise ValueError(f"Refusing to enumerate 2**{len(candidates)} FB subsets; raise the guard deliberately if needed")
        return [list(combo) for size in range(len(candidates) + 1) for combo in combinations(candidates, size)]
    raise ValueError(f"Unsupported task {task!r}")
