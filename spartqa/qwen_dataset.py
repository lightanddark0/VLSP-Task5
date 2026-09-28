"""Tokenization, label masking, collation and deterministic sampling (requires torch).

Not imported by test_spartqa.py; keep heavy imports out of spartqa/qwen_data.py.
"""

from __future__ import annotations

import random
from typing import Any

from spartqa.qwen_data import build_prompt_messages, completion_text

IGNORE_INDEX = -100


def _chat_template_ids(processor, messages: list[dict[str, Any]], add_generation_prompt: bool) -> list[int]:
    """Normalizes apply_chat_template output across processor/tokenizer return shapes."""
    encoded = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def encode_example(processor, example: dict[str, Any]) -> dict[str, Any]:
    """Tokenizes prompt+completion; supervises only the assistant completion tokens plus EOS."""
    messages = build_prompt_messages(example["payload"])
    prompt_ids = _chat_template_ids(processor, messages, add_generation_prompt=True)
    completion = completion_text(example["gold"], example["payload"])
    full_messages = messages + [{"role": "assistant", "content": [{"type": "text", "text": completion}]}]
    full_ids = _chat_template_ids(processor, full_messages, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Chat template prefix mismatch for key {example['key']!r}; cannot mask reliably. "
            "Verify the processor's chat template keeps a stable prompt prefix before assistant content."
        )
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    if len(labels) != len(full_ids):
        raise ValueError("Label/length mismatch while masking prompt tokens")
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError(f"No supervised completion tokens for key {example['key']!r}")
    return {"key": example["key"], "input_ids": list(full_ids), "labels": labels}


class SpartQATokenizedDataset:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_length), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(item["labels"], dtype=torch.long)
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def select_stage_a_examples(
    auto_train_examples: list[dict[str, Any]], assignment: dict[str, str], target_count: int, seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministic diverse subset of Auto TRAIN examples, balanced across q_type where available."""
    train_examples = [example for example in auto_train_examples if assignment[example["key"]] == "train"]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for example in train_examples:
        by_task.setdefault(example["payload"]["q_type"], []).append(example)
    rng = random.Random(seed)
    for bucket in by_task.values():
        bucket.sort(key=lambda example: example["key"])
        rng.shuffle(bucket)
    per_task_target = max(1, target_count // max(len(by_task), 1))
    selected: list[dict[str, Any]] = []
    for bucket in by_task.values():
        selected.extend(bucket[:per_task_target])
    remaining_target = target_count - len(selected)
    if remaining_target > 0:
        leftovers = [example for bucket in by_task.values() for example in bucket[per_task_target:]]
        leftovers.sort(key=lambda example: example["key"])
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining_target])
    selected.sort(key=lambda example: example["key"])
    rng.shuffle(selected)
    return selected


def _balanced_repeat_sample(pool: list[dict[str, Any]], target_count: int, seed: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for example in pool:
        by_task.setdefault(example["payload"]["q_type"], []).append(example)
    rng = random.Random(seed)
    tasks = sorted(by_task)
    order = {task: rng.sample(by_task[task], len(by_task[task])) for task in tasks}
    cursors = {task: 0 for task in tasks}
    schedule: list[dict[str, Any]] = []
    while len(schedule) < target_count and tasks:
        for task in tasks:
            if len(schedule) >= target_count:
                break
            bucket = order[task]
            schedule.append(bucket[cursors[task] % len(bucket)])
            cursors[task] += 1
    return schedule


def build_stage_b_schedule(
    human_train_examples: list[dict[str, Any]], auto_train_examples: list[dict[str, Any]],
    assignment: dict[str, str], human_passes: float = 3.0, seed: int = 42,
) -> list[dict[str, Any]]:
    """Human:Auto = 1:1 by source, q_type-balanced within each source, capped at human_passes over Human TRAIN."""
    human_pool = [example for example in human_train_examples if assignment[example["key"]] == "train"]
    auto_pool = [example for example in auto_train_examples if assignment[example["key"]] == "train"]
    if not human_pool:
        raise ValueError("No Human TRAIN examples available for stage B")
    half_target = int(round(len(human_pool) * human_passes))
    human_schedule = _balanced_repeat_sample(human_pool, half_target, seed)
    auto_schedule = _balanced_repeat_sample(auto_pool, half_target, seed + 1)
    schedule = human_schedule + auto_schedule
    random.Random(seed + 2).shuffle(schedule)
    return schedule
