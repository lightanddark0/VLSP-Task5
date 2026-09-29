"""Constrained decoding and eight-cell evaluation for Qwen3-VL QLoRA (requires torch).

Not imported by test_spartqa.py; keep heavy imports out of spartqa/qwen_data.py.
"""

from __future__ import annotations

import json
from typing import Any

from spartqa.agent_data import build_extraction_prompt_messages, build_reasoning_prompt_messages, canonical_graph
from spartqa.data import validate_answer
from spartqa.metrics import finalize_metrics, update_metric_state
from spartqa.qwen_data import build_prompt_messages, legal_answers

# Never a legal YN/CO/FR/FB label; guarantees a missing prediction always scores as wrong,
# even against an empty FB gold answer (an empty prediction would otherwise look correct).
_MISSING_PREDICTION_SENTINEL = ["__invalid_missing_prediction__"]


class LegalAnswerTrie:
    """Token-prefix trie of canonical `{"answer": [...]}` completions for one payload."""

    def __init__(self, processor, prompt_length: int, payload: dict[str, Any]):
        self.prompt_length = prompt_length
        self.eos_token_id = processor.tokenizer.eos_token_id
        self._root: dict[Any, Any] = {}
        for answer in legal_answers(payload):
            text = json.dumps({"answer": answer}, ensure_ascii=False)
            token_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
            self._insert(list(token_ids) + ([self.eos_token_id] if self.eos_token_id is not None else []))

    def _insert(self, token_ids: list[int]) -> None:
        node = self._root
        for token_id in token_ids:
            node = node.setdefault(token_id, {})
        node[None] = True

    def allowed_tokens(self, generated_ids: list[int]) -> list[int]:
        """Batched generate() keeps calling this for every row until the whole batch is
        done, even rows that already emitted EOS for a shorter answer. Once a row is past
        its own terminal node, keep allowing EOS so HF can pad it instead of crashing on
        an empty constraint set.
        """
        completion_ids = generated_ids[self.prompt_length:]
        node = self._root
        for token_id in completion_ids:
            if token_id not in node:
                return [self.eos_token_id] if self.eos_token_id is not None else []
            node = node[token_id]
        children = [token_id for token_id in node if token_id is not None]
        if not children:
            return [self.eos_token_id] if self.eos_token_id is not None else []
        return children


def build_prefix_allowed_tokens_fn(processor, prompt_lengths: list[int], payloads: list[dict[str, Any]]):
    tries = [LegalAnswerTrie(processor, length, payload) for length, payload in zip(prompt_lengths, payloads)]

    def prefix_allowed_tokens_fn(batch_id: int, input_ids) -> list[int]:
        return tries[batch_id].allowed_tokens(input_ids.tolist())

    return prefix_allowed_tokens_fn


def generate_predictions(
    model, processor, examples: list[dict[str, Any]], max_new_tokens: int = 64, batch_size: int = 4,
) -> dict[str, list[Any] | None]:
    import torch

    predictions: dict[str, list[Any] | None] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        rendered = [
            processor.apply_chat_template(build_prompt_messages(example["payload"]), tokenize=False, add_generation_prompt=True)
            for example in batch
        ]
        encoded = processor.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]
        prefix_fn = build_prefix_allowed_tokens_fn(
            processor, [prompt_length] * len(batch), [example["payload"] for example in batch],
        )
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn, pad_token_id=processor.tokenizer.pad_token_id,
            )
        for row, example in enumerate(batch):
            completion_ids = generated[row, prompt_length:]
            text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
            predictions[example["key"]] = parse_prediction(text, example["payload"])
    return predictions


def parse_prediction(text: str, payload: dict[str, Any]) -> list[Any] | None:
    try:
        answer = json.loads(text)["answer"]
        validate_answer(answer, payload)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
    return answer


def parse_graph_prediction(text: str) -> dict[str, Any] | None:
    try:
        return canonical_graph(json.loads(text))
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None


def generate_graphs(
    model, processor, examples: list[dict[str, Any]], max_new_tokens: int = 512, batch_size: int = 4,
) -> dict[str, dict[str, Any] | None]:
    """Agent 1 free-form generation (no constrained decoding: graphs are not a small enumerable set)."""
    import torch

    graphs: dict[str, dict[str, Any] | None] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        rendered = [
            processor.apply_chat_template(
                build_extraction_prompt_messages(example["payload"]), tokenize=False, add_generation_prompt=True,
            )
            for example in batch
        ]
        encoded = processor.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        for row, example in enumerate(batch):
            completion_ids = generated[row, prompt_length:]
            text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
            graphs[example["key"]] = parse_graph_prediction(text)
    return graphs


def generate_predictions_with_graphs(
    model, processor, examples: list[dict[str, Any]], graphs: dict[str, dict[str, Any] | None],
    max_new_tokens: int = 64, batch_size: int = 4,
) -> dict[str, list[Any] | None]:
    """Agent 2 constrained generation, same legal-answer trie as generate_predictions but with
    each example's Agent-1-extracted facts (or None, on a failed/missing extraction) in the prompt.
    """
    import torch

    predictions: dict[str, list[Any] | None] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        rendered = [
            processor.apply_chat_template(
                build_reasoning_prompt_messages(example["payload"], graphs.get(example["key"])),
                tokenize=False, add_generation_prompt=True,
            )
            for example in batch
        ]
        encoded = processor.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]
        prefix_fn = build_prefix_allowed_tokens_fn(
            processor, [prompt_length] * len(batch), [example["payload"] for example in batch],
        )
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn, pad_token_id=processor.tokenizer.pad_token_id,
            )
        for row, example in enumerate(batch):
            completion_ids = generated[row, prompt_length:]
            text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
            predictions[example["key"]] = parse_prediction(text, example["payload"])
    return predictions


def evaluate_examples(predictions: dict[str, list[Any] | None], examples: list[dict[str, Any]]) -> dict[str, Any]:
    state: dict[str, dict[str, float]] = {}
    invalid = 0
    for example in examples:
        gold = example["gold"]
        if gold is None:
            continue
        prediction = predictions.get(example["key"])
        if prediction is None:
            invalid += 1
            prediction = _MISSING_PREDICTION_SENTINEL
        update_metric_state(state, example["payload"]["q_type"], prediction, gold)
    metrics: dict[str, Any] = dict(finalize_metrics(state))
    metrics["_invalid_count"] = invalid
    metrics["_total_count"] = len(examples)
    return metrics


def macro_score(metrics_by_source: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Averages each task's primary metric across sources, plus the internal eight-cell macro mean.

    This macro is an internal selection aid only; the task documentation defines no single
    official all-task leaderboard scalar.
    """
    primary_by_task: dict[str, dict[str, float]] = {}
    for source, metrics in metrics_by_source.items():
        for task, values in metrics.items():
            if task.startswith("_"):
                continue
            primary = values.get("accuracy", values.get("exact_match"))
            if primary is not None:
                primary_by_task.setdefault(task, {})[source] = primary
    task_source_average = {task: sum(sources.values()) / len(sources) for task, sources in primary_by_task.items()}
    all_cells = [value for sources in primary_by_task.values() for value in sources.values()]
    return {
        "official_task_source_average": task_source_average,
        "internal_macro_of_eight_cells": (sum(all_cells) / len(all_cells)) if all_cells else 0.0,
        "worst_cell": min(all_cells) if all_cells else 0.0,
        "cell_count": len(all_cells),
    }


def regression_gate(
    baseline_by_source: dict[str, dict[str, Any]], candidate_by_source: dict[str, dict[str, Any]], max_drop_pp: float = 2.0,
) -> dict[str, Any]:
    """Predeclared guardrail: no (source, task) primary cell may drop more than max_drop_pp below baseline."""
    failures = []
    for source, tasks in baseline_by_source.items():
        for task, values in tasks.items():
            if task.startswith("_"):
                continue
            baseline_primary = values.get("accuracy", values.get("exact_match"))
            candidate_values = candidate_by_source.get(source, {}).get(task, {})
            candidate_primary = candidate_values.get("accuracy", candidate_values.get("exact_match"))
            if baseline_primary is None or candidate_primary is None:
                failures.append(f"{source}/{task}: missing metric for comparison")
                continue
            if (baseline_primary - candidate_primary) * 100 > max_drop_pp:
                failures.append(f"{source}/{task}: baseline={baseline_primary:.4f} candidate={candidate_primary:.4f}")
    return {"passed": not failures, "failures": failures, "max_drop_pp": max_drop_pp}
