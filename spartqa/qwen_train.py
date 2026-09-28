"""GPU training/evaluation orchestration for Qwen3-VL-8B-Instruct QLoRA.

Called from qwen_finetune.py (local CLI) or modal_app.py (Modal remote functions).
Requires torch/transformers/peft/bitsandbytes; not covered by offline CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spartqa.data import examples_from_data, read_json, write_json
from spartqa.qwen_data import SOURCES, load_source_examples
from spartqa.qwen_dataset import (
    SpartQATokenizedDataset,
    build_stage_b_schedule,
    collate_batch,
    encode_example,
    select_stage_a_examples,
)
from spartqa.qwen_eval import evaluate_examples, generate_predictions, macro_score, regression_gate
from spartqa.qwen_model import LoraSettings, attach_lora, load_base_model, load_processor, trainable_parameter_report


def _load_all_examples(human_path: Path, auto_path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "human": load_source_examples("human", read_json(human_path)),
        "auto": load_source_examples("auto", read_json(auto_path)),
    }


def _build_trainer(model, processor, schedule: list[dict[str, Any]], args, learning_rate: float):
    from transformers import Trainer, TrainingArguments

    records = [encode_example(processor, example) for example in schedule]
    dataset = SpartQATokenizedDataset(records)
    pad_token_id = processor.tokenizer.pad_token_id

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_batch(batch, pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(1, args.max_steps // 4),
        seed=args.seed,
        report_to=[],
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    return Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate)


def run_stage_a(args) -> None:
    examples = _load_all_examples(args.human, args.auto)
    assignment = read_json(args.manifest)["assignment"]
    schedule = select_stage_a_examples(examples["auto"], assignment, args.auto_subset_size, seed=args.seed)
    processor = load_processor(args.model_id, args.revision)
    model = load_base_model(args.model_id, args.revision)
    peft_model = attach_lora(model, LoraSettings())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "trainable_parameters.json", trainable_parameter_report(peft_model))
    write_json(args.output_dir / "schedule_summary.json", {"selected_count": len(schedule)})
    trainer = _build_trainer(peft_model, processor, schedule, args, args.learning_rate)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    peft_model.save_pretrained(str(args.output_dir / "adapter"))
    processor.save_pretrained(str(args.output_dir / "adapter"))


def run_stage_b(args) -> None:
    from peft import PeftModel

    examples = _load_all_examples(args.human, args.auto)
    assignment = read_json(args.manifest)["assignment"]
    schedule = build_stage_b_schedule(examples["human"], examples["auto"], assignment, human_passes=args.human_passes, seed=args.seed)
    processor = load_processor(args.model_id, args.revision)
    base_model = load_base_model(args.model_id, args.revision)
    peft_model = PeftModel.from_pretrained(base_model, str(args.init_adapter), is_trainable=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "schedule_summary.json", {"selected_count": len(schedule)})
    trainer = _build_trainer(peft_model, processor, schedule, args, args.learning_rate)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    peft_model.save_pretrained(str(args.output_dir / "adapter"))
    processor.save_pretrained(str(args.output_dir / "adapter"))


def _validation_examples(examples: dict[str, list[dict[str, Any]]], assignment: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    return {
        source: [example for example in source_examples if assignment[example["key"]] == "val"]
        for source, source_examples in examples.items()
    }


def run_evaluate(args) -> None:
    examples = _load_all_examples(args.human, args.auto)
    assignment = read_json(args.manifest)["assignment"]
    validation = _validation_examples(examples, assignment)
    processor = load_processor(args.model_id)
    model = load_base_model(args.model_id)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    metrics_by_source: dict[str, dict[str, Any]] = {}
    max_questions = getattr(args, "max_questions", None)
    for source in SOURCES:
        source_examples = validation[source]
        if max_questions is not None:
            source_examples = source_examples[:max_questions]
        if not source_examples:
            metrics_by_source[source] = {}
            continue
        predictions = generate_predictions(model, processor, source_examples, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        metrics_by_source[source] = evaluate_examples(predictions, source_examples)
    report: dict[str, Any] = {"by_source": metrics_by_source, "summary": macro_score(metrics_by_source)}
    if args.baseline_metrics is not None:
        report["regression_gate"] = regression_gate(read_json(args.baseline_metrics)["by_source"], metrics_by_source)
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


def run_predict(args) -> None:
    from peft import PeftModel

    data = read_json(args.input)
    examples = examples_from_data(data)
    max_questions = getattr(args, "max_questions", None)
    if max_questions is not None:
        examples = examples[:max_questions]
    processor = load_processor(args.model_id)
    model = load_base_model(args.model_id)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    predictions = generate_predictions(model, processor, examples, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    missing = [example["key"] for example in examples if predictions.get(example["key"]) is None]
    if missing:
        raise ValueError(f"Missing or invalid predictions for {len(missing)} questions, e.g. {missing[:5]}")
    output = json.loads(json.dumps(data))
    for example in examples:
        item = output["data"][example["story_index"]]
        item["questions"][example["question_index"]]["answer"] = predictions[example["key"]]
    write_json(args.output, output)
