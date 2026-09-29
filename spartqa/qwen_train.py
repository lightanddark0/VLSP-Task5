"""GPU training/evaluation orchestration for Qwen3-VL-8B-Instruct QLoRA.

Called from qwen_finetune.py (local CLI) or modal_app.py (Modal remote functions).
Requires torch/transformers/peft/bitsandbytes; not covered by offline CI.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from spartqa.agent_data import graphs_from_pot_log
from spartqa.data import examples_from_data, read_json, write_json
from spartqa.qwen_data import SOURCES, load_source_examples
from spartqa.qwen_dataset import (
    SpartQATokenizedDataset,
    build_stage_b_schedule,
    collate_batch,
    encode_example,
    encode_extraction_example,
    encode_reasoning_example,
    select_stage_a_examples,
)
from spartqa.qwen_eval import (
    evaluate_examples,
    generate_graphs,
    generate_predictions,
    generate_predictions_with_graphs,
    macro_score,
    regression_gate,
)
from spartqa.qwen_model import LoraSettings, attach_lora, load_base_model, load_processor, trainable_parameter_report


def _load_all_examples(human_path: Path, auto_path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "human": load_source_examples("human", read_json(human_path)),
        "auto": load_source_examples("auto", read_json(auto_path)),
    }


def _build_trainer(model, processor, schedule: list[dict[str, Any]], args, learning_rate: float, encode_fn=None):
    from transformers import Trainer, TrainingArguments

    encode = encode_fn or (lambda example: encode_example(processor, example))
    records = [encode(example) for example in schedule]
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
    from peft import PeftModel, prepare_model_for_kbit_training

    from spartqa.qwen_model import trainable_parameter_report

    examples = _load_all_examples(args.human, args.auto)
    assignment = read_json(args.manifest)["assignment"]
    schedule = build_stage_b_schedule(examples["human"], examples["auto"], assignment, human_passes=args.human_passes, seed=args.seed)
    processor = load_processor(args.model_id, args.revision)
    base_model = load_base_model(args.model_id, args.revision)
    # Reloading a saved adapter skips attach_lora(); without this, frozen embeddings feed
    # requires_grad=False activations into gradient checkpointing and no LoRA param gets a gradient.
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    peft_model = PeftModel.from_pretrained(base_model, str(args.init_adapter), is_trainable=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "trainable_parameters.json", trainable_parameter_report(peft_model))
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


def _load_graphs_file(path: Path) -> dict[str, dict[str, Any]]:
    return read_json(path)["graphs"]


def run_distill_graphs(args) -> None:
    """Converts a completed gpt_experiment.py --method pot run's responses.jsonl into a
    compact key -> graph mapping, used as Agent 1's training target and (optionally)
    as Agent 2's training-time facts context.

    gpt_experiment.py keys graphs by plain "story_index:question_index" (no source
    prefix), while qwen_data.py's examples are keyed "human:story_index:question_index"
    or "auto:...". Re-keying with --source here is what makes the two line up; running
    this twice (once per --source, same --output) merges into one graphs.json.
    """
    raw_graphs = graphs_from_pot_log(args.pot_log)
    graphs = {f"{args.source}:{key}": graph for key, graph in raw_graphs.items()}
    existing = _load_graphs_file(args.output) if args.output.exists() else {}
    merged = {**existing, **graphs}
    write_json(args.output, {"graphs": merged})
    print(json.dumps({
        "pot_log": str(args.pot_log), "source": args.source,
        "extracted_graphs": len(graphs), "total_graphs_in_output": len(merged),
    }, indent=2))


def run_train_agent_extraction(args) -> None:
    """Agent 1: single QLoRA run, story+question -> structured facts JSON distilled from GPT."""
    graphs = _load_graphs_file(args.graphs)
    examples = _load_all_examples(args.human, args.auto)
    assignment = read_json(args.manifest)["assignment"]
    pool = [
        example
        for source_examples in examples.values()
        for example in source_examples
        if assignment[example["key"]] == "train" and example["key"] in graphs
    ]
    pool.sort(key=lambda example: example["key"])
    random.Random(args.seed).shuffle(pool)
    if getattr(args, "max_examples", None) is not None:
        pool = pool[: args.max_examples]
    if not pool:
        raise ValueError("No TRAIN examples with a distilled graph found; run distill-graphs first")
    processor = load_processor(args.model_id, args.revision)
    model = load_base_model(args.model_id, args.revision)
    peft_model = attach_lora(model, LoraSettings())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "trainable_parameters.json", trainable_parameter_report(peft_model))
    write_json(args.output_dir / "schedule_summary.json", {"selected_count": len(pool)})
    encode_fn = lambda example: encode_extraction_example(processor, example, graphs[example["key"]])
    trainer = _build_trainer(peft_model, processor, pool, args, args.learning_rate, encode_fn=encode_fn)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    peft_model.save_pretrained(str(args.output_dir / "adapter"))
    processor.save_pretrained(str(args.output_dir / "adapter"))


def run_train_agent_reasoning(args) -> None:
    """Agent 2: single QLoRA run, story+question[+facts] -> answer JSON.

    Reuses the Human:Auto 1:1 balanced schedule from stage B; examples whose key has
    no distilled graph train the no-facts fallback path (graph=None in the prompt),
    matching what happens at inference when Agent 1's extraction fails to parse.
    """
    graphs = _load_graphs_file(args.graphs) if args.graphs else {}
    examples = _load_all_examples(args.human, args.auto)
    assignment = read_json(args.manifest)["assignment"]
    schedule = build_stage_b_schedule(examples["human"], examples["auto"], assignment, human_passes=args.human_passes, seed=args.seed)
    processor = load_processor(args.model_id, args.revision)
    model = load_base_model(args.model_id, args.revision)
    peft_model = attach_lora(model, LoraSettings())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "trainable_parameters.json", trainable_parameter_report(peft_model))
    write_json(args.output_dir / "schedule_summary.json", {
        "selected_count": len(schedule),
        "with_extracted_graph": sum(example["key"] in graphs for example in schedule),
    })
    encode_fn = lambda example: encode_reasoning_example(processor, example, graphs.get(example["key"]))
    trainer = _build_trainer(peft_model, processor, schedule, args, args.learning_rate, encode_fn=encode_fn)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    peft_model.save_pretrained(str(args.output_dir / "adapter"))
    processor.save_pretrained(str(args.output_dir / "adapter"))


def _load_pipeline_model(model_id: str, extraction_adapter: Path, reasoning_adapter: Path):
    """Loads the base model once and both agents as named adapters on the same PeftModel,
    switched with set_adapter() instead of reloading the quantized base model twice.
    """
    from peft import PeftModel

    base_model = load_base_model(model_id)
    peft_model = PeftModel.from_pretrained(base_model, str(extraction_adapter), adapter_name="extraction")
    peft_model.load_adapter(str(reasoning_adapter), adapter_name="reasoning")
    peft_model.eval()
    return peft_model


def run_evaluate_pipeline(args) -> None:
    """Two-agent evaluation: Agent 1 extracts facts, Agent 2 answers using them."""
    examples = _load_all_examples(args.human, args.auto)
    assignment = read_json(args.manifest)["assignment"]
    validation = _validation_examples(examples, assignment)
    processor = load_processor(args.model_id)
    peft_model = _load_pipeline_model(args.model_id, args.extraction_adapter, args.reasoning_adapter)
    metrics_by_source: dict[str, dict[str, Any]] = {}
    graph_stats: dict[str, Any] = {}
    max_questions = getattr(args, "max_questions", None)
    for source in SOURCES:
        source_examples = validation[source]
        if max_questions is not None:
            source_examples = source_examples[:max_questions]
        if not source_examples:
            metrics_by_source[source] = {}
            continue
        peft_model.set_adapter("extraction")
        graphs = generate_graphs(peft_model, processor, source_examples, max_new_tokens=args.graph_max_new_tokens, batch_size=args.batch_size)
        peft_model.set_adapter("reasoning")
        predictions = generate_predictions_with_graphs(
            peft_model, processor, source_examples, graphs,
            max_new_tokens=args.answer_max_new_tokens, batch_size=args.batch_size,
        )
        metrics_by_source[source] = evaluate_examples(predictions, source_examples)
        graph_stats[source] = {"count": len(source_examples), "parsed": sum(graph is not None for graph in graphs.values())}
    report: dict[str, Any] = {
        "by_source": metrics_by_source, "summary": macro_score(metrics_by_source), "graph_extraction": graph_stats,
    }
    if args.baseline_metrics is not None:
        report["regression_gate"] = regression_gate(read_json(args.baseline_metrics)["by_source"], metrics_by_source)
    write_json(args.output, report)
    print(json.dumps({"summary": report["summary"], "graph_extraction": graph_stats}, indent=2, ensure_ascii=False))


def run_predict_pipeline(args) -> None:
    """Two-agent prediction for an unlabeled public-test JSON file."""
    data = read_json(args.input)
    examples = examples_from_data(data)
    max_questions = getattr(args, "max_questions", None)
    if max_questions is not None:
        examples = examples[:max_questions]
    processor = load_processor(args.model_id)
    peft_model = _load_pipeline_model(args.model_id, args.extraction_adapter, args.reasoning_adapter)
    peft_model.set_adapter("extraction")
    graphs = generate_graphs(peft_model, processor, examples, max_new_tokens=args.graph_max_new_tokens, batch_size=args.batch_size)
    peft_model.set_adapter("reasoning")
    predictions = generate_predictions_with_graphs(
        peft_model, processor, examples, graphs, max_new_tokens=args.answer_max_new_tokens, batch_size=args.batch_size,
    )
    missing = [example["key"] for example in examples if predictions.get(example["key"]) is None]
    if missing:
        raise ValueError(f"Missing or invalid predictions for {len(missing)} questions, e.g. {missing[:5]}")
    output = json.loads(json.dumps(data))
    for example in examples:
        item = output["data"][example["story_index"]]
        item["questions"][example["question_index"]]["answer"] = predictions[example["key"]]
    write_json(args.output, output)
