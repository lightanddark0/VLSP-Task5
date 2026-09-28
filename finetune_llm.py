"""LoRA fine-tuning of a causal LLM that generates ViSPARTQA answers directly.

The model reads the dataset name, story, question, and options, and is trained
to emit the answer list in the task's JSON format. Loss covers answer tokens only.

Train on the story-level splits (one A100/H100):
    python finetune_llm.py train --output-dir outputs/qwen25_7b_lora
Predict a labeled dev file or an unlabeled test file:
    python finetune_llm.py predict --adapter outputs/qwen25_7b_lora --dataset human \\
        --input Data/splits/human_dev.json --fill-from Data/splits/human_train.json \\
        --output outputs/qwen25_7b_lora/human_dev_pred.json
Then score with evaluate.py or check a test prediction with validate_submission.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from spartqa.data import examples_from_data, read_json, write_json
from spartqa.metrics import question_records, score_records
from spartqa.prompting import build_prompt, format_answer, parse_answer
from spartqa.repro import run_metadata, set_seed
from spartqa.submission import majority_answers

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def named_paths(values: list[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {value!r}")
        paths[name] = Path(path)
    return paths


def load_items(paths: dict[str, Path], limit: int | None, seed: int) -> list[tuple[str, dict[str, Any]]]:
    items = []
    rng = random.Random(seed)
    for name, path in paths.items():
        examples = examples_from_data(read_json(path))
        if limit is not None and limit < len(examples):
            examples = rng.sample(examples, limit)
        items.extend((name, example) for example in examples)
    return items


def encode(tokenizer: Any, dataset: str, example: dict[str, Any], max_length: int) -> dict[str, list[int]] | None:
    prompt_ids = tokenizer(build_prompt(example["payload"], dataset))["input_ids"]
    target_ids = tokenizer(" " + format_answer(example["gold"], example["payload"]),
                           add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > max_length:
        return None
    return {"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids}


def encode_items(tokenizer: Any, items: list[tuple[str, dict[str, Any]]], max_length: int,
                 repeats: dict[str, int] | None = None) -> tuple[list[dict[str, list[int]]], int]:
    features, skipped = [], 0
    for dataset, example in items:
        feature = encode(tokenizer, dataset, example, max_length)
        if feature is None:
            skipped += 1
            continue
        features.extend([feature] * (repeats or {}).get(dataset, 1))
    return features, skipped


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append([1] * len(feature["input_ids"]) + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value) for key, value in batch.items()}


def load_tokenizer(name: str | Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def model_dtype(cpu: bool = False) -> torch.dtype:
    if torch.cuda.is_available() and not cpu:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_name)
    repeats = {name: int(count) for name, count in (item.split("=", 1) for item in args.repeat or [])}

    train_items = load_items(named_paths(args.train), args.max_train_per_dataset, args.seed)
    train_features, skipped = encode_items(tokenizer, train_items, args.max_length, repeats)
    random.Random(args.seed).shuffle(train_features)
    eval_features = None
    if args.dev:
        eval_items = load_items(named_paths(args.dev), args.eval_examples_per_dataset, args.seed)
        eval_features, _ = encode_items(tokenizer, eval_items, args.max_length)
    print(f"Training features: {len(train_features)} (skipped {skipped} longer than {args.max_length} tokens)")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=model_dtype(args.cpu),
                                                 attn_implementation=args.attn_implementation)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    dtype = model_dtype(args.cpu)
    evaluate_steps = eval_features is not None
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if evaluate_steps else "no",
        eval_steps=args.save_steps if evaluate_steps else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=evaluate_steps,
        metric_for_best_model="eval_loss" if evaluate_steps else None,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        use_cpu=args.cpu,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_features,
                      eval_dataset=eval_features, data_collator=Collator(tokenizer.pad_token_id))
    trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    write_json(output_dir / "run_config.json", run_metadata(
        command="train", args=vars(args) | {"func": None}, base_model=args.model_name,
        train_features=len(train_features), skipped_too_long=skipped,
        log_history=trainer.state.log_history,
    ))
    print(f"Saved adapter to {adapter_dir}")


def load_for_inference(adapter: Path, model_name: str | None, device: torch.device) -> tuple[Any, Any]:
    config_path = adapter.parent / "run_config.json"
    if model_name is None and config_path.exists():
        model_name = read_json(config_path)["base_model"]
    if model_name is None:
        model_name = DEFAULT_MODEL
    tokenizer = load_tokenizer(adapter)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype())
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def generate_answers(model: Any, tokenizer: Any, prompts: list[str], batch_size: int,
                     max_new_tokens: int, device: torch.device) -> list[str]:
    order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]), reverse=True)
    outputs: list[str] = [""] * len(prompts)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        encoded = tokenizer([prompts[index] for index in indices], return_tensors="pt", padding=True).to(device)
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        for index, text in zip(indices, texts):
            outputs[index] = text
        print(f"Generated {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return outputs


def predict_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_for_inference(Path(args.adapter), args.model_name, device)
    data = read_json(args.input)
    examples = examples_from_data(data)
    if args.max_questions is not None:
        examples = examples[:args.max_questions]
    fallback = majority_answers(read_json(args.fill_from)) if args.fill_from else None
    prompts = [build_prompt(example["payload"], args.dataset) for example in examples]
    texts = generate_answers(model, tokenizer, prompts, args.batch_size, args.max_new_tokens, device)

    output = copy.deepcopy(data)
    for story in output["data"]:
        for question in story["questions"]:
            question.pop("answer", None)
    log, invalid = [], 0
    for example, text in zip(examples, texts):
        answer = parse_answer(text, example["payload"])
        used_fallback = answer is None
        if used_fallback:
            invalid += 1
            if fallback is not None:
                answer = [label for label in fallback[example["payload"]["q_type"]]
                          if example["payload"]["q_type"] != "FB" or label in example["payload"]["candidate_answers"]]
        if answer is not None:
            output["data"][example["story_index"]]["questions"][example["question_index"]]["answer"] = answer
        log.append({"key": example["key"], "generation": text, "answer": answer, "fallback": used_fallback})

    output_path = Path(args.output)
    write_json(output_path, output)
    with output_path.with_suffix(".generations.jsonl").open("w", encoding="utf-8") as stream:
        for record in log:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {output_path}; unparseable generations: {invalid}/{len(examples)}"
          + ("" if fallback else " (left without answers; pass --fill-from to fill them)"))
    if examples and examples[0]["gold"] is not None and args.max_questions is None:
        print(json.dumps(score_records(question_records(output, data))["by_task"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="LoRA fine-tune on labeled NAME=PATH files.")
    train.add_argument("--model-name", default=DEFAULT_MODEL)
    train.add_argument("--train", nargs="+", default=["human=Data/splits/human_train.json",
                                                       "auto=Data/splits/auto_train.json"])
    train.add_argument("--dev", nargs="*", default=["human=Data/splits/human_dev.json",
                                                     "auto=Data/splits/auto_dev.json"],
                       help="Used for eval loss and best-checkpoint selection; pass nothing to disable.")
    train.add_argument("--repeat", nargs="*", default=["human=8"],
                       help="Oversample small datasets, e.g. human=8.")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--gradient-accumulation-steps", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--warmup-ratio", type=float, default=0.03)
    train.add_argument("--lora-r", type=int, default=32)
    train.add_argument("--lora-alpha", type=int, default=64)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--max-train-per-dataset", type=int, help="Random subset per dataset for quick pilots.")
    train.add_argument("--eval-examples-per-dataset", type=int, default=500)
    train.add_argument("--save-steps", type=int, default=500)
    train.add_argument("--logging-steps", type=int, default=20)
    train.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--attn-implementation", default="sdpa")
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--cpu", action="store_true", help="Force CPU, for smoke tests only.")
    train.set_defaults(func=train_command)

    predict = subparsers.add_parser("predict", help="Generate answers into a copy of the input JSON.")
    predict.add_argument("--adapter", required=True, help="The adapter directory saved by train.")
    predict.add_argument("--model-name", help="Base model; defaults to the one recorded at training time.")
    predict.add_argument("--dataset", required=True, help="Dataset name used in the prompt (human or auto).")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--fill-from", help="Labeled file for fallback answers when a generation is invalid.")
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--max-new-tokens", type=int, default=24)
    predict.add_argument("--max-questions", type=int, help="Only the first N questions, for smoke tests.")
    predict.add_argument("--seed", type=int, default=42)
    predict.add_argument("--cpu", action="store_true", help="Force CPU, for smoke tests only.")
    predict.set_defaults(func=predict_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
