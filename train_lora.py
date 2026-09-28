"""Branch F: LoRA fine-tuning of Qwen3-8B to print the answer JSON directly.

Stage 1 (Auto):
    python train_lora.py --train-file outputs/sft/auto/train.jsonl --dev-file outputs/sft/auto/dev.jsonl \\
        --output-dir /content/ckpt/f_auto --hub-repo vispatialqa-f-qwen3-8b-auto --stage stage1
Stage 2 (Human), continuing from the stage 1 adapter:
    python train_lora.py --train-file outputs/sft/human/train.jsonl --dev-file outputs/sft/human/dev.jsonl \\
        --init-adapter vispatialqa-f-qwen3-8b-auto --output-dir /content/ckpt/f_human \\
        --hub-repo vispatialqa-f-qwen3-8b-human --stage stage2 --learning-rate 5e-5 --epochs 3 \\
        --gradient-accumulation-steps 2 --save-strategy epoch
Smoke test: add --limit 200 --max-steps 20 --no-push.

Loss covers only the answer tokens. The prompt is the Qwen3 chat template with
thinking disabled, identical to infer_f.py. Checkpoints are pushed to the
private Hub repo (folder last-checkpoint/) while training; a rerun resumes from
the newest local checkpoint, or else from the Hub. Only the adapter is stored.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

from spartqa import hub, tracking
from spartqa.data import write_json
from spartqa.prompting import prompt_version, render_prompt
from spartqa.repro import run_metadata, set_seed

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def answer_end_id(tokenizer: Any) -> int:
    """Token that ends an assistant turn (<|im_end|> for Qwen), which vLLM stops on."""
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
        return token_id
    return tokenizer.eos_token_id


def tokenize_records(tokenizer: Any, path: str, max_len: int, limit: int | None) -> tuple[Any, int, int]:
    """Tokenized dataset, number of records dropped for length, and total tokens kept."""
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=path, split="train")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    end_id = answer_end_id(tokenizer)

    def encode(record: dict[str, Any]) -> dict[str, Any]:
        prompt_ids = tokenizer(render_prompt(tokenizer, record["messages"]), add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(record["target"], add_special_tokens=False)["input_ids"] + [end_id]
        return {"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids}

    encoded = dataset.map(encode, remove_columns=dataset.column_names, num_proc=min(8, os.cpu_count() or 1),
                          desc=f"tokenize {Path(path).name}")
    before = len(encoded)
    encoded = encoded.filter(lambda record: len(record["input_ids"]) <= max_len)
    tokens = sum(len(ids) for ids in encoded["input_ids"])
    return encoded, before - len(encoded), tokens


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch
        width = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(list(feature["input_ids"]) + [self.pad_token_id] * padding)
            batch["attention_mask"].append([1] * len(feature["input_ids"]) + [0] * padding)
            batch["labels"].append(list(feature["labels"]) + [-100] * padding)
        return {key: torch.tensor(value) for key, value in batch.items()}


def supported_kwargs(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Drop arguments this transformers version does not know (names change between releases)."""
    parameters = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in values.items() if key in parameters}


def load_base_model(name: str, attn_implementation: str) -> Any:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    return AutoModelForCausalLM.from_pretrained(name, **{dtype_key: torch.bfloat16},
                                                attn_implementation=attn_implementation)


def find_resume_checkpoint(output_dir: Path, hub_repo: str | None) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint
    local = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    if local:
        return local
    if (output_dir / "last-checkpoint" / "trainer_state.json").exists():
        return str(output_dir / "last-checkpoint")
    if hub.repo_has_path(hub_repo, "last-checkpoint"):
        print(f"Downloading last-checkpoint from {hub_repo} to resume")
        hub.download(hub_repo, output_dir, allow_patterns=["last-checkpoint/*", tracking.RUN_ID_FILE])
        return str(output_dir / "last-checkpoint")
    return None


def build_callback(tokens_per_step: float) -> Any:
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        """Prints an ETA and logs estimated token throughput at each logging step."""

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.start, self.start_step = time.time(), state.global_step

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
            done = state.global_step - self.start_step
            if done <= 0 or not state.max_steps:
                return
            elapsed = time.time() - self.start
            remaining = elapsed / done * (state.max_steps - state.global_step)
            tracking.log({"train/tokens_per_second_est": tokens_per_step * done / elapsed}, summary=False)
            print(f"[progress] step {state.global_step}/{state.max_steps}  elapsed {elapsed / 60:.1f} min  "
                  f"ETA {remaining / 60:.1f} min", flush=True)

    return Progress()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", help="Eval loss and best-checkpoint selection.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--init-adapter", help="Adapter (path or Hub repo) to continue training from, for stage 2.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hub-repo", help="Private Hub model repo for checkpoints and the final adapter.")
    parser.add_argument("--no-push", action="store_true", help="Do not push to the Hub (smoke tests).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints.")
    parser.add_argument("--stage", default="stage1", help="Tag used in W&B, e.g. stage1 or stage2.")
    parser.add_argument("--dataset-tag", default=None, help="W&B dataset tag; defaults from --stage.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, help="Use only the first N training records.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=1024, help="Longer records are dropped and counted.")
    parser.add_argument("--save-strategy", choices=("steps", "epoch"), default="steps")
    parser.add_argument("--save-steps", type=int, default=300)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smoke = bool(args.limit or args.max_steps > 0)
    push = bool(args.hub_repo) and not args.no_push and hub.hub_enabled()
    hub_repo = hub.resolve_repo(args.hub_repo) if push else None
    if hub_repo:
        hub.ensure_repo(hub_repo, "model")
    resume = None if args.no_resume else find_resume_checkpoint(args.output_dir, hub_repo)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_set, dropped, train_tokens = tokenize_records(tokenizer, args.train_file, args.max_len, args.limit)
    dev_set = None
    if args.dev_file:
        dev_set, _, _ = tokenize_records(tokenizer, args.dev_file, args.max_len,
                                         max(1, args.limit // 5) if args.limit else None)
    print(f"Train records: {len(train_set)} (dropped {dropped} longer than {args.max_len} tokens); "
          f"dev records: {len(dev_set) if dev_set is not None else 0}")

    dataset_tag = args.dataset_tag or ("auto" if args.stage == "stage1" else "human")
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    name = args.run_name or tracking.run_name("F", "train", dataset_tag, args.stage,
                                              f"r{args.lora_r}-{len(train_set) // 1000}k")
    config = {"method": "LoRA", "base_model": args.model_name, "init_adapter": args.init_adapter,
              "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                       "targets": LORA_TARGETS},
              "learning_rate": args.learning_rate, "scheduler": "cosine", "warmup_ratio": args.warmup_ratio,
              "epochs": args.epochs, "max_steps": args.max_steps, "batch_size": args.batch_size,
              "gradient_accumulation_steps": args.gradient_accumulation_steps, "effective_batch": effective_batch,
              "max_len": args.max_len, "seed": args.seed, "train_file": args.train_file, "dev_file": args.dev_file,
              "train_records": len(train_set), "dropped_too_long": dropped, "train_tokens": train_tokens,
              "prompt_version": prompt_version("F"), "resumed_from": resume}
    stats_path = Path(args.train_file).with_name("stats.json")
    if stats_path.exists():
        config["data_stats"] = json.loads(stats_path.read_text(encoding="utf-8"))
    tracking.init_run("F", "train", name, [dataset_tag, args.stage] + (["smoke"] if smoke else []),
                      config, run_dir=args.output_dir)
    if hub_repo:
        hub.upload_file(args.output_dir / tracking.RUN_ID_FILE, hub_repo, tracking.RUN_ID_FILE, "model")

    model = load_base_model(args.model_name, args.attn_implementation)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.init_adapter:
        source = args.init_adapter if Path(args.init_adapter).exists() else hub.resolve_repo(args.init_adapter)
        adapter_path, init_revision = hub.resolve_adapter(source, args.output_dir / "init_adapter")
        tracking.log_hf_link(None if Path(args.init_adapter).exists() else source, init_revision, "adapter_in")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                                 lora_dropout=args.lora_dropout, target_modules=LORA_TARGETS,
                                                 task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.3%})")
    tracking.update_config({"trainable_params": trainable, "total_params": total})

    has_eval = dev_set is not None and len(dev_set) > 0
    steps_per_epoch = max(1, -(-len(train_set) // effective_batch))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs + 0.999)
    warmup_steps = int(total_steps * args.warmup_ratio + 0.999)
    values = {
        "output_dir": str(args.output_dir), "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate, "num_train_epochs": args.epochs, "max_steps": args.max_steps,
        "lr_scheduler_type": "cosine", "warmup_steps": warmup_steps, "weight_decay": 0.0,
        "logging_steps": args.logging_steps, "save_strategy": args.save_strategy, "save_steps": args.save_steps,
        "save_total_limit": 2, "eval_strategy": args.save_strategy if has_eval else "no",
        "evaluation_strategy": args.save_strategy if has_eval else "no", "eval_steps": args.save_steps,
        "load_best_model_at_end": has_eval, "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False, "bf16": True, "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}, "group_by_length": True,
        "remove_unused_columns": False, "dataloader_num_workers": 2, "seed": args.seed, "data_seed": args.seed,
        "report_to": ["wandb"] if tracking.active() else "none", "run_name": name,
        "push_to_hub": bool(hub_repo), "hub_model_id": hub_repo, "hub_strategy": "checkpoint",
        "hub_private_repo": True,
    }
    training_args = TrainingArguments(**supported_kwargs(TrainingArguments, values))
    trainer_kwargs = {"model": model, "args": training_args, "train_dataset": train_set,
                      "eval_dataset": dev_set if has_eval else None,
                      "data_collator": Collator(tokenizer.pad_token_id),
                      "callbacks": [build_callback(train_tokens / steps_per_epoch)]}
    trainer_kwargs["processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters
                   else "tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    with tracking.timer("train"):
        result = trainer.train(resume_from_checkpoint=resume)
    runtime = result.metrics.get("train_runtime", 0.0)
    tracking.log({"train/total_tokens": train_tokens * args.epochs if args.max_steps <= 0 else None,
                  "train/tokens_per_second": train_tokens * args.epochs / runtime if runtime and args.max_steps <= 0
                  else None,
                  "resources/peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3
                  if torch.cuda.is_available() else None})

    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_json(adapter_dir / "run_config.json", run_metadata(
        base_model=args.model_name, prompt_version=prompt_version("F"), args={k: str(v) for k, v in vars(args).items()},
        train_metrics=result.metrics, best_checkpoint=trainer.state.best_model_checkpoint,
        log_history=trainer.state.log_history))
    print(f"Saved adapter to {adapter_dir}")
    if hub_repo:
        revision = hub.upload_folder(adapter_dir, hub_repo, "", "model", "Final adapter")
        tracking.log_hf_link(hub_repo, revision, "adapter_out")
        print(f"Pushed adapter to {hub.repo_url(hub_repo)} (commit {revision})")
    tracking.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
