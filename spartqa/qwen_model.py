"""Qwen3-VL-8B-Instruct QLoRA model setup (torch/transformers/peft/bitsandbytes required).

Not imported by test_spartqa.py or unittest discovery, so the offline CI stays
dependency-free. Import this module only inside a CUDA environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Linear-layer leaf names LoRA attaches to; restricted to the language decoder below.
LANGUAGE_LORA_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Name fragments that mark a module as vision/merger/projector, excluded even if the leaf name matches.
# Verify this list against `model.named_modules()` on the actual loaded revision (see `inspect-modules`).
NON_LANGUAGE_NAME_FRAGMENTS = (
    "visual", "vision", "merger", "projector", "patch_embed", "patch_merge", "image_",
)


@dataclass
class LoraSettings:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05


def build_quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_processor(model_id: str = MODEL_ID, revision: str | None = None):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # Batched generation on a decoder-only model needs left padding, or shorter
    # sequences resume generating from a trailing pad position instead of their
    # real last token. Training ignores this: collate_batch pads manually.
    processor.tokenizer.padding_side = "left"
    return processor


def load_base_model(model_id: str = MODEL_ID, revision: str | None = None, attn_implementation: str = "sdpa"):
    """Loads the quantized base model only; call attach_lora() before training."""
    import torch

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise ImportError(
            "This transformers install lacks Qwen3VLForConditionalGeneration. "
            "Install a release with Qwen3-VL support, e.g. 'pip install git+https://github.com/huggingface/transformers'."
        ) from error

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        quantization_config=build_quantization_config(),
        device_map={"": 0},
        attn_implementation=attn_implementation,
    )
    model.config.use_cache = False
    return model


def resolve_language_lora_target_modules(model) -> list[str]:
    """Full qualified names of language-decoder linear layers only.

    Excludes vision tower, merger/projector, embeddings and lm_head by construction:
    only leaves matching LANGUAGE_LORA_TARGET_SUFFIXES are considered, and any name
    containing a NON_LANGUAGE_NAME_FRAGMENTS fragment is dropped even then.
    """
    targets = []
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in LANGUAGE_LORA_TARGET_SUFFIXES:
            continue
        if any(fragment in name for fragment in NON_LANGUAGE_NAME_FRAGMENTS):
            continue
        targets.append(name)
    if not targets:
        raise ValueError(
            "No language-decoder LoRA target modules resolved; inspect model.named_modules() "
            "on this revision and update NON_LANGUAGE_NAME_FRAGMENTS/LANGUAGE_LORA_TARGET_SUFFIXES."
        )
    return targets


def attach_lora(model, settings: LoraSettings = LoraSettings()):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    target_modules = resolve_language_lora_target_modules(model)
    lora_config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    _assert_only_lora_trainable(peft_model)
    return peft_model


def _assert_only_lora_trainable(peft_model) -> None:
    non_lora_trainable = [name for name, parameter in peft_model.named_parameters() if parameter.requires_grad and "lora_" not in name]
    if non_lora_trainable:
        raise ValueError(f"Unexpected trainable non-LoRA parameters: {non_lora_trainable[:5]}")


def trainable_parameter_report(peft_model) -> dict[str, Any]:
    trainable = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in peft_model.parameters())
    _assert_only_lora_trainable(peft_model)
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": (trainable / total) if total else 0.0,
    }
