"""Modal orchestration for Qwen3-VL-8B-Instruct QLoRA, running against the "qwen" Volume.

Prerequisites: `pip install modal` and `modal setup` (or `modal token new`) done locally.
This file only defines the App; nothing runs on import, and nothing here is executed
automatically by this repository's offline tests.

Typical sequence (see Docs/qwen_qlora.md for full detail and budget notes):
    modal run modal_app.py::list_volume
    modal run modal_app.py::manifest
    modal run modal_app.py::inspect_modules
    modal run modal_app.py::audit
    modal run modal_app.py::evaluate --output-name metrics_base.json
    modal run modal_app.py::train_stage_a --auto-subset-size 30000 --max-steps 800
    modal run modal_app.py::evaluate --adapter /vol/outputs/qwen_qlora/stage_a/adapter --output-name metrics_stage_a.json --baseline-metrics /vol/outputs/qwen_qlora/metrics_base.json
    modal run modal_app.py::train_stage_b --max-steps 300 --human-passes 3.0
    modal run modal_app.py::evaluate --adapter /vol/outputs/qwen_qlora/stage_b/adapter --output-name metrics_stage_b.json --baseline-metrics /vol/outputs/qwen_qlora/metrics_base.json
    modal run modal_app.py::predict --input-path /vol/Data/human_public_test.json --adapter /vol/outputs/qwen_qlora/stage_b/adapter --output-name human_public_test_predictions.json

If your own upload put the repo/Data folder somewhere other than the default
/vol/Data, pass --root-dir to any command below (e.g. --root-dir /vol/Task5),
or run list_volume first to see the actual layout on the "qwen" Volume.
"""

from __future__ import annotations

import modal

VOLUME_NAME = "qwen"
VOLUME_PATH = "/vol"
GPU = "A100-80GB"
MINUTES = 60
TRAIN_TIMEOUT_SECONDS = 6 * 60 * MINUTES


def _data_dir(root_dir: str) -> str:
    return f"{root_dir}/Data"


def _output_dir(root_dir: str) -> str:
    return f"{root_dir}/outputs/qwen_qlora"

app = modal.App("vlsp-qwen-qlora")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Exact pins, not ranges: an unpinned torchvision let pip re-resolve torch's own
# CUDA/NCCL sub-wheels to a mismatched set, breaking `import torch` with
# `undefined symbol: ncclCommResume`. This triple installed cleanly together
# before; re-verify with `inspect-modules` after any change here.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "nvidia-nccl-cu12==2.27.3",
        "transformers>=4.57,<5",
        "accelerate>=1.1",
        "peft>=0.14",
        "bitsandbytes>=0.45",
        "datasets>=3.1",
        "sentencepiece>=0.2",
        "safetensors>=0.4.5",
    )
    .env({"HF_HOME": f"{VOLUME_PATH}/hf_cache"})
    .add_local_python_source("spartqa")
    .add_local_file("qwen_finetune.py", remote_path="/root/qwen_finetune.py")
)


def _run_cli(argv: list[str], output_dir: str) -> int:
    import pathlib

    import qwen_finetune

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    exit_code = qwen_finetune.main_with_argv(argv)
    volume.commit()
    return exit_code


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=30 * MINUTES)
def run_cli_cpu(argv: list[str], output_dir: str) -> int:
    return _run_cli(argv, output_dir)


@app.function(image=image, gpu=GPU, volumes={VOLUME_PATH: volume}, timeout=TRAIN_TIMEOUT_SECONDS)
def run_cli_gpu(argv: list[str], output_dir: str) -> int:
    return _run_cli(argv, output_dir)


@app.function(volumes={VOLUME_PATH: volume}, timeout=5 * MINUTES)
def _list_volume(path: str) -> str:
    import pathlib

    target = pathlib.Path(path)
    if not target.exists():
        return f"{path} does not exist on the volume"
    lines = [f"{path}:"]
    for entry in sorted(target.rglob("*")):
        if entry.is_file():
            lines.append(f"  {entry} ({entry.stat().st_size} bytes)")
    return "\n".join(lines) if len(lines) > 1 else f"{path} is empty"


@app.local_entrypoint()
def list_volume(path: str = VOLUME_PATH) -> None:
    """Cheap, GPU-free listing of what actually landed on the volume, to confirm --root-dir."""
    print(_list_volume.remote(path))


@app.local_entrypoint()
def upload_data(local_dir: str = "Data") -> None:
    """One-time client-side upload; equivalent to `modal volume put qwen ./Data /Data`."""
    import pathlib

    local_path = pathlib.Path(local_dir)
    if not local_path.is_dir():
        raise SystemExit(f"{local_path} is not a directory")
    with volume.batch_upload() as batch:
        batch.put_directory(str(local_path), "/Data")
    print(f"Uploaded {local_path} to volume '{VOLUME_NAME}:/Data'")


@app.local_entrypoint()
def manifest(root_dir: str = VOLUME_PATH, seed: int = 42, val_ratio: float = 0.2) -> None:
    data_dir, output_dir = _data_dir(root_dir), _output_dir(root_dir)
    argv = [
        "manifest",
        "--human", f"{data_dir}/human_train.json",
        "--auto", f"{data_dir}/auto_train.json",
        "--output", f"{output_dir}/manifest.json",
        "--seed", str(seed),
        "--val-ratio", str(val_ratio),
    ]
    print(f"manifest exit_code={run_cli_cpu.remote(argv, output_dir)}")


@app.local_entrypoint()
def audit(root_dir: str = VOLUME_PATH, model_id: str = "Qwen/Qwen3-VL-8B-Instruct") -> None:
    data_dir, output_dir = _data_dir(root_dir), _output_dir(root_dir)
    argv = [
        "audit",
        "--human", f"{data_dir}/human_train.json",
        "--auto", f"{data_dir}/auto_train.json",
        "--model-id", model_id,
        "--output", f"{output_dir}/audit.json",
    ]
    print(f"audit exit_code={run_cli_cpu.remote(argv, output_dir)}")


@app.local_entrypoint()
def inspect_modules(root_dir: str = VOLUME_PATH, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", revision: str | None = None) -> None:
    argv = ["inspect-modules", "--model-id", model_id]
    if revision:
        argv += ["--revision", revision]
    print(f"inspect-modules exit_code={run_cli_gpu.remote(argv, _output_dir(root_dir))}")


@app.local_entrypoint()
def train_stage_a(
    root_dir: str = VOLUME_PATH, auto_subset_size: int = 30000, max_steps: int = 800, micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 8, learning_rate: float = 1e-4, seed: int = 42,
) -> None:
    data_dir, output_dir = _data_dir(root_dir), _output_dir(root_dir)
    argv = [
        "train-stage-a",
        "--manifest", f"{output_dir}/manifest.json",
        "--human", f"{data_dir}/human_train.json",
        "--auto", f"{data_dir}/auto_train.json",
        "--output-dir", f"{output_dir}/stage_a",
        "--auto-subset-size", str(auto_subset_size),
        "--max-steps", str(max_steps),
        "--micro-batch-size", str(micro_batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--learning-rate", str(learning_rate),
        "--seed", str(seed),
    ]
    print(f"train-stage-a exit_code={run_cli_gpu.remote(argv, output_dir)}")


@app.local_entrypoint()
def train_stage_b(
    max_steps: int, root_dir: str = VOLUME_PATH, init_adapter: str | None = None, human_passes: float = 3.0,
    micro_batch_size: int = 4, gradient_accumulation_steps: int = 8, learning_rate: float = 3e-5, seed: int = 42,
) -> None:
    data_dir, output_dir = _data_dir(root_dir), _output_dir(root_dir)
    argv = [
        "train-stage-b",
        "--manifest", f"{output_dir}/manifest.json",
        "--human", f"{data_dir}/human_train.json",
        "--auto", f"{data_dir}/auto_train.json",
        "--output-dir", f"{output_dir}/stage_b",
        "--init-adapter", init_adapter or f"{output_dir}/stage_a/adapter",
        "--human-passes", str(human_passes),
        "--max-steps", str(max_steps),
        "--micro-batch-size", str(micro_batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--learning-rate", str(learning_rate),
        "--seed", str(seed),
    ]
    print(f"train-stage-b exit_code={run_cli_gpu.remote(argv, output_dir)}")


@app.local_entrypoint()
def evaluate(
    root_dir: str = VOLUME_PATH, adapter: str | None = None, baseline_metrics: str | None = None,
    output_name: str = "metrics_base.json", max_new_tokens: int = 64, batch_size: int = 4,
) -> None:
    data_dir, output_dir = _data_dir(root_dir), _output_dir(root_dir)
    argv = [
        "evaluate",
        "--manifest", f"{output_dir}/manifest.json",
        "--human", f"{data_dir}/human_train.json",
        "--auto", f"{data_dir}/auto_train.json",
        "--output", f"{output_dir}/{output_name}",
        "--max-new-tokens", str(max_new_tokens),
        "--batch-size", str(batch_size),
    ]
    if adapter:
        argv += ["--adapter", adapter]
    if baseline_metrics:
        argv += ["--baseline-metrics", baseline_metrics]
    print(f"evaluate exit_code={run_cli_gpu.remote(argv, output_dir)}")


@app.local_entrypoint()
def predict(
    input_path: str, adapter: str, output_name: str, root_dir: str = VOLUME_PATH,
    max_new_tokens: int = 64, batch_size: int = 4,
) -> None:
    output_dir = _output_dir(root_dir)
    argv = [
        "predict",
        "--input", input_path,
        "--adapter", adapter,
        "--output", f"{output_dir}/{output_name}",
        "--max-new-tokens", str(max_new_tokens),
        "--batch-size", str(batch_size),
    ]
    print(f"predict exit_code={run_cli_gpu.remote(argv, output_dir)}")


# Two-agent (extraction -> reasoning) pipeline. distill_graphs is CPU-only: it just
# parses a `gpt_experiment.py --method pot` responses.jsonl already produced/uploaded
# to the Volume; the API calls themselves are not run on Modal.
@app.local_entrypoint()
def distill_graphs(pot_log: str, source: str, root_dir: str = VOLUME_PATH, output_name: str = "graphs.json") -> None:
    output_dir = _output_dir(root_dir)
    argv = ["distill-graphs", "--pot-log", pot_log, "--source", source, "--output", f"{output_dir}/{output_name}"]
    print(f"distill-graphs exit_code={run_cli_cpu.remote(argv, output_dir)}")


@app.local_entrypoint()
def train_agent_extraction(
    max_steps: int, root_dir: str = VOLUME_PATH, graphs: str | None = None, max_examples: int | None = None,
    micro_batch_size: int = 4, gradient_accumulation_steps: int = 8, learning_rate: float = 1e-4, seed: int = 42,
) -> None:
    data_dir, output_dir = _data_dir(root_dir), _output_dir(root_dir)
    argv = [
        "train-agent-extraction",
        "--manifest", f"{output_dir}/manifest.json",
        "--human", f"{data_dir}/human_train.json",
        "--auto", f"{data_dir}/auto_train.json",
        "--output-dir", f"{output_dir}/agent_extraction",
        "--graphs", graphs or f"{output_dir}/graphs.json",
        "--max-steps", str(max_steps),
        "--micro-batch-size", str(micro_batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--learning-rate", str(learning_rate),
        "--seed", str(seed),
    ]
    if max_examples is not None:
        argv += ["--max-examples", str(max_examples)]
    print(f"train-agent-extraction exit_code={run_cli_gpu.remote(argv, output_dir)}")


@app.local_entrypoint()
def train_agent_reasoning(
    max_steps: int, root_dir: str = VOLUME_PATH, graphs: str | None = None, human_passes: float = 3.0,
    micro_batch_size: int = 4, gradient_accumulation_steps: int = 8, learning_rate: float = 3e-5, seed: int = 42,
) -> None:
    data_dir, output_dir = _data_dir(root_dir), _output_dir(root_dir)
    argv = [
        "train-agent-reasoning",
        "--manifest", f"{output_dir}/manifest.json",
        "--human", f"{data_dir}/human_train.json",
        "--auto", f"{data_dir}/auto_train.json",
        "--output-dir", f"{output_dir}/agent_reasoning",
        "--graphs", graphs or f"{output_dir}/graphs.json",
        "--human-passes", str(human_passes),
        "--max-steps", str(max_steps),
        "--micro-batch-size", str(micro_batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--learning-rate", str(learning_rate),
        "--seed", str(seed),
    ]
    print(f"train-agent-reasoning exit_code={run_cli_gpu.remote(argv, output_dir)}")


@app.local_entrypoint()
def evaluate_pipeline(
    root_dir: str = VOLUME_PATH, extraction_adapter: str | None = None, reasoning_adapter: str | None = None,
    baseline_metrics: str | None = None, output_name: str = "metrics_pipeline.json",
    graph_max_new_tokens: int = 512, answer_max_new_tokens: int = 64, batch_size: int = 4,
) -> None:
    data_dir, output_dir = _data_dir(root_dir), _output_dir(root_dir)
    argv = [
        "evaluate-pipeline",
        "--manifest", f"{output_dir}/manifest.json",
        "--human", f"{data_dir}/human_train.json",
        "--auto", f"{data_dir}/auto_train.json",
        "--extraction-adapter", extraction_adapter or f"{output_dir}/agent_extraction/adapter",
        "--reasoning-adapter", reasoning_adapter or f"{output_dir}/agent_reasoning/adapter",
        "--output", f"{output_dir}/{output_name}",
        "--graph-max-new-tokens", str(graph_max_new_tokens),
        "--answer-max-new-tokens", str(answer_max_new_tokens),
        "--batch-size", str(batch_size),
    ]
    if baseline_metrics:
        argv += ["--baseline-metrics", baseline_metrics]
    print(f"evaluate-pipeline exit_code={run_cli_gpu.remote(argv, output_dir)}")


@app.local_entrypoint()
def predict_pipeline(
    input_path: str, output_name: str, root_dir: str = VOLUME_PATH, extraction_adapter: str | None = None,
    reasoning_adapter: str | None = None, graph_max_new_tokens: int = 512, answer_max_new_tokens: int = 64,
    batch_size: int = 4,
) -> None:
    output_dir = _output_dir(root_dir)
    argv = [
        "predict-pipeline",
        "--input", input_path,
        "--extraction-adapter", extraction_adapter or f"{output_dir}/agent_extraction/adapter",
        "--reasoning-adapter", reasoning_adapter or f"{output_dir}/agent_reasoning/adapter",
        "--output", f"{output_dir}/{output_name}",
        "--graph-max-new-tokens", str(graph_max_new_tokens),
        "--answer-max-new-tokens", str(answer_max_new_tokens),
        "--batch-size", str(batch_size),
    ]
    print(f"predict-pipeline exit_code={run_cli_gpu.remote(argv, output_dir)}")
