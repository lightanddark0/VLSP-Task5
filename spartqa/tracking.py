"""Weights & Biases tracking: metrics, configs, timings, and small tables only.

Models and checkpoints are never uploaded to W&B (WANDB_LOG_MODEL=false);
runs record the Hugging Face repo and commit hash instead. Every function is a
no-op when wandb is not installed, WANDB_MODE=disabled, or no API key is set
(offline mode still records locally for a later ``wandb sync``).
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from spartqa.repro import run_metadata

PROJECT = "vispatialqa-vlsp2026"
RUN_ID_FILE = "wandb_run_id.txt"

_run: Any = None
_started = time.time()


def tracking_enabled() -> bool:
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode == "disabled":
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return mode == "offline" or bool(os.environ.get("WANDB_API_KEY")) or Path.home().joinpath(".netrc").exists()


def run_name(branch: str, job: str, dataset: str, split: str, note: str = "") -> str:
    parts = [branch, job, dataset, split, note, time.strftime("%Y%m%d-%H%M")]
    return "-".join(part for part in parts if part)


def environment() -> dict[str, Any]:
    info = run_metadata()
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name()
    except ImportError:
        pass
    for name in ("vllm", "wandb", "huggingface_hub"):
        try:
            from importlib import metadata
            info["packages"][name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            continue
    return info


def init_run(group: str, job_type: str, name: str, tags: list[str], config: dict[str, Any],
             run_dir: str | Path | None = None, run_id: str | None = None) -> Any:
    """Create a run, or resume the one whose id is stored in ``run_dir``."""
    global _run, _started
    _started = time.time()
    if not tracking_enabled():
        print("W&B tracking disabled (no wandb, no WANDB_API_KEY, or WANDB_MODE=disabled).")
        return None
    import wandb
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_WATCH", "false")
    id_path = Path(run_dir) / RUN_ID_FILE if run_dir else None
    if run_id is None and id_path is not None and id_path.exists():
        run_id = id_path.read_text().strip() or None
    run_id = run_id or uuid.uuid4().hex[:12]
    if id_path is not None:
        id_path.parent.mkdir(parents=True, exist_ok=True)
        id_path.write_text(run_id)
    _run = wandb.init(project=os.environ.get("WANDB_PROJECT", PROJECT), entity=os.environ.get("WANDB_ENTITY") or None,
                      group=group, job_type=job_type, name=name, tags=[tag for tag in tags if tag],
                      config={**config, "env": environment()}, id=run_id, resume="allow")
    return _run


def active() -> bool:
    return _run is not None


def update_config(values: dict[str, Any]) -> None:
    if _run is not None:
        _run.config.update(values, allow_val_change=True)


def log_hf_link(repo_id: str | None, revision: str | None, kind: str, repo_type: str = "model") -> None:
    """Record where a model or prediction file lives; nothing is uploaded to W&B."""
    if _run is None or not repo_id:
        return
    from spartqa.hub import repo_url
    update_config({f"hf/{kind}/repo": repo_id, f"hf/{kind}/revision": revision,
                   f"hf/{kind}/url": repo_url(repo_id, repo_type, revision)})


def log(values: dict[str, Any], summary: bool = True) -> None:
    if _run is None:
        return
    values = {key: value for key, value in values.items() if value is not None}
    _run.log(values)
    if summary:
        for key, value in values.items():
            _run.summary[key] = value


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}/{key}", inner, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value


def log_eval(report: dict[str, Any], prefix: str) -> None:
    """Log ``by_task`` metrics as <prefix>/<q_type>/<metric>, plus detail numbers."""
    values: dict[str, Any] = {}
    _flatten(prefix, report.get("by_task", {}), values)
    if "primary_macro_unofficial" in report:
        values[f"{prefix}/primary_macro_unofficial"] = report["primary_macro_unofficial"]
    _flatten(f"{prefix}/details", report.get("details", {}), values)
    log(values)


def log_table(name: str, rows: list[list[Any]], columns: list[str], max_rows: int = 200) -> None:
    if _run is None:
        return
    import wandb
    _run.log({name: wandb.Table(columns=columns, data=[list(row) for row in rows[:max_rows]])})


def log_confusion(name: str, gold: list[Any], predicted: list[Any], classes: list[Any]) -> None:
    if _run is None or not gold:
        return
    import wandb
    index = {label: position for position, label in enumerate(classes)}
    pairs = [(index[g], index[p]) for g, p in zip(gold, predicted) if g in index and p in index]
    if pairs:
        _run.log({name: wandb.plot.confusion_matrix(y_true=[g for g, _ in pairs], preds=[p for _, p in pairs],
                                                    class_names=[str(label) for label in classes])})


@contextmanager
def timer(name: str, gpu: bool = True) -> Iterator[dict[str, float]]:
    """Measure wall time; logs time/<name>_sec and GPU hours (1 GPU)."""
    result: dict[str, float] = {}
    start = time.time()
    try:
        yield result
    finally:
        result["seconds"] = time.time() - start
        values = {f"time/{name}_sec": result["seconds"]}
        if gpu:
            values[f"time/{name}_gpu_hours"] = result["seconds"] / 3600
        print(f"[time] {name}: {result['seconds'] / 60:.1f} min")
        log(values)


def finish() -> None:
    global _run
    if _run is None:
        return
    values = {"time/total_sec": time.time() - _started}
    try:
        import torch
        if torch.cuda.is_available():
            values["resources/peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    except ImportError:
        pass
    log(values)
    _run.finish()
    _run = None
