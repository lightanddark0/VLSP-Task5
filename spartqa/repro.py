"""Seeding and run metadata for reproducible fine-tuning experiments."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRACKED_PACKAGES = ("numpy", "torch", "transformers", "peft", "accelerate", "datasets", "sentencepiece")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch when they are installed.

    deterministic=True also requests deterministic PyTorch kernels, which can be
    slower. Call this before building models, data loaders, or CUDA tensors.
    """
    random.seed(seed)
    try:
        import numpy
        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def run_metadata(**extra: Any) -> dict[str, Any]:
    """Code version and environment details to save next to each run's outputs."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        **extra,
    }
