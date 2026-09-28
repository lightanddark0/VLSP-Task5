"""Hugging Face Hub storage: private repos for adapters, predictions, and data.

Every helper is a no-op (returning None) when no HF token is available, so the
same scripts run locally without the Hub. Upload failures only print a
warning: a network error must not abort a long inference run. Downloads that a
run depends on (adapters) raise instead.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def hub_enabled() -> bool:
    if os.environ.get("HF_HUB_DISABLE", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token
    except ImportError:
        return False
    return get_token() is not None


@lru_cache(maxsize=1)
def hub_user() -> str:
    if os.environ.get("HF_USER"):
        return os.environ["HF_USER"]
    from huggingface_hub import HfApi
    return HfApi().whoami()["name"]


def resolve_repo(name: str | None) -> str | None:
    """'name' becomes '<user>/name'; 'org/name' is kept; empty stays None."""
    if not name:
        return None
    return name if "/" in name else f"{hub_user()}/{name}"


def repo_url(repo_id: str, repo_type: str = "model", revision: str | None = None) -> str:
    prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[repo_type]
    return f"https://huggingface.co/{prefix}{repo_id}" + (f"/tree/{revision}" if revision else "")


def ensure_repo(repo_id: str | None, repo_type: str = "model") -> str | None:
    if not repo_id or not hub_enabled():
        return None
    from huggingface_hub import HfApi
    HfApi().create_repo(repo_id, private=True, exist_ok=True, repo_type=repo_type)
    return repo_id


def _commit_oid(info: Any) -> str | None:
    return getattr(info, "oid", None)


def upload_file(path: str | Path, repo_id: str | None, path_in_repo: str,
                repo_type: str = "dataset", message: str | None = None) -> str | None:
    """Upload one file; returns the commit hash, or None if skipped or failed."""
    if not repo_id or not hub_enabled() or not Path(path).exists():
        return None
    try:
        from huggingface_hub import HfApi
        ensure_repo(repo_id, repo_type)
        info = HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=repo_id,
                                   repo_type=repo_type, commit_message=message or f"Upload {path_in_repo}")
        return _commit_oid(info)
    except Exception as error:  # noqa: BLE001 - never abort a run over an upload
        print(f"Warning: upload of {path} to {repo_id} failed: {error}")
        return None


def upload_folder(folder: str | Path, repo_id: str | None, path_in_repo: str = "",
                  repo_type: str = "dataset", message: str | None = None,
                  allow_patterns: list[str] | None = None, ignore_patterns: list[str] | None = None) -> str | None:
    if not repo_id or not hub_enabled() or not Path(folder).is_dir():
        return None
    try:
        from huggingface_hub import HfApi
        ensure_repo(repo_id, repo_type)
        info = HfApi().upload_folder(folder_path=str(folder), path_in_repo=path_in_repo or None, repo_id=repo_id,
                                     repo_type=repo_type, commit_message=message or f"Upload {path_in_repo or folder}",
                                     allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
        return _commit_oid(info)
    except Exception as error:  # noqa: BLE001
        print(f"Warning: upload of {folder} to {repo_id} failed: {error}")
        return None


def repo_has_path(repo_id: str | None, prefix: str, repo_type: str = "model") -> bool:
    if not repo_id or not hub_enabled():
        return False
    from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files(repo_id, repo_type=repo_type)
    except Exception:  # noqa: BLE001 - a missing repo means nothing to resume
        return False
    return any(name == prefix or name.startswith(prefix.rstrip("/") + "/") for name in files)


def download(repo_id: str, local_dir: str | Path, repo_type: str = "model",
             allow_patterns: list[str] | None = None, ignore_patterns: list[str] | None = None,
             revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(local_dir), revision=revision,
                             allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
    return Path(path)


def latest_commit(repo_id: str | None, repo_type: str = "model") -> str | None:
    if not repo_id or not hub_enabled():
        return None
    try:
        from huggingface_hub import HfApi
        return HfApi().list_repo_commits(repo_id, repo_type=repo_type)[0].commit_id
    except Exception:  # noqa: BLE001
        return None


def resolve_adapter(adapter: str, cache_dir: str | Path) -> tuple[Path, str | None]:
    """Local adapter directory for a path or a Hub repo id, plus the Hub commit if any.

    Training checkpoints under last-checkpoint/ are not downloaded.
    """
    if Path(adapter).exists():
        return Path(adapter), None
    local = download(adapter, Path(cache_dir) / adapter.replace("/", "__"),
                     ignore_patterns=["last-checkpoint/*", "checkpoint-*/*", "*.pt", "*.bin.index.json"])
    return local, latest_commit(adapter)
