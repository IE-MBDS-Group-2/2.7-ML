"""Read config.yaml (the single source of truth for non-secret settings)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repository root: the folder that holds config.yaml.

    ML2_REPO_ROOT overrides the search (used inside the Docker image).
    """
    env = os.environ.get("ML2_REPO_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [Path.cwd(), *Path.cwd().parents, *here.parents]:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("config.yaml not found; run from inside the repository or set ML2_REPO_ROOT")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml as a dictionary."""
    cfg_path = Path(path) if path else repo_root() / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative: str | Path) -> Path:
    """Turn a path from config.yaml into an absolute path under the repo root."""
    p = Path(relative)
    return p if p.is_absolute() else repo_root() / p
