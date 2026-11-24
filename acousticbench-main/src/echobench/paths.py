"""Locate the project root and the runtime folders."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    override = os.environ.get("ECHOBENCH_HOME")
    if override:
        return Path(override).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "pyproject.toml").is_file() and (cand / "src" / "echobench").is_dir():
            return cand
    return here


def data_root(root: Path | None = None) -> Path:
    folder = (root or find_root()) / "data"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def db_file(root: Path | None = None) -> Path:
    return data_root(root) / "echobench.db"


def runs_root(root: Path | None = None) -> Path:
    folder = data_root(root) / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def ensure_importable(root: Path) -> None:
    text = str(root.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)
