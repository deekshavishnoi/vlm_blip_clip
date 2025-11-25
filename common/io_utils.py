"""
common/io_utils.py

Small, dependency-light utilities shared across tasks.

WHY THIS FILE EXISTS
--------------------
Each task (captioning, analysis, chatbot) needs the same boring-but-important
plumbing: create folders, read/write simple files, consistent logging, etc.
Putting these here avoids copy–pasting and keeps task code focused on logic.

WHAT'S INSIDE
-------------
- ensure_dir: safely create folders
- read_yaml: load small configs
- save_jsonl / load_jsonl: appendable, streaming-friendly records
- save_parquet: write DataFrames (optional; used in Task 2)
- relpath: stable relative paths for manifests
- list_images: quick discovery of frame files
- set_seed: basic reproducibility
- init_logger: consistent logging setup
- write_text: save tiny text artifacts (descriptions, notes)
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, List
import json
import logging
import os
import random

try:
    import yaml  # for reading configs
except ImportError:  # pragma: no cover
    yaml = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# -----------------------
# Paths & filesystem
# -----------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist; return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def relpath(p: str | Path, base: str | Path = ".") -> str:
    """Return relative path string from 'base' to 'p'."""
    return os.path.relpath(str(p), str(base))


def list_images(root: str | Path, exts: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    """Recursively list image files under a directory."""
    root = Path(root)
    paths: List[Path] = []
    for ext in exts:
        paths.extend(root.rglob(f"*{ext}"))
    return sorted(paths)


# -----------------------
# Configs & small files
# -----------------------

def read_yaml(path: str | Path) -> dict:
    """Load a YAML config into a dict."""
    if yaml is None:
        raise ImportError("PyYAML is not installed. Add 'pyyaml' to requirements.txt.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """Write an iterable of dicts to JSONL (one JSON per line)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_text(path: str | Path, text: str) -> None:
    """Write plain text to a file (UTF-8)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# -----------------------
# DataFrames & Parquet (Task 2)
# -----------------------

def save_parquet(df, path: str | Path) -> None:
    """
    Save a pandas DataFrame to Parquet.
    Used for objects/scenes tables in Task 2.
    """
    if pd is None:
        raise ImportError("pandas/pyarrow not installed. Add 'pandas' and 'pyarrow' to requirements.txt.")
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


# -----------------------
# Reproducibility & logging
# -----------------------

def set_seed(seed: int = 42) -> None:
    """Set seeds for random & numpy (if available) to make results repeatable."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def init_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create a console logger with timestamps.
    Use the same format across all tasks to keep outputs consistent.
    """
    logger = logging.getLogger(name if name else "bmw")
    if logger.handlers:
        # Reuse existing handler if already configured (avoid duplicated logs)
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger
