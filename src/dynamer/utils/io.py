from __future__ import annotations

from pathlib import Path
from typing import Iterable


def resolve_existing(root: str | Path, relative_path: str | Path) -> Path:
    fp = Path(root) / Path(relative_path)
    if not fp.exists():
        raise FileNotFoundError(fp)
    return fp


def file_exists(root: str | Path, relative_path: str | Path) -> bool:
    return (Path(root) / Path(relative_path)).exists()


def count_existing(root: str | Path, relative_paths: Iterable[str | Path]) -> int:
    return sum(file_exists(root, p) for p in relative_paths)
