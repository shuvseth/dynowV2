"""Repository-relative path helpers."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(p: str | Path) -> Path:
    """Resolve a path relative to the repository root unless it is absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
