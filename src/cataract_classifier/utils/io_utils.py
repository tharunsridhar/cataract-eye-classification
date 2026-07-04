from __future__ import annotations

from pathlib import Path


def make_versioned_filename(base_name: str, accuracy: float, ext: str) -> str:
    accuracy_str = f"{accuracy:.2f}".replace(".", "_")
    return f"{base_name}_{accuracy_str}.{ext.lstrip('.')}"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
