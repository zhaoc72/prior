"""Input/output helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_index(index: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, index, allow_pickle=True)

__all__ = ["load_json", "save_index"]
