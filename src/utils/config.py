from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
