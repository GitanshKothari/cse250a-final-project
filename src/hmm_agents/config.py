from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = ROOT / "configs" / "local.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if LOCAL_CONFIG.exists():
        with open(LOCAL_CONFIG, encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}
        unknown = set(local) - {"paths"}
        if unknown:
            raise ValueError(f"{LOCAL_CONFIG.name} may only override `paths`, got {sorted(unknown)}")
        cfg["paths"].update(local.get("paths") or {})
    for k, v in cfg["paths"].items():
        p = ROOT / v
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][k] = p
    return cfg
