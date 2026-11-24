from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = ROOT / v
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][k] = p
    return cfg
