"""Load and resolve project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    for key in ("data_dir", "raw_dir", "processed_dir", "models_dir", "reports_dir"):
        rel = cfg["paths"][key]
        cfg["paths"][key] = str(ROOT / rel)

    cfg["root"] = str(ROOT)
    return cfg


def day_to_int(day: str) -> int:
    """Convert M5 day column name (e.g. d_1885 or d1885) to integer index."""
    s = day.strip()
    if s.startswith("d_"):
        return int(s[2:])
    if s.startswith("d"):
        return int(s[1:])
    return int(s)
