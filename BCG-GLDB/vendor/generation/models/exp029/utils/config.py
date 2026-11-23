"""Load Exp029 as a recursive override of the Exp028 configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from exp027.utils.config import validate_config
from exp028.utils.config import _deep_merge


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle)
    if not isinstance(override, dict):
        raise ValueError("Exp029 YAML root must be a mapping")
    base_name = override.pop("base_config", None)
    if not base_name:
        raise ValueError("Exp029 config must declare base_config")
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = path.parents[1] / base_path
    from exp028.utils.config import load_config as load_exp028_config

    base = load_exp028_config(base_path.resolve())
    config = _deep_merge(base, override)
    validate_config(config)
    config["experiment"]["config_path"] = str(path)
    config["experiment"]["base_config_path"] = str(base_path.resolve())
    return config
