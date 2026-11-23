"""Config loader with a small, explicit Exp027 base-config inheritance layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from exp027.utils.config import validate_config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle)
    if not isinstance(override, dict):
        raise ValueError("Exp028 YAML root must be a mapping")
    base_name = override.pop("base_config", None)
    if not base_name:
        raise ValueError("Exp028 config must declare base_config")
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = path.parents[1] / base_path
    with base_path.resolve().open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    if not isinstance(base, dict):
        raise ValueError("Exp028 base config must be a mapping")
    config = _deep_merge(base, override)
    validate_config(config)
    config["experiment"]["config_path"] = str(path)
    config["experiment"]["base_config_path"] = str(base_path.resolve())
    return config
