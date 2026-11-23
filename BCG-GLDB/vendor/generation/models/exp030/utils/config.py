"""Load Exp030 as a recursive override of the Exp029 configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from exp027.utils.config import validate_config
from exp028.utils.config import _deep_merge


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle)
    if not isinstance(override, dict):
        raise ValueError("Exp030 YAML root must be a mapping")
    base_name = override.pop("base_config", None)
    if not base_name:
        raise ValueError("Exp030 config must declare base_config")
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = path.parents[1] / base_path
    base_path = base_path.resolve()
    # The base may itself be an Exp030-style config (it declares the Exp030
    # ``model.performance_refiner`` section).  In that case recursively resolve
    # the full ancestor chain with this loader instead of handing the path to
    # exp029's loader, which would misalign the chain one level and make
    # exp028's loader treat exp028 (a partial override, no ``model`` section)
    # as the terminal base.
    base_yaml = yaml.safe_load(base_path.open("r", encoding="utf-8"))
    is_exp030_base = (
        isinstance(base_yaml, dict)
        and isinstance(base_yaml.get("model"), dict)
        and "performance_refiner" in base_yaml["model"]
    )
    if is_exp030_base:
        base = load_config(base_path)
    else:
        from exp029.utils.config import load_config as load_exp029_config

        base = load_exp029_config(base_path)
    config = _deep_merge(base, override)
    validate_config(config)
    config["experiment"]["config_path"] = str(path)
    config["experiment"]["base_config_path"] = str(base_path.resolve())
    return config


__all__ = ["load_config"]
