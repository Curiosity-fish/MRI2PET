"""Configuration loading and geometry validation for Exp027."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from exp027.training.stages import STAGE_NAMES


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three integers")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def validate_config(config: dict) -> dict:
    for section in ("experiment", "data", "model", "diffusion", "training", "stages"):
        if section not in config:
            raise ValueError(f"Missing Exp027 config section: {section}")
    image_size = _triple(config["data"].get("image_size"), "data.image_size")
    patch_size = _triple(config["data"].get("patch_size"), "data.patch_size")
    if any(image_size[index] % patch_size[index] for index in range(3)):
        raise ValueError("data.patch_size must divide data.image_size exactly")
    if any(value % 4 for value in (*image_size, *patch_size)):
        raise ValueError("image and patch dimensions must be divisible by the DVAE factor 4")
    missing = [name for name in STAGE_NAMES if name not in config["stages"]]
    extra = [name for name in config["stages"] if name not in STAGE_NAMES]
    if missing or extra:
        raise ValueError(f"Exp027 stage names mismatch; missing={missing}, extra={extra}")
    for name in STAGE_NAMES:
        stage = config["stages"][name]
        epochs = stage.get("epochs")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 0:
            raise ValueError(f"stages.{name}.epochs must be a non-negative integer")
        if stage.get("enabled", True) and epochs > 0 and float(stage.get("learning_rate", 0)) <= 0:
            raise ValueError(f"stages.{name}.learning_rate must be positive")
    patch_count = 1
    for index in range(3):
        patch_count *= image_size[index] // patch_size[index]
    patch_batch = int(config["training"].get("patch_batch_size", 1))
    if not 0 < patch_batch <= patch_count:
        raise ValueError(f"training.patch_batch_size must be in [1,{patch_count}]")
    if int(config["diffusion"].get("timesteps", 0)) < 2:
        raise ValueError("diffusion.timesteps must be at least 2")
    global_config = config["model"].get("global_mamba", {})
    backend = str(global_config.get("backend", "mamba"))
    if backend not in {"mamba", "debug_gru"}:
        raise ValueError("model.global_mamba.backend must be 'mamba' or 'debug_gru'")
    if backend == "debug_gru" and not bool(global_config.get("allow_test_backend", False)):
        raise ValueError("debug_gru requires model.global_mamba.allow_test_backend=true")
    return config


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")
    validate_config(config)
    config["experiment"]["config_path"] = str(path)
    return config
