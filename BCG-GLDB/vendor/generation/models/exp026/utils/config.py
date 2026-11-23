from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


STAGE_NAMES = (
    "patch_vae",
    "mri_patch_prior",
    "local_control",
    "table_mri_prior",
    "global_mamba",
    "joint_finetune",
    "decoder_calibration",
)


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three integers")
    result = tuple(int(v) for v in value)
    if any(v <= 0 for v in result):
        raise ValueError(f"{name} values must be positive")
    return result


def validate_config(config: dict) -> dict:
    for section in ("experiment", "data", "model", "diffusion", "training", "stages"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    image_size = _triple(config["data"].get("image_size"), "data.image_size")
    patch_size = _triple(config["data"].get("patch_size"), "data.patch_size")
    if any(image_size[i] % patch_size[i] != 0 for i in range(3)):
        raise ValueError("data.patch_size must divide data.image_size exactly")
    if any(v % 4 != 0 for v in (*image_size, *patch_size)):
        raise ValueError("image and patch dimensions must be divisible by the DVAE factor 4")
    missing = [name for name in STAGE_NAMES if name not in config["stages"]]
    extra = [name for name in config["stages"] if name not in STAGE_NAMES]
    if missing or extra:
        raise ValueError(f"Stage names mismatch; missing={missing}, extra={extra}")
    for name in STAGE_NAMES:
        stage = config["stages"][name]
        if not isinstance(stage.get("enabled", True), bool):
            raise ValueError(f"stages.{name}.enabled must be boolean")
        epochs = stage.get("epochs")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 0:
            raise ValueError(f"stages.{name}.epochs must be a non-negative integer")
        if stage.get("enabled", True) and epochs > 0 and float(stage.get("learning_rate", 0)) <= 0:
            raise ValueError(f"stages.{name}.learning_rate must be positive")
    patch_count = 1
    for i in range(3):
        patch_count *= image_size[i] // patch_size[i]
    patch_batch = int(config["training"].get("patch_batch_size", 1))
    if patch_batch <= 0 or patch_batch > patch_count:
        raise ValueError(f"training.patch_batch_size must be in [1,{patch_count}]")
    if int(config["diffusion"].get("timesteps", 0)) < 2:
        raise ValueError("diffusion.timesteps must be at least 2")
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
