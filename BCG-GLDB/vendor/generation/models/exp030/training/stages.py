"""Exp030 freeze policy: Exp027's audited policy plus the performance refiner.

The new `performance_refiner` module is enabled only during `joint_finetune`
(and the terminal calibration stages that share its trainable set, matching how
`image_refiner` is handled).  All other stages keep the exact Exp027 behaviour.
"""

from __future__ import annotations

from typing import Any, Iterable

from exp027.training.stages import (
    STAGE_NAMES,
    configure_trainable_modules as _configure_exp027,
    set_stage_mode as _set_stage_mode_exp027,
    trainable_parameter_count,
)


def _set_prefixes(model: Any, prefixes: Iterable[str]) -> None:
    prefixes = tuple(prefixes)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        )


# Stages that own the image-domain refinement stack (image_refiner is included by
# the base Exp027 policy; performance_refiner is the Exp030 addition).
_REFINER_STAGES = (
    "joint_finetune",
    "decoder_calibration",
    "structural_decoder",
    "final_calibration",
)


def configure_trainable_modules(model: Any, stage_name: str) -> list[str]:
    # Exp027's policy freezes every parameter then unlocks the stage's own
    # modules.  We must ONLY additionally unlock performance_refiner here --
    # calling _set_prefixes again would re-freeze all stage modules (e.g.
    # image_refiner / mri_control) and silently disable joint_finetune.
    prefixes = _configure_exp027(model, stage_name)
    if stage_name in _REFINER_STAGES:
        for name, parameter in model.named_parameters():
            if name == "performance_refiner" or name.startswith("performance_refiner."):
                parameter.requires_grad = True
        prefixes = list(prefixes) + ["performance_refiner"]
    return prefixes


def set_stage_mode(model: Any, stage_name: str, training: bool = True) -> None:
    """Keep frozen modules in eval mode and only active modules in train mode."""

    model.eval()
    configure_trainable_modules(model, stage_name)
    if training:
        for module in model.modules():
            module.training = any(
                parameter.requires_grad for parameter in module.parameters()
            )


__all__ = [
    "STAGE_NAMES",
    "configure_trainable_modules",
    "set_stage_mode",
    "trainable_parameter_count",
]
