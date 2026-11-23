from __future__ import annotations

import torch

from exp026.utils.config import STAGE_NAMES
from exp025t.models.exp025t_model import Exp025TMRI2PET


def _requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _enable_calibration(model: Exp025TMRI2PET) -> None:
    _requires_grad(model.metric_head, True)
    model.smooth_weight_logit.requires_grad_(True)
    model.atlas_weight_logit.requires_grad_(True)


def configure_trainable_modules(
    model: Exp025TMRI2PET, stage: str
) -> list[str]:
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage}")
    _requires_grad(model, False)
    if stage == "joint_finetune":
        _enable_calibration(model)
        return ["metric_head", "smooth_weight", "atlas_weight"]
    if stage == "decoder_calibration":
        _requires_grad(model.image_refiner, True)
        _enable_calibration(model)
        return [
            "image_refiner",
            "metric_head",
            "smooth_weight",
            "atlas_weight",
        ]
    raise ValueError(
        f"Exp025t only trains joint_finetune and decoder_calibration, got {stage}"
    )


def set_stage_mode(model: Exp025TMRI2PET, stage: str, training: bool) -> None:
    model.eval()
    if not training:
        return
    if stage == "joint_finetune":
        model.metric_head.train()
    elif stage == "decoder_calibration":
        model.image_refiner.train()
        model.metric_head.train()
    else:
        raise ValueError(f"Unsupported Exp025t training stage: {stage}")


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
