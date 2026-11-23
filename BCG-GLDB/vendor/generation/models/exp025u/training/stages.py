from __future__ import annotations

import torch

from exp026.utils.config import STAGE_NAMES
from exp025u.models.exp025u_model import Exp025UMRI2PET


def _requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _enable_spatial_scalars(model: Exp025UMRI2PET) -> None:
    model.post_atlas_weight_logit.requires_grad_(True)
    model.residual_prior_strength_logit.requires_grad_(True)
    model.post_smooth_weight_logit.requires_grad_(True)


def configure_trainable_modules(
    model: Exp025UMRI2PET, stage: str
) -> list[str]:
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage}")
    _requires_grad(model, False)
    if stage == "joint_finetune":
        _enable_spatial_scalars(model)
        return [
            "post_atlas_weight",
            "residual_prior_strength",
            "post_smooth_weight",
        ]
    if stage == "decoder_calibration":
        _requires_grad(model.image_refiner, True)
        _requires_grad(model.metric_head, True)
        model.smooth_weight_logit.requires_grad_(True)
        model.atlas_weight_logit.requires_grad_(True)
        _enable_spatial_scalars(model)
        return [
            "image_refiner",
            "metric_head",
            "metric_calibration_scalars",
            "spatial_calibration_scalars",
        ]
    raise ValueError(
        f"Exp025u only trains joint_finetune and decoder_calibration, got {stage}"
    )


def set_stage_mode(model: Exp025UMRI2PET, stage: str, training: bool) -> None:
    model.eval()
    if not training:
        return
    if stage == "joint_finetune":
        return
    if stage == "decoder_calibration":
        model.image_refiner.train()
        model.metric_head.train()
        return
    raise ValueError(f"Unsupported Exp025u training stage: {stage}")
