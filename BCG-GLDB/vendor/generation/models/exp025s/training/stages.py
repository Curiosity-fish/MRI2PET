from __future__ import annotations

import torch

from exp026.utils.config import STAGE_NAMES
from exp025s.models.exp025s_model import Exp025SMRI2PET


def _requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def configure_trainable_modules(
    model: Exp025SMRI2PET, stage: str
) -> list[str]:
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage}")
    _requires_grad(model, False)
    groups: list[str] = []
    if stage == "patch_vae":
        _requires_grad(model.patch_vae, True)
        groups = ["patch_vae"]
    elif stage == "mri_patch_prior":
        _requires_grad(model.mri_control, True)
        _requires_grad(model.mri_prior_heads, True)
        model.coarse_gate_logit.requires_grad_(True)
        groups = ["mri_control", "mri_prior_heads", "coarse_gate"]
    elif stage == "local_control":
        _requires_grad(model.mri_control, True)
        _requires_grad(model.mri_prior_heads, True)
        _requires_grad(model.local_denoiser, True)
        model.direct_residual_scale.requires_grad_(True)
        model.coarse_gate_logit.requires_grad_(True)
        groups = [
            "mri_control",
            "mri_prior_heads",
            "local_denoiser",
            "direct_residual_scale",
            "coarse_gate",
        ]
    elif stage == "table_mri_prior":
        _requires_grad(model.table_interaction, True)
        model.coarse_gate_logit.requires_grad_(True)
        groups = ["table_interaction", "coarse_gate"]
    elif stage == "global_mamba":
        _requires_grad(model.global_mamba, True)
        _requires_grad(model.local_denoiser.global_fusions, True)
        _requires_grad(model.local_denoiser.global_to_condition, True)
        model.direct_residual_scale.requires_grad_(True)
        model.coarse_gate_logit.requires_grad_(True)
        groups = [
            "global_mamba",
            "local_denoiser.global_fusions",
            "local_denoiser.global_to_condition",
            "direct_residual_scale",
            "coarse_gate",
        ]
    elif stage == "joint_finetune":
        for module in (
            model.mri_control,
            model.mri_prior_heads,
            model.table_interaction,
            model.global_mamba,
            model.local_denoiser,
            model.patch_vae.decoder,
            model.image_refiner,
        ):
            _requires_grad(module, True)
        model.direct_residual_scale.requires_grad_(True)
        model.coarse_gate_logit.requires_grad_(True)
        groups = [
            "mri_control",
            "mri_prior_heads",
            "table_interaction",
            "global_mamba",
            "local_denoiser",
            "patch_vae.decoder",
            "image_refiner",
            "direct_path_scalars",
        ]
    elif stage == "decoder_calibration":
        _requires_grad(model.patch_vae.decoder, True)
        _requires_grad(model.image_refiner, True)
        model.coarse_gate_logit.requires_grad_(True)
        groups = ["patch_vae.decoder", "image_refiner", "coarse_gate"]
    return groups


def set_stage_mode(model: Exp025SMRI2PET, stage: str, training: bool) -> None:
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage}")
    model.eval()
    if not training:
        return
    if stage == "patch_vae":
        model.patch_vae.train()
    elif stage == "mri_patch_prior":
        model.mri_control.train()
        model.mri_prior_heads.train()
    elif stage == "local_control":
        model.mri_control.train()
        model.mri_prior_heads.train()
        model.local_denoiser.train()
    elif stage == "table_mri_prior":
        model.table_interaction.train()
    elif stage == "global_mamba":
        model.global_mamba.train()
        model.local_denoiser.global_fusions.train()
        model.local_denoiser.global_to_condition.train()
    elif stage == "joint_finetune":
        model.mri_control.train()
        model.mri_prior_heads.train()
        model.table_interaction.train()
        model.global_mamba.train()
        model.local_denoiser.train()
        model.patch_vae.decoder.train()
        model.image_refiner.train()
    elif stage == "decoder_calibration":
        model.patch_vae.decoder.train()
        model.image_refiner.train()


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
