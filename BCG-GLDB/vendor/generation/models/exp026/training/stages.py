from __future__ import annotations

import torch

from exp026.models.exp026_model import Exp026MRI2PET
from exp026.utils.config import STAGE_NAMES


def _requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def configure_trainable_modules(model: Exp026MRI2PET, stage: str) -> list[str]:
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage}")
    _requires_grad(model, False)
    trainable_groups: list[str] = []
    if stage == "patch_vae":
        _requires_grad(model.patch_vae, True)
        trainable_groups = ["patch_vae.encoder", "patch_vae.decoder"]
    elif stage == "mri_patch_prior":
        _requires_grad(model.mri_control, True)
        _requires_grad(model.mri_prior_heads, True)
        trainable_groups = ["mri_control", "mri_prior_heads"]
    elif stage == "local_control":
        _requires_grad(model.mri_control, True)
        _requires_grad(model.mri_prior_heads, True)
        _requires_grad(model.local_denoiser, True)
        _requires_grad(model.local_denoiser.table_injections, False)
        _requires_grad(model.local_denoiser.global_fusions, False)
        _requires_grad(model.local_denoiser.subject_to_condition, False)
        _requires_grad(model.local_denoiser.global_to_condition, False)
        trainable_groups = [
            "mri_control",
            "mri_prior_heads",
            "local_denoiser.core",
            "local_denoiser.mri_injections",
        ]
    elif stage == "table_mri_prior":
        _requires_grad(model.table_interaction, True)
        trainable_groups = ["table_interaction"]
    elif stage == "global_mamba":
        _requires_grad(model.global_mamba, True)
        _requires_grad(model.local_denoiser.global_fusions, True)
        _requires_grad(model.local_denoiser.global_to_condition, True)
        trainable_groups = [
            "global_mamba",
            "local_denoiser.global_fusions",
            "local_denoiser.global_to_condition",
        ]
    elif stage == "joint_finetune":
        _requires_grad(model.mri_control, True)
        _requires_grad(model.mri_prior_heads, True)
        _requires_grad(model.table_interaction, True)
        _requires_grad(model.global_mamba, True)
        _requires_grad(model.local_denoiser, True)
        trainable_groups = [
            "mri_control",
            "mri_prior_heads",
            "table_interaction",
            "global_mamba",
            "local_denoiser",
        ]
    elif stage == "decoder_calibration":
        _requires_grad(model.patch_vae.decoder, True)
        trainable_groups = ["patch_vae.decoder"]
    return trainable_groups


def set_stage_mode(model: Exp026MRI2PET, stage: str, training: bool) -> None:
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
    elif stage == "decoder_calibration":
        model.patch_vae.decoder.train()


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
