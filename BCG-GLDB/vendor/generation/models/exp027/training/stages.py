"""The explicit 13-stage freeze policy for Exp027."""

from __future__ import annotations

from typing import Any, Iterable


STAGE_NAMES = (
    "patch_vae",
    "mri_patch_prior",
    "local_diffusion",
    "deterministic_prior",
    "deterministic_local",
    "table_mri_prior",
    "global_mamba",
    "joint_finetune",
    "decoder_calibration",
    "structural_head",
    "structural_decoder",
    "spatial_scalars",
    "final_calibration",
)


def _set_prefixes(model: Any, prefixes: Iterable[str]) -> None:
    prefixes = tuple(prefixes)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def configure_trainable_modules(model: Any, stage_name: str) -> list[str]:
    """Freeze every parameter, then enable only the modules owned by a stage."""

    if stage_name not in STAGE_NAMES:
        raise KeyError(f"Unknown Exp027 stage: {stage_name}")
    common_mri = ("mri_control", "mri_prior_heads")
    if stage_name == "patch_vae":
        prefixes = ("patch_vae",)
    elif stage_name == "mri_patch_prior":
        prefixes = common_mri
    elif stage_name == "local_diffusion":
        prefixes = common_mri + ("local_denoiser",)
    elif stage_name == "deterministic_prior":
        prefixes = common_mri + ("coarse_gate_logit",)
    elif stage_name == "deterministic_local":
        prefixes = common_mri + (
            "local_denoiser",
            "direct_residual_scale",
            "coarse_gate_logit",
        )
    elif stage_name == "table_mri_prior":
        prefixes = ("table_interaction",)
    elif stage_name == "global_mamba":
        # The context projection layers live in the local denoiser in the
        # inherited Exp025 implementation.  Enable them when present without
        # making the policy depend on a particular historical class version.
        prefixes = ("global_mamba", "local_denoiser.global_fusions")
    elif stage_name == "joint_finetune":
        prefixes = (
            "mri_control",
            "mri_prior_heads",
            "table_interaction",
            "global_mamba",
            "local_denoiser",
            "patch_vae.decoder",
            "image_refiner",
            "direct_residual_scale",
            "coarse_gate_logit",
        )
    elif stage_name == "decoder_calibration":
        prefixes = ("patch_vae.decoder", "image_refiner", "coarse_gate_logit")
    elif stage_name == "structural_head":
        prefixes = ("metric_head", "smooth_weight_logit", "atlas_weight_logit")
    elif stage_name == "structural_decoder":
        prefixes = ("image_refiner", "metric_head", "smooth_weight_logit", "atlas_weight_logit")
    elif stage_name == "spatial_scalars":
        prefixes = (
            "post_atlas_weight_logit",
            "residual_prior_strength_logit",
            "post_smooth_weight_logit",
        )
    else:  # final_calibration
        prefixes = (
            "image_refiner",
            "metric_head",
            "smooth_weight_logit",
            "atlas_weight_logit",
            "post_atlas_weight_logit",
            "residual_prior_strength_logit",
            "post_smooth_weight_logit",
        )
    _set_prefixes(model, prefixes)
    return list(prefixes)


def set_stage_mode(model: Any, stage_name: str, training: bool = True) -> None:
    """Keep frozen modules in eval mode and only active modules in train mode."""

    model.eval()
    configure_trainable_modules(model, stage_name)
    if training:
        # Set each module from its own trainable descendants. Calling parent.train()
        # would also put frozen sibling branches in train mode.
        for module in model.modules():
            module.training = any(parameter.requires_grad for parameter in module.parameters())


def trainable_parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
