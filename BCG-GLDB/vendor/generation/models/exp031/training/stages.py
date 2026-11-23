"""The explicit 4-stage freeze policy for Exp031_HMRD.

    residual_patch_vae  : train the residual Patch-DVAE on Y_res
    metabolic_prior     : train MRI control + GlobalMetabolicPrior3D (+ Parc98 tokens)
    residual_diffusion  : train the global/local diffusion on the residual latent
                          (residual VAE encoder and metabolic prior frozen)
    joint_finetune      : fine-tune everything except the residual VAE encoder
"""

from __future__ import annotations

from typing import Any, Iterable


STAGE_NAMES = (
    "residual_patch_vae",
    "metabolic_prior",
    "residual_diffusion",
    "joint_finetune",
)


def _set_prefixes(model: Any, prefixes: Iterable[str]) -> None:
    prefixes = tuple(prefixes)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        )


def configure_trainable_modules(model: Any, stage_name: str) -> list[str]:
    """Freeze every parameter, then enable only the modules owned by a stage."""

    if stage_name not in STAGE_NAMES:
        raise KeyError(f"Unknown Exp031 stage: {stage_name}")
    if stage_name == "residual_patch_vae":
        prefixes = ("residual_vae",)
    elif stage_name == "metabolic_prior":
        # table_interaction is NOT trained here (it is random until stage 3);
        # the metabolic prior is supervised purely from MRI + the GT ROI targets.
        prefixes = ("mri_control", "metabolic_prior", "roi_tokens")
    elif stage_name == "residual_diffusion":
        # The residual VAE encoder and the metabolic prior stay frozen; their
        # outputs are used as conditioning only.
        prefixes = (
            "mri_control",
            "mri_prior_heads",
            "table_interaction",
            "local_denoiser",
            "global_mamba",
            "roi_tokens",
            "routing_attn",
        )
    else:  # joint_finetune
        # VAE encoder stays frozen (the latent space must not drift); decoder,
        # metabolic prior, conditioning and routing are all trainable.
        prefixes = (
            "mri_control",
            "mri_prior_heads",
            "table_interaction",
            "global_mamba",
            "local_denoiser",
            "metabolic_prior",
            "roi_tokens",
            "routing_attn",
            "residual_vae.decoder",
        )
    _set_prefixes(model, prefixes)
    return list(prefixes)


def set_stage_mode(model: Any, stage_name: str, training: bool = True) -> None:
    """Keep frozen modules in eval mode and only active modules in train mode."""

    model.eval()
    configure_trainable_modules(model, stage_name)
    if training:
        # Set each module from its own trainable descendants.  Calling parent.train()
        # would also put frozen sibling branches in train mode.
        for module in model.modules():
            module.training = any(
                parameter.requires_grad for parameter in module.parameters()
            )


def trainable_parameter_count(model: Any) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


__all__ = [
    "STAGE_NAMES",
    "configure_trainable_modules",
    "set_stage_mode",
    "trainable_parameter_count",
]
