from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from exp026.models.exp026_model import Exp026MRI2PET
from models.global_fusion_refiner3d import GlobalFusionRefiner3D


class Exp025SMRI2PET(Exp026MRI2PET):
    """Exp026 backbone with a deterministic, metric-oriented reconstruction path."""

    def __init__(
        self,
        *,
        atlas_slope: torch.Tensor,
        atlas_bias: torch.Tensor,
        direct_timestep: int = 0,
        direct_residual_scale: float = 0.1,
        coarse_latent_weight: float = 0.5,
        refiner_base_channels: int = 12,
        refiner_channel_mults: Sequence[int] = (1, 2, 4, 4),
        refiner_blocks_per_level: int = 1,
        refiner_dropout: float = 0.0,
        refiner_correction_scale: float = 0.35,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        expected = (1, 1, *self.image_size)
        slope = self._atlas_tensor(atlas_slope, expected, "atlas_slope")
        bias = self._atlas_tensor(atlas_bias, expected, "atlas_bias")
        self.register_buffer("atlas_slope", slope, persistent=True)
        self.register_buffer("atlas_bias", bias, persistent=True)

        weight = min(max(float(coarse_latent_weight), 1e-4), 1.0 - 1e-4)
        self.coarse_gate_logit = nn.Parameter(
            torch.logit(torch.tensor(weight, dtype=torch.float32))
        )
        self.direct_residual_scale = nn.Parameter(
            torch.tensor(float(direct_residual_scale), dtype=torch.float32)
        )
        self.direct_timestep = int(direct_timestep)
        self.image_refiner = GlobalFusionRefiner3D(
            in_channels=3,
            out_channels=1,
            base_channels=int(refiner_base_channels),
            channel_mults=tuple(int(v) for v in refiner_channel_mults),
            blocks_per_level=int(refiner_blocks_per_level),
            norm_groups=int(kwargs.get("norm_groups", 8)),
            dropout=float(refiner_dropout),
            correction_scale=float(refiner_correction_scale),
            use_checkpoint=bool(kwargs.get("gradient_checkpointing", True)),
        )

    @staticmethod
    def _atlas_tensor(
        value: torch.Tensor,
        expected: tuple[int, int, int, int, int],
        name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor[None, None]
        elif tensor.ndim == 4:
            tensor = tensor[None]
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        return tensor.contiguous()

    @staticmethod
    def mri_foreground_mask(mri: torch.Tensor) -> torch.Tensor:
        return (mri.abs().amax(dim=1, keepdim=True) > 1e-6).to(dtype=mri.dtype)

    def atlas_prediction(self, mri: torch.Tensor) -> torch.Tensor:
        prediction = self.atlas_slope.to(dtype=mri.dtype) * mri + self.atlas_bias.to(
            dtype=mri.dtype
        )
        return prediction.clamp(-1.0, 1.0) * self.mri_foreground_mask(mri)

    def _direct_context(
        self,
        base_latent: torch.Tensor,
        timesteps: torch.Tensor,
        control_patches: dict[str, torch.Tensor],
        mri_prior_pyramid: dict[str, torch.Tensor],
        table_context: Optional[dict[str, torch.Tensor]],
        use_global: bool,
        inject_table_locally: bool,
    ) -> dict[str, object]:
        prior_patches = None
        if table_context is not None and inject_table_locally:
            prior_patches = self.build_table_prior_pyramid(
                table_context["prior_patches"], mri_prior_pyramid
            )

        global_tokens = None
        updated_table_tokens = None
        if use_global:
            global_tokens, updated_table_tokens = self.global_mamba(
                base_latent,
                control_patches["high"],
                self.local_denoiser.embed_time(timesteps),
                table_tokens=(
                    table_context["table_tokens"] if table_context is not None else None
                ),
                table_key_padding_mask=(
                    table_context["table_key_padding_mask"]
                    if table_context is not None
                    else None
                ),
                reliability=(
                    table_context["reliability"] if table_context is not None else None
                ),
            )
        return {
            "control_patches": control_patches,
            "mri_prior_pyramid": mri_prior_pyramid,
            "table_context": table_context,
            "prior_patches": prior_patches,
            "global_tokens": global_tokens,
            "updated_table_tokens": updated_table_tokens,
            "inject_table_locally": bool(inject_table_locally),
        }

    def predict_direct(
        self,
        mri: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        patch_batch_size: int,
        *,
        use_table: bool,
        use_global: bool,
        use_local_refiner: bool,
        use_image_refiner: bool,
        clamp_output: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | dict | None]]:
        controls = self.mri_control(mri)
        control_patches = self.build_control_patch_pyramid(controls)
        mri_prior_pyramid = self.mri_prior_heads(control_patches)
        table_context = None
        base_latent = mri_prior_pyramid["high"]
        if use_table:
            if table is None:
                raise ValueError("The direct table path requires table values")
            table_context = self.table_interaction(
                control_patches["high"],
                table,
                table_mask,
                base_prior_patches=base_latent,
            )
            base_latent = table_context["prior_patches"]

        refined_latent = base_latent
        direct_correction = None
        if use_local_refiner:
            timesteps = torch.full(
                (mri.shape[0],),
                self.direct_timestep,
                device=mri.device,
                dtype=torch.long,
            )
            context = self._direct_context(
                base_latent,
                timesteps,
                control_patches,
                mri_prior_pyramid,
                table_context,
                use_global=use_global,
                inject_table_locally=use_table,
            )
            chunks = []
            for start in range(0, self.num_patches, int(patch_batch_size)):
                end = min(start + int(patch_batch_size), self.num_patches)
                chunks.append(
                    self.denoise_patch_chunk(
                        base_latent,
                        timesteps,
                        context,
                        start,
                        end,
                    )
                )
            direct_correction = torch.cat(chunks, dim=1)
            refined_latent = (
                base_latent
                + self.direct_residual_scale.to(dtype=base_latent.dtype)
                * direct_correction
            )

        decoded_patches = self.decode_latent_patches(
            refined_latent, int(patch_batch_size)
        )
        latent_pet = self.stitch_image_patches(decoded_patches)
        atlas_pet = self.atlas_prediction(mri)
        latent_weight = torch.sigmoid(self.coarse_gate_logit).to(dtype=latent_pet.dtype)
        coarse_pet = latent_weight * latent_pet + (1.0 - latent_weight) * atlas_pet
        foreground = self.mri_foreground_mask(mri)
        coarse_pet = coarse_pet * foreground

        prediction = coarse_pet
        refiner_output = None
        if use_image_refiner:
            refiner_output = self.image_refiner(
                local_pet=coarse_pet,
                mri=mri,
                stitch_mask=foreground,
                return_dict=True,
            )
            prediction = refiner_output["refined"] * foreground
        if clamp_output:
            prediction = prediction.clamp(-1.0, 1.0)

        return prediction, {
            "base_latent": base_latent,
            "refined_latent": refined_latent,
            "direct_correction": direct_correction,
            "mri_prior_pyramid": mri_prior_pyramid,
            "table_context": table_context,
            "latent_pet": latent_pet,
            "atlas_pet": atlas_pet,
            "coarse_pet": coarse_pet,
            "foreground": foreground,
            "refiner_output": refiner_output,
            "latent_weight": latent_weight,
        }


def build_exp025s_model(
    config: dict,
    table_num_features: int,
    atlas_prior: dict,
    feature_type_ids: Optional[Sequence[int]] = None,
) -> Exp025SMRI2PET:
    data = config["data"]
    model = config["model"]
    mamba = model["global_mamba"]
    table = model["table"]
    direct = model.get("direct_path", {})
    refiner = model.get("image_refiner", {})
    return Exp025SMRI2PET(
        image_size=tuple(data["image_size"]),
        patch_size=tuple(data["patch_size"]),
        table_num_features=table_num_features,
        feature_type_ids=feature_type_ids,
        image_channels=int(model.get("image_channels", 1)),
        latent_channels=int(model.get("latent_channels", 4)),
        vae_base_channels=int(model.get("vae_base_channels", 16)),
        mri_channels=tuple(model.get("mri_channels", [8, 16, 32, 64])),
        local_channels=tuple(model.get("local_channels", [32, 64, 96])),
        time_dim=int(model.get("time_dim", 128)),
        table_token_dim=int(table.get("token_dim", 128)),
        table_token_grid=tuple(table.get("token_grid", [4, 4, 4])),
        table_depth=int(table.get("depth", 2)),
        table_heads=int(table.get("heads", 4)),
        num_feature_types=int(table.get("num_feature_types", 1)),
        global_token_dim=int(mamba.get("token_dim", 128)),
        mamba_depth=int(mamba.get("depth", 2)),
        mamba_state_dim=int(mamba.get("state_dim", 16)),
        mamba_conv_width=int(mamba.get("conv_width", 4)),
        mamba_expand=int(mamba.get("expand", 2)),
        mamba_backend=str(mamba.get("backend", "mamba")),
        allow_test_backend=bool(mamba.get("allow_test_backend", False)),
        norm_groups=int(model.get("norm_groups", 8)),
        dropout=float(model.get("dropout", 0.0)),
        gradient_checkpointing=bool(
            config["training"].get("gradient_checkpointing", True)
        ),
        atlas_slope=atlas_prior["slope"],
        atlas_bias=atlas_prior["bias"],
        direct_timestep=int(direct.get("timestep", 0)),
        direct_residual_scale=float(direct.get("residual_scale", 0.1)),
        coarse_latent_weight=float(direct.get("coarse_latent_weight", 0.5)),
        refiner_base_channels=int(refiner.get("base_channels", 12)),
        refiner_channel_mults=tuple(refiner.get("channel_mults", [1, 2, 4, 4])),
        refiner_blocks_per_level=int(refiner.get("blocks_per_level", 1)),
        refiner_dropout=float(refiner.get("dropout", 0.0)),
        refiner_correction_scale=float(refiner.get("correction_scale", 0.35)),
    )
