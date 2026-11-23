from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from exp025t.models.exp025t_model import Exp025TMRI2PET, _bounded_logit


class Exp025UMRI2PET(Exp025TMRI2PET):
    """Exp025t with train-only spatial residual and post-atlas calibration."""

    def __init__(
        self,
        *,
        atlas_residual_beta: torch.Tensor,
        atlas_residual_bias: torch.Tensor,
        post_atlas_weight: float = 0.075,
        post_atlas_weight_max: float = 0.20,
        residual_prior_strength: float = 0.25,
        residual_prior_strength_max: float = 0.50,
        post_smooth_weight: float = 0.0,
        post_smooth_weight_max: float = 0.20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        expected = (1, 1, *self.image_size)
        beta = self._atlas_tensor(
            atlas_residual_beta, expected, "atlas_residual_beta"
        )
        bias = self._atlas_tensor(
            atlas_residual_bias, expected, "atlas_residual_bias"
        )
        self.register_buffer("atlas_residual_beta", beta, persistent=True)
        self.register_buffer("atlas_residual_bias", bias, persistent=True)

        self.post_atlas_weight_max = float(post_atlas_weight_max)
        self.residual_prior_strength_max = float(
            residual_prior_strength_max
        )
        self.post_smooth_weight_max = float(post_smooth_weight_max)
        self.post_atlas_weight_logit = torch.nn.Parameter(
            _bounded_logit(post_atlas_weight, self.post_atlas_weight_max)
        )
        self.residual_prior_strength_logit = torch.nn.Parameter(
            _bounded_logit(
                residual_prior_strength, self.residual_prior_strength_max
            )
        )
        self.post_smooth_weight_logit = torch.nn.Parameter(
            _bounded_logit(post_smooth_weight, self.post_smooth_weight_max)
        )

    @property
    def post_atlas_weight(self) -> torch.Tensor:
        return self.post_atlas_weight_max * torch.sigmoid(
            self.post_atlas_weight_logit
        )

    @property
    def residual_prior_strength(self) -> torch.Tensor:
        return self.residual_prior_strength_max * torch.sigmoid(
            self.residual_prior_strength_logit
        )

    @property
    def post_smooth_weight(self) -> torch.Tensor:
        return self.post_smooth_weight_max * torch.sigmoid(
            self.post_smooth_weight_logit
        )

    def apply_spatial_calibration(
        self,
        prediction: torch.Tensor,
        atlas_pet: torch.Tensor,
        foreground: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction32 = prediction.float()
        atlas32 = atlas_pet.float()
        foreground32 = foreground.float()

        atlas_weight = self.post_atlas_weight.to(dtype=prediction32.dtype)
        atlas_blend = prediction32 + atlas_weight * (
            atlas32 - prediction32
        )
        spatial_residual = self.atlas_residual_beta.to(
            dtype=prediction32.dtype
        ) * (
            atlas32 - atlas_blend
        ) + self.atlas_residual_bias.to(dtype=prediction32.dtype)
        prior_strength = self.residual_prior_strength.to(
            dtype=prediction32.dtype
        )
        calibrated = atlas_blend + prior_strength * spatial_residual

        local_mean = F.avg_pool3d(
            calibrated,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=True,
        )
        local_mean = local_mean * foreground32
        smooth_weight = self.post_smooth_weight.to(dtype=prediction32.dtype)
        calibrated = calibrated + smooth_weight * (
            local_mean - calibrated
        )
        calibrated = calibrated * foreground32
        return calibrated, {
            "input": prediction32,
            "atlas_blend": atlas_blend,
            "spatial_residual": spatial_residual,
            "local_mean": local_mean,
            "post_atlas_weight": atlas_weight,
            "residual_prior_strength": prior_strength,
            "post_smooth_weight": smooth_weight,
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
    ) -> tuple[torch.Tensor, dict[str, object]]:
        prediction, auxiliary = super().predict_direct(
            mri,
            table,
            table_mask,
            patch_batch_size,
            use_table=use_table,
            use_global=use_global,
            use_local_refiner=use_local_refiner,
            use_image_refiner=use_image_refiner,
            clamp_output=False,
        )
        spatial_calibration = None
        if use_image_refiner:
            prediction, spatial_calibration = self.apply_spatial_calibration(
                prediction,
                auxiliary["atlas_pet"],
                auxiliary["foreground"],
            )
        if clamp_output:
            prediction = prediction.clamp(-1.0, 1.0)
        auxiliary["spatial_calibration"] = spatial_calibration
        return prediction, auxiliary


def build_exp025u_model(
    config: dict,
    table_num_features: int,
    atlas_prior: dict,
    residual_prior: dict,
    feature_type_ids: Optional[Sequence[int]] = None,
) -> Exp025UMRI2PET:
    data = config["data"]
    model = config["model"]
    mamba = model["global_mamba"]
    table = model["table"]
    direct = model.get("direct_path", {})
    refiner = model.get("image_refiner", {})
    calibration = model.get("metric_calibration", {})
    spatial = model.get("spatial_calibration", {})
    return Exp025UMRI2PET(
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
        refiner_channel_mults=tuple(
            refiner.get("channel_mults", [1, 2, 4, 4])
        ),
        refiner_blocks_per_level=int(
            refiner.get("blocks_per_level", 1)
        ),
        refiner_dropout=float(refiner.get("dropout", 0.0)),
        refiner_correction_scale=float(
            refiner.get("correction_scale", 0.35)
        ),
        smooth_weight=float(calibration.get("smooth_weight", 0.35)),
        smooth_weight_max=float(
            calibration.get("smooth_weight_max", 0.60)
        ),
        atlas_weight=float(calibration.get("atlas_weight", 0.05)),
        atlas_weight_max=float(
            calibration.get("atlas_weight_max", 0.20)
        ),
        metric_head_channels=int(calibration.get("channels", 8)),
        metric_head_depth=int(calibration.get("depth", 2)),
        metric_head_correction_scale=float(
            calibration.get("correction_scale", 0.08)
        ),
        atlas_residual_beta=residual_prior["atlas_beta"],
        atlas_residual_bias=residual_prior["atlas_bias"],
        post_atlas_weight=float(spatial.get("atlas_weight", 0.075)),
        post_atlas_weight_max=float(
            spatial.get("atlas_weight_max", 0.20)
        ),
        residual_prior_strength=float(
            spatial.get("residual_prior_strength", 0.25)
        ),
        residual_prior_strength_max=float(
            spatial.get("residual_prior_strength_max", 0.50)
        ),
        post_smooth_weight=float(spatial.get("smooth_weight", 0.0)),
        post_smooth_weight_max=float(
            spatial.get("smooth_weight_max", 0.20)
        ),
    )
