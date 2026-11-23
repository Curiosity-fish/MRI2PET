from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from exp026.utils.patch_codec import (
    build_patch_grid,
    crop_feature_grid,
    extract_patches,
    grid_indices,
    stitch_patches,
    validate_patch_geometry,
)

from .global_mamba import GlobalBiMamba3D
from .local_denoiser import LocalControlDenoiser3D
from .mri_control import MRIControlEncoder3D, MRIPatchPriorHeads3D
from .patch_vae import PatchDVAE3D
from .table_interaction import JointTableMRIPrior3D


def _ceil_half(size: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((int(v) + 1) // 2 for v in size)


class Exp026MRI2PET(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        table_num_features: int,
        feature_type_ids: Optional[Sequence[int]] = None,
        image_channels: int = 1,
        latent_channels: int = 4,
        vae_base_channels: int = 16,
        mri_channels: tuple[int, int, int, int] = (8, 16, 32, 64),
        local_channels: tuple[int, int, int] = (32, 64, 96),
        time_dim: int = 128,
        table_token_dim: int = 128,
        table_token_grid: tuple[int, int, int] = (4, 4, 4),
        table_depth: int = 2,
        table_heads: int = 4,
        num_feature_types: int = 1,
        global_token_dim: int = 128,
        mamba_depth: int = 2,
        mamba_state_dim: int = 16,
        mamba_conv_width: int = 4,
        mamba_expand: int = 2,
        mamba_backend: str = "mamba",
        allow_test_backend: bool = False,
        norm_groups: int = 8,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = tuple(int(v) for v in image_size)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.grid_size = validate_patch_geometry(self.image_size, self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        if any(v % PatchDVAE3D.downsample_factor != 0 for v in self.patch_size):
            raise ValueError("Every patch dimension must be divisible by the DVAE factor 4")
        if any(v % PatchDVAE3D.downsample_factor != 0 for v in self.image_size):
            raise ValueError("Every image dimension must be divisible by the DVAE factor 4")
        self.latent_patch_size = tuple(v // PatchDVAE3D.downsample_factor for v in self.patch_size)
        self.full_latent_size = tuple(v // PatchDVAE3D.downsample_factor for v in self.image_size)
        self.latent_channels = int(latent_channels)
        self.image_patch_coords = build_patch_grid(self.image_size, self.patch_size)
        self.latent_patch_coords = build_patch_grid(self.full_latent_size, self.latent_patch_size)
        self.register_buffer(
            "grid_coords",
            torch.tensor(grid_indices(self.grid_size), dtype=torch.long),
            persistent=False,
        )

        self.patch_vae = PatchDVAE3D(
            image_channels=image_channels,
            latent_channels=latent_channels,
            base_channels=vae_base_channels,
            norm_groups=norm_groups,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.mri_control = MRIControlEncoder3D(
            in_channels=image_channels,
            channels=mri_channels,
            norm_groups=norm_groups,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        control_channels = tuple(int(v) for v in mri_channels[1:])
        self.mri_prior_heads = MRIPatchPriorHeads3D(
            mri_channels=control_channels,
            latent_channels=latent_channels,
            norm_groups=norm_groups,
            gradient_checkpointing=gradient_checkpointing,
        )
        if tuple(int(v) for v in table_token_grid) != self.grid_size:
            raise ValueError(
                f"table_token_grid={table_token_grid} must match patch grid={self.grid_size}"
            )
        self.table_interaction = JointTableMRIPrior3D(
            num_features=table_num_features,
            mri_channels=control_channels[0],
            latent_channels=latent_channels,
            token_dim=table_token_dim,
            num_patches=self.num_patches,
            depth=table_depth,
            num_heads=table_heads,
            feature_type_ids=feature_type_ids,
            num_feature_types=num_feature_types,
        )
        self.global_mamba = GlobalBiMamba3D(
            grid_size=self.grid_size,
            latent_channels=latent_channels,
            mri_channels=control_channels[0],
            token_dim=global_token_dim,
            depth=mamba_depth,
            state_dim=mamba_state_dim,
            conv_width=mamba_conv_width,
            expand=mamba_expand,
            table_heads=table_heads,
            table_token_dim=table_token_dim,
            time_dim=time_dim,
            backend=mamba_backend,
            allow_test_backend=allow_test_backend,
        )
        self.local_denoiser = LocalControlDenoiser3D(
            latent_channels=latent_channels,
            mri_channels=control_channels,
            local_channels=local_channels,
            time_dim=time_dim,
            table_token_dim=table_token_dim,
            global_token_dim=global_token_dim,
            grid_size=self.grid_size,
            norm_groups=norm_groups,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

    def extract_image_patches(self, volume: torch.Tensor) -> torch.Tensor:
        return extract_patches(volume, self.image_patch_coords, self.patch_size)

    def stitch_image_patches(self, patches: torch.Tensor) -> torch.Tensor:
        return stitch_patches(patches, self.image_patch_coords, self.image_size)

    def stitch_latent_patches(self, patches: torch.Tensor) -> torch.Tensor:
        return stitch_patches(patches, self.latent_patch_coords, self.full_latent_size)

    def encode_pet_patches(
        self,
        pet: torch.Tensor,
        patch_batch_size: int,
        sample: bool = False,
        encoder_grad: bool = False,
    ) -> torch.Tensor:
        patches = self.extract_image_patches(pet)
        chunks = []
        grad_context = nullcontext() if encoder_grad else torch.no_grad()
        with grad_context:
            for start in range(0, self.num_patches, int(patch_batch_size)):
                patch = patches[:, start : start + int(patch_batch_size)]
                batch, count = patch.shape[:2]
                mu, logvar = self.patch_vae.encoder(
                    patch.reshape(batch * count, patch.shape[2], *patch.shape[-3:])
                )
                latent = self.patch_vae.reparameterize(mu, logvar) if sample else mu
                chunks.append(latent.reshape(batch, count, *latent.shape[1:]))
        return torch.cat(chunks, dim=1)

    def decode_latent_patches(
        self,
        latent_patches: torch.Tensor,
        patch_batch_size: int,
    ) -> torch.Tensor:
        """Decode in chunks without no_grad so PET loss can reach predicted latents."""
        chunks = []
        for start in range(0, self.num_patches, int(patch_batch_size)):
            latent = latent_patches[:, start : start + int(patch_batch_size)]
            chunks.append(self.decode_latent_chunk(latent))
        return torch.cat(chunks, dim=1)

    def decode_latent_chunk(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 6 or latent.shape[1] <= 0:
            raise ValueError("latent must be a non-empty [B,K,C,D,H,W] tensor")
        batch, count = latent.shape[:2]
        decoded = self.patch_vae.decoder(
            latent.reshape(batch * count, latent.shape[2], *latent.shape[-3:])
        )
        return decoded.reshape(batch, count, *decoded.shape[1:])

    def build_control_patch_pyramid(
        self, controls: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        middle_size = _ceil_half(self.latent_patch_size)
        low_size = _ceil_half(middle_size)
        return {
            "high": crop_feature_grid(
                controls["high"], self.grid_coords, self.grid_size, self.latent_patch_size
            ),
            "middle": crop_feature_grid(
                controls["middle"], self.grid_coords, self.grid_size, middle_size
            ),
            "low": crop_feature_grid(
                controls["low"], self.grid_coords, self.grid_size, low_size
            ),
        }

    def build_latent_patch_pyramid(
        self,
        high: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if high.ndim != 6 or high.shape[1] != self.num_patches:
            raise ValueError(
                f"Expected latent patches [B,{self.num_patches},C,D,H,W], got {tuple(high.shape)}"
            )
        batch, patches, channels = high.shape[:3]
        high_flat = high.reshape(batch * patches, channels, *high.shape[-3:])
        middle_size = _ceil_half(self.latent_patch_size)
        low_size = _ceil_half(middle_size)
        middle = F.interpolate(
            high_flat, size=middle_size, mode="trilinear", align_corners=False
        ).reshape(batch, patches, channels, *middle_size)
        low = F.interpolate(
            high_flat, size=low_size, mode="trilinear", align_corners=False
        ).reshape(batch, patches, channels, *low_size)
        return {"high": high, "middle": middle, "low": low}

    def build_table_prior_pyramid(
        self,
        table_prior_high: torch.Tensor,
        mri_prior_pyramid: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        residual_high = table_prior_high - mri_prior_pyramid["high"]
        residual_pyramid = self.build_latent_patch_pyramid(residual_high)
        return {
            level: mri_prior_pyramid[level] + residual_pyramid[level]
            for level in ("high", "middle", "low")
        }

    @staticmethod
    def _flatten_selected(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        selected = x[:, start:end]
        return selected.reshape(selected.shape[0] * selected.shape[1], *selected.shape[2:])

    def prepare_denoise_context(
        self,
        noisy_latent_patches: torch.Tensor,
        timesteps: torch.Tensor,
        mri: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        use_table: bool,
        use_global: bool,
        inject_table_locally: bool,
    ) -> dict[str, object]:
        controls = self.mri_control(mri)
        control_patches = self.build_control_patch_pyramid(controls)
        table_context = None
        mri_prior_pyramid = None
        if use_table:
            if table is None:
                raise ValueError("This stage requires table values")
            mri_prior_pyramid = self.mri_prior_heads(control_patches)
            table_context = self.table_interaction(
                control_patches["high"],
                table,
                table_mask,
                base_prior_patches=mri_prior_pyramid["high"],
            )
        prior_patches = (
            self.build_table_prior_pyramid(
                table_context["prior_patches"], mri_prior_pyramid
            )
            if table_context is not None
            and mri_prior_pyramid is not None
            and inject_table_locally
            else None
        )

        global_tokens = None
        updated_table_tokens = None
        if use_global:
            high_control_patches = control_patches["high"]
            time_embedding = self.local_denoiser.embed_time(timesteps)
            global_tokens, updated_table_tokens = self.global_mamba(
                noisy_latent_patches,
                high_control_patches,
                time_embedding,
                table_tokens=(table_context["table_tokens"] if table_context is not None else None),
                table_key_padding_mask=(
                    table_context["table_key_padding_mask"] if table_context is not None else None
                ),
                reliability=(table_context["reliability"] if table_context is not None else None),
            )

        return {
            "controls": controls,
            "control_patches": control_patches,
            "mri_prior_pyramid": mri_prior_pyramid,
            "table_context": table_context,
            "prior_patches": prior_patches,
            "global_tokens": global_tokens,
            "updated_table_tokens": updated_table_tokens,
            "inject_table_locally": bool(inject_table_locally),
        }

    def denoise_patch_chunk(
        self,
        noisy_latent_patches: torch.Tensor,
        timesteps: torch.Tensor,
        denoise_context: dict[str, object],
        start: int,
        end: int,
    ) -> torch.Tensor:
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > self.num_patches:
            raise ValueError(f"Invalid patch range [{start},{end})")
        batch = noisy_latent_patches.shape[0]
        count = end - start
        control_patches = denoise_context["control_patches"]
        prior_patches = denoise_context["prior_patches"]
        table_context = denoise_context["table_context"]
        global_tokens = denoise_context["global_tokens"]
        inject_table_locally = bool(denoise_context["inject_table_locally"])

        local_control = {
            level: self._flatten_selected(value, start, end)
            for level, value in control_patches.items()
        }
        local_prior = (
            {
                level: self._flatten_selected(value, start, end)
                for level, value in prior_patches.items()
            }
            if prior_patches is not None
            else None
        )
        coords = self.grid_coords[start:end][None].expand(batch, -1, -1).reshape(-1, 3)
        time_chunk = timesteps.repeat_interleave(count)
        subject = None
        if table_context is not None and inject_table_locally:
            subject = table_context["subject_token"].repeat_interleave(count, dim=0)
        global_chunk = (
            self._flatten_selected(global_tokens, start, end)
            if global_tokens is not None
            else None
        )
        noisy_chunk = self._flatten_selected(noisy_latent_patches, start, end)
        prediction = self.local_denoiser(
            noisy_chunk,
            time_chunk,
            coords,
            local_control,
            table_prior=local_prior,
            subject_token=subject,
            global_token=global_chunk,
        )
        return prediction.reshape(batch, count, *prediction.shape[1:])

    def denoise_patches(
        self,
        noisy_latent_patches: torch.Tensor,
        timesteps: torch.Tensor,
        mri: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        patch_batch_size: int,
        use_table: bool,
        use_global: bool,
        inject_table_locally: bool,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        denoise_context = self.prepare_denoise_context(
            noisy_latent_patches,
            timesteps,
            mri,
            table,
            table_mask,
            use_table=use_table,
            use_global=use_global,
            inject_table_locally=inject_table_locally,
        )

        predictions = []
        for start in range(0, self.num_patches, int(patch_batch_size)):
            end = min(start + int(patch_batch_size), self.num_patches)
            predictions.append(
                self.denoise_patch_chunk(
                    noisy_latent_patches,
                    timesteps,
                    denoise_context,
                    start,
                    end,
                )
            )
        return torch.cat(predictions, dim=1), denoise_context


def build_exp026_model(
    config: dict,
    table_num_features: int,
    feature_type_ids: Optional[Sequence[int]] = None,
) -> Exp026MRI2PET:
    data = config["data"]
    model = config["model"]
    mamba = model["global_mamba"]
    table = model["table"]
    return Exp026MRI2PET(
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
        gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True)),
    )
