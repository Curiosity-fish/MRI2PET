from __future__ import annotations

import torch
from torch import nn

from models.exp025d_global_region_patch_denoiser import (
    ConditionalResBlock3D,
    Downsample3D,
    FusionGate3D,
    PatchPositionEncoder,
    ResidualConvBlock3D,
    SafeGroupNorm,
    SinusoidalTimeEmbedding,
    Swish,
    Upsample3D,
)


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class ControlGate(nn.Module):
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Swish(),
            nn.Linear(int(cond_dim), int(channels)),
            nn.Sigmoid(),
        )
        linear = self.net[1]
        nn.init.zeros_(linear.weight)
        nn.init.constant_(linear.bias, -2.0)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class MRIToLatentConditionEncoder3D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: tuple[int, int, int] = (8, 16, 32),
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c0, c80, c40 = [int(v) for v in hidden_channels]
        self.stem = nn.Conv3d(1, c0, kernel_size=3, padding=1)
        self.block0 = ResidualConvBlock3D(c0, c0, norm_groups, dropout)
        self.down80 = Downsample3D(c0, c80)
        self.block80 = ResidualConvBlock3D(c80, c80, norm_groups, dropout)
        self.down40 = Downsample3D(c80, c40)
        self.block40 = ResidualConvBlock3D(c40, c40, norm_groups, dropout)
        self.out = nn.Conv3d(c40, int(out_channels), kernel_size=1)

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        x = self.block0(self.stem(mri))
        x = self.block80(self.down80(x))
        x = self.block40(self.down40(x))
        return self.out(x)


class GlobalControlEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, int] = (32, 64),
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c40, c20 = [int(v) for v in channels]
        self.stem = nn.Conv3d(int(in_channels), c40, kernel_size=3, padding=1)
        self.block40a = ResidualConvBlock3D(c40, c40, norm_groups, dropout)
        self.block40b = ResidualConvBlock3D(c40, c40, norm_groups, dropout)
        self.down20 = Downsample3D(c40, c20)
        self.block20a = ResidualConvBlock3D(c20, c20, norm_groups, dropout)
        self.block20b = ResidualConvBlock3D(c20, c20, norm_groups, dropout)

    def forward(self, cond_full: torch.Tensor) -> dict[str, torch.Tensor]:
        x40 = self.block40b(self.block40a(self.stem(cond_full)))
        x20 = self.block20b(self.block20a(self.down20(x40)))
        return {"40": x40, "20": x20}


class GlobalLatentControlEncoder3D(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        cond_channels: int,
        channels: tuple[int, int] = (32, 64),
        cond_dim: int = 256,
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c40, c20 = [int(v) for v in channels]
        self.stem = nn.Conv3d(int(latent_channels), c40, kernel_size=3, padding=1)
        self.block40a = ConditionalResBlock3D(c40, c40, cond_dim, norm_groups, dropout)
        self.block40b = ConditionalResBlock3D(c40, c40, cond_dim, norm_groups, dropout)
        self.down20 = Downsample3D(c40, c20)
        self.block20a = ConditionalResBlock3D(c20, c20, cond_dim, norm_groups, dropout)
        self.block20b = ConditionalResBlock3D(c20, c20, cond_dim, norm_groups, dropout)
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.control_encoder = GlobalControlEncoder3D(
            in_channels=int(cond_channels),
            channels=tuple(int(v) for v in channels),
            norm_groups=int(norm_groups),
            dropout=float(dropout),
        )
        self.inject40 = _zero_module(nn.Conv3d(c40, c40, kernel_size=1))
        self.inject20 = _zero_module(nn.Conv3d(c20, c20, kernel_size=1))
        self.gate40 = ControlGate(cond_dim, c40)
        self.gate20 = ControlGate(cond_dim, c20)

    @staticmethod
    def _inject(
        x: torch.Tensor,
        control: torch.Tensor,
        gate: ControlGate,
        inject: nn.Conv3d,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        scale = gate(cond)[:, :, None, None, None].to(dtype=x.dtype)
        return x + scale * inject(control).to(dtype=x.dtype)

    def forward(self, x_t_full: torch.Tensor, cond_full: torch.Tensor, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        control = self.control_encoder(cond_full)
        x40 = self.block40a(self.stem(x_t_full), cond)
        x40 = self.block40b(x40, cond)
        x40 = self._inject(x40, control["40"], self.gate40, self.inject40, cond)
        x20 = self.down20(x40)
        x20 = self.block20a(x20, cond)
        x20 = self.block20b(x20, cond)
        x20 = self._inject(x20, control["20"], self.gate20, self.inject20, cond)
        pooled = self.pool(x20).flatten(1)
        return {"40": x40, "20": x20, "pooled": pooled}


class LocalControlEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, int] = (32, 64),
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c10, c5 = [int(v) for v in channels]
        self.stem = nn.Conv3d(int(in_channels), c10, kernel_size=3, padding=1)
        self.block10a = ResidualConvBlock3D(c10, c10, norm_groups, dropout)
        self.block10b = ResidualConvBlock3D(c10, c10, norm_groups, dropout)
        self.down5 = Downsample3D(c10, c5)
        self.block5a = ResidualConvBlock3D(c5, c5, norm_groups, dropout)
        self.block5b = ResidualConvBlock3D(c5, c5, norm_groups, dropout)

    def forward(self, cond_patch: torch.Tensor) -> dict[str, torch.Tensor]:
        x10 = self.block10b(self.block10a(self.stem(cond_patch)))
        x5 = self.block5b(self.block5a(self.down5(x10)))
        return {"10": x10, "5": x5}


class Exp025KPatchLatentControlDenoiser3D(nn.Module):
    def __init__(
        self,
        full_latent_size: tuple[int, int, int],
        patch_latent_size: tuple[int, int, int],
        latent_channels: int = 4,
        cond_channels: int = 4,
        mri_encoder_channels: tuple[int, int, int] = (8, 16, 32),
        global_channels: tuple[int, int] = (32, 64),
        local_channels: tuple[int, int] = (32, 64),
        time_dim: int = 256,
        pos_dim: int = 256,
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.full_latent_size = tuple(int(v) for v in full_latent_size)
        self.patch_latent_size = tuple(int(v) for v in patch_latent_size)
        self.grid_size = tuple(self.full_latent_size[i] // self.patch_latent_size[i] for i in range(3))
        self.latent_channels = int(latent_channels)
        self.cond_channels = int(cond_channels)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )
        self.pos_embed = PatchPositionEncoder(pos_dim, self.grid_size)
        self.global_proj = nn.Sequential(
            nn.Linear(int(global_channels[-1]), time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )

        self.condition_encoder = MRIToLatentConditionEncoder3D(
            out_channels=int(cond_channels),
            hidden_channels=tuple(int(v) for v in mri_encoder_channels),
            norm_groups=int(norm_groups),
            dropout=float(dropout),
        )
        self.global_encoder = GlobalLatentControlEncoder3D(
            latent_channels=int(latent_channels),
            cond_channels=int(cond_channels),
            channels=tuple(int(v) for v in global_channels),
            cond_dim=int(time_dim),
            norm_groups=int(norm_groups),
            dropout=float(dropout),
        )

        c10, c5 = [int(v) for v in local_channels]
        self.local_control = LocalControlEncoder3D(
            in_channels=int(cond_channels),
            channels=(c10, c5),
            norm_groups=int(norm_groups),
            dropout=float(dropout),
        )
        self.local_stem = nn.Conv3d(int(latent_channels), c10, kernel_size=3, padding=1)
        self.enc10a = ConditionalResBlock3D(c10, c10, time_dim, norm_groups, dropout)
        self.enc10b = ConditionalResBlock3D(c10, c10, time_dim, norm_groups, dropout)
        self.down5 = Downsample3D(c10, c5)
        self.mid5a = ConditionalResBlock3D(c5, c5, time_dim, norm_groups, dropout)
        self.mid5b = ConditionalResBlock3D(c5, c5, time_dim, norm_groups, dropout)
        self.up10 = Upsample3D(c5, c10)
        self.dec10a = ConditionalResBlock3D(c10 + c10, c10, time_dim, norm_groups, dropout)
        self.dec10b = ConditionalResBlock3D(c10, c10, time_dim, norm_groups, dropout)

        self.fuse_global10e = FusionGate3D(c10, int(global_channels[0]), norm_groups)
        self.fuse_global5 = FusionGate3D(c5, int(global_channels[1]), norm_groups)
        self.fuse_global10d = FusionGate3D(c10, int(global_channels[0]), norm_groups)

        self.inject10e = _zero_module(nn.Conv3d(c10, c10, kernel_size=1))
        self.inject5 = _zero_module(nn.Conv3d(c5, c5, kernel_size=1))
        self.inject10d = _zero_module(nn.Conv3d(c10, c10, kernel_size=1))
        self.gate10e = ControlGate(time_dim, c10)
        self.gate5 = ControlGate(time_dim, c5)
        self.gate10d = ControlGate(time_dim, c10)

        self.out = nn.Sequential(
            SafeGroupNorm(c10, norm_groups),
            nn.SiLU(),
            nn.Conv3d(c10, int(latent_channels), kernel_size=3, padding=1),
        )

    def encode_condition(self, mri: torch.Tensor) -> torch.Tensor:
        cond = self.condition_encoder(mri)
        if tuple(cond.shape[-3:]) != self.full_latent_size:
            raise ValueError(
                f"Condition encoder produced {tuple(cond.shape[-3:])}, expected {self.full_latent_size}."
            )
        return cond

    def encode_global(
        self,
        x_t_full: torch.Tensor,
        cond_full: torch.Tensor,
        time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.global_encoder(x_t_full, cond_full, self.time_embed(time))

    def _global_condition(
        self,
        time: torch.Tensor,
        pooled: torch.Tensor,
        patch_coords: torch.Tensor,
    ) -> torch.Tensor:
        time_cond = self.time_embed(time)
        if time_cond.shape[0] == 1 and patch_coords.shape[0] > 1:
            time_cond = time_cond.expand(patch_coords.shape[0], -1)
        pooled_cond = self.global_proj(pooled)
        if pooled_cond.shape[0] == 1 and patch_coords.shape[0] > 1:
            pooled_cond = pooled_cond.expand(patch_coords.shape[0], -1)
        return time_cond + pooled_cond + self.pos_embed(patch_coords)

    def _crop_feature(self, feature: torch.Tensor, patch_coords: torch.Tensor) -> torch.Tensor:
        batch = feature.shape[0]
        if batch != 1:
            raise ValueError("Exp025k stage2 currently assumes training.batch_size=1.")
        size = tuple(feature.shape[-3 + i] // self.grid_size[i] for i in range(3))
        patches = []
        for coord in patch_coords.long().tolist():
            z, y, x = coord
            patches.append(
                feature[
                    :,
                    :,
                    z * size[0] : (z + 1) * size[0],
                    y * size[1] : (y + 1) * size[1],
                    x * size[2] : (x + 1) * size[2],
                ]
            )
        return torch.cat(patches, dim=0)

    @staticmethod
    def _inject(
        x: torch.Tensor,
        control: torch.Tensor,
        gate: ControlGate,
        inject: nn.Conv3d,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        scale = gate(cond)[:, :, None, None, None].to(dtype=x.dtype)
        return x + scale * inject(control).to(dtype=x.dtype)

    def denoise_patches(
        self,
        x_t_patch: torch.Tensor,
        cond_patch: torch.Tensor,
        patch_coords: torch.Tensor,
        time: torch.Tensor,
        global_context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cond_vec = self._global_condition(time, global_context["pooled"], patch_coords)
        local_control = self.local_control(cond_patch)
        global10 = self._crop_feature(global_context["40"], patch_coords)
        global5 = self._crop_feature(global_context["20"], patch_coords)

        x = self.local_stem(x_t_patch)
        x = self.enc10a(x, cond_vec)
        x = self.enc10b(x, cond_vec)
        x = self._inject(x, local_control["10"], self.gate10e, self.inject10e, cond_vec)
        x = self.fuse_global10e(x, global10)
        skip10 = x

        x = self.down5(x)
        x = self.mid5a(x, cond_vec)
        x = self._inject(x, local_control["5"], self.gate5, self.inject5, cond_vec)
        x = self.fuse_global5(x, global5)
        x = self.mid5b(x, cond_vec)

        x = self.up10(x, skip10.shape[-3:])
        x = self.dec10a(torch.cat([x, skip10], dim=1), cond_vec)
        x = self.dec10b(x, cond_vec)
        x = self._inject(x, local_control["10"], self.gate10d, self.inject10d, cond_vec)
        x = self.fuse_global10d(x, global10)
        return self.out(x)
