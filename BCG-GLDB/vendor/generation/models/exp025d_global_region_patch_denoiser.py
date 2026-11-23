from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_groups(channels: int, groups: int) -> int:
    groups = min(int(groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class SafeGroupNorm(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_safe_groups(channels, groups), int(channels), eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim > 1:
            t = t.view(t.shape[0], -1)[:, 0]
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-scale))
        emb = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
        return self.conv(F.interpolate(x, size=size, mode="trilinear", align_corners=False))


class ResidualConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(in_channels, norm_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = SafeGroupNorm(out_channels, norm_groups)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class ConditionalResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, norm_groups: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(in_channels, norm_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = SafeGroupNorm(out_channels, norm_groups)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.cond_proj(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + gamma[:, :, None, None, None].to(dtype=h.dtype))
        h = h + beta[:, :, None, None, None].to(dtype=h.dtype)
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class FusionGate3D(nn.Module):
    def __init__(self, local_channels: int, guide_channels: int, norm_groups: int) -> None:
        super().__init__()
        self.guide_proj = nn.Conv3d(guide_channels, local_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv3d(local_channels * 2, local_channels, kernel_size=3, padding=1),
            SafeGroupNorm(local_channels, norm_groups),
            Swish(),
            nn.Conv3d(local_channels, local_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        guide = self.guide_proj(guide)
        gate = self.gate(torch.cat([x, guide], dim=1))
        return x + self.alpha.to(dtype=x.dtype) * gate * guide


class PatchPositionEncoder(nn.Module):
    def __init__(self, dim: int, grid_size: tuple[int, int, int]) -> None:
        super().__init__()
        gz, gy, gx = [int(v) for v in grid_size]
        self.grid_size = (gz, gy, gx)
        self.ez = nn.Embedding(gz, dim)
        self.ey = nn.Embedding(gy, dim)
        self.ex = nn.Embedding(gx, dim)
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.long()
        gz, gy, gx = self.grid_size
        cz = (coords[:, 0].float() + 0.5) / float(gz)
        cy = (coords[:, 1].float() + 0.5) / float(gy)
        cx = (coords[:, 2].float() + 0.5) / float(gx)
        return (
            self.ez(coords[:, 0]) + self.ey(coords[:, 1]) + self.ex(coords[:, 2]) + self.mlp(torch.stack([cz, cy, cx], dim=1))
        )


def _region_boundary(mask: torch.Tensor) -> torch.Tensor:
    labels = mask.long()
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    diff = labels[:, :, 1:, :, :] != labels[:, :, :-1, :, :]
    boundary[:, :, 1:, :, :] |= diff
    boundary[:, :, :-1, :, :] |= diff
    diff = labels[:, :, :, 1:, :] != labels[:, :, :, :-1, :]
    boundary[:, :, :, 1:, :] |= diff
    boundary[:, :, :, :-1, :] |= diff
    diff = labels[:, :, :, :, 1:] != labels[:, :, :, :, :-1]
    boundary[:, :, :, :, 1:] |= diff
    boundary[:, :, :, :, :-1] |= diff
    return boundary.float()


class MRIPatchEncoder(nn.Module):
    def __init__(self, channels: Iterable[int], norm_groups: int, dropout: float) -> None:
        super().__init__()
        c40, c20, c10, c5 = [int(v) for v in channels]
        self.stem = nn.Conv3d(1, c40, kernel_size=3, padding=1)
        self.block40 = ResidualConvBlock3D(c40, c40, norm_groups, dropout)
        self.down20 = Downsample3D(c40, c20)
        self.block20 = ResidualConvBlock3D(c20, c20, norm_groups, dropout)
        self.down10 = Downsample3D(c20, c10)
        self.block10 = ResidualConvBlock3D(c10, c10, norm_groups, dropout)
        self.down5 = Downsample3D(c10, c5)
        self.block5 = ResidualConvBlock3D(c5, c5, norm_groups, dropout)

    def forward(self, mri_patch: torch.Tensor) -> dict[str, torch.Tensor]:
        x40 = self.block40(self.stem(mri_patch))
        x20 = self.block20(self.down20(x40))
        x10 = self.block10(self.down10(x20))
        x5 = self.block5(self.down5(x10))
        return {"40": x40, "20": x20, "10": x10, "5": x5}


class GlobalEncoder3D(nn.Module):
    def __init__(self, channels: Iterable[int], cond_dim: int, norm_groups: int, dropout: float) -> None:
        super().__init__()
        c0, c80, c40 = [int(v) for v in channels]
        self.stem = nn.Conv3d(2, c0, kernel_size=3, padding=1)
        self.block0 = ResidualConvBlock3D(c0, c0, norm_groups, dropout)
        self.down80 = Downsample3D(c0, c80)
        self.block80a = ConditionalResBlock3D(c80, c80, cond_dim, norm_groups, dropout)
        self.block80b = ConditionalResBlock3D(c80, c80, cond_dim, norm_groups, dropout)
        self.down40 = Downsample3D(c80, c40)
        self.block40a = ConditionalResBlock3D(c40, c40, cond_dim, norm_groups, dropout)
        self.block40b = ConditionalResBlock3D(c40, c40, cond_dim, norm_groups, dropout)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x_t_full: torch.Tensor, mri_full: torch.Tensor, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.block0(self.stem(torch.cat([x_t_full, mri_full], dim=1)))
        x80 = self.down80(x)
        x80 = self.block80a(x80, cond)
        x80 = self.block80b(x80, cond)
        x40 = self.down40(x80)
        x40 = self.block40a(x40, cond)
        x40 = self.block40b(x40, cond)
        pooled = self.pool(x40).flatten(1)
        return {"80": x80, "40": x40, "pooled": pooled}


class Exp025DGlobalRegionPatchDenoiser3D(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        num_regions: int = 98,
        base_channels: int = 32,
        global_channels: Iterable[int] = (16, 32, 64),
        local_channels: Iterable[int] = (32, 64, 128, 256),
        time_dim: int = 256,
        pos_dim: int = 256,
        norm_groups: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = tuple(int(v) for v in image_size)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.grid_size = tuple(self.image_size[i] // self.patch_size[i] for i in range(3))
        self.num_regions = int(num_regions)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )
        self.pos_embed = PatchPositionEncoder(pos_dim, self.grid_size)
        self.global_proj = nn.Sequential(
            nn.Linear(int(list(global_channels)[-1]), time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )
        cond_dim = int(time_dim)
        self.global_encoder = GlobalEncoder3D(global_channels, cond_dim, norm_groups, dropout)
        c40, c20, c10, c5 = [int(v) for v in local_channels]
        self.mri_encoder = MRIPatchEncoder(local_channels, norm_groups, dropout)
        self.local_stem = nn.Conv3d(3, c40, kernel_size=3, padding=1)
        self.enc40a = ConditionalResBlock3D(c40, c40, cond_dim, norm_groups, dropout)
        self.enc40b = ConditionalResBlock3D(c40, c40, cond_dim, norm_groups, dropout)
        self.down20 = Downsample3D(c40, c20)
        self.enc20a = ConditionalResBlock3D(c20, c20, cond_dim, norm_groups, dropout)
        self.enc20b = ConditionalResBlock3D(c20, c20, cond_dim, norm_groups, dropout)
        self.down10 = Downsample3D(c20, c10)
        self.enc10a = ConditionalResBlock3D(c10, c10, cond_dim, norm_groups, dropout)
        self.enc10b = ConditionalResBlock3D(c10, c10, cond_dim, norm_groups, dropout)
        self.down5 = Downsample3D(c10, c5)
        self.mid1 = ConditionalResBlock3D(c5, c5, cond_dim, norm_groups, dropout)
        self.mid2 = ConditionalResBlock3D(c5, c5, cond_dim, norm_groups, dropout)
        self.up10 = Upsample3D(c5, c10)
        self.dec10a = ConditionalResBlock3D(c10 + c10, c10, cond_dim, norm_groups, dropout)
        self.dec10b = ConditionalResBlock3D(c10, c10, cond_dim, norm_groups, dropout)
        self.up20 = Upsample3D(c10, c20)
        self.dec20a = ConditionalResBlock3D(c20 + c20, c20, cond_dim, norm_groups, dropout)
        self.dec20b = ConditionalResBlock3D(c20, c20, cond_dim, norm_groups, dropout)
        self.up40 = Upsample3D(c20, c40)
        self.dec40a = ConditionalResBlock3D(c40 + c40, c40, cond_dim, norm_groups, dropout)
        self.dec40b = ConditionalResBlock3D(c40, c40, cond_dim, norm_groups, dropout)
        self.fuse_mri40e = FusionGate3D(c40, c40, norm_groups)
        self.fuse_mri20e = FusionGate3D(c20, c20, norm_groups)
        self.fuse_mri10e = FusionGate3D(c10, c10, norm_groups)
        self.fuse_mri5 = FusionGate3D(c5, c5, norm_groups)
        self.fuse_mri10d = FusionGate3D(c10, c10, norm_groups)
        self.fuse_mri20d = FusionGate3D(c20, c20, norm_groups)
        self.fuse_mri40d = FusionGate3D(c40, c40, norm_groups)
        global_ch = [int(v) for v in global_channels]
        self.fuse_global20e = FusionGate3D(c20, global_ch[1], norm_groups)
        self.fuse_global10e = FusionGate3D(c10, global_ch[2], norm_groups)
        self.fuse_global10d = FusionGate3D(c10, global_ch[2], norm_groups)
        self.fuse_global20d = FusionGate3D(c20, global_ch[1], norm_groups)
        self.out_norm = SafeGroupNorm(c40, norm_groups)
        self.out_conv = nn.Conv3d(c40, 1, kernel_size=3, padding=1)

    def _global_condition(self, time: torch.Tensor, pooled: torch.Tensor, patch_coords: torch.Tensor) -> torch.Tensor:
        time_cond = self.time_embed(time)
        if time_cond.shape[0] == 1 and patch_coords.shape[0] > 1:
            time_cond = time_cond.expand(patch_coords.shape[0], -1)
        global_cond = self.global_proj(pooled)
        if global_cond.shape[0] == 1 and patch_coords.shape[0] > 1:
            global_cond = global_cond.expand(patch_coords.shape[0], -1)
        return time_cond + global_cond + self.pos_embed(patch_coords)

    def encode_global(self, x_t_full: torch.Tensor, mri_full: torch.Tensor, time: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.time_embed(time)
        return self.global_encoder(x_t_full, mri_full, cond)

    def _crop_feature(self, feature: torch.Tensor, patch_coords: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = feature.shape
        if batch != 1:
            raise ValueError("Exp025d currently assumes one case per batch for global patch cropping.")
        size = (depth // self.grid_size[0], height // self.grid_size[1], width // self.grid_size[2])
        patches = []
        for coord in patch_coords.long().tolist():
            z, y, x = coord
            patches.append(feature[:, :, z * size[0] : (z + 1) * size[0], y * size[1] : (y + 1) * size[1], x * size[2] : (x + 1) * size[2]])
        return torch.cat(patches, dim=0)

    def denoise_patches(
        self,
        x_t_patch: torch.Tensor,
        mri_patch: torch.Tensor,
        region_mask_patch: torch.Tensor,
        patch_coords: torch.Tensor,
        time: torch.Tensor,
        global_context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        region_mask = region_mask_patch.float() / max(float(self.num_regions), 1.0)
        boundary = _region_boundary(region_mask_patch)
        x = self.local_stem(torch.cat([x_t_patch, region_mask, boundary], dim=1))
        mri_features = self.mri_encoder(mri_patch)
        cond = self._global_condition(time, global_context["pooled"], patch_coords)
        global20 = self._crop_feature(global_context["80"], patch_coords)
        global10 = self._crop_feature(global_context["40"], patch_coords)

        x40 = self.enc40a(x, cond)
        x40 = self.enc40b(x40, cond)
        x40 = self.fuse_mri40e(x40, mri_features["40"])
        skip40 = x40

        x20 = self.down20(x40)
        x20 = self.enc20a(x20, cond)
        x20 = self.enc20b(x20, cond)
        x20 = self.fuse_mri20e(x20, mri_features["20"])
        x20 = self.fuse_global20e(x20, global20)
        skip20 = x20

        x10 = self.down10(x20)
        x10 = self.enc10a(x10, cond)
        x10 = self.enc10b(x10, cond)
        x10 = self.fuse_mri10e(x10, mri_features["10"])
        x10 = self.fuse_global10e(x10, global10)
        skip10 = x10

        x5 = self.down5(x10)
        x5 = self.mid1(x5, cond)
        x5 = self.mid2(x5, cond)
        x5 = self.fuse_mri5(x5, mri_features["5"])

        y10 = self.up10(x5, skip10.shape[-3:])
        y10 = self.dec10a(torch.cat([y10, skip10], dim=1), cond)
        y10 = self.dec10b(y10, cond)
        y10 = self.fuse_mri10d(y10, mri_features["10"])
        y10 = self.fuse_global10d(y10, global10)

        y20 = self.up20(y10, skip20.shape[-3:])
        y20 = self.dec20a(torch.cat([y20, skip20], dim=1), cond)
        y20 = self.dec20b(y20, cond)
        y20 = self.fuse_mri20d(y20, mri_features["20"])
        y20 = self.fuse_global20d(y20, global20)

        y40 = self.up40(y20, skip40.shape[-3:])
        y40 = self.dec40a(torch.cat([y40, skip40], dim=1), cond)
        y40 = self.dec40b(y40, cond)
        y40 = self.fuse_mri40d(y40, mri_features["40"])

        return self.out_conv(F.silu(self.out_norm(y40)))
