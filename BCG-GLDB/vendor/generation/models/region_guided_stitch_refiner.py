from __future__ import annotations

from typing import Sequence

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
        self.norm = nn.GroupNorm(
            _safe_groups(int(channels), int(groups)),
            int(channels),
            eps=1e-6,
            affine=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class RegionModulation3D(nn.Module):
    """Spatial FiLM modulation generated from categorical region embeddings."""

    def __init__(self, region_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(int(region_dim), int(channels) * 2, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, region_feature: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(region_feature).chunk(2, dim=1)
        return x * (1.0 + gamma) + beta


class DepthwiseResidualBlock3D(nn.Module):
    """Lightweight residual block with region-conditioned modulation."""

    def __init__(
        self,
        channels: int,
        region_dim: int,
        norm_groups: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        dilation = int(dilation)
        self.norm1 = SafeGroupNorm(channels, norm_groups)
        self.region_mod1 = RegionModulation3D(region_dim, channels)
        self.depthwise1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm2 = SafeGroupNorm(channels, norm_groups)
        self.region_mod2 = RegionModulation3D(region_dim, channels)
        self.depthwise2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.pointwise2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout3d(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, region_feature: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.region_mod1(self.norm1(x), region_feature)
        h = self.pointwise1(self.depthwise1(F.silu(h)))
        h = self.region_mod2(self.norm2(h), region_feature)
        h = self.pointwise2(self.depthwise2(self.dropout(F.silu(h))))
        return residual + h


class RegionGuidedLightweightStitchRefiner3D(nn.Module):
    """Lightweight full-volume residual refiner.

    Inputs:
        stitched_pet: [B, 1, D, H, W]
        region_mask:  [B, 1, D, H, W], integer labels in [0, num_regions]

    The fixed patch seam map is constructed internally from image_size/patch_size.
    The anatomical region-boundary map is derived from region_mask at runtime.
    """

    def __init__(
        self,
        image_size: tuple[int, int, int] = (160, 192, 160),
        patch_size: tuple[int, int, int] = (40, 48, 40),
        num_regions: int = 98,
        region_embedding_dim: int = 4,
        channels: tuple[int, int, int] = (8, 16, 32),
        blocks_per_level: tuple[int, int, int] = (1, 1, 2),
        bottleneck_dilations: Sequence[int] = (1, 2),
        norm_groups: int = 4,
        dropout: float = 0.0,
        seam_width: int = 3,
        boundary_suppression: float = 0.5,
        seam_prior_bias: float = 1.5,
        max_residual_scale: float = 0.2,
        residual_strength_init: float = 0.1,
        zero_init_residual_head: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = tuple(int(v) for v in image_size)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.num_regions = int(num_regions)
        self.region_embedding_dim = int(region_embedding_dim)
        self.boundary_suppression = float(boundary_suppression)
        self.seam_prior_bias = float(seam_prior_bias)
        self.max_residual_scale = float(max_residual_scale)

        if any(i % p != 0 for i, p in zip(self.image_size, self.patch_size)):
            raise ValueError(
                f"image_size={self.image_size} must be divisible by patch_size={self.patch_size}."
            )
        if len(channels) != 3:
            raise ValueError("Refiner channels must contain exactly three scales.")
        if len(blocks_per_level) != 3:
            raise ValueError("blocks_per_level must contain exactly three values.")

        seam_map = self._build_seam_map(self.image_size, self.patch_size, int(seam_width))
        self.register_buffer("seam_map", seam_map, persistent=True)

        c0, c1, c2 = [int(v) for v in channels]
        self.region_embedding = nn.Embedding(self.num_regions + 1, self.region_embedding_dim)
        nn.init.normal_(self.region_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.region_embedding.weight[0].zero_()

        # Network input remains only stitched PET plus internally derived seam/boundary maps.
        self.stem = nn.Conv3d(3, c0, kernel_size=3, padding=1)
        self.encoder0 = nn.ModuleList(
            [
                DepthwiseResidualBlock3D(c0, self.region_embedding_dim, norm_groups, dropout=dropout)
                for _ in range(int(blocks_per_level[0]))
            ]
        )
        self.down1 = nn.Conv3d(c0, c1, kernel_size=3, stride=2, padding=1)
        self.encoder1 = nn.ModuleList(
            [
                DepthwiseResidualBlock3D(c1, self.region_embedding_dim, norm_groups, dropout=dropout)
                for _ in range(int(blocks_per_level[1]))
            ]
        )
        self.down2 = nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1)

        dilations = [int(v) for v in bottleneck_dilations]
        if not dilations:
            dilations = [1]
        bottleneck_count = max(int(blocks_per_level[2]), len(dilations))
        self.bottleneck = nn.ModuleList(
            [
                DepthwiseResidualBlock3D(
                    c2,
                    self.region_embedding_dim,
                    norm_groups,
                    dilation=dilations[index % len(dilations)],
                    dropout=dropout,
                )
                for index in range(bottleneck_count)
            ]
        )

        self.up1 = nn.Conv3d(c2, c1, kernel_size=3, padding=1)
        self.decoder1 = DepthwiseResidualBlock3D(
            c1,
            self.region_embedding_dim,
            norm_groups,
            dropout=dropout,
        )
        self.up0 = nn.Conv3d(c1, c0, kernel_size=3, padding=1)
        self.decoder0 = DepthwiseResidualBlock3D(
            c0,
            self.region_embedding_dim,
            norm_groups,
            dropout=dropout,
        )

        self.output_norm = SafeGroupNorm(c0, norm_groups)
        self.residual_head = nn.Conv3d(c0, 1, kernel_size=3, padding=1)
        self.gate_head = nn.Conv3d(c0, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.gate_head.bias, -1.5)
        if zero_init_residual_head:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

        init = torch.tensor(float(residual_strength_init)).clamp(-0.99, 0.99)
        self.residual_strength = nn.Parameter(torch.atanh(init))

    @staticmethod
    def _build_seam_map(
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        seam_width: int,
    ) -> torch.Tensor:
        seam_width = max(int(seam_width), 0)
        seam = torch.zeros((1, 1, *image_size), dtype=torch.float32)
        sigma = max(float(seam_width) / 2.0, 1.0)
        for axis, (size, patch) in enumerate(zip(image_size, patch_size)):
            for center in range(int(patch), int(size), int(patch)):
                low = max(0, center - seam_width)
                high = min(size, center + seam_width + 1)
                positions = torch.arange(low, high, dtype=torch.float32)
                weights = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
                if axis == 0:
                    current = seam[:, :, low:high, :, :]
                    seam[:, :, low:high, :, :] = torch.maximum(
                        current,
                        weights.view(1, 1, -1, 1, 1),
                    )
                elif axis == 1:
                    current = seam[:, :, :, low:high, :]
                    seam[:, :, :, low:high, :] = torch.maximum(
                        current,
                        weights.view(1, 1, 1, -1, 1),
                    )
                else:
                    current = seam[:, :, :, :, low:high]
                    seam[:, :, :, :, low:high] = torch.maximum(
                        current,
                        weights.view(1, 1, 1, 1, -1),
                    )
        return seam

    @staticmethod
    def build_region_boundary(region_mask: torch.Tensor) -> torch.Tensor:
        labels = region_mask.long()
        if labels.ndim != 5 or labels.shape[1] != 1:
            raise ValueError(f"region_mask must be [B,1,D,H,W], got {labels.shape}.")
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

        # Background transitions are useful boundaries, but the all-background area is not.
        return boundary.float()

    def _region_feature(
        self,
        region_mask: torch.Tensor,
        size: tuple[int, int, int],
    ) -> torch.Tensor:
        labels = F.interpolate(region_mask.float(), size=size, mode="nearest").long()
        labels = labels[:, 0].clamp(0, self.num_regions)
        feature = self.region_embedding(labels)
        return feature.permute(0, 4, 1, 2, 3).contiguous()

    def forward(
        self,
        stitched_pet: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if stitched_pet.ndim != 5 or stitched_pet.shape[1] != 1:
            raise ValueError(f"stitched_pet must be [B,1,D,H,W], got {stitched_pet.shape}.")
        if tuple(stitched_pet.shape[2:]) != self.image_size:
            raise ValueError(
                f"Expected stitched PET size {self.image_size}, got {tuple(stitched_pet.shape[2:])}."
            )
        region_mask = region_mask.long()
        if region_mask.shape != stitched_pet.shape:
            raise ValueError(
                f"region_mask must match stitched_pet shape, got {region_mask.shape} and {stitched_pet.shape}."
            )
        if region_mask.numel() > 0:
            min_label = int(region_mask.min().item())
            max_label = int(region_mask.max().item())
            if min_label < 0 or max_label > self.num_regions:
                raise ValueError(
                    f"Region labels must be in [0,{self.num_regions}], got [{min_label},{max_label}]."
                )

        batch = stitched_pet.shape[0]
        seam = self.seam_map.to(device=stitched_pet.device, dtype=stitched_pet.dtype).expand(
            batch, -1, -1, -1, -1
        )
        boundary = self.build_region_boundary(region_mask).to(dtype=stitched_pet.dtype)
        repair_prior = seam * (1.0 - self.boundary_suppression * boundary).clamp(0.0, 1.0)

        region0 = self._region_feature(region_mask, tuple(stitched_pet.shape[2:]))
        x0 = self.stem(torch.cat([stitched_pet, seam, boundary], dim=1))
        for block in self.encoder0:
            x0 = block(x0, region0)

        x1 = self.down1(x0)
        region1 = self._region_feature(region_mask, tuple(x1.shape[2:]))
        for block in self.encoder1:
            x1 = block(x1, region1)

        x2 = self.down2(x1)
        region2 = self._region_feature(region_mask, tuple(x2.shape[2:]))
        for block in self.bottleneck:
            x2 = block(x2, region2)

        y1 = F.interpolate(x2, size=x1.shape[2:], mode="trilinear", align_corners=False)
        y1 = self.up1(y1) + x1
        y1 = self.decoder1(y1, region1)

        y0 = F.interpolate(y1, size=x0.shape[2:], mode="trilinear", align_corners=False)
        y0 = self.up0(y0) + x0
        y0 = self.decoder0(y0, region0)
        y0 = F.silu(self.output_norm(y0))

        raw_residual = torch.tanh(self.residual_head(y0))
        gate = torch.sigmoid(self.gate_head(y0) + self.seam_prior_bias * repair_prior)
        strength = self.max_residual_scale * torch.tanh(self.residual_strength)
        correction = strength.to(dtype=stitched_pet.dtype) * gate * raw_residual
        refined_pet = stitched_pet + correction

        return {
            "refined_pet": refined_pet,
            "correction": correction,
            "raw_residual": raw_residual,
            "gate": gate,
            "seam_map": seam,
            "region_boundary": boundary,
            "repair_prior": repair_prior,
        }
