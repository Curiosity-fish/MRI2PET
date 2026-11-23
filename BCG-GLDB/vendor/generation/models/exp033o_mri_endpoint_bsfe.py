from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SafeGroupNorm(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        g = min(int(groups), int(channels))
        while g > 1 and channels % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(in_ch, groups)
        self.act1 = SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = SafeGroupNorm(out_ch, groups)
        self.act2 = SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.drop(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@dataclass
class Exp033OEndpointConfig:
    in_channels: int = 1
    latent_channels: int = 4
    pre_down_hidden: int = 16
    base_channels: int = 24
    channel_mults: tuple[int, ...] = (2, 4, 8, 16)
    num_res_blocks: int = 2
    norm_groups: int = 8
    dropout: float = 0.05
    head_channels: tuple[int, ...] = (32, 16, 8)


class MRIPETEndpointBSFE(nn.Module):
    """MRI -> z_mri endpoint -> latent-coordinate pyramid.

    z_mri is the actual bridge endpoint. The pyramid is extracted after z_mri so
    the bridge condition and Ada features share the same endpoint representation.
    """

    def __init__(self, cfg: Exp033OEndpointConfig) -> None:
        super().__init__()
        self.cfg = cfg
        g = int(cfg.norm_groups)
        h = int(cfg.pre_down_hidden)
        zc = int(cfg.latent_channels)

        self.endpoint_encoder = nn.Sequential(
            ResBlock3D(int(cfg.in_channels), int(cfg.in_channels), g, 0.0),
            nn.Conv3d(int(cfg.in_channels), h, 4, stride=2, padding=1),
            SafeGroupNorm(h, g),
            SiLU(),
            ResBlock3D(h, h, g, 0.0),
            nn.Conv3d(h, h, 4, stride=2, padding=1),
            SafeGroupNorm(h, g),
            SiLU(),
            ResBlock3D(h, h, g, float(cfg.dropout)),
            nn.Conv3d(h, zc, 3, padding=1),
        )

        base = int(cfg.base_channels)
        self.stem = nn.Sequential(
            nn.Conv3d(zc, base, 3, padding=1),
            SafeGroupNorm(base, g),
            SiLU(),
        )
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = base
        for i, mult in enumerate(tuple(cfg.channel_mults)):
            out_ch = base * int(mult)
            blocks = []
            for _ in range(int(cfg.num_res_blocks)):
                blocks.append(ResBlock3D(in_ch, out_ch, g, float(cfg.dropout)))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))
            if i != len(tuple(cfg.channel_mults)) - 1:
                self.downs.append(Downsample3D(out_ch))
            in_ch = out_ch

        self.pet_head = self._build_pet_head(zc, tuple(int(c) for c in cfg.head_channels), g, float(cfg.dropout))

    @staticmethod
    def _build_pet_head(in_ch: int, channels: tuple[int, ...], groups: int, dropout: float) -> nn.Module:
        c0, c1, c2 = channels
        return nn.Sequential(
            nn.Conv3d(in_ch, c0, 3, padding=1),
            SafeGroupNorm(c0, groups),
            SiLU(),
            ResBlock3D(c0, c0, groups, dropout),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(c0, c1, 3, padding=1),
            SafeGroupNorm(c1, groups),
            SiLU(),
            ResBlock3D(c1, c1, groups, dropout),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(c1, c2, 3, padding=1),
            SafeGroupNorm(c2, groups),
            SiLU(),
            nn.Conv3d(c2, 1, 3, padding=1),
        )

    def encode_endpoint(self, mri: torch.Tensor) -> torch.Tensor:
        return self.endpoint_encoder(mri)

    def encode_pyramid_from_endpoint(self, z_mri: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(z_mri)
        pyramid = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            pyramid.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return pyramid

    def forward(self, mri: torch.Tensor, with_pet: bool = True) -> dict[str, Any]:
        z_mri = self.encode_endpoint(mri)
        pyramid = self.encode_pyramid_from_endpoint(z_mri)
        out: dict[str, Any] = {"z_mri": z_mri, "pyramid": pyramid}
        if with_pet:
            out["pet_aux"] = self.pet_head(z_mri)
        return out


class Exp033OBridgeModel(nn.Module):
    def __init__(self, endpoint: MRIPETEndpointBSFE, unet: nn.Module) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.unet = unet
        self.freeze_endpoint()

    def freeze_endpoint(self) -> None:
        self.endpoint.eval()
        for p in self.endpoint.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.endpoint.eval()
        return self

    @torch.no_grad()
    def encode_condition(self, mri: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        out = self.endpoint(mri, with_pet=False)
        return out["z_mri"], out["pyramid"]

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        pyramid,
        table: torch.Tensor,
        table_mask: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.unet.time_encoder(t)
        return self.unet(x_t, condition, t_emb, pyramid, table, table_mask)
