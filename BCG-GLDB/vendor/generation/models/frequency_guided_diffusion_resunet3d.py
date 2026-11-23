from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SafeGroupNorm(nn.Module):
    def __init__(self, channels: int, num_groups: int = 16) -> None:
        super().__init__()
        groups = min(int(num_groups), int(channels))
        while groups > 1 and channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-scale))
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditioningMLP(nn.Module):
    def __init__(self, time_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Swish(),
            nn.Linear(time_dim * 4, cond_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ConditioningResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0, num_groups: int = 16) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(in_channels, num_groups=num_groups)
        self.act1 = Swish()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Sequential(Swish(), nn.Linear(cond_dim, out_channels * 2))

        self.norm2 = SafeGroupNorm(out_channels, num_groups=num_groups)
        self.act2 = Swish()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        scale, shift = torch.chunk(self.cond_proj(cond_emb), chunks=2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(self.dropout(self.act2(h)))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MRIMultiScaleEncoder3D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, channel_mults: Iterable[int], num_groups: int = 8) -> None:
        super().__init__()
        channels = [int(base_channels) * int(mult) for mult in channel_mults]
        self.stem = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_ch = channels[0]
        for idx, ch in enumerate(channels):
            self.blocks.append(
                nn.Sequential(
                    SafeGroupNorm(current_ch, num_groups=num_groups),
                    Swish(),
                    nn.Conv3d(current_ch, ch, kernel_size=3, padding=1),
                    SafeGroupNorm(ch, num_groups=num_groups),
                    Swish(),
                    nn.Conv3d(ch, ch, kernel_size=3, padding=1),
                )
            )
            current_ch = ch
            if idx < len(channels) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv3d(current_ch, channels[idx + 1], kernel_size=3, padding=1),
                        Downsample3D(channels[idx + 1]),
                    )
                )
                current_ch = channels[idx + 1]

        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, mri: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        feats = []
        x = self.stem(mri)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            feats.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        global_vec = self.global_pool(feats[-1]).flatten(1)
        return feats, global_vec


class FrequencyGuidanceBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, input_mode: str = "xt_mri") -> None:
        super().__init__()
        self.input_mode = str(input_mode).lower()
        if self.input_mode not in {"mri_only", "xt_mri"}:
            raise ValueError(f"Unsupported frequency input mode: {input_mode}")
        input_dim = in_channels * (3 if self.input_mode == "mri_only" else 6)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _summarize(self, volume: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.rfftn(volume, dim=(-3, -2, -1), norm="ortho")
        amp = torch.abs(freq)
        low_d = max(1, amp.shape[-3] // 8)
        low_h = max(1, amp.shape[-2] // 8)
        low_w = max(1, amp.shape[-1] // 8)
        low = amp[:, :, :low_d, :low_h, :low_w]
        return torch.cat(
            [
                low.mean(dim=(-3, -2, -1)),
                low.amax(dim=(-3, -2, -1)),
                low.std(dim=(-3, -2, -1), unbiased=False),
            ],
            dim=1,
        )

    def forward(self, x_t: torch.Tensor, mri: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "mri_only":
            pooled = self._summarize(mri)
        else:
            pooled = torch.cat([self._summarize(x_t), self._summarize(mri)], dim=1)
        return self.proj(pooled)


class FrequencyGuidedDiffusionResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 8,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        time_dim: int = 128,
        dropout: float = 0.1,
        num_groups: int = 8,
        mri_in_channels: int = 1,
        frequency_hidden_dim: int = 128,
        condition_dim: int = 128,
        frequency_input_mode: str = "xt_mri",
    ) -> None:
        super().__init__()
        self.channels = [int(base_channels) * int(mult) for mult in channel_mults]
        self.condition_dim = int(condition_dim)

        self.time_mlp = ConditioningMLP(time_dim, self.condition_dim)
        self.mri_encoder = MRIMultiScaleEncoder3D(
            in_channels=mri_in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_groups=num_groups,
        )
        self.mri_global_proj = nn.Sequential(
            nn.Linear(self.channels[-1], self.condition_dim),
            Swish(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.frequency_branch = FrequencyGuidanceBranch(
            in_channels=mri_in_channels,
            hidden_dim=int(frequency_hidden_dim),
            out_dim=self.condition_dim,
            input_mode=frequency_input_mode,
        )
        self.condition_fuse = nn.Sequential(
            nn.Linear(self.condition_dim * 3, self.condition_dim),
            Swish(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )

        self.input_conv = nn.Conv3d(in_channels, self.channels[0], kernel_size=3, padding=1)
        self.mri_skip_projs = nn.ModuleList([nn.Conv3d(ch, ch, kernel_size=1) for ch in self.channels])

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_ch = self.channels[0]
        for idx, ch in enumerate(self.channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ConditioningResidualBlock3D(current_ch, ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups),
                        ConditioningResidualBlock3D(ch, ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups),
                    ]
                )
            )
            current_ch = ch
            if idx < len(self.channels) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv3d(current_ch, self.channels[idx + 1], kernel_size=3, padding=1),
                        Downsample3D(self.channels[idx + 1]),
                    )
                )
                current_ch = self.channels[idx + 1]

        self.mid1 = ConditioningResidualBlock3D(current_ch, current_ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups)
        self.mid2 = ConditioningResidualBlock3D(current_ch, current_ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_channels = list(reversed(self.channels))
        current_ch = rev_channels[0]
        for skip_ch in rev_channels[1:]:
            self.upsamples.append(
                nn.Sequential(
                    Upsample3D(current_ch),
                    nn.Conv3d(current_ch, skip_ch, kernel_size=3, padding=1),
                )
            )
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ConditioningResidualBlock3D(skip_ch * 2, skip_ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups),
                        ConditioningResidualBlock3D(skip_ch, skip_ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups),
                    ]
                )
            )
            current_ch = skip_ch

        self.out_norm = SafeGroupNorm(self.channels[0], num_groups=num_groups)
        self.out_act = Swish()
        self.out_conv = nn.Conv3d(self.channels[0], out_channels, kernel_size=3, padding=1)

    def build_condition(self, x_t: torch.Tensor, mri: torch.Tensor, t: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        mri_feats, mri_global = self.mri_encoder(mri)
        time_emb = self.time_mlp(t)
        mri_emb = self.mri_global_proj(mri_global)
        freq_emb = self.frequency_branch(x_t, mri)
        cond = self.condition_fuse(torch.cat([time_emb, mri_emb, freq_emb], dim=1))
        return mri_feats, cond

    def forward(self, x_t: torch.Tensor, mri: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mri_feats, cond = self.build_condition(x_t, mri, t)
        x = self.input_conv(torch.cat([x_t, mri], dim=1))

        skips = []
        for idx, blocks in enumerate(self.down_blocks):
            x = x + self.mri_skip_projs[idx](mri_feats[idx])
            for block in blocks:
                x = block(x, cond)
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        skips = list(reversed(skips[:-1]))
        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, skips):
            x = upsample(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, cond)

        return self.out_conv(self.out_act(self.out_norm(x)))
