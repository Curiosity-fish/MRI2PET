from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from models.resunet3d import SafeGroupNorm, SelfAttention3D, Swish


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-scale))
        emb = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualTimeBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0, num_groups: int = 16) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(in_channels, num_groups=num_groups)
        self.act1 = Swish()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            Swish(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = SafeGroupNorm(out_channels, num_groups=num_groups)
        self.act2 = Swish()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBridgeResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        time_dim: int = 256,
        dropout: float = 0.1,
        num_groups: int = 16,
        use_attention: bool = True,
        attention_level: int = 3,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        mults = [int(m) for m in channel_mults]
        channels = [int(base_channels) * m for m in mults]
        if len(channels) < 2:
            raise ValueError("channel_mults must have at least two stages.")

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Swish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.input_conv = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_ch = channels[0]
        for idx, ch in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualTimeBlock3D(current_ch, ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                        ResidualTimeBlock3D(ch, ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                        SelfAttention3D(ch, num_heads=num_heads, num_groups=num_groups)
                        if use_attention and idx == attention_level
                        else nn.Identity(),
                    ]
                )
            )
            current_ch = ch
            if idx < len(channels) - 1:
                self.downsamples.append(Downsample3D(current_ch, channels[idx + 1]))
                current_ch = channels[idx + 1]

        self.mid_blocks = nn.ModuleList(
            [
                ResidualTimeBlock3D(current_ch, current_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                SelfAttention3D(current_ch, num_heads=num_heads, num_groups=num_groups) if use_attention else nn.Identity(),
                ResidualTimeBlock3D(current_ch, current_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
            ]
        )

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        current_ch = rev_channels[0]
        for idx, skip_ch in enumerate(rev_channels[1:]):
            level = len(rev_channels) - idx - 2
            self.upsamples.append(Upsample3D(current_ch, skip_ch))
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualTimeBlock3D(skip_ch * 2, skip_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                        ResidualTimeBlock3D(skip_ch, skip_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                        SelfAttention3D(skip_ch, num_heads=num_heads, num_groups=num_groups)
                        if use_attention and level == attention_level
                        else nn.Identity(),
                    ]
                )
            )
            current_ch = skip_ch

        self.out_norm = SafeGroupNorm(channels[0], num_groups=num_groups)
        self.out_act = Swish()
        self.out_conv = nn.Conv3d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_t, condition], dim=1)
        t_emb = self.time_mlp(t)
        x = self.input_conv(x)

        skips = []
        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResidualTimeBlock3D):
                    x = block(x, t_emb)
                else:
                    x = block(x)
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        for block in self.mid_blocks:
            if isinstance(block, ResidualTimeBlock3D):
                x = block(x, t_emb)
            else:
                x = block(x)

        skips = list(reversed(skips[:-1]))
        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, skips):
            x = upsample(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                if isinstance(block, ResidualTimeBlock3D):
                    x = block(x, t_emb)
                else:
                    x = block(x)

        return self.out_conv(self.out_act(self.out_norm(x)))
