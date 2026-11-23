from __future__ import annotations

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


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, num_groups: int = 16) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            SafeGroupNorm(in_channels, num_groups=num_groups),
            Swish(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            SafeGroupNorm(out_channels, num_groups=num_groups),
            Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x)) + self.skip(x)


class SelfAttention3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 16) -> None:
        super().__init__()
        heads = max(1, min(int(num_heads), int(channels)))
        while heads > 1 and channels % heads != 0:
            heads -= 1
        self.num_heads = heads
        self.norm = SafeGroupNorm(channels, num_groups=num_groups)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, d * h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, c // self.num_heads, d * h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, d * h * w).permute(0, 1, 3, 2)
        attn = torch.matmul(q, k) * (c // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(b, c, d, h, w)
        return x + self.proj(out)


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


class ResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        channel_mults: Iterable[int] = (1, 2, 4),
        dropout: float = 0.1,
        num_groups: int = 16,
        use_attention: bool = True,
        attention_level: int = 2,
    ) -> None:
        super().__init__()
        mults = [int(m) for m in channel_mults]
        channels = [int(base_channels) * m for m in mults]
        if len(channels) < 2:
            raise ValueError("channel_mults must have at least two stages.")

        self.stem = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx, ch in enumerate(channels):
            in_ch = channels[0] if idx == 0 else ch
            block = nn.Sequential(
                ResidualBlock3D(in_ch, ch, dropout=dropout, num_groups=num_groups),
                ResidualBlock3D(ch, ch, dropout=dropout, num_groups=num_groups),
                SelfAttention3D(ch, num_heads=4, num_groups=num_groups)
                if use_attention and idx == attention_level
                else nn.Identity(),
            )
            self.encoders.append(block)
            if idx < len(channels) - 1:
                self.downsamples.append(Downsample3D(ch, channels[idx + 1]))

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(bottleneck_ch, bottleneck_ch, dropout=dropout, num_groups=num_groups),
            SelfAttention3D(bottleneck_ch, num_heads=4, num_groups=num_groups) if use_attention else nn.Identity(),
            ResidualBlock3D(bottleneck_ch, bottleneck_ch, dropout=dropout, num_groups=num_groups),
        )

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_channels = list(reversed(channels))
        current_ch = rev_channels[0]
        for idx, skip_ch in enumerate(rev_channels[1:]):
            self.upsamples.append(Upsample3D(current_ch, skip_ch))
            self.decoders.append(
                nn.Sequential(
                    ResidualBlock3D(skip_ch * 2, skip_ch, dropout=dropout, num_groups=num_groups),
                    ResidualBlock3D(skip_ch, skip_ch, dropout=dropout, num_groups=num_groups),
                    SelfAttention3D(skip_ch, num_heads=4, num_groups=num_groups)
                    if use_attention and (len(rev_channels) - idx - 2) == attention_level
                    else nn.Identity(),
                )
            )
            current_ch = skip_ch

        self.head = nn.Sequential(
            SafeGroupNorm(channels[0], num_groups=num_groups),
            Swish(),
            nn.Conv3d(channels[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        x = self.bottleneck(x)

        skips = list(reversed(skips[:-1]))
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, skips):
            x = upsample(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.head(x)
