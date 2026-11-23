from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SafeGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int) -> None:
        super().__init__()
        groups = min(int(num_groups), int(num_channels))
        while int(num_channels) % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, int(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * (-scale)
        )
        emb = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Swish(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ResBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.norm1 = SafeGroupNorm(num_groups, self.in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(int(time_dim), self.out_channels)

        self.norm2 = SafeGroupNorm(num_groups, self.out_channels)
        self.act2 = Swish()
        self.dropout = nn.Dropout3d(float(dropout))
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)


class SelfAttention3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        if self.channels % self.num_heads != 0:
            self.num_heads = 1
        self.norm = SafeGroupNorm(num_groups, self.channels)
        self.qkv = nn.Conv3d(self.channels, self.channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(self.channels, self.channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        head_dim = c // self.num_heads
        n = d * h * w

        q = q.reshape(b, self.num_heads, head_dim, n).transpose(-1, -2)
        k = k.reshape(b, self.num_heads, head_dim, n)
        v = v.reshape(b, self.num_heads, head_dim, n).transpose(-1, -2)

        attn = torch.matmul(q, k) * (head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).reshape(b, c, d, h, w)
        out = self.proj(out)
        return residual + out


class HaarDWT3D(nn.Module):
    """
    3D Haar 小波分解。

    输入:
        x: [B, C, D, H, W]

    输出:
        subbands: [B, 8C, D/2, H/2, W/2]

    8 个子带包括:
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(8.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 为了安全，如果某个维度不是偶数，就在右侧 pad 1 个 voxel。
        d_pad = x.shape[-3] % 2
        h_pad = x.shape[-2] % 2
        w_pad = x.shape[-1] % 2
        if d_pad or h_pad or w_pad:
            x = F.pad(x, (0, w_pad, 0, h_pad, 0, d_pad))

        x000 = x[:, :, 0::2, 0::2, 0::2]
        x001 = x[:, :, 0::2, 0::2, 1::2]
        x010 = x[:, :, 0::2, 1::2, 0::2]
        x011 = x[:, :, 0::2, 1::2, 1::2]
        x100 = x[:, :, 1::2, 0::2, 0::2]
        x101 = x[:, :, 1::2, 0::2, 1::2]
        x110 = x[:, :, 1::2, 1::2, 0::2]
        x111 = x[:, :, 1::2, 1::2, 1::2]

        lll = (x000 + x001 + x010 + x011 + x100 + x101 + x110 + x111) * self.scale
        llh = (x000 - x001 + x010 - x011 + x100 - x101 + x110 - x111) * self.scale
        lhl = (x000 + x001 - x010 - x011 + x100 + x101 - x110 - x111) * self.scale
        lhh = (x000 - x001 - x010 + x011 + x100 - x101 - x110 + x111) * self.scale
        hll = (x000 + x001 + x010 + x011 - x100 - x101 - x110 - x111) * self.scale
        hlh = (x000 - x001 + x010 - x011 - x100 + x101 - x110 + x111) * self.scale
        hhl = (x000 + x001 - x010 - x011 - x100 - x101 + x110 + x111) * self.scale
        hhh = (x000 - x001 - x010 + x011 - x100 + x101 + x110 - x111) * self.scale

        return torch.cat([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], dim=1)


class WaveletDownBlock3D(nn.Module):
    """
    用 3D Haar DWT 代替普通 stride conv / pooling。

    作用:
        [B, C, D, H, W]
        -> DWT 得到 [B, 8C, D/2, H/2, W/2]
        -> 1x1 conv 融合子带
        -> ResBlock 提炼特征
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dwt = HaarDWT3D()
        self.fuse = nn.Conv3d(int(in_channels) * 8, int(out_channels), kernel_size=1)
        self.block = ResBlock3D(
            in_channels=out_channels,
            out_channels=out_channels,
            time_dim=time_dim,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.dwt(x)
        x = self.fuse(x)
        x = self.block(x, t_emb)
        return x


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pre = nn.Conv3d(int(in_channels), int(out_channels), kernel_size=1)
        self.block = ResBlock3D(
            in_channels=int(out_channels) + int(skip_channels),
            out_channels=int(out_channels),
            time_dim=time_dim,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = self.pre(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_emb)
        return x


class ResidualBridgeWaveletResUNet3D(nn.Module):
    """
    用于 Residual Bridge 的 3D Wavelet-ResUNet。

    接口保持和 ResidualBridgeResUNet3D 一致:
        forward(x_t, condition, t) -> residual_pred

    输入:
        x_t: [B, 1, D, H, W]
        condition: [B, 1, D, H, W]
        t: [B]

    输出:
        residual_pred: [B, 1, D, H, W]
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 8,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        time_dim: int = 256,
        dropout: float = 0.1,
        num_groups: int = 16,
        use_attention: bool = True,
        attention_level: int = 3,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(v) for v in channel_mults)
        self.time_dim = int(time_dim)
        self.use_attention = bool(use_attention)
        self.attention_level = int(attention_level)

        channels = [self.base_channels * m for m in self.channel_mults]
        if len(channels) < 2:
            raise ValueError("channel_mults 至少需要两个层级，例如 [1, 2, 4, 8]")

        self.time_mlp = TimeMLP(self.time_dim)

        self.input_conv = nn.Conv3d(self.in_channels, channels[0], kernel_size=3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        # encoder level 0
        self.enc_blocks.append(
            ResBlock3D(
                in_channels=channels[0],
                out_channels=channels[0],
                time_dim=self.time_dim,
                num_groups=num_groups,
                dropout=dropout,
            )
        )

        # wavelet down levels
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                WaveletDownBlock3D(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    time_dim=self.time_dim,
                    num_groups=num_groups,
                    dropout=dropout,
                )
            )
            self.enc_blocks.append(
                ResBlock3D(
                    in_channels=channels[i + 1],
                    out_channels=channels[i + 1],
                    time_dim=self.time_dim,
                    num_groups=num_groups,
                    dropout=dropout,
                )
            )

        bottleneck_channels = channels[-1]
        self.mid1 = ResBlock3D(
            bottleneck_channels,
            bottleneck_channels,
            time_dim=self.time_dim,
            num_groups=num_groups,
            dropout=dropout,
        )

        if self.use_attention:
            self.mid_attn = SelfAttention3D(
                bottleneck_channels,
                num_heads=num_heads,
                num_groups=num_groups,
            )
        else:
            self.mid_attn = nn.Identity()

        self.mid2 = ResBlock3D(
            bottleneck_channels,
            bottleneck_channels,
            time_dim=self.time_dim,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.up_blocks = nn.ModuleList()

        # decoder:
        # channels: [8, 16, 32, 64]
        # bottleneck 64 -> up with skip 32 -> 32
        # 32 -> up with skip 16 -> 16
        # 16 -> up with skip 8 -> 8
        for i in reversed(range(len(channels) - 1)):
            self.up_blocks.append(
                UpBlock3D(
                    in_channels=channels[i + 1],
                    skip_channels=channels[i],
                    out_channels=channels[i],
                    time_dim=self.time_dim,
                    num_groups=num_groups,
                    dropout=dropout,
                )
            )

        self.out_norm = SafeGroupNorm(num_groups, channels[0])
        self.out_act = Swish()
        self.out_conv = nn.Conv3d(channels[0], self.out_channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 和原 ResidualBridgeResUNet3D 一样，输入是 x_t 和 condition 拼接
        x = torch.cat([x_t, condition], dim=1)
        t_emb = self.time_mlp(t)

        x = self.input_conv(x)

        skips: list[torch.Tensor] = []

        # level 0
        x = self.enc_blocks[0](x, t_emb)
        skips.append(x)

        # wavelet encoder
        for i, down in enumerate(self.down_blocks):
            x = down(x, t_emb)
            x = self.enc_blocks[i + 1](x, t_emb)
            if i < len(self.down_blocks) - 1:
                skips.append(x)

        # bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        # decoder
        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip, t_emb)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x