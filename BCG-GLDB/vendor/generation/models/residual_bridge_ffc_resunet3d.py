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


class FourierUnit3D(nn.Module):
    """
    3D Fourier Unit.

    输入:
        x: [B, C, D, H, W]

    流程:
        x -> rFFTN -> real/imag concat -> 1x1x1 conv -> complex -> irFFTN

    注意:
        torch.fft 对 bf16 支持不稳定，所以这里强制在 float32 中做 FFT。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels * 2, kernel_size=1, bias=True),
            nn.GroupNorm(1, channels * 2),
            Swish(),
            nn.Conv3d(channels * 2, channels * 2, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        spatial_size = tuple(int(v) for v in x.shape[-3:])

        # FFT 建议用 float32，避免 bf16 / fp16 下不稳定
        if x.device.type == "cuda":
            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
        else:
            autocast_ctx = torch.amp.autocast(device_type="cpu", enabled=False)

        with autocast_ctx:
            x_float = x.float()

            ffted = torch.fft.rfftn(
                x_float,
                dim=(-3, -2, -1),
                norm="ortho",
            )

            fft_feature = torch.cat(
                [ffted.real, ffted.imag],
                dim=1,
            )

            fft_feature = self.conv(fft_feature)

            real, imag = torch.chunk(fft_feature, 2, dim=1)
            ffted_out = torch.complex(real.contiguous(), imag.contiguous())

            out = torch.fft.irfftn(
                ffted_out,
                s=spatial_size,
                dim=(-3, -2, -1),
                norm="ortho",
            )

        return out.to(dtype=original_dtype)


class SpectralResidual3D(nn.Module):
    """
    频域残差增强模块。

    为了省显存，不直接对全部 channels 做 FFT，而是:
        C -> C_spectral -> FFT -> C_spectral -> C

    global_ratio 控制参与 FFT 的通道比例。
    """

    def __init__(
        self,
        channels: int,
        global_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        channels = int(channels)
        spectral_channels = max(1, int(round(channels * float(global_ratio))))

        self.reduce = nn.Conv3d(channels, spectral_channels, kernel_size=1)
        self.fourier = FourierUnit3D(spectral_channels)
        self.expand = nn.Conv3d(spectral_channels, channels, kernel_size=1)

        # 初始设小一点，避免训练初期频域分支破坏原始特征
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        z = self.fourier(z)
        z = self.expand(z)
        return x + self.alpha.to(dtype=x.dtype) * z


class LocalGlobalConv3D(nn.Module):
    """
    FFC-like 3D 卷积模块。

    local branch:
        普通 3D Conv，负责局部细节

    global branch:
        Fourier residual，负责全局 / 长距离关系
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.local = nn.Conv3d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            padding=1,
        )
        self.spectral = SpectralResidual3D(
            channels=int(out_channels),
            global_ratio=float(global_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        x = self.spectral(x)
        return x


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


class FFCResBlock3D(nn.Module):
    """
    带 FFC-like 分支的 ResBlock。

    与普通 ResBlock3D 的区别:
        第二个 conv 换成 LocalGlobalConv3D。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 16,
        dropout: float = 0.1,
        global_ratio: float = 0.25,
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

        self.local_global = LocalGlobalConv3D(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            global_ratio=global_ratio,
        )

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.local_global(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 16,
        dropout: float = 0.1,
        use_ffc: bool = False,
        global_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.down = nn.Conv3d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            stride=2,
            padding=1,
        )

        block_cls = FFCResBlock3D if use_ffc else ResBlock3D
        if use_ffc:
            self.block = block_cls(
                out_channels,
                out_channels,
                time_dim=time_dim,
                num_groups=num_groups,
                dropout=dropout,
                global_ratio=global_ratio,
            )
        else:
            self.block = block_cls(
                out_channels,
                out_channels,
                time_dim=time_dim,
                num_groups=num_groups,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
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
        use_ffc: bool = False,
        global_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.pre = nn.Conv3d(int(in_channels), int(out_channels), kernel_size=1)

        block_cls = FFCResBlock3D if use_ffc else ResBlock3D
        if use_ffc:
            self.block = block_cls(
                int(out_channels) + int(skip_channels),
                int(out_channels),
                time_dim=time_dim,
                num_groups=num_groups,
                dropout=dropout,
                global_ratio=global_ratio,
            )
        else:
            self.block = block_cls(
                int(out_channels) + int(skip_channels),
                int(out_channels),
                time_dim=time_dim,
                num_groups=num_groups,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        x = self.pre(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_emb)
        return x


class SelfAttention3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_groups: int = 16,
    ) -> None:
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


class ResidualBridgeFFCResUNet3D(nn.Module):
    """
    Residual Bridge 使用的 3D FFC-ResUNet。

    forward 接口和 ResidualBridgeResUNet3D 保持一致:
        forward(x_t, condition, t) -> residual_pred
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
        use_attention: bool = False,
        attention_level: int = 3,
        num_heads: int = 4,
        ffc_levels: Sequence[int] = (2, 3),
        global_ratio: float = 0.25,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(v) for v in channel_mults)
        self.time_dim = int(time_dim)
        self.use_attention = bool(use_attention)
        self.attention_level = int(attention_level)
        self.ffc_levels = {int(v) for v in ffc_levels}
        self.global_ratio = float(global_ratio)

        channels = [self.base_channels * m for m in self.channel_mults]
        if len(channels) < 2:
            raise ValueError("channel_mults 至少需要两个层级，例如 [1, 2, 4, 8]")

        self.time_mlp = TimeMLP(self.time_dim)
        self.input_conv = nn.Conv3d(self.in_channels, channels[0], kernel_size=3, padding=1)

        self.enc0 = self._make_block(
            level=0,
            in_channels=channels[0],
            out_channels=channels[0],
            num_groups=num_groups,
            dropout=dropout,
        )

        self.down_blocks = nn.ModuleList()
        for level in range(1, len(channels)):
            self.down_blocks.append(
                DownBlock3D(
                    in_channels=channels[level - 1],
                    out_channels=channels[level],
                    time_dim=self.time_dim,
                    num_groups=num_groups,
                    dropout=dropout,
                    use_ffc=level in self.ffc_levels,
                    global_ratio=self.global_ratio,
                )
            )

        bottleneck_channels = channels[-1]

        self.mid1 = self._make_block(
            level=len(channels) - 1,
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
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

        self.mid2 = self._make_block(
            level=len(channels) - 1,
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(channels) - 1)):
            self.up_blocks.append(
                UpBlock3D(
                    in_channels=channels[level + 1],
                    skip_channels=channels[level],
                    out_channels=channels[level],
                    time_dim=self.time_dim,
                    num_groups=num_groups,
                    dropout=dropout,
                    use_ffc=level in self.ffc_levels,
                    global_ratio=self.global_ratio,
                )
            )

        self.out_norm = SafeGroupNorm(num_groups, channels[0])
        self.out_act = Swish()
        self.out_conv = nn.Conv3d(channels[0], self.out_channels, kernel_size=3, padding=1)

    def _make_block(
        self,
        level: int,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        dropout: float,
    ) -> nn.Module:
        if int(level) in self.ffc_levels:
            return FFCResBlock3D(
                in_channels=in_channels,
                out_channels=out_channels,
                time_dim=self.time_dim,
                num_groups=num_groups,
                dropout=dropout,
                global_ratio=self.global_ratio,
            )

        return ResBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            time_dim=self.time_dim,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x_t, condition], dim=1)
        t_emb = self.time_mlp(t)

        x = self.input_conv(x)

        skips: list[torch.Tensor] = []

        x = self.enc0(x, t_emb)
        skips.append(x)

        for i, down in enumerate(self.down_blocks):
            x = down(x, t_emb)

            # 最深层作为 bottleneck，不进入 skip
            if i < len(self.down_blocks) - 1:
                skips.append(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip, t_emb)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x