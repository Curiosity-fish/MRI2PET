from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from models.residual_bridge_resunet3d import (
    ResidualTimeBlock3D,
    Downsample3D,
    Upsample3D,
    SinusoidalTimeEmbedding,
)
from models.resunet3d import SafeGroupNorm, SelfAttention3D, Swish


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class MRIPatchControlEncoder3D(nn.Module):
    """Patch MRI encoder aligned to the PET latent patch UNet scales."""

    def __init__(
        self,
        base_channels: int = 8,
        out_channels: Iterable[int] = (16, 32, 64),
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        h, m, l = [int(v) for v in out_channels]
        c0 = int(base_channels)
        c1 = c0 * 2
        self.stem = nn.Sequential(
            nn.Conv3d(1, c0, kernel_size=3, stride=2, padding=1),
            SafeGroupNorm(c0, num_groups=norm_groups),
            Swish(),
            nn.Conv3d(c0, c1, kernel_size=3, stride=2, padding=1),
            SafeGroupNorm(c1, num_groups=norm_groups),
            Swish(),
        )
        self.h_block = nn.Sequential(
            nn.Conv3d(c1, h, kernel_size=3, padding=1),
            SafeGroupNorm(h, num_groups=norm_groups),
            Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.m_block = nn.Sequential(
            nn.Conv3d(h, m, kernel_size=4, stride=2, padding=1),
            SafeGroupNorm(m, num_groups=norm_groups),
            Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.l_block = nn.Sequential(
            nn.Conv3d(m, l, kernel_size=4, stride=2, padding=1),
            SafeGroupNorm(l, num_groups=norm_groups),
            Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, mri_patch: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.h_block(self.stem(mri_patch))
        m = self.m_block(h)
        l = self.l_block(m)
        return {"h": h, "m": m, "l": l}


class ZeroConvControl3D(nn.Module):
    def __init__(self, channels: Iterable[int]) -> None:
        super().__init__()
        h, m, l = [int(v) for v in channels]
        self.h = zero_module(nn.Conv3d(h, h, kernel_size=1))
        self.m = zero_module(nn.Conv3d(m, m, kernel_size=1))
        self.l = zero_module(nn.Conv3d(l, l, kernel_size=1))
        self.last_stats: dict[str, float] = {}

    def forward(self, x: torch.Tensor, mri: torch.Tensor, level: str, enabled: bool = True) -> torch.Tensor:
        layer = getattr(self, level)
        out = layer(mri)
        self.last_stats[f"{level}_abs_mean"] = float(out.detach().abs().mean().item())
        if not enabled:
            return x
        return x + out.to(dtype=x.dtype)

    def control_weight_norm(self) -> float:
        total = 0.0
        for p in self.parameters():
            total += float(p.detach().square().sum().item())
        return total ** 0.5


class AdaptiveControl3D(nn.Module):
    def __init__(self, channels: Iterable[int], gamma_scale: float = 0.5, beta_scale: float = 0.5) -> None:
        super().__init__()
        h, m, l = [int(v) for v in channels]
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.h = zero_module(nn.Conv3d(h, h * 2, kernel_size=1))
        self.m = zero_module(nn.Conv3d(m, m * 2, kernel_size=1))
        self.l = zero_module(nn.Conv3d(l, l * 2, kernel_size=1))
        self.last_stats: dict[str, float] = {}

    def forward(self, x: torch.Tensor, mri: torch.Tensor, level: str, enabled: bool = True) -> torch.Tensor:
        raw = getattr(self, level)(mri)
        gamma_raw, beta_raw = torch.chunk(raw, 2, dim=1)
        gamma = self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)
        self.last_stats[f"{level}_gamma_mean"] = float(gamma.detach().mean().item())
        self.last_stats[f"{level}_gamma_abs_mean"] = float(gamma.detach().abs().mean().item())
        self.last_stats[f"{level}_beta_mean"] = float(beta.detach().mean().item())
        self.last_stats[f"{level}_beta_abs_mean"] = float(beta.detach().abs().mean().item())
        if not enabled:
            return x
        return (1.0 + gamma.to(dtype=x.dtype)) * x + beta.to(dtype=x.dtype)


class Exp032GMRIControlUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        base_channels: int = 16,
        channel_mults: Iterable[int] = (1, 2, 4),
        time_dim: int = 256,
        dropout: float = 0.0,
        num_groups: int = 8,
        use_attention: bool = False,
        attention_level: int = 1,
        num_heads: int = 4,
        mri_base_channels: int = 8,
        control_type: str = "zero_conv",
        gamma_scale: float = 0.5,
        beta_scale: float = 0.5,
    ) -> None:
        super().__init__()
        mults = [int(m) for m in channel_mults]
        channels = [int(base_channels) * m for m in mults]
        if len(channels) != 3:
            raise ValueError("Exp032g uses exactly three Local UNet scales: high/mid/low.")

        self.control_type = str(control_type)
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

        self.mri_encoder = MRIPatchControlEncoder3D(
            base_channels=mri_base_channels,
            out_channels=channels,
            norm_groups=num_groups,
            dropout=dropout,
        )
        if self.control_type == "adaptive":
            self.control = AdaptiveControl3D(channels, gamma_scale=gamma_scale, beta_scale=beta_scale)
        elif self.control_type == "zero_conv":
            self.control = ZeroConvControl3D(channels)
        else:
            raise ValueError(f"Unsupported control_type={control_type}")

    def local_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("mri_encoder.") and not k.startswith("control.")
        }

    def load_local_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        own.update({k: v for k, v in state.items() if k in own})
        self.load_state_dict(own, strict=True)

    def mri_encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return self.mri_encoder.state_dict()

    def load_mri_encoder_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mri_encoder.load_state_dict(state, strict=True)

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        mri_patch: torch.Tensor,
        control_enabled: bool = True,
    ) -> torch.Tensor:
        x = torch.cat([x_t, condition], dim=1)
        t_emb = self.time_mlp(t)
        mri = self.mri_encoder(mri_patch)
        x = self.input_conv(x)

        skips = []
        levels = ["h", "m", "l"]
        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResidualTimeBlock3D):
                    x = block(x, t_emb)
                else:
                    x = block(x)
            x = self.control(x, mri[levels[idx]], levels[idx], enabled=control_enabled)
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


class MRILatentProjectionHead3D(nn.Module):
    def __init__(self, mri_encoder: MRIPatchControlEncoder3D, latent_channels: int = 4) -> None:
        super().__init__()
        self.mri_encoder = mri_encoder
        high_channels = mri_encoder.h_block[0].out_channels
        self.head = nn.Sequential(
            nn.Conv3d(high_channels, high_channels, kernel_size=3, padding=1),
            SafeGroupNorm(high_channels, num_groups=8),
            Swish(),
            nn.Conv3d(high_channels, int(latent_channels), kernel_size=1),
        )

    def forward(self, mri_patch: torch.Tensor) -> torch.Tensor:
        features = self.mri_encoder(mri_patch)
        return self.head(features["h"])
