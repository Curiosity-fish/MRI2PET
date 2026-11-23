from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.residual_bridge_resunet3d import (
    Downsample3D,
    ResidualTimeBlock3D,
    SinusoidalTimeEmbedding,
    Upsample3D,
)
from models.resunet3d import SafeGroupNorm, SelfAttention3D, Swish


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


class OptionalTableFiLM(nn.Module):
    def __init__(self, num_features: int, out_dim: int) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.out_dim = int(out_dim)
        hidden = max(self.num_features * 2, 32)
        self.net = nn.Sequential(
            nn.Linear(self.num_features * 2 + 1, hidden),
            Swish(),
            nn.Linear(hidden, self.out_dim),
        )
        _zero_module(self.net[-1])

    def forward(
        self,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if table is None or self.num_features <= 0:
            raise ValueError("table input is required when table_num_features > 0")
        if table_mask is None:
            table_mask = torch.zeros_like(table)
        valid = (table_mask <= 0.5).float()
        safe_table = table.float() * valid
        quality = valid.mean(dim=1, keepdim=True)
        x = torch.cat([safe_table, valid, quality], dim=1)
        return self.net(x.to(device=device, dtype=dtype))


class ControlEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        time_dim: int,
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv3d(int(in_channels), channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_ch = channels[0]
        for idx, ch in enumerate(channels):
            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualTimeBlock3D(current_ch, ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                        ResidualTimeBlock3D(ch, ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                    ]
                )
            )
            current_ch = ch
            if idx < len(channels) - 1:
                self.downsamples.append(Downsample3D(current_ch, channels[idx + 1]))
                current_ch = channels[idx + 1]

        self.mid = nn.ModuleList(
            [
                ResidualTimeBlock3D(current_ch, current_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                ResidualTimeBlock3D(current_ch, current_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
            ]
        )

    def forward(self, control: torch.Tensor, t_emb: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        x = self.input_conv(control)
        feats: list[torch.Tensor] = []
        for idx, blocks in enumerate(self.blocks):
            for block in blocks:
                x = block(x, t_emb)
            feats.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        for block in self.mid:
            x = block(x, t_emb)
        return feats, x


class BSFEControlNetLatentBridgeDenoiser3D(nn.Module):
    """PET latent denoiser with a separate BSFE/MRI control branch.

    The PET trunk only sees x_t. The frozen BSFE latent condition is routed
    through a dedicated control encoder and injected as zero-conv residuals.
    """

    def __init__(
        self,
        latent_channels: int = 1,
        control_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        time_dim: int = 256,
        dropout: float = 0.1,
        num_groups: int = 16,
        use_attention: bool = True,
        attention_level: int = 2,
        num_heads: int = 4,
        table_num_features: int = 0,
    ) -> None:
        super().__init__()
        mults = [int(m) for m in channel_mults]
        channels = [int(base_channels) * m for m in mults]
        if len(channels) < 2:
            raise ValueError("channel_mults must have at least two stages.")

        self.table_num_features = int(table_num_features)
        cond_dim = int(time_dim)
        if self.table_num_features > 0:
            self.table_film = OptionalTableFiLM(self.table_num_features, cond_dim)
        else:
            self.table_film = None

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Swish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.input_conv = nn.Conv3d(int(latent_channels), channels[0], kernel_size=3, padding=1)
        self.control_encoder = ControlEncoder3D(
            in_channels=int(control_channels),
            channels=channels,
            time_dim=time_dim,
            dropout=dropout,
            num_groups=num_groups,
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.control_inject_down = nn.ModuleList()
        self.control_gate_down = nn.ModuleList()
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
            self.control_inject_down.append(_zero_module(nn.Conv3d(ch, ch, kernel_size=1)))
            self.control_gate_down.append(ControlGate(cond_dim, ch))
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
        self.control_inject_mid = _zero_module(nn.Conv3d(current_ch, current_ch, kernel_size=1))
        self.control_gate_mid = ControlGate(cond_dim, current_ch)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.control_inject_up = nn.ModuleList()
        self.control_gate_up = nn.ModuleList()
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
            self.control_inject_up.append(_zero_module(nn.Conv3d(skip_ch, skip_ch, kernel_size=1)))
            self.control_gate_up.append(ControlGate(cond_dim, skip_ch))
            current_ch = skip_ch

        self.out_norm = SafeGroupNorm(channels[0], num_groups=num_groups)
        self.out_act = Swish()
        self.out_conv = nn.Conv3d(channels[0], int(latent_channels), kernel_size=3, padding=1)

    def _make_cond_vector(
        self,
        t: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cond = self.time_mlp(t)
        if self.table_film is not None and table is not None:
            cond = cond + self.table_film(table, table_mask, device=device, dtype=dtype)
        return cond

    def _inject(self, x: torch.Tensor, control: torch.Tensor, gate_module: nn.Module, inject_module: nn.Module, cond: torch.Tensor) -> torch.Tensor:
        gate = gate_module(cond)[:, :, None, None, None].to(dtype=x.dtype)
        return x + gate * inject_module(control).to(dtype=x.dtype)

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        table: Optional[torch.Tensor] = None,
        table_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tuple(condition.shape[-3:]) != tuple(x_t.shape[-3:]):
            condition = F.interpolate(condition, size=x_t.shape[-3:], mode="trilinear", align_corners=False)
        cond = self._make_cond_vector(t, table, table_mask, x_t.device, x_t.dtype)
        control_skips, control_mid = self.control_encoder(condition, cond)

        x = self.input_conv(x_t)
        skips = []
        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResidualTimeBlock3D):
                    x = block(x, cond)
                else:
                    x = block(x)
            x = self._inject(x, control_skips[idx], self.control_gate_down[idx], self.control_inject_down[idx], cond)
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        for block in self.mid_blocks:
            if isinstance(block, ResidualTimeBlock3D):
                x = block(x, cond)
            else:
                x = block(x)
        x = self._inject(x, control_mid, self.control_gate_mid, self.control_inject_mid, cond)

        skips = list(reversed(skips[:-1]))
        control_up = list(reversed(control_skips[:-1]))
        for upsample, blocks, skip, control_skip, gate, inject in zip(
            self.upsamples,
            self.up_blocks,
            skips,
            control_up,
            self.control_gate_up,
            self.control_inject_up,
        ):
            x = upsample(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                if isinstance(block, ResidualTimeBlock3D):
                    x = block(x, cond)
                else:
                    x = block(x)
            x = self._inject(x, control_skip, gate, inject, cond)

        return self.out_conv(self.out_act(self.out_norm(x)))
