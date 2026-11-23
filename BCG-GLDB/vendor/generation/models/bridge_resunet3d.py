from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from models.resunet3d import ResUNet3D


class BridgeResUNet3D(nn.Module):
    """Exp002 ResUNet3D backbone wrapped for conditional bridge modeling.

    The core spatial network stays identical to Exp002. We only augment the
    input with the current bridge state `x_t`, the MRI condition, and a
    broadcast time map channel.
    """

    def __init__(
        self,
        state_channels: int = 1,
        condition_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        channel_mults: Iterable[int] = (1, 2, 4),
        dropout: float = 0.1,
        num_groups: int = 16,
        use_attention: bool = True,
        attention_level: int = 2,
        time_max: int = 200,
    ) -> None:
        super().__init__()
        self.state_channels = int(state_channels)
        self.condition_channels = int(condition_channels)
        self.time_max = max(int(time_max), 2)
        self.backbone = ResUNet3D(
            in_channels=self.state_channels + self.condition_channels + 1,
            out_channels=int(out_channels),
            base_channels=int(base_channels),
            channel_mults=tuple(channel_mults),
            dropout=float(dropout),
            num_groups=int(num_groups),
            use_attention=bool(use_attention),
            attention_level=int(attention_level),
        )

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.shape[0] != condition.shape[0]:
            raise ValueError("x_t and condition must have the same batch size.")
        t_norm = t.float() / float(self.time_max - 1)
        t_map = t_norm[:, None, None, None, None].expand(-1, 1, *x_t.shape[-3:])
        return self.backbone(torch.cat([x_t, condition, t_map], dim=1))
