from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from models.region_local_unet_denoiser import GatedCrossAttention3D, TableTokenizer
from models.residual_bridge_resunet3d import (
    Downsample3D,
    ResidualTimeBlock3D,
    SinusoidalTimeEmbedding,
    Upsample3D,
)
from models.resunet3d import SafeGroupNorm, SelfAttention3D, Swish


class BSFETableGCALatentBridgeDenoiser3D(nn.Module):
    """Latent bridge denoiser conditioned by BSFE coarse latent plus tabular GCA."""

    def __init__(
        self,
        latent_channels: int = 1,
        cond_channels: int = 1,
        table_num_features: int = 0,
        base_channels: int = 32,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        time_dim: int = 256,
        dropout: float = 0.1,
        num_groups: int = 16,
        use_attention: bool = True,
        attention_level: int = 2,
        num_heads: int = 4,
        table_token_dim: int = 128,
        table_cross_attn_heads: int = 4,
        use_table_gca: bool = True,
    ) -> None:
        super().__init__()
        mults = [int(m) for m in channel_mults]
        channels = [int(base_channels) * m for m in mults]
        if len(channels) < 2:
            raise ValueError("channel_mults must have at least two stages.")

        self.latent_channels = int(latent_channels)
        self.cond_channels = int(cond_channels)
        self.table_num_features = int(table_num_features)
        self.use_table_gca = bool(use_table_gca) and self.table_num_features > 0
        self.table_token_dim = int(table_token_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Swish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.table_tokenizer = (
            TableTokenizer(self.table_num_features, self.table_token_dim)
            if self.use_table_gca
            else None
        )
        self.input_conv = nn.Conv3d(self.latent_channels + self.cond_channels, channels[0], kernel_size=3, padding=1)
        self.input_table_gca = (
            GatedCrossAttention3D(
                in_channel=channels[0],
                table_token_dim=self.table_token_dim,
                n_head=int(table_cross_attn_heads),
                norm_groups=int(num_groups),
            )
            if self.use_table_gca
            else None
        )

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
        self.mid_table_gca = (
            GatedCrossAttention3D(
                in_channel=current_ch,
                table_token_dim=self.table_token_dim,
                n_head=int(table_cross_attn_heads),
                norm_groups=int(num_groups),
            )
            if self.use_table_gca
            else None
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
        self.out_conv = nn.Conv3d(channels[0], self.latent_channels, kernel_size=3, padding=1)

    def _build_table_context(
        self,
        x: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_table_gca or self.table_tokenizer is None or table is None:
            return None, None, None
        miss = table_mask if table_mask is not None else torch.zeros_like(table)
        table_tokens, table_kpm, table_global = self.table_tokenizer(table.float(), miss.float())
        if table_tokens is not None:
            table_tokens = table_tokens.to(device=x.device, dtype=x.dtype)
        if table_global is not None:
            table_global = table_global.to(device=x.device, dtype=x.dtype)
        return table_tokens, table_kpm, table_global

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        table: Optional[torch.Tensor] = None,
        table_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.cat([x_t, condition], dim=1)
        t_emb = self.time_mlp(t)
        table_tokens, table_kpm, table_global = self._build_table_context(x, table, table_mask)
        x = self.input_conv(x)
        if self.input_table_gca is not None:
            x = self.input_table_gca(x, table_tokens, table_kpm, table_global)

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
        if self.mid_table_gca is not None:
            x = self.mid_table_gca(x, table_tokens, table_kpm, table_global)

        skips = list(reversed(skips[:-1]))
        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, skips):
            x = upsample(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                if isinstance(block, ResidualTimeBlock3D):
                    x = block(x, t_emb)
                else:
                    x = block(x)

        return self.out_conv(self.out_act(self.out_norm(x)))
