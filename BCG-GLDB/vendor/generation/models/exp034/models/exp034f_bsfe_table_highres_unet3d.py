"""Exp034F: Exp034A with all-stage BSFE controls and high-resolution table BiCA."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from exp026.models.table_interaction import MissingAwareTableTokenizer
from models.exp034a_basic_unet3d import TimeEmbedding, TimeResBlock3D
from models.exp034c_table_bica_unet3d import BidirectionalTableStage


class BSFETableHighResUNet3D(nn.Module):
    """All seven BSFE injections; image/table interaction only at stages 0, 1, 5, 6."""

    def __init__(self, table_num_features: int, feature_type_ids: Sequence[int], in_channels: int = 4,
                 out_channels: int = 4, channels: Sequence[int] = (96, 192, 384, 768),
                 bsfe_channels: Sequence[int] = (48, 96, 192, 384, 192, 96, 48),
                 time_dim: int = 384, table_token_dim: int = 128, table_heads: int = 4,
                 table_token_grids: Sequence[Sequence[int]] = ((10, 12, 10), (10, 12, 10), (10, 12, 10), (10, 12, 10)),
                 norm_groups: int = 8, dropout: float = 0.0, table_lambda_init: float = 0.05) -> None:
        super().__init__()
        c0, c1, c2, c3 = (int(v) for v in channels)
        if len(bsfe_channels) != 7 or len(table_token_grids) != 4:
            raise ValueError('Expected 7 BSFE channel entries and 4 table token grids.')
        self.time_embedding = TimeEmbedding(time_dim)
        self.table_tokenizer = MissingAwareTableTokenizer(table_num_features, table_token_dim, feature_type_ids, num_feature_types=5)
        self.input = nn.Conv3d(in_channels, c0, 3, padding=1)
        self.enc0a = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout); self.enc0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.down1 = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.enc1a = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout); self.enc1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.down2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.enc2a = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout); self.enc2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.down3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.mid1 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout); self.mid2 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout)
        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1); self.dec2a = TimeResBlock3D(c2 * 2, c2, time_dim, norm_groups, dropout); self.dec2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1); self.dec1a = TimeResBlock3D(c1 * 2, c1, time_dim, norm_groups, dropout); self.dec1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.up0 = nn.ConvTranspose3d(c1, c0, 4, stride=2, padding=1); self.dec0a = TimeResBlock3D(c0 * 2, c0, time_dim, norm_groups, dropout); self.dec0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        stage_channels = (c0, c1, c2, c3, c2, c1, c0)
        self.bsfe_zero = nn.ModuleList([nn.Conv3d(int(i), int(o), 1) for i, o in zip(bsfe_channels, stage_channels)])
        for layer in self.bsfe_zero:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.table_stages = nn.ModuleList([
            BidirectionalTableStage(ch, table_token_dim, time_dim, grid, table_heads, table_lambda_init)
            for ch, grid in zip((c0, c1, c1, c0), table_token_grids)
        ])
        self.out = nn.Sequential(nn.GroupNorm(min(norm_groups, c0), c0), nn.SiLU(), nn.Conv3d(c0, out_channels, 3, padding=1))

    def _bsfe(self, x: torch.Tensor, pyramid: Sequence[torch.Tensor], index: int) -> torch.Tensor:
        return x + self.bsfe_zero[index](pyramid[index])

    def forward(self, x1: torch.Tensor, t: torch.Tensor, pyramid: Sequence[torch.Tensor], table: torch.Tensor, table_mask: torch.Tensor) -> torch.Tensor:
        if len(pyramid) != 4:
            raise ValueError(f'Expected four BSFE pyramid levels, got {len(pyramid)}')
        p = (pyramid[0], pyramid[1], pyramid[2], pyramid[3], pyramid[2], pyramid[1], pyramid[0])
        te = self.time_embedding(t)
        table_tokens, table_kpm, _ = self.table_tokenizer(table, table_mask)
        s0 = self._bsfe(self.enc0b(self.enc0a(self.input(x1), te), te), p, 0)
        s0, table_tokens = self.table_stages[0](s0, table_tokens, table_kpm, te)
        s1 = self._bsfe(self.enc1b(self.enc1a(self.down1(s0), te), te), p, 1)
        s1, table_tokens = self.table_stages[1](s1, table_tokens, table_kpm, te)
        s2 = self._bsfe(self.enc2b(self.enc2a(self.down2(s1), te), te), p, 2)
        h = self._bsfe(self.mid2(self.mid1(self.down3(s2), te), te), p, 3)
        h = self._bsfe(self.dec2b(self.dec2a(torch.cat([self.up2(h), s2], 1), te), te), p, 4)
        h = self._bsfe(self.dec1b(self.dec1a(torch.cat([self.up1(h), s1], 1), te), te), p, 5)
        h, table_tokens = self.table_stages[2](h, table_tokens, table_kpm, te)
        h = self._bsfe(self.dec0b(self.dec0a(torch.cat([self.up0(h), s0], 1), te), te), p, 6)
        h, _ = self.table_stages[3](h, table_tokens, table_kpm, te)
        return self.out(h)
