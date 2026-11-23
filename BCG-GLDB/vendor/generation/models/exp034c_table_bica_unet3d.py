"""Exp034C: basic 3D U-Net with full-stage bidirectional table interaction."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.exp026.models.table_interaction import MissingAwareTableTokenizer
from models.exp034a_basic_unet3d import TimeEmbedding, TimeResBlock3D
from models.global_local_mamba_patch_denoiser import AxialBiMambaBlock


class GlobalMambaContext3D(nn.Module):
    """Global bottleneck context with a zero-initialized residual projection."""
    def __init__(self, channels: int, time_dim: int, token_dim: int = 128) -> None:
        super().__init__()
        self.to_tokens = nn.Conv3d(channels, token_dim, 1)
        self.block = AxialBiMambaBlock(
            dim=token_dim, condition_dim=time_dim, d_state=8, d_conv=3,
            expand=1, bidirectional=True, active_axes=('x', 'y', 'z'),
            line_chunk_size=256, gradient_checkpointing=True,
        )
        self.to_context = nn.Conv3d(token_dim, channels, 1)
        nn.init.zeros_(self.to_context.weight)
        nn.init.zeros_(self.to_context.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        tokens = self.to_tokens(x)
        empty_table = torch.zeros_like(time_emb[:, :1])
        empty_condition = torch.zeros_like(time_emb)
        context = self.block(tokens, base_condition=time_emb,
                             table_condition=empty_condition,
                             table_reliability=empty_table)
        return self.to_context(context)


class CrossAttentionDelta(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int) -> None:
        super().__init__()
        if query_dim % heads or context_dim % heads:
            raise ValueError('token dimensions must divide attention heads')
        self.q_norm = nn.LayerNorm(query_dim)
        self.ctx_norm = nn.LayerNorm(context_dim)
        self.q_proj = nn.Linear(query_dim, context_dim) if query_dim != context_dim else nn.Identity()
        self.out_proj = nn.Linear(context_dim, query_dim)
        self.attn = nn.MultiheadAttention(context_dim, heads, batch_first=True)

    def forward(self, query: torch.Tensor, context: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        q = self.q_proj(self.q_norm(query))
        ctx = self.ctx_norm(context)
        out = self.attn(q, ctx, ctx, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.out_proj(out)


class BidirectionalTableStage(nn.Module):
    """Image <-> table cross-attention plus gated residual feature injection."""

    def __init__(self, channels: int, token_dim: int, time_dim: int, token_grid: Sequence[int], heads: int, lambda_init: float) -> None:
        super().__init__()
        self.grid = tuple(int(v) for v in token_grid)
        self.image_to_tokens = nn.Sequential(nn.Conv3d(channels, token_dim, 1), nn.GroupNorm(min(8, token_dim), token_dim), nn.SiLU())
        self.tokens_to_image = nn.Conv3d(token_dim, channels, 1)
        nn.init.zeros_(self.tokens_to_image.weight)
        nn.init.zeros_(self.tokens_to_image.bias)
        self.image_reads_table = CrossAttentionDelta(token_dim, token_dim, heads)
        self.table_reads_image = CrossAttentionDelta(token_dim, token_dim, heads)
        self.table_time = nn.Linear(time_dim, token_dim)
        self.gate_spatial = nn.Conv3d(channels, channels, 1)
        self.gate_time = nn.Linear(time_dim, channels)
        nn.init.zeros_(self.gate_spatial.weight)
        nn.init.zeros_(self.gate_spatial.bias)
        nn.init.zeros_(self.gate_time.weight)
        nn.init.zeros_(self.gate_time.bias)
        self.lambda_scale = nn.Parameter(torch.tensor(float(lambda_init)))

    def forward(self, image: torch.Tensor, table_tokens: torch.Tensor, table_kpm: torch.Tensor, time_emb: torch.Tensor):
        pooled = F.adaptive_avg_pool3d(self.image_to_tokens(image), self.grid)
        image_tokens = pooled.flatten(2).transpose(1, 2)
        table_tokens = table_tokens + self.table_time(time_emb)[:, None, :]
        image_delta = self.image_reads_table(image_tokens, table_tokens, table_kpm)
        image_updated = image_tokens + image_delta
        table_delta = self.table_reads_image(table_tokens, image_updated)
        table_tokens = table_tokens + table_delta
        # Read the updated table tokens back into the image path. This keeps
        # the interaction genuinely bidirectional and gives table_reads_image
        # a gradient through the generated image residual.
        image_delta = image_delta + self.image_reads_table(image_tokens, table_tokens, table_kpm)
        delta_grid = image_delta.transpose(1, 2).reshape(image.shape[0], -1, *self.grid)
        delta_feature = self.tokens_to_image(F.interpolate(delta_grid, size=image.shape[-3:], mode='trilinear', align_corners=False))
        gate = torch.sigmoid(self.gate_spatial(image) + self.gate_time(time_emb)[:, :, None, None, None])
        return image + self.lambda_scale * gate * delta_feature, table_tokens


class TableBiCAUNet3D(nn.Module):
    """Exp034A backbone with progressive table interaction at all seven stages."""

    def __init__(self, table_num_features: int, feature_type_ids: Sequence[int], in_channels: int = 4, out_channels: int = 4,
                 channels: Sequence[int] = (96, 192, 384, 768), time_dim: int = 384, token_dim: int = 128,
                 token_grids: Sequence[Sequence[int]] = ((10, 12, 10), (10, 12, 10), (5, 6, 5), (5, 6, 5), (5, 6, 5), (10, 12, 10), (10, 12, 10)),
                 heads: int = 4, norm_groups: int = 8, dropout: float = 0.0, lambda_init: float = 0.05,
                 table_stage_indices: Sequence[int] = (0, 1, 2, 3, 4, 5, 6)) -> None:
        super().__init__()
        c0, c1, c2, c3 = [int(v) for v in channels]
        if len(token_grids) != 7:
            raise ValueError('Exp034C requires one token grid for every UNet stage')
        self.table_stage_indices = frozenset(int(index) for index in table_stage_indices)
        if not self.table_stage_indices.issubset(set(range(7))):
            raise ValueError('table_stage_indices must be chosen from 0..6')
        self.time_embedding = TimeEmbedding(time_dim)
        self.table_tokenizer = MissingAwareTableTokenizer(table_num_features, token_dim, feature_type_ids, num_feature_types=5)
        self.input = nn.Conv3d(in_channels, c0, 3, padding=1)
        self.enc0a = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout); self.enc0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.down1 = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.enc1a = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout); self.enc1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.down2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.enc2a = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout); self.enc2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.down3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.mid1 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout); self.mid2 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout)
        self.global_mamba = GlobalMambaContext3D(c3, time_dim, token_dim=128)
        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1)
        self.dec2a = TimeResBlock3D(c2 * 2, c2, time_dim, norm_groups, dropout); self.dec2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1)
        self.dec1a = TimeResBlock3D(c1 * 2, c1, time_dim, norm_groups, dropout); self.dec1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.up0 = nn.ConvTranspose3d(c1, c0, 4, stride=2, padding=1)
        self.dec0a = TimeResBlock3D(c0 * 2, c0, time_dim, norm_groups, dropout); self.dec0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        stage_channels = (c0, c1, c2, c3, c2, c1, c0)
        self.table_stages = nn.ModuleList([BidirectionalTableStage(ch, token_dim, time_dim, grid, heads, lambda_init) for ch, grid in zip(stage_channels, token_grids)])
        self.out = nn.Sequential(nn.GroupNorm(min(norm_groups, c0), c0), nn.SiLU(), nn.Conv3d(c0, out_channels, 3, padding=1))

    def _fuse_table(self, index: int, image: torch.Tensor, table_tokens: torch.Tensor, table_kpm: torch.Tensor, time_emb: torch.Tensor):
        if index not in self.table_stage_indices:
            return image, table_tokens
        return self.table_stages[index](image, table_tokens, table_kpm, time_emb)

    def forward(self, x1: torch.Tensor, t: torch.Tensor, table: torch.Tensor, table_mask: torch.Tensor) -> torch.Tensor:
        te = self.time_embedding(t)
        table_tokens, table_kpm, _ = self.table_tokenizer(table, table_mask)
        s0 = self.enc0b(self.enc0a(self.input(x1), te), te); s0, table_tokens = self._fuse_table(0, s0, table_tokens, table_kpm, te)
        s1 = self.enc1b(self.enc1a(self.down1(s0), te), te); s1, table_tokens = self._fuse_table(1, s1, table_tokens, table_kpm, te)
        s2 = self.enc2b(self.enc2a(self.down2(s1), te), te); s2, table_tokens = self._fuse_table(2, s2, table_tokens, table_kpm, te)
        h = self.mid2(self.mid1(self.down3(s2), te), te)
        h = h + self.global_mamba(h, te)
        h, table_tokens = self._fuse_table(3, h, table_tokens, table_kpm, te)
        h = self.dec2b(self.dec2a(torch.cat([self.up2(h), s2], dim=1), te), te); h, table_tokens = self._fuse_table(4, h, table_tokens, table_kpm, te)
        h = self.dec1b(self.dec1a(torch.cat([self.up1(h), s1], dim=1), te), te); h, table_tokens = self._fuse_table(5, h, table_tokens, table_kpm, te)
        h = self.dec0b(self.dec0a(torch.cat([self.up0(h), s0], dim=1), te), te); h, _ = self._fuse_table(6, h, table_tokens, table_kpm, te)
        return self.out(h)
