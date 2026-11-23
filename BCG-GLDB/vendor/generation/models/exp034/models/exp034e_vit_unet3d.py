"""Exp034E: Exp034A U-Net with high-resolution 3D ViT residual adapters."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.exp034a_basic_unet3d import TimeEmbedding, TimeResBlock3D


class ViT3DResidualAdapter(nn.Module):
    """Patch-token ViT that returns a gated residual at the input resolution."""

    def __init__(
        self,
        channels: int,
        spatial_shape: Sequence[int],
        patch_size: Sequence[int],
        time_dim: int,
        token_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.patch_size = tuple(int(v) for v in patch_size)
        shape = tuple(int(v) for v in spatial_shape)
        if len(shape) != 3 or any(s % p for s, p in zip(shape, self.patch_size)):
            raise ValueError(f"spatial_shape={shape} must be divisible by patch_size={self.patch_size}")
        self.grid = tuple(s // p for s, p in zip(shape, self.patch_size))
        n_tokens = self.grid[0] * self.grid[1] * self.grid[2]
        if token_dim % heads:
            raise ValueError(f"token_dim={token_dim} must be divisible by heads={heads}")

        self.to_tokens = nn.Conv3d(channels, token_dim, self.patch_size, stride=self.patch_size)
        self.position = nn.Parameter(torch.zeros(1, n_tokens, token_dim))
        self.time = nn.Linear(time_dim, token_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=heads,
                dim_feedforward=token_dim * int(mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ) for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(token_dim)
        self.from_tokens = nn.ConvTranspose3d(token_dim, channels, self.patch_size, stride=self.patch_size)
        self.gate = nn.Parameter(torch.full((1, channels, 1, 1, 1), 1e-3))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        tokens = self.to_tokens(x).flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.position.shape[1]:
            raise RuntimeError(
                f"ViT token grid mismatch: got {tokens.shape[1]} tokens, "
                f"expected {self.position.shape[1]}"
            )
        tokens = tokens + self.position + self.time(time_emb).unsqueeze(1)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        update = self.from_tokens(tokens.transpose(1, 2).reshape(x.shape[0], -1, *self.grid))
        return x + self.gate * update


class ViTHighResUNet3D(nn.Module):
    """Exp034A backbone with ViT adapters at encoder and decoder high resolutions."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: Sequence[int] = (96, 192, 384, 768),
        time_dim: int = 384,
        norm_groups: int = 8,
        dropout: float = 0.0,
        latent_shape: Sequence[int] = (40, 48, 40),
        vit_token_dim: int = 128,
        vit_depth: int = 2,
        vit_heads: int = 4,
        vit_mlp_ratio: int = 4,
        vit_dropout: float = 0.0,
        vit_use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = (int(v) for v in channels)
        latent_shape = tuple(int(v) for v in latent_shape)
        half_shape = tuple(v // 2 for v in latent_shape)
        self.time_embedding = TimeEmbedding(time_dim)
        self.input = nn.Conv3d(in_channels, c0, 3, padding=1)
        self.enc0a = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.enc0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.down1 = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.enc1a = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.enc1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.down2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.enc2a = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.enc2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.down3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.mid1 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout)
        self.mid2 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout)
        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1)
        self.dec2a = TimeResBlock3D(c2 * 2, c2, time_dim, norm_groups, dropout)
        self.dec2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1)
        self.dec1a = TimeResBlock3D(c1 * 2, c1, time_dim, norm_groups, dropout)
        self.dec1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.up0 = nn.ConvTranspose3d(c1, c0, 4, stride=2, padding=1)
        self.dec0a = TimeResBlock3D(c0 * 2, c0, time_dim, norm_groups, dropout)
        self.dec0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)

        vit_args = dict(
            time_dim=time_dim,
            token_dim=vit_token_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
            use_checkpoint=vit_use_checkpoint,
        )
        self.vit_enc0 = ViT3DResidualAdapter(c0, latent_shape, (4, 4, 4), **vit_args)
        self.vit_enc1 = ViT3DResidualAdapter(c1, half_shape, (2, 2, 2), **vit_args)
        self.vit_dec1 = ViT3DResidualAdapter(c1, half_shape, (2, 2, 2), **vit_args)
        self.vit_dec0 = ViT3DResidualAdapter(c0, latent_shape, (4, 4, 4), **vit_args)
        self.out = nn.Sequential(
            nn.GroupNorm(min(norm_groups, c0), c0),
            nn.SiLU(),
            nn.Conv3d(c0, out_channels, 3, padding=1),
        )

    def forward(self, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_embedding(t)
        s0 = self.enc0b(self.enc0a(self.input(x1), te), te)
        s0 = self.vit_enc0(s0, te)
        s1 = self.enc1b(self.enc1a(self.down1(s0), te), te)
        s1 = self.vit_enc1(s1, te)
        s2 = self.enc2b(self.enc2a(self.down2(s1), te), te)
        h = self.mid2(self.mid1(self.down3(s2), te), te)
        h = self.dec2b(self.dec2a(torch.cat([self.up2(h), s2], dim=1), te), te)
        h = self.dec1b(self.dec1a(torch.cat([self.up1(h), s1], dim=1), te), te)
        h = self.vit_dec1(h, te)
        h = self.dec0b(self.dec0a(torch.cat([self.up0(h), s0], dim=1), te), te)
        h = self.vit_dec0(h, te)
        return self.out(h)
