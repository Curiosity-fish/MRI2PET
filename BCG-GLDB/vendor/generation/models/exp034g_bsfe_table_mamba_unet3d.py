"""Exp034G: Exp034F with corrected direct-x1 axial Mamba low-resolution fusion."""
from __future__ import annotations

from typing import Sequence

import torch

from models.exp034e_mamba_unet3d import X1MambaFusion
from models.exp034f_bsfe_table_highres_unet3d import BSFETableHighResUNet3D


class BSFETableMambaUNet3D(BSFETableHighResUNet3D):
    """Exp034F plus x1 Mamba fusion at encoder s2, bottleneck, and decoder s2."""

    def __init__(self, *args, latent_shape: Sequence[int] = (40, 48, 40),
                 mamba_token_dim: int = 128, mamba_d_state: int = 8,
                 mamba_d_conv: int = 3, mamba_expand: int = 1,
                 mamba_line_chunk_size: int = 256, mamba_depth: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        c0, c1, c2, c3 = (int(v) for v in kwargs.get('channels', (96, 192, 384, 768)))
        quarter = tuple(int(v) // 4 for v in latent_shape)
        eighth = tuple(int(v) // 8 for v in latent_shape)
        common = dict(time_dim=int(kwargs.get('time_dim', 384)), token_dim=int(mamba_token_dim),
                      d_state=int(mamba_d_state), d_conv=int(mamba_d_conv), expand=int(mamba_expand),
                      line_chunk_size=int(mamba_line_chunk_size), depth=int(mamba_depth))
        self.mamba_enc_s2 = X1MambaFusion(4, c2, quarter, **common)
        self.mamba_mid = X1MambaFusion(4, c3, eighth, **common)
        self.mamba_dec_s2 = X1MambaFusion(4, c2, quarter, **common)

    def forward(self, x1: torch.Tensor, t: torch.Tensor, pyramid, table: torch.Tensor, table_mask: torch.Tensor) -> torch.Tensor:
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
        s2 = self.mamba_enc_s2(x1, te, s2)
        h = self._bsfe(self.mid2(self.mid1(self.down3(s2), te), te), p, 3)
        h = self.mamba_mid(x1, te, h)
        h = self._bsfe(self.dec2b(self.dec2a(torch.cat([self.up2(h), s2], 1), te), te), p, 4)
        h = self.mamba_dec_s2(x1, te, h)
        h = self._bsfe(self.dec1b(self.dec1a(torch.cat([self.up1(h), s1], 1), te), te), p, 5)
        h, table_tokens = self.table_stages[2](h, table_tokens, table_kpm, te)
        h = self._bsfe(self.dec0b(self.dec0a(torch.cat([self.up0(h), s0], 1), te), te), p, 6)
        h, _ = self.table_stages[3](h, table_tokens, table_kpm, te)
        return self.out(h)
