
"""Exp037C-ViT: Exp037A BasicUNet3D + bottleneck ViT + global pyramid + multi-scale Global-Local fusion.

The ONLY architectural change relative to Exp037C-BiMamba is the global
sequence operator:
    Bi-Mamba -> ViT / Transformer self-attention

The Exp037A local backbone, Global Pyramid, five Global-Local fusion points,
auxiliary global heads, diffusion interface, and global losses remain unchanged.

Input latent:
    x_t [B,4,40,48,40]

Local hierarchy:
    Enc0: [B,c0,40,48,40]
    Enc1: [B,c1,20,24,20]
    Enc2: [B,c2,10,12,10]
    Mid : [B,c3,5,6,5]

ViT global branch:
    Mid [B,c3,5,6,5]
      -> 1x1x1 projection
      -> 150 dense spatial tokens [B,150,D]
      -> learned factorized 3D positional embedding
      -> pre-norm Transformer blocks
      -> reshape
      -> 1x1x1 projection
      -> G3 [B,c3,5,6,5]
      -> G2 [B,c2,10,12,10]
      -> G1 [B,c1,20,24,20]

Fusion points:
    Mid       <- G3
    Enc2 skip <- G2
    Dec2      <- G2
    Enc1 skip <- G1
    Dec1      <- G1

Dec0 stays purely local.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * -scale
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.net(emb)


class TimeResBlock3D(nn.Module):
    """Identical residual block to Exp037A."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(time_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class ViTBlock(nn.Module):
    """Pre-norm Transformer block for dense bottleneck spatial tokens."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(
                f"token_dim ({dim}) must be divisible by heads ({heads})"
            )

        hidden = int(round(dim * float(mlp_ratio)))
        if hidden <= 0:
            raise ValueError("ViT MLP hidden dimension must be > 0")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=float(attention_dropout),
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(float(dropout))

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.norm1(x)
        attn_out, _ = self.attn(
            qkv,
            qkv,
            qkv,
            need_weights=False,
        )
        x = x + self.attn_drop(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalViT3D(nn.Module):
    """ViT global modeling over the 5x6x5 bottleneck (150 tokens).

    No CLS token is used because Exp037C needs a dense global spatial feature
    map for the Global Pyramid and multi-scale fusion.

    Position is encoded with learnable factorized 3D embeddings:
        P(d,h,w) = P_D(d) + P_H(h) + P_W(w)
    """

    def __init__(
        self,
        in_channels: int,
        token_dim: int = 256,
        depth: int = 3,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bottleneck_shape: Tuple[int, int, int] = (5, 6, 5),
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.bottleneck_shape = tuple(int(v) for v in bottleneck_shape)

        if len(self.bottleneck_shape) != 3 or min(self.bottleneck_shape) <= 0:
            raise ValueError(
                f"Invalid bottleneck_shape={self.bottleneck_shape}"
            )
        if int(depth) <= 0:
            raise ValueError("ViT depth must be > 0")
        if int(heads) <= 0:
            raise ValueError("ViT heads must be > 0")
        if self.token_dim % int(heads) != 0:
            raise ValueError(
                f"token_dim={self.token_dim} must be divisible by heads={heads}"
            )

        self.in_proj = nn.Conv3d(
            self.in_channels,
            self.token_dim,
            kernel_size=1,
        )

        d, h, w = self.bottleneck_shape
        self.pos_d = nn.Parameter(torch.zeros(1, d, 1, 1, self.token_dim))
        self.pos_h = nn.Parameter(torch.zeros(1, 1, h, 1, self.token_dim))
        self.pos_w = nn.Parameter(torch.zeros(1, 1, 1, w, self.token_dim))
        nn.init.trunc_normal_(self.pos_d, std=0.02)
        nn.init.trunc_normal_(self.pos_h, std=0.02)
        nn.init.trunc_normal_(self.pos_w, std=0.02)

        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=self.token_dim,
                    heads=int(heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                    attention_dropout=float(attention_dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.out_norm = nn.LayerNorm(self.token_dim)
        self.out_proj = nn.Conv3d(
            self.token_dim,
            self.in_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")

        b, _, d, h, w = x.shape
        if (d, h, w) != self.bottleneck_shape:
            raise ValueError(
                "Unexpected bottleneck spatial shape: "
                f"got {(d,h,w)}, expected {self.bottleneck_shape}. "
                "For the current 160x192x160 image / 40x48x40 latent, "
                "the Exp037A bottleneck is expected to be 5x6x5."
            )

        volume = self.in_proj(x)  # [B,E,D,H,W]
        tokens_3d = volume.permute(0, 2, 3, 4, 1)  # [B,D,H,W,E]
        tokens_3d = tokens_3d + self.pos_d + self.pos_h + self.pos_w
        tokens = tokens_3d.reshape(b, d * h * w, self.token_dim)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.out_norm(tokens)
        volume = tokens.reshape(b, d, h, w, self.token_dim)
        volume = volume.permute(0, 4, 1, 2, 3).contiguous()
        return self.out_proj(volume)


class GlobalPyramidUp3D(nn.Module):
    """Upsample a global feature by one U-Net level."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.refine = nn.Sequential(
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


class GlobalLocalGatedFusion3D(nn.Module):
    """Residual Global->Local gated fusion.

    F = Local + alpha * gate(Local, Global) * Project(Global)
    """

    def __init__(
        self,
        channels: int,
        gate_bias: float = -2.0,
        init_alpha: float = 0.10,
    ) -> None:
        super().__init__()
        self.global_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.gate = nn.Conv3d(channels * 2, channels, kernel_size=1)
        nn.init.constant_(self.gate.bias, float(gate_bias))
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(
        self,
        local: torch.Tensor,
        global_feature: torch.Tensor,
        *,
        return_gate: bool = False,
    ):
        if local.shape != global_feature.shape:
            raise ValueError(
                "Fusion inputs must have identical shape, got "
                f"local={tuple(local.shape)} global={tuple(global_feature.shape)}"
            )

        projected_global = self.global_proj(global_feature)
        gate = torch.sigmoid(
            self.gate(torch.cat([local, projected_global], dim=1))
        )
        fused = local + self.alpha * gate * projected_global

        if return_gate:
            return fused, gate.float().mean()
        return fused


class Exp037CGlobalLocalViTUNet3D(nn.Module):
    """Exp037A local backbone + one bottleneck ViT global branch."""

    EXP037A_PREFIXES = (
        "time_embedding.",
        "input.",
        "enc0a.",
        "enc0b.",
        "down1.",
        "enc1a.",
        "enc1b.",
        "down2.",
        "enc2a.",
        "enc2b.",
        "down3.",
        "mid1.",
        "mid2.",
        "up2.",
        "dec2a.",
        "dec2b.",
        "up1.",
        "dec1a.",
        "dec1b.",
        "up0.",
        "dec0a.",
        "dec0b.",
        "out.",
    )

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        time_dim: int = 128,
        norm_groups: int = 8,
        dropout: float = 0.0,
        global_token_dim: int = 256,
        vit_depth: int = 3,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.0,
        vit_attention_dropout: float = 0.0,
        vit_bottleneck_shape: Tuple[int, int, int] = (5, 6, 5),
        fusion_gate_bias: float = -2.0,
        fusion_init_alpha: float = 0.10,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels

        # Exp037A local backbone: names are intentionally unchanged.
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
        self.out = nn.Sequential(
            nn.GroupNorm(min(norm_groups, c0), c0),
            nn.SiLU(),
            nn.Conv3d(c0, out_channels, 3, padding=1),
        )

        # The only global operator: ViT at the 150-token bottleneck.
        self.global_vit = GlobalViT3D(
            in_channels=c3,
            token_dim=int(global_token_dim),
            depth=int(vit_depth),
            heads=int(vit_heads),
            mlp_ratio=float(vit_mlp_ratio),
            dropout=float(vit_dropout),
            attention_dropout=float(vit_attention_dropout),
            bottleneck_shape=tuple(vit_bottleneck_shape),
        )

        # Same Global Pyramid as the Mamba version.
        self.global_up2 = GlobalPyramidUp3D(c3, c2, norm_groups)
        self.global_up1 = GlobalPyramidUp3D(c2, c1, norm_groups)

        # Same five fusion points as the Mamba version.
        self.fuse_mid = GlobalLocalGatedFusion3D(
            c3, fusion_gate_bias, fusion_init_alpha
        )
        self.fuse_enc2 = GlobalLocalGatedFusion3D(
            c2, fusion_gate_bias, fusion_init_alpha
        )
        self.fuse_dec2 = GlobalLocalGatedFusion3D(
            c2, fusion_gate_bias, fusion_init_alpha
        )
        self.fuse_enc1 = GlobalLocalGatedFusion3D(
            c1, fusion_gate_bias, fusion_init_alpha
        )
        self.fuse_dec1 = GlobalLocalGatedFusion3D(
            c1, fusion_gate_bias, fusion_init_alpha
        )

        # Same auxiliary global heads / loss interface as Exp037C-BiMamba.
        self.global_head_g3 = nn.Conv3d(c3, out_channels, kernel_size=1)
        self.global_head_g2 = nn.Conv3d(c2, out_channels, kernel_size=1)

    def forward(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        te = self.time_embedding(t)

        # Local encoder.
        s0 = self.enc0b(self.enc0a(self.input(x1), te), te)
        s1 = self.enc1b(self.enc1a(self.down1(s0), te), te)
        s2 = self.enc2b(self.enc2a(self.down2(s1), te), te)

        # Local bottleneck.
        h = self.mid2(self.mid1(self.down3(s2), te), te)

        # ViT global representation + Global Pyramid.
        g3 = self.global_vit(h)
        g2 = self.global_up2(g3)
        g1 = self.global_up1(g2)

        # Global-aware cached encoder skips.
        s2_fused, gate_enc2 = self.fuse_enc2(s2, g2, return_gate=True)
        s1_fused, gate_enc1 = self.fuse_enc1(s1, g1, return_gate=True)

        # Bottleneck Global-Local fusion.
        h, gate_mid = self.fuse_mid(h, g3, return_gate=True)

        # Decoder level 2.
        h = self.dec2b(
            self.dec2a(torch.cat([self.up2(h), s2_fused], dim=1), te),
            te,
        )
        h, gate_dec2 = self.fuse_dec2(h, g2, return_gate=True)

        # Decoder level 1.
        h = self.dec1b(
            self.dec1a(torch.cat([self.up1(h), s1_fused], dim=1), te),
            te,
        )
        h, gate_dec1 = self.fuse_dec1(h, g1, return_gate=True)

        # Highest-resolution decoder remains local.
        h = self.dec0b(
            self.dec0a(torch.cat([self.up0(h), s0], dim=1), te),
            te,
        )
        pred_v = self.out(h)

        if not return_aux:
            return pred_v

        aux: Dict[str, torch.Tensor] = {
            "global_g3": g3,
            "global_g2": g2,
            "global_g1": g1,
            "global_pred_g3": self.global_head_g3(g3),
            "global_pred_g2": self.global_head_g2(g2),
            "gate_mid": gate_mid,
            "gate_enc2": gate_enc2,
            "gate_dec2": gate_dec2,
            "gate_enc1": gate_enc1,
            "gate_dec1": gate_dec1,
            "global_g3_norm": g3.float().flatten(1).norm(dim=1).mean(),
            "global_g2_norm": g2.float().flatten(1).norm(dim=1).mean(),
            "global_g1_norm": g1.float().flatten(1).norm(dim=1).mean(),
        }
        return pred_v, aux


def load_exp037a_backbone(
    model: Exp037CGlobalLocalViTUNet3D,
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, int]:
    """Load only Exp037A-compatible backbone weights.

    ViT / Global Pyramid / Fusion / Auxiliary heads remain freshly initialized.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = checkpoint.get("model", checkpoint)

    current = model.state_dict()
    loaded = {}
    skipped_shape = 0
    skipped_new = 0

    for key, value in state.items():
        if not key.startswith(model.EXP037A_PREFIXES):
            skipped_new += 1
            continue
        if key not in current or current[key].shape != value.shape:
            skipped_shape += 1
            continue
        loaded[key] = value

    missing, unexpected = model.load_state_dict(loaded, strict=False)
    return {
        "loaded": len(loaded),
        "skipped_shape": int(skipped_shape),
        "skipped_non_backbone": int(skipped_new),
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
    }
