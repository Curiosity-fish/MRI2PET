
# model/bsfe_latent.py
# BSFE for latent diffusion conditioning:
#   Input MRI:  (B, 1, 96, 160, 160)
#   First downsample to latent spatial size: (B, 1, 24, 40, 40)
#   Then extract structural features, tokens, global vector for diffusion UNet.
#
# Outputs are all in LATENT spatial coordinates (24,40,40) pyramid.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

try:
    from einops import rearrange
except Exception:
    rearrange = None


# -------------------------
# Basic blocks
# -------------------------
class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GN(nn.Module):
    """GroupNorm with safe group count."""
    def __init__(self, channels: int, num_groups: int = 16, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        g = min(num_groups, channels)
        while g > 1 and (channels % g != 0):
            g -= 1
        self.gn = nn.GroupNorm(num_groups=g, num_channels=channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_groups: int = 16, dropout: float = 0.0):
        super().__init__()
        self.norm1 = GN(in_ch, norm_groups)
        self.act1 = SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = GN(out_ch, norm_groups)
        self.act2 = SiLU()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -------------------------
# Pre-downsample (MRI -> latent spatial size)
# -------------------------
class PreDownsample(nn.Module):
    """
    Two stride-2 conv downsamples:
      (96,160,160) -> (48,80,80) -> (24,40,40)
    Keeps learnable filtering (preferred over plain interpolate).
    """
    def __init__(self, in_ch: int = 1, hidden_ch: int = 8, norm_groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock3D(in_ch, in_ch, norm_groups=norm_groups, dropout=0.0),
            nn.Conv3d(in_ch, hidden_ch, kernel_size=4, stride=2, padding=1),
            GN(hidden_ch, norm_groups),
            SiLU(),
            ResBlock3D(hidden_ch, hidden_ch, norm_groups=norm_groups, dropout=0.0),
            nn.Conv3d(hidden_ch, hidden_ch, kernel_size=4, stride=2, padding=1),
            GN(hidden_ch, norm_groups),
            SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Attention (used only at low-res stages)
# -------------------------
class MHAttention3D(nn.Module):
    """Multi-head self-attention over flattened 3D tokens (use only on low-res maps)."""
    def __init__(self, ch: int, n_heads: int = 4, head_dim: int = 32, norm_groups: int = 16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim

        self.norm = GN(ch, norm_groups)
        self.to_qkv = nn.Conv1d(ch, inner * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(inner, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        n = D * H * W
        h = self.norm(x).view(B, C, n)

        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.n_heads, self.head_dim, n).transpose(2, 3)  # (B, heads, N, head_dim)
        k = k.view(B, self.n_heads, self.head_dim, n).transpose(2, 3)
        v = v.view(B, self.n_heads, self.head_dim, n).transpose(2, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(2, 3).contiguous().view(B, self.n_heads * self.head_dim, n)
        out = self.proj(out).view(B, C, D, H, W)
        return x + out


# -------------------------
# Tokenization
# -------------------------
class TokenLearner(nn.Module):
    """Learnable queries attend to deep 3D features to produce compact tokens."""
    def __init__(self, in_ch: int, n_tokens: int = 32, token_dim: int = 128):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim

        self.q = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)
        self.k_proj = nn.Linear(in_ch, token_dim, bias=False)
        self.v_proj = nn.Linear(in_ch, token_dim, bias=False)
        self.ln = nn.LayerNorm(token_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if rearrange is None:
            B, C, D, H, W = feat.shape
            x = feat.view(B, C, D * H * W).transpose(1, 2)  # (B, N, C)
        else:
            x = rearrange(feat, "b c d h w -> b (d h w) c")

        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.q.expand(x.size(0), -1, -1)
        scale = 1.0 / math.sqrt(self.token_dim)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        t = torch.matmul(attn, v)
        return self.ln(t)


# -------------------------
# Config + BSFE
# -------------------------
@dataclass
class BSFEConfig:
    # MRI input shape (for sanity checks only)
    input_shape: Tuple[int, int, int] = (96, 160, 160)
    # latent spatial shape (must match PET latent: 24,40,40)
    latent_shape: Tuple[int, int, int] = (24, 40, 40)

    in_channels: int = 1
    pre_down_hidden: int = 32  # channels inside pre-downsample
    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2

    norm_groups: int = 16
    dropout: float = 0.0

    # attention (deep stages only)
    attn_stages: Tuple[int, ...] = ( 1, 3)
    attn_heads: int = 4
    attn_head_dim: int = 32

    # tokens + global embedding
    n_tokens: int = 32
    token_dim: int = 128
    global_dim: int = 256

    # optionally project token dim to match diffusion context dim
    out_context_dim: Optional[int] = None  # e.g. tab_out_dim


class BrainStructuralFeatureExtractor(nn.Module):
    """
    MRI -> (pre-downsample to latent) -> pyramid/tokens/global in latent coordinates.
    """
    def __init__(self, cfg: BSFEConfig):
        super().__init__()
        self.cfg = cfg

        self.pre_down = PreDownsample(cfg.in_channels, cfg.pre_down_hidden, norm_groups=min(8, cfg.pre_down_hidden))
        # stem after pre-downsample
        ch0 = cfg.base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(cfg.pre_down_hidden, ch0, kernel_size=3, padding=1),
            GN(ch0, cfg.norm_groups),
            SiLU(),
        )

        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.attn = nn.ModuleDict()

        in_ch = ch0
        for si, mult in enumerate(cfg.channel_mults):
            out_ch = ch0 * mult
            blocks = nn.ModuleList()
            for bi in range(cfg.num_res_blocks):
                blocks.append(ResBlock3D(in_ch if bi == 0 else out_ch, out_ch, cfg.norm_groups, cfg.dropout))
                blocks.append(ResBlock3D(out_ch, out_ch, cfg.norm_groups, cfg.dropout))
            self.stages.append(blocks)

            if si != len(cfg.channel_mults) - 1:
                self.downs.append(Downsample3D(out_ch))

            if si in cfg.attn_stages:
                self.attn[str(si)] = MHAttention3D(out_ch, cfg.attn_heads, cfg.attn_head_dim, cfg.norm_groups)

            in_ch = out_ch

        deep_ch = ch0 * cfg.channel_mults[-1]
        self.tokenizer = TokenLearner(deep_ch, cfg.n_tokens, cfg.token_dim)

        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(deep_ch, cfg.global_dim),
            nn.SiLU(),
        )

        self.token_proj = None
        if cfg.out_context_dim is not None and cfg.out_context_dim != cfg.token_dim:
            self.token_proj = nn.Linear(cfg.token_dim, cfg.out_context_dim, bias=False)

    def forward(self, mri: torch.Tensor) -> Dict[str, Any]:
        # mri: (B, 1, 96,160,160)
        x = self.pre_down(mri)  # (B, pre_down_hidden, 24,40,40) if input matches
        # If your input is not exactly (96,160,160), this will still downsample by 4x.
        x = self.stem(x)

        pyramid: List[torch.Tensor] = []
        for si, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x)
            if str(si) in self.attn:
                x = self.attn[str(si)](x)
            pyramid.append(x)
            if si < len(self.downs):
                x = self.downs[si](x)

        deep = pyramid[-1]
        tokens = self.tokenizer(deep)
        if self.token_proj is not None:
            tokens = self.token_proj(tokens)
        global_vec = self.global_head(deep)

        return {"pyramid": pyramid, "tokens": tokens, "global": global_vec}


if __name__ == "__main__":
    cfg = BSFEConfig(out_context_dim=8)
    net = BrainStructuralFeatureExtractor(cfg)
    x = torch.randn(2, 1, 96, 160, 160)
    y = net(x)
    print("pyramid:", [t.shape for t in y["pyramid"]])
    print("tokens:", y["tokens"].shape, "global:", y["global"].shape)
