from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def make_boundary_mask(mask: torch.Tensor) -> torch.Tensor:
    """Create a one-voxel boundary band from a 3D binary mask.

    Args:
        mask: Tensor with shape [B, 1, D, H, W].
    """
    mask = mask.float()
    dilated = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def _choose_num_groups(channels: int, requested: int) -> int:
    requested = max(int(requested), 1)
    channels = int(channels)
    for g in range(min(requested, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        t = t.float()
        half = self.dim // 2
        if half <= 0:
            return t[:, None]
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class MissingAwareTableEncoder(nn.Module):
    """Missing-aware table encoder with region-conditioned gated cross attention.

    table:      [B, F]
    table_mask: [B, F], 1 means observed, 0 means missing.
    region_cond is used as query so different brain regions can attend to different table variables.
    """

    def __init__(
        self,
        num_features: int,
        cond_dim: int,
        token_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.cond_dim = int(cond_dim)
        self.token_dim = int(token_dim)
        self.value_proj = nn.Linear(1, token_dim)
        self.feature_embed = nn.Embedding(max(self.num_features, 1), token_dim)
        self.missing_embed = nn.Embedding(2, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.query_proj = nn.Linear(cond_dim, token_dim)
        self.cross_attn = nn.MultiheadAttention(token_dim, num_heads=int(heads), dropout=float(dropout), batch_first=True)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        region_cond: torch.Tensor,
    ) -> torch.Tensor:
        b = region_cond.shape[0]
        device = region_cond.device
        dtype = region_cond.dtype
        if self.num_features <= 0 or table is None:
            return torch.zeros_like(region_cond)

        table = table.to(device=device, dtype=dtype)
        if table.ndim == 1:
            table = table.unsqueeze(0)
        if table.shape[0] != b:
            table = table.expand(b, -1)
        table = table[:, : self.num_features]
        if table.shape[1] < self.num_features:
            table = F.pad(table, (0, self.num_features - table.shape[1]))

        if table_mask is None:
            table_mask = torch.ones_like(table)
        else:
            table_mask = table_mask.to(device=device, dtype=dtype)
            if table_mask.ndim == 1:
                table_mask = table_mask.unsqueeze(0)
            if table_mask.shape[0] != b:
                table_mask = table_mask.expand(b, -1)
            table_mask = table_mask[:, : self.num_features]
            if table_mask.shape[1] < self.num_features:
                table_mask = F.pad(table_mask, (0, self.num_features - table_mask.shape[1]))

        observed = (table_mask > 0.5).long()
        value = table * table_mask
        tokens = self.value_proj(value.unsqueeze(-1))
        feature_ids = torch.arange(self.num_features, device=device, dtype=torch.long)[None, :].expand(b, -1)
        tokens = tokens + self.feature_embed(feature_ids) + self.missing_embed(observed)
        tokens = self.token_norm(tokens)

        q = self.query_proj(region_cond).unsqueeze(1)
        ctx, _ = self.cross_attn(q, tokens, tokens, need_weights=False)
        ctx = self.out_proj(ctx.squeeze(1))
        g = self.gate(torch.cat([region_cond, ctx], dim=-1))
        return g * ctx


class FiLMResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.norm1 = nn.GroupNorm(_choose_num_groups(self.in_ch, groups), self.in_ch)
        self.conv1 = nn.Conv3d(self.in_ch, self.out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_choose_num_groups(self.out_ch, groups), self.out_ch)
        self.dropout = nn.Dropout3d(float(dropout)) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(self.out_ch, self.out_ch, kernel_size=3, padding=1)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), self.out_ch * 2))
        self.skip = nn.Conv3d(self.in_ch, self.out_ch, kernel_size=1) if self.in_ch != self.out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.cond_proj(cond).to(dtype=h.dtype)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class MaskGuidedSE3D(nn.Module):
    """Mask-guided channel recalibration.

    This complements attention: attention models spatial/token interaction, while this module explicitly
    recalibrates channels using target-region and whole-brain pooled statistics.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(int(channels) // int(reduction), 8)
        self.net = nn.Sequential(
            nn.Linear(int(channels) * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(channels)),
            nn.Sigmoid(),
        )

    @staticmethod
    def _masked_pool(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(mask.float(), size=x.shape[2:], mode="nearest")
        mask = mask.to(dtype=x.dtype, device=x.device)
        denom = mask.sum(dim=(2, 3, 4)).clamp_min(eps)
        return (x * mask).sum(dim=(2, 3, 4)) / denom

    def forward(self, x: torch.Tensor, brain_mask: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        brain_pool = self._masked_pool(x, brain_mask)
        region_pool = self._masked_pool(x, region_mask)
        gate = self.net(torch.cat([brain_pool, region_pool], dim=1)).to(dtype=x.dtype)
        return x * (1.0 + gate[:, :, None, None, None])


class Downsample3D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose3d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _partition_windows_3d(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B,C,D,H,W] -> [B*nW, N, C]. D/H/W must be divisible by window_size."""
    b, c, d, h, w = x.shape
    ws = int(window_size)
    x = x.view(b, c, d // ws, ws, h // ws, ws, w // ws, ws)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    return x.view(b * (d // ws) * (h // ws) * (w // ws), ws * ws * ws, c)


def _reverse_windows_3d(windows: torch.Tensor, window_size: int, b: int, d: int, h: int, w: int) -> torch.Tensor:
    """[B*nW, N, C] -> [B,C,D,H,W]."""
    ws = int(window_size)
    c = windows.shape[-1]
    x = windows.view(b, d // ws, h // ws, w // ws, ws, ws, ws, c)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return x.view(b, c, d, h, w)


def _pad_to_multiple_3d(x: torch.Tensor, multiple: int, mode: str = "replicate") -> tuple[torch.Tensor, tuple[int, int, int]]:
    d, h, w = x.shape[-3:]
    m = int(multiple)
    pd = (m - d % m) % m
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if pd or ph or pw:
        x = F.pad(x, (0, pw, 0, ph, 0, pd), mode=mode)
    return x, (pd, ph, pw)


def _unpad_3d(x: torch.Tensor, pads: tuple[int, int, int]) -> torch.Tensor:
    pd, ph, pw = pads
    if pd:
        x = x[..., :-pd, :, :]
    if ph:
        x = x[..., :, :-ph, :]
    if pw:
        x = x[..., :, :, :-pw]
    return x


class MaskedWindowAttention3D(nn.Module):
    """3D window self-attention with region-mask pair bias.

    This is the main upgrade over the previous light model. It performs local token mixing inside 3D windows
    and explicitly boosts attention pairs that both lie inside the selected target-region mask.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        window_size: int = 8,
        cond_dim: int = 256,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        region_pair_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.heads = int(heads)
        self.window_size = int(window_size)
        if self.channels % self.heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.head_dim = self.channels // self.heads
        self.scale = self.head_dim ** -0.5
        self.region_pair_bias = float(region_pair_bias)
        self.norm1 = nn.LayerNorm(self.channels)
        self.qkv = nn.Linear(self.channels, self.channels * 3)
        self.proj = nn.Linear(self.channels, self.channels)
        self.attn_drop = nn.Dropout(float(dropout))
        self.proj_drop = nn.Dropout(float(dropout))
        self.cond_proj1 = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), self.channels * 2))
        hidden = max(int(self.channels * float(mlp_ratio)), self.channels)
        self.norm2 = nn.LayerNorm(self.channels)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.channels),
            nn.Dropout(float(dropout)),
        )
        self.cond_proj2 = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), self.channels * 2))

    @staticmethod
    def _apply_film_tokens(tokens: torch.Tensor, scale_shift: torch.Tensor, repeat_factor: int) -> torch.Tensor:
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.repeat_interleave(repeat_factor, dim=0).unsqueeze(1)
        shift = shift.repeat_interleave(repeat_factor, dim=0).unsqueeze(1)
        return tokens * (1.0 + scale.to(dtype=tokens.dtype)) + shift.to(dtype=tokens.dtype)

    def forward(self, x: torch.Tensor, region_mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, d0, h0, w0 = x.shape
        ws = min(self.window_size, d0, h0, w0)
        # If the resolution is smaller than the configured window, fall back to one global window.
        if d0 % ws != 0 or h0 % ws != 0 or w0 % ws != 0:
            x_pad, pads = _pad_to_multiple_3d(x, ws)
            mask_pad, _ = _pad_to_multiple_3d(region_mask.float(), ws)
        else:
            x_pad, pads = x, (0, 0, 0)
            mask_pad = region_mask.float()
        if mask_pad.shape[2:] != x_pad.shape[2:]:
            mask_pad = F.interpolate(mask_pad, size=x_pad.shape[2:], mode="nearest")
        d, h, w = x_pad.shape[-3:]
        nwin = (d // ws) * (h // ws) * (w // ws)

        tokens = _partition_windows_3d(x_pad, ws)  # [B*nW,N,C]
        residual = tokens
        qkv_in = self.norm1(tokens)
        qkv_in = self._apply_film_tokens(qkv_in, self.cond_proj1(cond), repeat_factor=nwin)
        qkv = self.qkv(qkv_in).view(qkv_in.shape[0], qkv_in.shape[1], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.region_pair_bias != 0:
            r_tokens = _partition_windows_3d(mask_pad.to(dtype=x_pad.dtype), ws)[..., 0]
            inside = (r_tokens > 0.5).to(dtype=attn.dtype)
            pair = inside[:, None, :, None] * inside[:, None, None, :]
            attn = attn + self.region_pair_bias * pair

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(tokens.shape[0], tokens.shape[1], c)
        tokens = residual + self.proj_drop(self.proj(out))

        residual = tokens
        mlp_in = self.norm2(tokens)
        mlp_in = self._apply_film_tokens(mlp_in, self.cond_proj2(cond), repeat_factor=nwin)
        tokens = residual + self.mlp(mlp_in)

        out = _reverse_windows_3d(tokens, ws, b, d, h, w)
        out = _unpad_3d(out, pads)
        return out


class GlobalAttention3D(nn.Module):
    """Full global attention for low-resolution bottleneck features, e.g. 8^3 tokens."""

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        cond_dim: int = 256,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.heads = int(heads)
        self.norm1 = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(self.channels, num_heads=self.heads, dropout=float(dropout), batch_first=True)
        self.cond_proj1 = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), self.channels * 2))
        self.norm2 = nn.LayerNorm(self.channels)
        hidden = max(int(self.channels * float(mlp_ratio)), self.channels)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.channels),
            nn.Dropout(float(dropout)),
        )
        self.cond_proj2 = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), self.channels * 2))

    @staticmethod
    def _film(tokens: torch.Tensor, scale_shift: torch.Tensor) -> torch.Tensor:
        scale, shift = scale_shift.chunk(2, dim=-1)
        return tokens * (1.0 + scale[:, None, :].to(tokens.dtype)) + shift[:, None, :].to(tokens.dtype)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        residual = tokens
        q = self._film(self.norm1(tokens), self.cond_proj1(cond))
        out, _ = self.attn(q, q, q, need_weights=False)
        tokens = residual + out
        residual = tokens
        tokens = residual + self.mlp(self._film(self.norm2(tokens), self.cond_proj2(cond)))
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)


class CrossAttention3D(nn.Module):
    """Image-query cross-attention to MRI/mask context tokens at bottleneck resolution."""

    def __init__(
        self,
        channels: int,
        context_channels: int,
        heads: int = 8,
        cond_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.context_channels = int(context_channels)
        self.heads = int(heads)
        self.norm_q = nn.LayerNorm(self.channels)
        self.norm_kv = nn.LayerNorm(self.context_channels)
        self.kv_proj = nn.Linear(self.context_channels, self.channels)
        self.q_film = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), self.channels * 2))
        self.attn = nn.MultiheadAttention(self.channels, num_heads=self.heads, dropout=float(dropout), batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(self.channels), nn.Linear(self.channels, self.channels))

    @staticmethod
    def _film(tokens: torch.Tensor, scale_shift: torch.Tensor) -> torch.Tensor:
        scale, shift = scale_shift.chunk(2, dim=-1)
        return tokens * (1.0 + scale[:, None, :].to(tokens.dtype)) + shift[:, None, :].to(tokens.dtype)

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)
        kv = context.flatten(2).transpose(1, 2)
        residual = q
        qn = self._film(self.norm_q(q), self.q_film(cond))
        kvn = self.kv_proj(self.norm_kv(kv))
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = residual + self.out(out)
        return q.transpose(1, 2).reshape(b, c, d, h, w)


class CondTokenCrossAttention3D(nn.Module):
    """Image-query cross-attention to learned condition tokens derived from time+region+table condition."""

    def __init__(self, channels: int, cond_dim: int = 256, heads: int = 8, num_tokens: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.cond_dim = int(cond_dim)
        self.num_tokens = int(num_tokens)
        self.norm_q = nn.LayerNorm(self.channels)
        self.token_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.num_tokens * self.channels),
        )
        self.attn = nn.MultiheadAttention(self.channels, num_heads=int(heads), dropout=float(dropout), batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(self.channels), nn.Linear(self.channels, self.channels))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)
        tokens = self.token_proj(cond).view(b, self.num_tokens, c).to(dtype=q.dtype)
        out, _ = self.attn(self.norm_q(q), tokens, tokens, need_weights=False)
        q = q + self.out(out)
        return q.transpose(1, 2).reshape(b, c, d, h, w)


class PatchMTGBridgeDenoiser(nn.Module):
    """Strong attention version of Mask-Topology Guided 64^3 Patch Diffusion Bridge denoiser.

    Compared with the lightweight version, this model adds:
      1) mask-biased 3D window self-attention at selected encoder/decoder levels;
      2) full global self-attention at the bottleneck;
      3) bottleneck cross-attention to MRI + mask context tokens;
      4) condition-token cross-attention from time, parc98 region label, and table condition.

    Inputs:
        x_t:         [B, 1, 64, 64, 64]
        mri:         [B, 1, 64, 64, 64]
        brain_mask:  [B, 1, 64, 64, 64]
        region_mask: [B, 1, 64, 64, 64]
        time:        [B]
        region_label:[B], parc98 label id mapped by your dataset/atlas
        table:       [B, F]
        table_mask:  [B, F]

    Output:
        residual prediction R_t = x_t - PET, shape [B, 1, 64, 64, 64].
    """

    def __init__(
        self,
        in_channels: int = 1,
        mri_in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 24,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        blocks_per_level: int = 3,
        norm_groups: int = 8,
        dropout: float = 0.05,
        time_dim: int = 256,
        cond_dim: int = 256,
        num_regions: int = 98,
        table_num_features: int = 0,
        table_token_dim: int = 128,
        table_cross_attn_heads: int = 8,
        use_coords: bool = True,
        use_mask_guided_se: bool = True,
        use_boundary_head: bool = True,
        use_checkpoint: bool = False,
        use_window_attention: bool = True,
        attention_levels: Sequence[int] = (1, 2, 3),
        decoder_attention_levels: Sequence[int] = (1, 2),
        attention_heads: Sequence[int] | int = (4, 4, 8, 8),
        window_size: int = 8,
        attention_mlp_ratio: float = 2.0,
        region_pair_bias: float = 1.0,
        use_global_mid_attn: bool = True,
        use_mri_cross_attn: bool = True,
        use_cond_cross_attn: bool = True,
        cond_cross_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.use_coords = bool(use_coords)
        self.use_mask_guided_se = bool(use_mask_guided_se)
        self.use_boundary_head = bool(use_boundary_head)
        self.use_checkpoint = bool(use_checkpoint)
        self.num_regions = int(num_regions)
        self.use_window_attention = bool(use_window_attention)
        self.attention_levels = set(int(v) for v in attention_levels)
        self.decoder_attention_levels = set(int(v) for v in decoder_attention_levels)
        self.use_global_mid_attn = bool(use_global_mid_attn)
        self.use_mri_cross_attn = bool(use_mri_cross_attn)
        self.use_cond_cross_attn = bool(use_cond_cross_attn)

        if isinstance(attention_heads, int):
            self.attention_heads = [int(attention_heads)] * len(tuple(channel_mults))
        else:
            self.attention_heads = [int(v) for v in attention_heads]

        # x_t + MRI + brain mask + selected region mask + selected region boundary + optional xyz coordinates.
        input_ch = int(in_channels) + int(mri_in_channels) + 3 + (3 if self.use_coords else 0)
        self.input_proj = nn.Conv3d(input_ch, int(base_channels), kernel_size=3, padding=1)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(int(time_dim)),
            nn.Linear(int(time_dim), int(cond_dim)),
            nn.SiLU(),
            nn.Linear(int(cond_dim), int(cond_dim)),
        )
        self.region_embed = nn.Embedding(self.num_regions + 1, int(cond_dim))
        self.table_encoder = MissingAwareTableEncoder(
            num_features=int(table_num_features),
            cond_dim=int(cond_dim),
            token_dim=int(table_token_dim),
            heads=int(table_cross_attn_heads),
            dropout=float(dropout),
        )
        self.cond_mlp = nn.Sequential(
            nn.LayerNorm(int(cond_dim)),
            nn.Linear(int(cond_dim), int(cond_dim) * 2),
            nn.SiLU(),
            nn.Linear(int(cond_dim) * 2, int(cond_dim)),
            nn.LayerNorm(int(cond_dim)),
        )

        channels = [int(base_channels) * int(m) for m in channel_mults]
        self.channels = channels

        self.down_blocks = nn.ModuleList()
        self.down_se = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = int(base_channels)
        for level, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(int(blocks_per_level)):
                blocks.append(FiLMResBlock3D(ch, out_ch, cond_dim=int(cond_dim), groups=int(norm_groups), dropout=float(dropout)))
                ch = out_ch
            self.down_blocks.append(blocks)
            self.down_se.append(MaskGuidedSE3D(ch) if self.use_mask_guided_se else nn.Identity())
            heads = self._heads_for_level(level, ch)
            if self.use_window_attention and level in self.attention_levels:
                self.down_attn.append(
                    MaskedWindowAttention3D(
                        ch,
                        heads=heads,
                        window_size=int(window_size),
                        cond_dim=int(cond_dim),
                        mlp_ratio=float(attention_mlp_ratio),
                        dropout=float(dropout),
                        region_pair_bias=float(region_pair_bias),
                    )
                )
            else:
                self.down_attn.append(nn.Identity())
            if level != len(channels) - 1:
                self.downsamples.append(Downsample3D(ch))

        mid_ch = ch
        self.mid1 = FiLMResBlock3D(mid_ch, mid_ch, cond_dim=int(cond_dim), groups=int(norm_groups), dropout=float(dropout))
        self.mid_global_attn = (
            GlobalAttention3D(
                mid_ch,
                heads=self._heads_for_level(len(channels) - 1, mid_ch),
                cond_dim=int(cond_dim),
                mlp_ratio=float(attention_mlp_ratio),
                dropout=float(dropout),
            )
            if self.use_global_mid_attn
            else nn.Identity()
        )
        self.mri_context_proj = nn.Conv3d(int(mri_in_channels) + 2, mid_ch, kernel_size=3, padding=1)
        self.mid_mri_cross = (
            CrossAttention3D(
                channels=mid_ch,
                context_channels=mid_ch,
                heads=self._heads_for_level(len(channels) - 1, mid_ch),
                cond_dim=int(cond_dim),
                dropout=float(dropout),
            )
            if self.use_mri_cross_attn
            else nn.Identity()
        )
        self.mid_cond_cross = (
            CondTokenCrossAttention3D(
                channels=mid_ch,
                cond_dim=int(cond_dim),
                heads=self._heads_for_level(len(channels) - 1, mid_ch),
                num_tokens=int(cond_cross_tokens),
                dropout=float(dropout),
            )
            if self.use_cond_cross_attn
            else nn.Identity()
        )
        self.mid_se = MaskGuidedSE3D(mid_ch) if self.use_mask_guided_se else nn.Identity()
        self.mid2 = FiLMResBlock3D(mid_ch, mid_ch, cond_dim=int(cond_dim), groups=int(norm_groups), dropout=float(dropout))

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_se = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        for level in reversed(range(len(channels))):
            skip_ch = channels[level]
            blocks = nn.ModuleList()
            blocks.append(FiLMResBlock3D(ch + skip_ch, skip_ch, cond_dim=int(cond_dim), groups=int(norm_groups), dropout=float(dropout)))
            ch = skip_ch
            for _ in range(max(int(blocks_per_level) - 1, 0)):
                blocks.append(FiLMResBlock3D(ch, ch, cond_dim=int(cond_dim), groups=int(norm_groups), dropout=float(dropout)))
            self.up_blocks.append(blocks)
            self.up_se.append(MaskGuidedSE3D(ch) if self.use_mask_guided_se else nn.Identity())
            heads = self._heads_for_level(level, ch)
            if self.use_window_attention and level in self.decoder_attention_levels:
                self.up_attn.append(
                    MaskedWindowAttention3D(
                        ch,
                        heads=heads,
                        window_size=int(window_size),
                        cond_dim=int(cond_dim),
                        mlp_ratio=float(attention_mlp_ratio),
                        dropout=float(dropout),
                        region_pair_bias=float(region_pair_bias),
                    )
                )
            else:
                self.up_attn.append(nn.Identity())
            if level != 0:
                self.upsamples.append(Upsample3D(ch))

        self.out_norm = nn.GroupNorm(_choose_num_groups(ch, norm_groups), ch)
        self.out_conv = nn.Conv3d(ch, int(out_channels), kernel_size=3, padding=1)
        if self.use_boundary_head:
            self.boundary_head = nn.Sequential(
                nn.GroupNorm(_choose_num_groups(ch, norm_groups), ch),
                nn.SiLU(),
                nn.Conv3d(ch, ch, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv3d(ch, int(out_channels), kernel_size=3, padding=1),
            )
        else:
            self.boundary_head = None

    def _heads_for_level(self, level: int, channels: int) -> int:
        if not self.attention_heads:
            return 1
        h = int(self.attention_heads[min(int(level), len(self.attention_heads) - 1)])
        h = max(h, 1)
        while h > 1 and int(channels) % h != 0:
            h -= 1
        return h

    @staticmethod
    def _coords_like(x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape
        zz = torch.linspace(-1.0, 1.0, d, device=x.device, dtype=x.dtype).view(1, 1, d, 1, 1)
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, 1, w)
        zz = zz.expand(b, 1, d, h, w)
        yy = yy.expand(b, 1, d, h, w)
        xx = xx.expand(b, 1, d, h, w)
        return torch.cat([zz, yy, xx], dim=1)

    @staticmethod
    def _ensure_5d(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"{name} must be 5D [B,C,D,H,W], got {tuple(x.shape)}")
        return x

    def _module_forward(self, module: nn.Module, *args):
        if self.use_checkpoint and self.training:
            try:
                return checkpoint(lambda *inner_args: module(*inner_args), *args, use_reentrant=False)
            except (TypeError, ValueError):
                return checkpoint(lambda *inner_args: module(*inner_args), *args)
        return module(*args)

    def _attn_forward(self, module: nn.Module, h: torch.Tensor, region_mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if isinstance(module, MaskedWindowAttention3D):
            return self._module_forward(module, h, region_mask, cond)
        return module(h)

    def forward(
        self,
        x_t: torch.Tensor,
        mri: torch.Tensor,
        brain_mask: torch.Tensor,
        region_mask: torch.Tensor,
        time: torch.Tensor,
        region_label: torch.Tensor,
        table: Optional[torch.Tensor] = None,
        table_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_t = self._ensure_5d(x_t, "x_t")
        mri = self._ensure_5d(mri, "mri")
        brain_mask = self._ensure_5d(brain_mask, "brain_mask")
        region_mask = self._ensure_5d(region_mask, "region_mask")
        dtype = x_t.dtype
        device = x_t.device

        mri = mri.to(device=device, dtype=dtype)
        brain_mask = brain_mask.to(device=device, dtype=dtype)
        region_mask = region_mask.to(device=device, dtype=dtype)
        if brain_mask.shape[2:] != x_t.shape[2:]:
            brain_mask = F.interpolate(brain_mask, size=x_t.shape[2:], mode="nearest")
        if region_mask.shape[2:] != x_t.shape[2:]:
            region_mask = F.interpolate(region_mask, size=x_t.shape[2:], mode="nearest")
        boundary = make_boundary_mask(region_mask).to(dtype=dtype)

        parts = [x_t, mri, brain_mask, region_mask, boundary]
        if self.use_coords:
            parts.append(self._coords_like(x_t))
        x = torch.cat(parts, dim=1)
        h = self.input_proj(x)

        region_label = region_label.to(device=device, dtype=torch.long).clamp(0, self.num_regions)
        time_cond = self.time_embed(time.to(device=device))
        region_cond = self.region_embed(region_label)
        table_cond = self.table_encoder(table, table_mask, region_cond=region_cond)
        cond = self.cond_mlp(time_cond + region_cond + table_cond)

        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = self._module_forward(block, h, cond)
            se = self.down_se[level]
            h = se(h, brain_mask, region_mask) if isinstance(se, MaskGuidedSE3D) else se(h)
            h = self._attn_forward(self.down_attn[level], h, region_mask, cond)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self._module_forward(self.mid1, h, cond)
        h = self._module_forward(self.mid_global_attn, h, cond) if isinstance(self.mid_global_attn, GlobalAttention3D) else self.mid_global_attn(h)
        if self.use_mri_cross_attn:
            mri_ctx = torch.cat([mri, brain_mask, region_mask], dim=1)
            mri_ctx = F.interpolate(mri_ctx, size=h.shape[2:], mode="trilinear", align_corners=False)
            mri_ctx = self.mri_context_proj(mri_ctx)
            h = self._module_forward(self.mid_mri_cross, h, mri_ctx, cond)
        if self.use_cond_cross_attn:
            h = self._module_forward(self.mid_cond_cross, h, cond)
        h = self.mid_se(h, brain_mask, region_mask) if isinstance(self.mid_se, MaskGuidedSE3D) else self.mid_se(h)
        h = self._module_forward(self.mid2, h, cond)

        upsample_idx = 0
        for level, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = self._module_forward(block, h, cond)
            se = self.up_se[level]
            h = se(h, brain_mask, region_mask) if isinstance(se, MaskGuidedSE3D) else se(h)
            h = self._attn_forward(self.up_attn[level], h, region_mask, cond)
            if level != len(self.up_blocks) - 1:
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1

        feat = F.silu(self.out_norm(h))
        residual = self.out_conv(feat)
        if self.boundary_head is not None:
            bd = self.boundary_head(h)
            boundary_gate = boundary
            if boundary_gate.shape[2:] != residual.shape[2:]:
                boundary_gate = F.interpolate(boundary_gate, size=residual.shape[2:], mode="nearest")
            residual = residual + boundary_gate * bd
        return residual
