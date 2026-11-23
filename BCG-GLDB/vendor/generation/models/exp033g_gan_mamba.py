"""Exp033g: GAN-Mamba for MRI->PET (port of baseline Exp151_ganmamba).

Method (from baseline/Exp151_ganmamba_t1mri_fdgpet/gan_mamba.py + aee.py):
- Stage 1: train MRI and PET autoencoders (AAE) in image space with MSE
  reconstruction (Encoder -> 4ch latent at image/4, Decoder -> image).
- Stage 2: GAN training. Generator = frozen MRI-AE encoder (no-grad) +
  trainable Mamba U-Net ("GEN") operating in the 4ch latent space +
  fine-tunable PET-AE decoder. Discriminator = PatchGAN CNN. Loss:
  hinge adversarial + MSE reconstruction.

Adaptations for Exp033g:
- Table conditioning (cate/cont features) is dropped. In the original
  baseline the table cross-attention is only used when `attn_res` is
  non-empty, and the default (attn_res=()) never activates it, so the
  image-only variant keeps the full effective architecture.
- The original Mamba SSM was a per-timestep Python loop (~76k steps for a
  40x48x40 latent), which took ~110 min/epoch for GAN. Here we use the
  CUDA mamba_ssm kernel (same SSM semantics, dramatically faster).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

try:  # CUDA kernel
    from mamba_ssm import Mamba as MambaSSM
except Exception:  # pragma: no cover - fallback for CPU-only import checks
    MambaSSM = None


def valid_group_count(num_channels: int, requested_groups: int) -> int:
    requested_groups = min(int(requested_groups), int(num_channels))
    for g in range(requested_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm(nn.Module):
    def __init__(self, channels, num_proup=16):
        super().__init__()
        groups = valid_group_count(channels, num_proup)
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


# ---------------------------------------------------------------------------
# AAE (autoencoder) from baseline/aee.py
# ---------------------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim, 4, 2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 4, 2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AAResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
        )
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block(x)


class AAEncoder(nn.Module):
    def __init__(self, image_channels: int = 1, output_channels: int = 4) -> None:
        super().__init__()
        channels = [16, 32, 64]
        layers: list[nn.Module] = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for idx in range(len(channels) - 1):
            layers.append(AAResidualBlock(channels[idx], channels[idx + 1]))
            layers.append(Downsample(channels[idx + 1]))
        layers.extend(
            [
                AAResidualBlock(channels[-1], channels[-1]),
                GroupNorm(channels[-1]),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels[-1], output_channels, 3, 1, 1),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AADecoder(nn.Module):
    def __init__(self, image_channels: int = 1, input_channels: int = 4) -> None:
        super().__init__()
        channels = [64, 32, 16]
        in_channels = channels[0]
        layers: list[nn.Module] = [
            nn.Conv3d(input_channels, in_channels, 3, 1, 1),
            AAResidualBlock(in_channels, in_channels),
            AAResidualBlock(in_channels, channels[0]),
            Upsample(channels[0]),
            AAResidualBlock(channels[0], channels[1]),
            Upsample(channels[1]),
            AAResidualBlock(channels[1], channels[2]),
            GroupNorm(channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels[2], image_channels, 3, 1, 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AAE(nn.Module):
    def __init__(self, latent_dim: int = 4) -> None:
        super().__init__()
        self.encoder = AAEncoder(output_channels=latent_dim)
        self.decoder = AADecoder(input_channels=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Mamba U-Net generator (from baseline/gan_mamba.py, table removed)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        groups = valid_group_count(dim, groups)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        return self.block2(self.block1(x)) + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()
        self.n_head = n_head
        groups = valid_group_count(in_channel, norm_groups)
        self.norm = nn.GroupNorm(groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, kernel_size=1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, kernel_size=1)

    def forward(self, input):
        batch, channel, height, width, depth = input.shape
        head_dim = channel // self.n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, self.n_head, head_dim * 3, height, width, depth)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum("bnchwd,bncyxz->bnhwdyxz", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, self.n_head, height, width, depth, -1)
        attn = torch.softmax(attn, dim=-1)
        attn = attn.view(batch, self.n_head, height, width, depth, height, width, depth)
        out = torch.einsum("bnhwdyxz,bncyxz->bnchwd", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width, depth))
        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        self.attn = SelfAttention(dim_out, norm_groups=norm_groups) if with_attn else None

    def forward(self, x):
        x = self.res_block(x)
        if self.with_attn:
            x = self.attn(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


@dataclass
class MambaConfig:
    d_model: int
    n_layers: int = 2
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4


class Mamba3DBottleneck(nn.Module):
    """Global Mamba SSM over the full latent volume, using the CUDA kernel.

    Replaces the per-timestep python-loop SSM of the baseline (same SSM
    semantics; O(L) scan with the mamba_ssm kernel instead of O(L) python
    iterations).
    """

    def __init__(self, dim: int, depth: int = 1, d_state: int = 16, expand_factor: int = 2,
                 d_conv: int = 4, dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.local_pos = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        if MambaSSM is None:
            raise RuntimeError('mamba_ssm (CUDA) is required for Exp033g')
        self.forward_mamba = MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand_factor)
        self.backward_mamba = MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand_factor) if bidirectional else None
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)
        self.local_out = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w, d = x.shape
        residual = x
        tokens = rearrange(x + self.local_pos(x), "b c h w d -> b (h w d) c")
        y_fwd = self.forward_mamba(tokens)
        if self.bidirectional:
            y_bwd = torch.flip(self.backward_mamba(torch.flip(tokens, dims=[1])), dims=[1])
            tokens = 0.5 * (y_fwd + y_bwd)
        else:
            tokens = y_fwd
        x = rearrange(self.out_proj(self.dropout(self.out_norm(tokens))), "b (h w d) c -> b c h w d", h=h, w=w, d=d)
        return self.local_out(x) + x + residual


class GEN(nn.Module):
    """Latent-space Mamba U-Net generator (baseline GEN, table removed)."""

    def __init__(self, latent_dim, inner_channel=8, norm_groups=8, channel_mults=(1, 2),
                 attn_res=(), res_blocks=1, dropout=0.0, image_size=40,
                 mamba_d_state=16, mamba_expand_factor=2, mamba_d_conv=4,
                 mamba_depth=1, mamba_bidirectional=False):
        super().__init__()
        in_channel = latent_dim * 2
        out_channel = latent_dim
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv3d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            use_attn = now_res in attn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                downs.append(ResnetBlocWithAttn(pre_channel, channel_mult, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res //= 2
        self.downs = nn.ModuleList(downs)
        self.mid = Mamba3DBottleneck(
            dim=pre_channel, depth=mamba_depth, d_state=mamba_d_state,
            expand_factor=mamba_expand_factor, d_conv=mamba_d_conv, dropout=dropout,
            bidirectional=mamba_bidirectional,
        )
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = ind < 1
            use_attn = now_res in attn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks + 1):
                ups.append(ResnetBlocWithAttn(pre_channel + feat_channels.pop(), channel_mult, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res *= 2
        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, default_out(out_channel, in_channel), groups=norm_groups)
        self.cond = nn.Sequential(
            nn.Conv3d(1, latent_dim * 2, kernel_size=4, stride=2, padding=1),
            AAResidualBlock(latent_dim * 2, latent_dim * 2),
            GroupNorm(latent_dim * 2, num_proup=latent_dim),
            Swish(),
            nn.Conv3d(latent_dim * 2, latent_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x, mri):
        x = self.cond(x)
        if x.shape[-3:] != mri.shape[-3:]:
            x = F.interpolate(x, size=mri.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, mri], dim=1)

        feats = []
        for layer in self.downs:
            x = layer(x)
            feats.append(x)

        x = self.mid(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                skip = feats.pop()
                if x.shape[-3:] != skip.shape[-3:]:
                    x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
                x = layer(torch.cat((x, skip), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x)


def default_out(out_channel, in_channel):
    return out_channel


class GANMamba(nn.Module):
    """Frozen MRI encoder + trainable GEN + fine-tunable PET decoder."""

    def __init__(self, latent_dim, encoder, decoder, freeze_mri_encoder=True, **gen_kwargs):
        super().__init__()
        self.mri_encoder = encoder
        self.freeze_mri_encoder = freeze_mri_encoder
        self.gen = GEN(latent_dim=latent_dim, **gen_kwargs)
        self.decoder = decoder
        if freeze_mri_encoder:
            for p in self.mri_encoder.parameters():
                p.requires_grad = False

    def forward(self, data):
        if self.freeze_mri_encoder:
            self.mri_encoder.eval()
            with torch.no_grad():
                mri = self.mri_encoder(data)
        else:
            mri = self.mri_encoder(data)
        return self.decoder(self.gen(data, mri))


class Discriminator(nn.Module):
    def __init__(self, image_channels=1, channels=(8, 16, 32, 64)):
        super().__init__()
        layers = [
            nn.Conv3d(image_channels, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels[0], channels[1], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels[1], channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels[2], channels[3], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels[3], image_channels, 4, 2, 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
