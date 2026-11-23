from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class SafeGroupNorm(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        channels = int(channels)
        groups = min(int(groups), channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SE3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(int(channels) // int(reduction), 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock3D(nn.Module):
    """Memory-light 3D residual block for full-volume / patch refinement."""

    def __init__(self, channels: int, norm_groups: int = 8, dropout: float = 0.0, dilation: int = 1) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(channels, norm_groups)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = SafeGroupNorm(channels, norm_groups)
        self.dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
            groups=channels,
        )
        self.pw = nn.Conv3d(channels, channels, kernel_size=1)
        self.se = SE3D(channels)
        self.drop = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.dw(F.silu(self.norm2(h)))
        h = self.se(h)
        h = self.drop(h)
        h = self.pw(h)
        return x + h


class Downsample3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_groups: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SafeGroupNorm(in_ch, norm_groups),
            nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpsampleAdd3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            SafeGroupNorm(in_ch, norm_groups),
            nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.skip_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_ch, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.proj(x)
        return x + self.skip_gate(skip) * skip


class GlobalFusionRefiner3D(nn.Module):
    """Exp021b global fusion refiner.

    Input channels are normally:
      local_stitched_pet, MRI, local_stitched_mask_or_weight

    The network predicts a bounded correction instead of directly replacing the
    local PET. This keeps region-wise details from Exp021 and only repairs global
    inconsistency / stitching artifacts.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 12,
        channel_mults: Sequence[int] = (1, 2, 4, 4),
        blocks_per_level: int = 2,
        norm_groups: int = 8,
        dropout: float = 0.05,
        correction_scale: float = 0.15,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.correction_scale = float(correction_scale)
        self.use_checkpoint = bool(use_checkpoint)

        chs = [int(base_channels) * int(m) for m in channel_mults]
        self.stem = nn.Conv3d(self.in_channels, chs[0], kernel_size=3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, ch in enumerate(chs):
            self.enc_blocks.append(
                nn.ModuleList([
                    ResidualBlock3D(
                        ch,
                        norm_groups=int(norm_groups),
                        dropout=float(dropout),
                        dilation=1 if j % 2 == 0 else 2,
                    )
                    for j in range(int(blocks_per_level))
                ])
            )
            if i < len(chs) - 1:
                self.downs.append(Downsample3D(ch, chs[i + 1], norm_groups=int(norm_groups)))

        mid_ch = chs[-1]
        self.mid = nn.ModuleList([
            ResidualBlock3D(mid_ch, int(norm_groups), float(dropout), dilation=2),
            ResidualBlock3D(mid_ch, int(norm_groups), float(dropout), dilation=3),
        ])

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.ups.append(UpsampleAdd3D(chs[i + 1], chs[i], norm_groups=int(norm_groups)))
            self.dec_blocks.append(
                nn.ModuleList([
                    ResidualBlock3D(
                        chs[i],
                        norm_groups=int(norm_groups),
                        dropout=float(dropout),
                        dilation=1 if j % 2 == 0 else 2,
                    )
                    for j in range(int(blocks_per_level))
                ])
            )

        self.final = nn.Sequential(
            SafeGroupNorm(chs[0], int(norm_groups)),
            nn.SiLU(),
            nn.Conv3d(chs[0], self.out_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(lambda _x: block(_x), x, use_reentrant=False)
        return block(x)

    def forward(
        self,
        local_pet: torch.Tensor,
        mri: torch.Tensor,
        stitch_mask: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        if stitch_mask is None:
            stitch_mask = torch.ones_like(local_pet)
        if mri.shape[2:] != local_pet.shape[2:]:
            mri = F.interpolate(mri, size=local_pet.shape[2:], mode="trilinear", align_corners=False)
        if stitch_mask.shape[2:] != local_pet.shape[2:]:
            stitch_mask = F.interpolate(stitch_mask.float(), size=local_pet.shape[2:], mode="nearest")
        mri = mri.to(device=local_pet.device, dtype=local_pet.dtype)
        stitch_mask = stitch_mask.to(device=local_pet.device, dtype=local_pet.dtype)

        x = torch.cat([local_pet, mri, stitch_mask], dim=1)
        h = self.stem(x)
        skips = []
        for level, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                h = self._run_block(block, h)
            skips.append(h)
            if level < len(self.downs):
                h = self.downs[level](h)

        for block in self.mid:
            h = self._run_block(block, h)

        skips = skips[:-1]
        for up, blocks in zip(self.ups, self.dec_blocks):
            h = up(h, skips.pop())
            for block in blocks:
                h = self._run_block(block, h)

        raw = self.final(h)
        correction = self.correction_scale * torch.tanh(raw)
        refined = local_pet + correction
        if return_dict:
            return {"refined": refined, "correction": correction, "raw": raw}
        return refined


# Backward-friendly aliases
UNet = GlobalFusionRefiner3D
Model = GlobalFusionRefiner3D
