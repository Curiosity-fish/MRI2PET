from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from exp.exp037B_v4.exp037B_v4_unet3d import Exp037Bv3T6UNet3D
from models.exp037c_global_local_vit_unet3d import (
    GlobalLocalGatedFusion3D,
    GlobalPyramidUp3D,
    GlobalViT3D,
)


class SelfAttention3D(nn.Module):
    """V2-style full spatial self-attention applied after a stage's ResBlock B."""

    def __init__(self, channels: int, heads: int = 1, norm_groups: int = 8):
        super().__init__()
        heads = max(1, min(int(heads), int(channels)))
        while heads > 1 and int(channels) % heads:
            heads -= 1
        groups = min(int(norm_groups), int(channels))
        while groups > 1 and int(channels) % groups:
            groups -= 1
        self.heads = heads
        self.norm = nn.GroupNorm(groups, int(channels))
        self.qkv = nn.Conv3d(int(channels), int(channels) * 3, 1, bias=False)
        self.out = nn.Conv3d(int(channels), int(channels), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        n = d * h * w
        q = q.reshape(b, self.heads, c // self.heads, n).transpose(-2, -1)
        k = k.reshape(b, self.heads, c // self.heads, n)
        v = v.reshape(b, self.heads, c // self.heads, n).transpose(-2, -1)
        weights = torch.softmax((q @ k) * (c // self.heads) ** -0.5, dim=-1)
        y = (weights @ v).transpose(-2, -1).reshape(b, c, d, h, w)
        return x + self.out(y)


class Exp037EGlobalBSFEUNet3D(Exp037Bv3T6UNet3D):
    """037C with frozen-BSFE residual adapters initialized as exact no-ops."""

    def __init__(
        self,
        *args,
        bsfe_channels=(48, 96, 192, 384),
        global_vit_enabled=False,
        global_token_dim=256,
        global_vit_depth=3,
        global_vit_heads=8,
        global_vit_mlp_ratio=4.0,
        global_vit_dropout=0.0,
        global_vit_attention_dropout=0.0,
        global_fusion_gate_bias=-2.0,
        global_fusion_init_alpha=0.1,
        second_resblock_attention_stages=(),
        second_resblock_attention_heads=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        c0, c1, c2, c3 = [int(x) for x in kwargs.get("channels", (96, 192, 384, 768))]
        stage_modules = {
            "enc2": (self.enc2b, c2),
            "mid": (self.mid2, c3),
            "dec2": (self.dec2b, c2),
        }
        requested_stages = tuple(second_resblock_attention_stages)
        unknown_stages = set(requested_stages) - set(stage_modules)
        if unknown_stages:
            raise ValueError(f"Unknown second-ResBlock attention stages: {sorted(unknown_stages)}")
        self.second_resblock_attention = nn.ModuleDict({
            name: SelfAttention3D(channels, second_resblock_attention_heads, kwargs.get("norm_groups", 8))
            for name, (_module, channels) in stage_modules.items()
            if name in requested_stages
        })
        self._second_resblock_attention_hooks = []
        for name, attention in self.second_resblock_attention.items():
            module, _channels = stage_modules[name]
            self._second_resblock_attention_hooks.append(
                module.register_forward_hook(lambda _module, _inputs, output, attn=attention: attn(output))
            )

        stage_channels = (c1, c2, c3, c2, c1, c0)
        # 037C has no BSFE adapter at Enc0; align the remaining stages with
        # the 036C pyramid mapping: Enc1, Enc2, Mid, Dec2, Dec1, Dec0.
        source_channels = (bsfe_channels[1], bsfe_channels[2], bsfe_channels[3], bsfe_channels[2], bsfe_channels[1], bsfe_channels[0])
        self.bsfe_zero = nn.ModuleList([nn.Conv3d(si, so, 1) for si, so in zip(source_channels, stage_channels)])
        for layer in self.bsfe_zero:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self._bsfe_pyramid = None
        self._bsfe_hooks = []
        for module, idx in ((self.enc1b, 0), (self.enc2b, 1), (self.mid2, 2), (self.dec2b, 3), (self.dec1b, 4), (self.dec0b, 5)):
            self._bsfe_hooks.append(module.register_forward_hook(self._make_hook(idx)))

        self.global_vit_enabled = bool(global_vit_enabled)
        self._global_cache = {}
        self._global_hooks = []
        if self.global_vit_enabled:
            self.global_vit = GlobalViT3D(
                c3, token_dim=global_token_dim, depth=global_vit_depth,
                heads=global_vit_heads, mlp_ratio=global_vit_mlp_ratio,
                dropout=global_vit_dropout,
                attention_dropout=global_vit_attention_dropout,
                bottleneck_shape=(5, 6, 5),
            )
            self.global_up2 = GlobalPyramidUp3D(c3, c2, kwargs.get("norm_groups", 8))
            self.global_up1 = GlobalPyramidUp3D(c2, c1, kwargs.get("norm_groups", 8))
            self.fuse_mid = GlobalLocalGatedFusion3D(c3, global_fusion_gate_bias, global_fusion_init_alpha)
            self.fuse_enc2 = GlobalLocalGatedFusion3D(c2, global_fusion_gate_bias, global_fusion_init_alpha)
            self.fuse_dec2 = GlobalLocalGatedFusion3D(c2, global_fusion_gate_bias, global_fusion_init_alpha)
            self.fuse_enc1 = GlobalLocalGatedFusion3D(c1, global_fusion_gate_bias, global_fusion_init_alpha)
            self.fuse_dec1 = GlobalLocalGatedFusion3D(c1, global_fusion_gate_bias, global_fusion_init_alpha)
            self.global_head_g3 = nn.Conv3d(c3, 4, 1)
            self.global_head_g2 = nn.Conv3d(c2, 4, 1)
            self._global_hooks = [
                self.up2.register_forward_pre_hook(self._global_mid_hook),
                self.dec2a.register_forward_pre_hook(self._global_enc2_hook),
                self.dec2b.register_forward_hook(self._global_dec2_hook),
                self.dec1a.register_forward_pre_hook(self._global_enc1_hook),
                self.dec1b.register_forward_hook(self._global_dec1_hook),
            ]

    def _make_hook(self, idx):
        def hook(_module, _inputs, output):
            if self._bsfe_pyramid is None:
                return output
            pyramid_idx = (1, 2, 3, 2, 1, 0)[idx]
            source = self._bsfe_pyramid[min(pyramid_idx, len(self._bsfe_pyramid) - 1)]
            if source.shape[-3:] != output.shape[-3:]:
                source = F.interpolate(source, size=output.shape[-3:], mode="trilinear", align_corners=False)
            return output + self.bsfe_zero[idx](source.to(dtype=output.dtype))
        return hook

    def _global_mid_hook(self, _module, inputs):
        h = inputs[0]
        g3 = self.global_vit(h)
        g2 = self.global_up2(g3)
        g1 = self.global_up1(g2)
        h, gate = self.fuse_mid(h, g3, return_gate=True)
        self._global_cache = {"g3": g3, "g2": g2, "g1": g1, "gate_mid": gate}
        return (h,)

    def _global_enc2_hook(self, _module, inputs):
        x, te = inputs
        c = x.shape[1] // 2
        skip, gate = self.fuse_enc2(x[:, c:], self._global_cache["g2"], return_gate=True)
        self._global_cache["gate_enc2"] = gate
        return (torch.cat([x[:, :c], skip], dim=1), te)

    def _global_dec2_hook(self, _module, _inputs, output):
        output, gate = self.fuse_dec2(output, self._global_cache["g2"], return_gate=True)
        self._global_cache["gate_dec2"] = gate
        return output

    def _global_enc1_hook(self, _module, inputs):
        x, te = inputs
        c = x.shape[1] // 2
        skip, gate = self.fuse_enc1(x[:, c:], self._global_cache["g1"], return_gate=True)
        self._global_cache["gate_enc1"] = gate
        return (torch.cat([x[:, :c], skip], dim=1), te)

    def _global_dec1_hook(self, _module, _inputs, output):
        output, gate = self.fuse_dec1(output, self._global_cache["g1"], return_gate=True)
        self._global_cache["gate_dec1"] = gate
        return output

    def forward(self, x1, t, table, table_mask, *, bsfe_pyramid=None, return_aux=False, disable_table=False):
        self._bsfe_pyramid = bsfe_pyramid
        self._global_cache = {}
        try:
            result = super().forward(x1, t, table, table_mask, return_aux=return_aux, disable_table=disable_table)
            if not return_aux or not self.global_vit_enabled:
                return result
            out, aux = result
            g = self._global_cache
            aux.update({
                "global_pred_g3": self.global_head_g3(g["g3"]),
                "global_pred_g2": self.global_head_g2(g["g2"]),
                "global_g3_norm": g["g3"].float().flatten(1).norm(dim=1).mean(),
                "global_g2_norm": g["g2"].float().flatten(1).norm(dim=1).mean(),
                "global_g1_norm": g["g1"].float().flatten(1).norm(dim=1).mean(),
                **{key: g[key] for key in ("gate_mid", "gate_enc2", "gate_dec2", "gate_enc1", "gate_dec1")},
            })
            return out, aux
        finally:
            self._bsfe_pyramid = None
