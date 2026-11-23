# model/bsfe_pet_head.py
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from model.BSFE import ResBlock3D


class _FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = feat_dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=1)
        return (1.0 + gamma[:, :, None, None, None]) * x + beta[:, :, None, None, None]


class _MergeRefine(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_groups: int = 16, dropout: float = 0.0, n_blocks: int = 2):
        super().__init__()
        self.merge = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.refine = nn.Sequential(
            *[ResBlock3D(out_ch, out_ch, norm_groups=norm_groups, dropout=dropout) for _ in range(int(n_blocks))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.merge(x))


class UpsampleDeconv3D(nn.Module):
    """ConvTranspose upsample with configurable in/out channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_ch, out_ch, 4, 2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _match_spatial(x: torch.Tensor, ref_spatial: Tuple[int, int, int]) -> torch.Tensor:
    """Center-crop or symmetric pad x to match ref spatial size."""
    rd, rh, rw = ref_spatial
    xd, xh, xw = x.shape[-3:]

    # crop
    if xd > rd:
        s = (xd - rd) // 2
        x = x[:, :, s:s + rd, :, :]
    if xh > rh:
        s = (xh - rh) // 2
        x = x[:, :, :, s:s + rh, :]
    if xw > rw:
        s = (xw - rw) // 2
        x = x[:, :, :, :, s:s + rw]

    # pad
    pd = max(rd - x.shape[-3], 0)
    ph = max(rh - x.shape[-2], 0)
    pw = max(rw - x.shape[-1], 0)
    if pd or ph or pw:
        x = F.pad(
            x,
            (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2),
            mode="constant",
            value=0.0,
        )
    return x


class BSFEPetLatentHead(nn.Module):
    """
    UNet-style decoder head:
      - pyramid decode: Fk(low-res) -> ... -> F1(24,40,40) with skip-concat ALL pyramid
      - extra 2 upsample stages: 24,40,40 -> 48,80,80 -> 96,160,160
      - output PET-space tensor (B, pet_out_channels, 96,160,160)  (default)

    decoder_channels: length k, channels aligned with [F1..Fk] (high-res -> low-res)
    extra_up_channels: length extra_upsamples, channels after each extra upsample (low->high)
    """
    def __init__(
        self,
        in_channels_list: List[int],
        global_dim: int,
        latent_dim: int,  # 保留参数名不破坏旧调用（此处不一定等于 pet_out_channels）

        decoder_channels: Optional[List[int]] = None,   # [F1..Fk] 通道
        n_fuse_blocks: int = 2,
        out_blocks: int = 1,

        # extra upsample to PET space
        extra_upsamples: int = 2,
        extra_up_channels: Optional[List[int]] = None,  # 例如 [64, 32]
        pet_out_channels: int = 1,
        pet_target_spatial: Tuple[int, int, int] = (96, 160, 160),

        norm_groups: int = 16,
        dropout: float = 0.0,

        use_global_film: bool = True,
        token_dim: int = 128,
        use_token_film: bool = True,

        use_token_cross_attn: bool = True,
        attn_heads: int = 4,
        max_cross_tokens: int = 4096,
    ):
        super().__init__()
        n_levels = len(in_channels_list)
        assert n_levels >= 1

        if decoder_channels is None:
            decoder_channels = [96] * n_levels
        assert len(decoder_channels) == n_levels, "decoder_channels length must equal pyramid levels."

        self.dec_ch = [int(c) for c in decoder_channels]  # aligned with F1..Fk
        self.use_token_cross_attn = bool(use_token_cross_attn)
        self.max_cross_tokens = int(max_cross_tokens)

        self.extra_upsamples = int(extra_upsamples)
        self.pet_out_channels = int(pet_out_channels)
        self.pet_target_spatial = tuple(int(x) for x in pet_target_spatial)

        if extra_up_channels is None:
            # 默认：逐步变窄，利于高分辨率细节
            extra_up_channels = [max(self.dec_ch[0] // 2, 32), max(self.dec_ch[0] // 4, 16)]
        assert len(extra_up_channels) == self.extra_upsamples, "extra_up_channels length must equal extra_upsamples."
        self.extra_up_channels = [int(c) for c in extra_up_channels]

        # lateral: Fi -> dec_ch[i]
        self.lateral = nn.ModuleList([
            nn.Conv3d(int(in_channels_list[i]), self.dec_ch[i], kernel_size=1)
            for i in range(n_levels)
        ])

        # bottleneck channels = dec_ch[-1]
        bott_ch = self.dec_ch[-1]
        self.mid1 = ResBlock3D(bott_ch, bott_ch, norm_groups=norm_groups, dropout=dropout)
        self.mid2 = ResBlock3D(bott_ch, bott_ch, norm_groups=norm_groups, dropout=dropout)

        # per-channel FiLM creators (because channels change across stages)
        self.use_global_film = bool(use_global_film)
        self.use_token_film = bool(use_token_film)
        if self.use_global_film:
            self.global_film = nn.ModuleList([_FiLM(global_dim, ch) for ch in self.dec_ch] +
                                             [_FiLM(global_dim, ch) for ch in self.extra_up_channels])
        else:
            self.global_film = None
        if self.use_token_film:
            self.token_film = nn.ModuleList([_FiLM(token_dim, ch) for ch in self.dec_ch] +
                                            [_FiLM(token_dim, ch) for ch in self.extra_up_channels])
        else:
            self.token_film = None

        # token cross-attn at bottleneck only
        if self.use_token_cross_attn:
            assert bott_ch % int(attn_heads) == 0, f"bottleneck channels {bott_ch} must be divisible by attn_heads {attn_heads}"
            self.token_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, bott_ch))
            self.q_ln = nn.LayerNorm(bott_ch)
            self.kv_ln = nn.LayerNorm(bott_ch)
            self.cross_attn = nn.MultiheadAttention(embed_dim=bott_ch, num_heads=int(attn_heads), batch_first=True)
        else:
            self.token_proj = self.q_ln = self.kv_ln = self.cross_attn = None

        # decoder for pyramid: from level+1 -> level
        self.up_layers = nn.ModuleList()
        self.decode_blocks = nn.ModuleList()
        for level in range(n_levels - 2, -1, -1):
            in_ch = self.dec_ch[level + 1]
            out_ch = self.dec_ch[level]
            self.up_layers.append(UpsampleDeconv3D(in_ch, out_ch))
            self.decode_blocks.append(
                _MergeRefine(
                    in_ch=out_ch + out_ch,  # up(out_ch) concat skip(out_ch)
                    out_ch=out_ch,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    n_blocks=n_fuse_blocks,
                )
            )

        # refine at F1
        f1_ch = self.dec_ch[0]
        self.out_refine = nn.Sequential(
            *[ResBlock3D(f1_ch, f1_ch, norm_groups=norm_groups, dropout=dropout) for _ in range(int(out_blocks))]
        )

        # extra upsample to PET space (2 stages)
        self.extra_up = nn.ModuleList()
        self.extra_refine = nn.ModuleList()
        cur_ch = f1_ch
        for j in range(self.extra_upsamples):
            nxt_ch = self.extra_up_channels[j]
            self.extra_up.append(UpsampleDeconv3D(cur_ch, nxt_ch))
            self.extra_refine.append(
                nn.Sequential(
                    ResBlock3D(nxt_ch, nxt_ch, norm_groups=norm_groups, dropout=dropout),
                    ResBlock3D(nxt_ch, nxt_ch, norm_groups=norm_groups, dropout=dropout),
                )
            )
            cur_ch = nxt_ch

        # final PET output
        self.pet_out = nn.Conv3d(cur_ch, self.pet_out_channels, kernel_size=3, padding=1)

        # 保留一个 latent_out（可选），如果你还想同时输出 latent 监督，可自己在 forward 里加返回
        self.latent_out = nn.Conv3d(f1_ch, int(latent_dim), kernel_size=3, padding=1)

    def _apply_film(self, x: torch.Tensor, idx: int, global_vec: torch.Tensor, tokens: Optional[torch.Tensor]) -> torch.Tensor:
        # idx：0..(len(dec_ch)+len(extra_up_channels)-1)
        if self.global_film is not None:
            x = self.global_film[idx](x, global_vec)
        if (self.token_film is not None) and (tokens is not None):
            x = self.token_film[idx](x, tokens.mean(dim=1))
        return x

    def _bottleneck_attn(self, x: torch.Tensor, tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if (not self.use_token_cross_attn) or (tokens is None) or (self.cross_attn is None):
            return x
        B, C, D, H, W = x.shape
        if D * H * W > self.max_cross_tokens:
            return x
        q = x.flatten(2).transpose(1, 2).contiguous()
        q = self.q_ln(q)
        kv = self.kv_ln(self.token_proj(tokens))
        out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        out = out.transpose(1, 2).contiguous().view(B, C, D, H, W)
        return x + out

    def forward(self, pyramid: List[torch.Tensor], global_vec: torch.Tensor, tokens: Optional[torch.Tensor] = None):
        # pyramid: [F1(highest-res), ..., Fk(lowest-res)]
        n_levels = len(self.lateral)
        assert len(pyramid) >= n_levels

        # start from lowest-res
        level_low = n_levels - 1
        x = self.lateral[level_low](pyramid[level_low])

        # bottleneck mid
        x = self.mid1(x)
        x = self._apply_film(x, idx=level_low, global_vec=global_vec, tokens=tokens)
        x = self._bottleneck_attn(x, tokens)
        x = self.mid2(x)
        x = self._apply_film(x, idx=level_low, global_vec=global_vec, tokens=tokens)

        # pyramid decode: use all skips
        bi = 0
        for level in range(n_levels - 2, -1, -1):
            skip = self.lateral[level](pyramid[level])
            x = self.up_layers[bi](x)
            x = _match_spatial(x, skip.shape[-3:])
            x = torch.cat([x, skip], dim=1)
            x = self.decode_blocks[bi](x)
            x = self._apply_film(x, idx=level, global_vec=global_vec, tokens=tokens)
            bi += 1

        # refine at F1
        x = self.out_refine(x)

        # extra upsample 2x: 24->48->96 / 40->80->160 / 40->80->160
        for j in range(self.extra_upsamples):
            x = self.extra_up[j](x)

        pet_hat = self.pet_out(x)  # (B,1,96,160,160)

        # 如果你还想同时监督 latent，可返回 latent_hat：
        # latent_hat = self.latent_out(x_at_f1)  # 需要你保留 F1 的特征
        return pet_hat


if __name__ == "__main__":
    from BSFE_v3 import *
    cfg = BSFEConfig(out_context_dim=None)
    net = BrainStructuralFeatureExtractor(cfg)
    x = torch.randn(2, 1, 160, 160, 96)
    y = net(x)
    print("pyramid:", [t.shape for t in y["pyramid"]])
    print("tokens:", y["tokens"].shape, "global:", y["global"].shape)

    head = BSFEPetLatentHead(
        in_channels_list=[32, 64, 128, 256],  # BSFE pyramid 原通道
        global_dim=256,
        latent_dim=1,
        decoder_channels=[32, 64, 128, 256],  # 你自定义：F1..F4 的通道（高->低）
        n_fuse_blocks=2,
        use_token_cross_attn=True,
        attn_heads=4,
    )

    z = head(y["pyramid"], y["global"], y["tokens"])
    print(z.shape)
