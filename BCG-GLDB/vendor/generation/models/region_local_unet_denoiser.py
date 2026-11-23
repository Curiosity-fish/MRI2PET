from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GroupNorm(nn.Module):
    def __init__(self, channels: int, num_groups: int = 16):
        super().__init__()
        channels = int(channels)
        groups = min(int(num_groups), channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: Optional[int] = None):
        super().__init__()
        out_ch = int(out_ch) if out_ch is not None else int(in_ch)
        self.conv = nn.Conv3d(int(in_ch), out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: Optional[int] = None):
        super().__init__()
        out_ch = int(out_ch) if out_ch is not None else int(in_ch)
        self.conv = nn.ConvTranspose3d(int(in_ch), out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t is None:
            raise ValueError("time/t must be provided for RegionLocalUNetDenoiser.")
        if t.ndim > 1:
            t = t.view(t.shape[0], -1)[:, 0]
        device = t.device
        half = self.dim // 2
        emb_scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-emb_scale))
        emb = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeRegionEmbedding(nn.Module):
    def __init__(self, time_dim: int = 128, num_regions: int = 18):
        super().__init__()
        self.time_dim = int(time_dim)
        self.num_regions = int(num_regions)
        self.time_embed = TimeEmbedding(self.time_dim)
        self.region_embed = nn.Embedding(self.num_regions + 1, self.time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.time_dim * 2, self.time_dim * 4),
            Swish(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )

    def _sanitize_region_label(
        self,
        region_label: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if region_label is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not torch.is_tensor(region_label):
            region_label = torch.as_tensor(region_label, device=device)
        region_label = region_label.to(device=device)
        if region_label.ndim > 1:
            region_label = region_label.view(region_label.shape[0], -1)[:, 0]
        region_label = region_label.long()
        if region_label.shape[0] != batch_size:
            raise ValueError(f"region_label batch size mismatch: got {region_label.shape[0]}, expected {batch_size}")
        if region_label.min().item() < 0 or region_label.max().item() > self.num_regions:
            raise ValueError(
                f"region_label must be in [0,{self.num_regions}] so that both 0-based and 1-based ids work; "
                f"got min={region_label.min().item()}, max={region_label.max().item()}"
            )
        return region_label

    def forward(
        self,
        time: torch.Tensor,
        region_label: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_emb = self.time_embed(time)
        r = self._sanitize_region_label(region_label, batch_size, device)
        r_emb = self.region_embed(r)
        return self.mlp(torch.cat([t_emb, r_emb], dim=-1))


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, norm_groups: int = 16, dropout: float = 0.0):
        super().__init__()
        self.norm = GroupNorm(dim, num_groups=norm_groups)
        self.act = Swish()
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.conv = nn.Conv3d(int(dim), int(dim_out), kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        x = self.act(x)
        x = self.drop(x)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, norm_groups: int = 16, dropout: float = 0.0):
        super().__init__()
        self.block1 = Block(dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, norm_groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(int(dim), int(dim_out), kernel_size=1) if int(dim) != int(dim_out) else nn.Identity()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        h = self.block1(x, scale_shift)
        h = self.block2(h, scale_shift)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel: int, n_head: int = 1, norm_groups: int = 16):
        super().__init__()
        self.in_channel = int(in_channel)
        heads = max(1, min(int(n_head), self.in_channel))
        while heads > 1 and self.in_channel % heads != 0:
            heads -= 1
        self.n_head = heads
        self.norm = GroupNorm(self.in_channel, num_groups=norm_groups)
        self.qkv = nn.Conv3d(self.in_channel, self.in_channel * 3, kernel_size=1, bias=False)
        self.out = nn.Conv3d(self.in_channel, self.in_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        n = d * h * w
        q = q.reshape(b, self.n_head, c // self.n_head, n).permute(0, 1, 3, 2)
        k = k.reshape(b, self.n_head, c // self.n_head, n)
        v = v.reshape(b, self.n_head, c // self.n_head, n).permute(0, 1, 3, 2)
        attn = torch.matmul(q, k) * (c // self.n_head) ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(b, c, d, h, w)
        return x + self.out(out)


class TableTokenizer(nn.Module):
    def __init__(self, num_features: int, token_dim: int = 128):
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)
        self.feat_embed = nn.Embedding(self.num_features, self.token_dim)
        self.val_proj = nn.Sequential(nn.Linear(1, self.token_dim), Swish(), nn.Linear(self.token_dim, self.token_dim))
        self.miss_proj = nn.Sequential(nn.Linear(1, self.token_dim), Swish(), nn.Linear(self.token_dim, self.token_dim))
        self.miss_ratio_proj = nn.Sequential(
            nn.Linear(1, self.token_dim),
            Swish(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.ln = nn.LayerNorm(self.token_dim)

    def forward(self, values: Optional[torch.Tensor], miss_mask: Optional[torch.Tensor] = None):
        if values is None:
            return None, None, None
        if values.ndim != 2:
            raise ValueError(f"table values must be (B,F), got {tuple(values.shape)}")
        b, f = values.shape
        if f != self.num_features:
            raise ValueError(
                f"table_num_features={self.num_features}, but got F={f}. "
                "Please set table_num_features to match your selected table columns."
            )
        values = values.float()
        if miss_mask is None:
            miss_mask = torch.zeros_like(values)
        miss_mask = miss_mask.float().to(device=values.device)
        if miss_mask.shape != values.shape:
            raise ValueError(f"miss_mask must match values shape: values={tuple(values.shape)}, miss={tuple(miss_mask.shape)}")
        feat_ids = torch.arange(self.num_features, device=values.device)
        feat_e = self.feat_embed(feat_ids)[None, :, :].expand(b, -1, -1)
        val_e = self.val_proj(values.unsqueeze(-1))
        miss_e = self.miss_proj(miss_mask.unsqueeze(-1))
        tokens = self.ln(feat_e + val_e + miss_e)
        key_padding_mask = miss_mask > 0.5
        valid = (~key_padding_mask).float()
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_vec = (tokens * valid.unsqueeze(-1)).sum(dim=1) / denom
        miss_ratio = miss_mask.mean(dim=1, keepdim=True)
        global_vec = self.ln(global_vec + self.miss_ratio_proj(miss_ratio))
        return tokens, key_padding_mask, global_vec


class GatedCrossAttention3D(nn.Module):
    def __init__(self, in_channel: int, table_token_dim: int = 128, n_head: int = 4, norm_groups: int = 16):
        super().__init__()
        self.in_channel = int(in_channel)
        self.table_token_dim = int(table_token_dim)
        heads = max(1, min(int(n_head), self.in_channel))
        while heads > 1 and self.in_channel % heads != 0:
            heads -= 1
        self.n_head = heads
        self.norm = GroupNorm(self.in_channel, num_groups=norm_groups)
        self.ctx_proj = nn.Linear(self.table_token_dim, self.in_channel)
        self.attn = nn.MultiheadAttention(embed_dim=self.in_channel, num_heads=self.n_head, batch_first=True)
        self.out_conv = nn.Conv3d(self.in_channel, self.in_channel, kernel_size=1)
        self.gate_proj = nn.Linear(self.table_token_dim, self.in_channel)
        self.x_gate_proj = nn.Linear(self.in_channel, self.in_channel)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        nn.init.zeros_(self.x_gate_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        table_tokens: Optional[torch.Tensor],
        table_key_padding_mask: Optional[torch.Tensor],
        table_global: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if table_tokens is None or table_global is None:
            return x
        b, c, d, h, w = x.shape
        x_n = self.norm(x)
        q = x_n.flatten(2).transpose(1, 2)
        kv = self.ctx_proj(table_tokens)
        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=table_key_padding_mask,
            need_weights=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, c, d, h, w)
        attn_out = self.out_conv(attn_out)
        x_pool = x_n.mean(dim=(2, 3, 4))
        gate = torch.sigmoid(self.gate_proj(table_global) + self.x_gate_proj(x_pool)).view(b, c, 1, 1, 1)
        return x + gate.to(dtype=x.dtype) * attn_out


class ResnetBlockWithAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        norm_groups: int = 16,
        dropout: float = 0.1,
        cond_emb_dim: int = 128,
        with_self_attn: bool = False,
        with_table_gca: bool = False,
        table_token_dim: int = 128,
        table_cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.res1 = ResnetBlock(dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        self.res2 = ResnetBlock(dim_out, dim_out, norm_groups=norm_groups, dropout=dropout)
        self.cond_mlp = nn.Sequential(Swish(), nn.Linear(int(cond_emb_dim), int(dim_out))) if cond_emb_dim else None
        self.self_attn = SelfAttention(dim_out, n_head=1, norm_groups=norm_groups) if with_self_attn else None
        self.table_gca = (
            GatedCrossAttention3D(
                in_channel=dim_out,
                table_token_dim=table_token_dim,
                n_head=table_cross_attn_heads,
                norm_groups=norm_groups,
            )
            if with_table_gca
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
        table_tokens: Optional[torch.Tensor] = None,
        table_key_padding_mask: Optional[torch.Tensor] = None,
        table_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.res1(x)
        if self.cond_mlp is not None and cond_emb is not None:
            x = x + self.cond_mlp(cond_emb)[:, :, None, None, None].to(dtype=x.dtype)
        x = self.res2(x)
        if self.self_attn is not None:
            x = self.self_attn(x)
        if self.table_gca is not None:
            x = self.table_gca(x, table_tokens, table_key_padding_mask, table_global)
        return x


def _looks_like_binary_mask(x: torch.Tensor) -> bool:
    if x is None or x.numel() == 0:
        return False
    x_det = x.detach()
    x_min = float(x_det.min().cpu())
    x_max = float(x_det.max().cpu())
    if x_min < -0.05 or x_max > 1.05:
        return False
    return (x_det - x_det.round()).abs().max().cpu().item() < 1e-3


def _center_crop_or_pad_to(x: torch.Tensor, target_spatial: Tuple[int, int, int]) -> torch.Tensor:
    td, th, tw = tuple(int(v) for v in target_spatial)
    _, _, d, h, w = x.shape
    sd = max((d - td) // 2, 0)
    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)
    x = x[:, :, sd:sd + min(d, td), sh:sh + min(h, th), sw:sw + min(w, tw)]
    pd = td - x.shape[2]
    ph = th - x.shape[3]
    pw = tw - x.shape[4]
    if pd > 0 or ph > 0 or pw > 0:
        x = F.pad(
            x,
            [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2],
            mode="constant",
            value=0,
        )
    return x


class RegionLocalUNetDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1,
        mri_in_channels: int = 1,
        inner_channel: int = 32,
        norm_groups: int = 16,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        attn_res: Tuple[int, ...] = (16,),
        res_blocks: int = 1,
        dropout: float = 0.1,
        image_size: int = 64,
        time_dim: int = 128,
        num_regions: int = 18,
        self_attention_mode: str = "pre_sampling_only",
        table_num_features: int = 17,
        table_token_dim: int = 128,
        table_cross_attn_heads: int = 4,
        table_cross_attn_res: Optional[Tuple[int, ...]] = None,
        table_has_missing_mask_in_conti: bool = False,
        use_table_cross_attn: bool = True,
        out_channels: Optional[int] = None,
        dual_pred: bool = False,
        dual_out_order: str = "v_eps",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.mri_in_channels = int(mri_in_channels)
        self.num_regions = int(num_regions)
        self.self_attention_mode = str(self_attention_mode)
        valid_attention_modes = {"disabled", "pre_sampling_only", "block", "mid_only"}
        if self.self_attention_mode not in valid_attention_modes:
            raise ValueError(
                f"self_attention_mode must be one of {sorted(valid_attention_modes)}, got {self.self_attention_mode}"
            )
        self.dual_pred = bool(dual_pred)
        self.dual_out_order = str(dual_out_order)
        if self.dual_out_order not in ("v_eps", "eps_v"):
            raise ValueError(f"dual_out_order must be 'v_eps' or 'eps_v', got {self.dual_out_order}")
        if out_channels is None:
            out_channels = self.latent_dim * (2 if self.dual_pred else 1)
        self.out_channels = int(out_channels)
        self.table_num_features = int(table_num_features)
        self.table_token_dim = int(table_token_dim)
        self.table_has_missing_mask_in_conti = bool(table_has_missing_mask_in_conti)
        self.use_table_cross_attn = bool(use_table_cross_attn) and self.table_num_features > 0
        self.table_tokenizer = TableTokenizer(self.table_num_features, self.table_token_dim) if self.use_table_cross_attn else None
        # Kept only for backward compatibility with existing configs.
        # The current design uses TBS-GCA in exactly two places:
        # 1) once at the input stem before encoder feature extraction
        # 2) inside the UNet mid blocks
        self.table_cross_attn_res = tuple(table_cross_attn_res) if table_cross_attn_res is not None else tuple(attn_res)
        self.table_cross_attn_heads = int(table_cross_attn_heads)
        self.cond_embed = TimeRegionEmbedding(time_dim=int(time_dim), num_regions=self.num_regions)
        self.cond_emb_dim = int(time_dim)

        in_channel = self.latent_dim + self.mri_in_channels
        self.input_conv = nn.Conv3d(in_channel, int(inner_channel), kernel_size=3, padding=1)
        self.input_table_gca = (
            GatedCrossAttention3D(
                in_channel=int(inner_channel),
                table_token_dim=self.table_token_dim,
                n_head=self.table_cross_attn_heads,
                norm_groups=norm_groups,
            )
            if self.use_table_cross_attn
            else None
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.pre_down_attns = nn.ModuleList()
        num_mults = len(channel_mults)
        now_res = int(image_size)
        pre_ch = int(inner_channel)

        for i in range(num_mults):
            ch = int(inner_channel) * int(channel_mults[i])
            use_attn = now_res in set(attn_res)
            blocks = nn.ModuleList()
            for j in range(int(res_blocks)):
                in_ch = pre_ch if j == 0 else ch
                use_block_self_attn = self.self_attention_mode == "block" and use_attn
                blocks.append(
                    ResnetBlockWithAttn(
                        in_ch,
                        ch,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        cond_emb_dim=self.cond_emb_dim,
                        with_self_attn=use_block_self_attn,
                        with_table_gca=False,
                        table_token_dim=self.table_token_dim,
                        table_cross_attn_heads=self.table_cross_attn_heads,
                    )
                )
                pre_ch = ch
            self.down_blocks.append(blocks)
            if i != num_mults - 1 and use_attn and self.self_attention_mode == "pre_sampling_only":
                self.pre_down_attns.append(SelfAttention(pre_ch, n_head=1, norm_groups=norm_groups))
            else:
                self.pre_down_attns.append(nn.Identity())
            if i != num_mults - 1:
                self.downsamples.append(Downsample(pre_ch, pre_ch))
                now_res //= 2
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = ResnetBlockWithAttn(
            pre_ch,
            pre_ch,
            norm_groups=norm_groups,
            dropout=dropout,
            cond_emb_dim=self.cond_emb_dim,
            with_self_attn=self.self_attention_mode == "mid_only",
            with_table_gca=self.use_table_cross_attn,
            table_token_dim=self.table_token_dim,
            table_cross_attn_heads=self.table_cross_attn_heads,
        )
        self.mid2 = ResnetBlockWithAttn(
            pre_ch,
            pre_ch,
            norm_groups=norm_groups,
            dropout=dropout,
            cond_emb_dim=self.cond_emb_dim,
            with_self_attn=False,
            with_table_gca=self.use_table_cross_attn,
            table_token_dim=self.table_token_dim,
            table_cross_attn_heads=self.table_cross_attn_heads,
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.pre_up_attns = nn.ModuleList()
        for i in reversed(range(num_mults)):
            ch = int(inner_channel) * int(channel_mults[i])
            use_attn = now_res in set(attn_res)
            blocks = nn.ModuleList()
            for _ in range(int(res_blocks)):
                use_block_self_attn = self.self_attention_mode == "block" and use_attn
                blocks.append(
                    ResnetBlockWithAttn(
                        pre_ch + ch,
                        ch,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        cond_emb_dim=self.cond_emb_dim,
                        with_self_attn=use_block_self_attn,
                        with_table_gca=False,
                        table_token_dim=self.table_token_dim,
                        table_cross_attn_heads=self.table_cross_attn_heads,
                    )
                )
                pre_ch = ch
            self.up_blocks.append(blocks)
            if i != 0 and use_attn and self.self_attention_mode == "pre_sampling_only":
                self.pre_up_attns.append(SelfAttention(pre_ch, n_head=1, norm_groups=norm_groups))
            else:
                self.pre_up_attns.append(nn.Identity())
            if i != 0:
                self.upsamples.append(Upsample(pre_ch, pre_ch))
                now_res *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.final_norm = GroupNorm(pre_ch, num_groups=norm_groups)
        self.final_act = Swish()
        self.final_conv = nn.Conv3d(pre_ch, self.out_channels, kernel_size=3, padding=1)

    def _split_table_and_mask(self, cate_x, conti_x):
        if conti_x is None or not self.use_table_cross_attn:
            return None, None
        if cate_x is not None:
            return conti_x, cate_x
        if self.table_has_missing_mask_in_conti and conti_x.ndim == 2 and conti_x.shape[1] == 2 * self.table_num_features:
            values = conti_x[:, :self.table_num_features]
            miss = conti_x[:, self.table_num_features:]
            if _looks_like_binary_mask(miss):
                return values, miss
        return conti_x, torch.zeros_like(conti_x)

    def _build_table_context(self, x: torch.Tensor, cate_x=None, conti_x=None):
        if not self.use_table_cross_attn or conti_x is None:
            return None, None, None
        values, miss = self._split_table_and_mask(cate_x, conti_x)
        if values is None:
            return None, None, None
        values = values.to(device=x.device)
        miss = miss.to(device=x.device) if miss is not None else None
        table_tokens, table_kpm, table_global = self.table_tokenizer(values, miss)
        if table_tokens is not None:
            table_tokens = table_tokens.to(device=x.device, dtype=x.dtype)
        if table_global is not None:
            table_global = table_global.to(device=x.device, dtype=x.dtype)
        if table_kpm is not None:
            table_kpm = table_kpm.to(device=x.device, dtype=torch.bool)
        return table_tokens, table_kpm, table_global

    def forward(
        self,
        x: torch.Tensor,
        mri: torch.Tensor,
        cate_x=None,
        conti_x=None,
        time: Optional[torch.Tensor] = None,
        region_label: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        tabular_values: Optional[torch.Tensor] = None,
        tabular_missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t is not None:
            time = t
        if tabular_values is not None:
            conti_x = tabular_values
        if tabular_missing_mask is not None:
            cate_x = tabular_missing_mask
        if x.ndim != 5:
            raise ValueError(f"x/Xt must be 5D (B,C,D,H,W), got {tuple(x.shape)}")
        if mri.ndim != 5:
            raise ValueError(f"mri must be 5D (B,C,D,H,W), got {tuple(mri.shape)}")
        if x.shape[0] != mri.shape[0]:
            raise ValueError(f"batch mismatch: x batch={x.shape[0]}, mri batch={mri.shape[0]}")
        if mri.shape[2:] != x.shape[2:]:
            mri = F.interpolate(mri, size=x.shape[2:], mode="trilinear", align_corners=False)
        mri = mri.to(device=x.device, dtype=x.dtype)
        cond_emb = self.cond_embed(time, region_label, batch_size=x.shape[0], device=x.device).to(dtype=x.dtype)
        table_tokens, table_kpm, table_global = self._build_table_context(x, cate_x=cate_x, conti_x=conti_x)
        x = torch.cat([x, mri], dim=1)
        x = self.input_conv(x)
        if self.input_table_gca is not None:
            x = self.input_table_gca(x, table_tokens, table_kpm, table_global)
        skips = []
        for blocks, pre_attn, down in zip(self.down_blocks, self.pre_down_attns, self.downsamples):
            for blk in blocks:
                x = blk(x, cond_emb, table_tokens, table_kpm, table_global)
                skips.append(x)
            x = pre_attn(x)
            x = down(x)
        x = self.mid1(x, cond_emb, table_tokens, table_kpm, table_global)
        x = self.mid2(x, cond_emb, table_tokens, table_kpm, table_global)
        for blocks, pre_attn, up in zip(self.up_blocks, self.pre_up_attns, self.upsamples):
            for blk in blocks:
                if not skips:
                    raise RuntimeError("skip list is empty. Check res_blocks/channel_mults construction.")
                skip = skips.pop()
                if skip.shape[2:] != x.shape[2:]:
                    skip = _center_crop_or_pad_to(skip, x.shape[2:])
                x = torch.cat([x, skip], dim=1)
                x = blk(x, cond_emb, table_tokens, table_kpm, table_global)
            x = pre_attn(x)
            x = up(x)
        x = self.final_act(self.final_norm(x))
        return self.final_conv(x)


UNet = RegionLocalUNetDenoiser
