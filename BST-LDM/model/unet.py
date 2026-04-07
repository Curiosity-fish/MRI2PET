# model/unet_vae.py
# ============================================================
# 3D DDPM UNet：用于 MRI->PET 的去噪网络（在 latent 空间上做扩散/去噪）
#
# 条件输入（conditioning）包含两部分：
# 1) MRI 条件：通过一个轻量 3D CNN 编码为 cond feature，
#    然后与扩散输入 x_t 在通道维上拼接（concat）作为 UNet 输入。
# 2) 表格条件（tabular）：通过“TableTokenizer -> token”后，
#    使用 **Gated Cross-Attention (GCA)** 融合进 UNet 各层特征。
#    - 支持缺失值掩码 missing-mask：
#        * 若提供 cate_x：视为缺失掩码（1=missing, 0=observed），conti_x 为数值
#        * 若 cate_x=None 且 conti_x 形状为 (B, 2F)：认为是 [values, missing_mask] 拼接
#    - 缺失 token 会在注意力中被忽略（key_padding_mask），同时缺失信息用于生成 gate
# ============================================================

from __future__ import annotations
import os
import math
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F


# -------------------------
# 基础模块：激活/归一化/上下采样
# -------------------------
class Swish(nn.Module):
    """Swish 激活：x * sigmoid(x)。扩散模型里常用。"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm(nn.Module):
    """
    安全版 GroupNorm：
    - 自动调整 groups，确保 channels % groups == 0
    - 避免某些通道数下 GroupNorm 报错
    """
    def __init__(self, channels, num_groups: int = 16):
        super().__init__()
        g = min(int(num_groups), int(channels))
        while g > 1 and channels % g != 0:
            g -= 1
        self.gn = nn.GroupNorm(num_groups=g, num_channels=int(channels), eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Downsample(nn.Module):
    """3D 下采样：Conv3d stride=2，相当于把空间尺寸 /2。"""
    def __init__(self, in_ch: int, out_ch: Optional[int] = None):
        super().__init__()
        out_ch = int(out_ch) if out_ch is not None else int(in_ch)
        self.conv = nn.Conv3d(int(in_ch), out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """3D 上采样：ConvTranspose3d stride=2，把空间尺寸 *2。"""
    def __init__(self, in_ch: int, out_ch: Optional[int] = None):
        super().__init__()
        out_ch = int(out_ch) if out_ch is not None else int(in_ch)
        self.conv = nn.ConvTranspose3d(int(in_ch), out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# -------------------------
# 时间步嵌入 Time Embedding
# -------------------------
class TimeEmbedding(nn.Module):
    """
    DDPM/UNet 常见的正弦时间嵌入：
    输入 t: (B,) -> 输出 emb: (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) long 或 float（扩散的时间步）
        return: (B, dim) 正弦/余弦位置编码
        """
        device = t.device
        half = self.dim // 2
        emb_scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-emb_scale))
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# -------------------------
# UNet 核心模块：卷积块 / 残差块 / 自注意力
# -------------------------
class Block(nn.Module):
    """
    基本卷积块：
    GN -> (可选FiLM scale/shift) -> Swish -> Dropout -> Conv3d
    """
    def __init__(self, dim: int, dim_out: int, norm_groups: int = 16, dropout: float = 0.0):
        super().__init__()
        self.norm = GroupNorm(dim, num_groups=norm_groups)
        self.act = Swish()
        self.conv = nn.Conv3d(dim, dim_out, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout) if dropout and float(dropout) > 0 else nn.Identity()

    def forward(self, x, scale_shift=None):
        x = self.norm(x)
        # scale_shift 预留：可做 FiLM/AdaGN（此脚本中未使用）
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        x = self.act(x)
        x = self.drop(x)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    """两层 Block + 残差旁路（通道不匹配时用 1x1 Conv 对齐）。"""
    def __init__(self, dim: int, dim_out: int, norm_groups: int = 16, dropout: float = 0.0):
        super().__init__()
        self.block1 = Block(dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, norm_groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, scale_shift=None):
        h = self.block1(x, scale_shift)
        h = self.block2(h, scale_shift)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    """
    3D 特征图自注意力（把 D*H*W 展平为 token 做 attention）：
    - 用于在某些分辨率层增强全局依赖
    """
    def __init__(self, in_channel: int, n_head: int = 1, norm_groups: int = 16):
        super().__init__()
        self.in_channel = int(in_channel)
        # 兼容不同 channel 配置：确保 embed_dim % num_heads == 0
        heads = int(n_head)
        heads = max(1, min(heads, self.in_channel))
        while heads > 1 and (self.in_channel % heads != 0):
            heads -= 1
        self.n_head = heads
        g = min(int(norm_groups), int(in_channel))
        while g > 1 and in_channel % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(num_groups=g, num_channels=int(in_channel))
        self.qkv = nn.Conv3d(int(in_channel), int(in_channel) * 3, 1, bias=False)
        self.out = nn.Conv3d(int(in_channel), int(in_channel), 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        # q: (b,head,N,ch), k: (b,head,ch,N), v: (b,head,N,ch)
        q = q.reshape(b, self.n_head, c // self.n_head, d * h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.n_head, c // self.n_head, d * h * w)
        v = v.reshape(b, self.n_head, c // self.n_head, d * h * w).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k) * (c // self.n_head) ** -0.5
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(b, c, d, h, w)
        out = self.out(out)
        return x + out


# -------------------------
# 表格条件：TableTokenizer + Gated Cross-Attention
# -------------------------
class TableTokenizer(nn.Module):
    """
    把表格标量特征（B,F）转为 token（B,F,D），供 cross-attn 使用。

    输入
      values: (B, F) 数值特征
      miss_mask: (B, F) 缺失掩码，1=missing，0=observed（可 None -> 默认全不缺失）

    输出
      tokens: (B, F, token_dim)
      key_padding_mask: (B, F) bool，True=忽略该 token（即 missing）
      global_vec: (B, token_dim) 对有效 token masked mean 得到全局摘要，同时加入缺失率信息
    """
    def __init__(self, num_features: int, token_dim: int = 128):
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)

        # 每个特征一个“ID embedding”：让模型区分“第 i 个标量是哪一项指标”
        self.feat_embed = nn.Embedding(self.num_features, self.token_dim)

        # 标量值 1维 -> token_dim
        self.val_proj = nn.Sequential(
            nn.Linear(1, self.token_dim),
            Swish(),
            nn.Linear(self.token_dim, self.token_dim),
        )

        # 缺失标记 0/1 -> token_dim（把缺失性作为信息注入 token）
        self.miss_proj = nn.Sequential(
            nn.Linear(1, self.token_dim),
            Swish(),
            nn.Linear(self.token_dim, self.token_dim),
        )

        # 把“整体缺失比例”也注入 global_vec（用于 gate）
        self.miss_ratio_proj = nn.Sequential(
            nn.Linear(1, self.token_dim),
            Swish(),
            nn.Linear(self.token_dim, self.token_dim),
        )

        self.ln = nn.LayerNorm(self.token_dim)

    def forward(self, values: torch.Tensor, miss_mask: Optional[torch.Tensor] = None):
        if values is None:
            return None, None, None
        if values.ndim != 2:
            raise ValueError(f"table values must be (B,F), got {tuple(values.shape)}")

        B, F = values.shape
        if F != self.num_features:
            raise ValueError(
                f"UNet expects table_num_features={self.num_features}, but got F={F}. "
                "Fix: pass correct table_num_features when building UNet."
            )

        if miss_mask is None:
            miss_mask = torch.zeros_like(values)
        if miss_mask.shape != values.shape:
            raise ValueError(
                f"miss_mask must match values shape (B,F). got values={tuple(values.shape)}, miss={tuple(miss_mask.shape)}"
            )

        # 1) 特征ID embedding
        feat_ids = torch.arange(self.num_features, device=values.device)
        feat_e = self.feat_embed(feat_ids)[None, :, :].expand(B, -1, -1)  # (B,F,D)

        # 2) 数值投影 + 缺失投影
        val_e = self.val_proj(values.unsqueeze(-1))        # (B,F,D)
        miss_e = self.miss_proj(miss_mask.unsqueeze(-1))   # (B,F,D)

        # token = value + feature_id + missingness，然后 LayerNorm
        tokens = self.ln(val_e + feat_e + miss_e)

        # key_padding_mask：missing=1 -> True（在 attention 中忽略）
        key_padding_mask = (miss_mask > 0.5)

        # masked mean pooling：仅对 observed token 求平均，得到 global_vec
        valid = (~key_padding_mask).float()  # (B,F) 1=valid
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_vec = (tokens * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)

        # 注入整体缺失比例，帮助 gate 感知缺失程度
        miss_ratio = miss_mask.float().mean(dim=1, keepdim=True)  # (B,1)
        global_vec = self.ln(global_vec + self.miss_ratio_proj(miss_ratio))

        return tokens, key_padding_mask, global_vec


class GatedCrossAttention3D(nn.Module):
    """
    Gated Cross-Attention（3D feature -> table tokens）：
        x <- x + sigmoid(gate(table_global, x)) * Attn(x, table_tokens)

    - gate 同时使用 table_global（包含缺失信息）和 x 的全局摘要（当前层内容自适应）
    """
    def __init__(self, in_channel: int, table_token_dim: int = 128, n_head: int = 4, norm_groups: int = 16):
        super().__init__()
        self.in_channel = int(in_channel)
        self.table_token_dim = int(table_token_dim)

        # 兼容不同 channel 配置：确保 embed_dim % num_heads == 0
        heads = int(n_head)
        heads = max(1, min(heads, self.in_channel))
        while heads > 1 and (self.in_channel % heads != 0):
            heads -= 1
        self.n_head = heads

        g = min(int(norm_groups), int(in_channel))
        while g > 1 and in_channel % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(num_groups=g, num_channels=int(in_channel))

        # table token -> C
        self.ctx_proj = nn.Linear(self.table_token_dim, self.in_channel)

        # (B, N, C)
        self.attn = nn.MultiheadAttention(embed_dim=self.in_channel, num_heads=self.n_head, batch_first=True)

        self.out_conv = nn.Conv3d(self.in_channel, self.in_channel, kernel_size=1)

        # gate(table_global) -> (B,C)
        self.gate_proj = nn.Linear(self.table_token_dim, self.in_channel)
        nn.init.constant_(self.gate_proj.bias, -2.0)  # 初始 gate 偏“关”

        # ✅ 新增：gate(x) -> (B,C)
        # 用当前层特征 x 的全局摘要来调节注入强度（content-adaptive）
        self.x_gate_proj = nn.Linear(self.in_channel, self.in_channel)
        nn.init.zeros_(self.x_gate_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        table_tokens: Optional[torch.Tensor],
        table_key_padding_mask: Optional[torch.Tensor],
        table_global: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (table_tokens is None) or (table_global is None):
            return x

        b, c, d, h, w = x.shape
        x_n = self.norm(x)

        # Q: (B, N, C)
        q = x_n.flatten(2).transpose(1, 2)

        # KV: (B, F, C)
        kv = self.ctx_proj(table_tokens)

        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=table_key_padding_mask
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, c, d, h, w)
        attn_out = self.out_conv(attn_out)

        # ✅ gate 同时使用 table_global + x（全局池化）
        x_pool = x_n.mean(dim=(2, 3, 4))  # (B,C)

        gate_logits = self.gate_proj(table_global) + self.x_gate_proj(x_pool)  # (B,C)
        gate = torch.sigmoid(gate_logits).view(b, c, 1, 1, 1)

        return x + gate * attn_out

class ResnetBlockWithAttn(nn.Module):
    """
    UNet 中的“增强残差块”：
    - 两个 ResnetBlock
    - 可选 Self-Attn（特征自注意力）
    - 可选 Cross-Attn（表格条件注入）
    - 时间嵌入 t_emb 通过 MLP 投影后加到特征上（类似 bias）
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        norm_groups: int = 16,
        dropout: float = 0.1,
        time_emb_dim: int = 128,
        with_self_attn: bool = False,
        with_cross_attn: bool = False,
        table_token_dim: int = 128,
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.res1 = ResnetBlock(dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        self.res2 = ResnetBlock(dim_out, dim_out, norm_groups=norm_groups, dropout=dropout)

        self.with_self_attn = bool(with_self_attn)
        self.self_attn = SelfAttention(dim_out, n_head=1, norm_groups=norm_groups) if self.with_self_attn else None

        self.with_cross_attn = bool(with_cross_attn)
        self.cross_attn = (
            GatedCrossAttention3D(
                in_channel=dim_out,
                table_token_dim=int(table_token_dim),
                n_head=int(cross_attn_heads),
                norm_groups=int(norm_groups),
            )
            if self.with_cross_attn else None
        )

        # 时间嵌入投影到 dim_out，用于注入扩散时间信息
        self.mlp = nn.Sequential(Swish(), nn.Linear(int(time_emb_dim), int(dim_out))) if time_emb_dim else None

    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        table_tokens: Optional[torch.Tensor] = None,
        table_key_padding_mask: Optional[torch.Tensor] = None,
        table_global: Optional[torch.Tensor] = None,
    ):
        x = self.res1(x, None)

        # 注入时间步：把 (B, time_dim) -> (B, C, 1,1,1) 加到特征图
        if (self.mlp is not None) and (t_emb is not None):
            temb = self.mlp(t_emb)[:, :, None, None, None]
            x = x + temb

        x = self.res2(x, None)

        # 可选自注意力：增强空间 token 间关系
        if self.self_attn is not None:
            x = self.self_attn(x)

        # 可选表格 cross-attn：将表格 token 注入到 3D 特征
        if self.cross_attn is not None:
            x = self.cross_attn(x, table_tokens, table_key_padding_mask, table_global)

        return x


# -------------------------
# MRI 条件编码器：MRI -> cond feature（对齐 latent 的空间分辨率）
# -------------------------
class MRICondEncoder(nn.Module):
    """
    将 MRI 体数据编码成与 x_t 同分辨率的条件特征 mri_cond：
    - 几层 Conv + Downsample
    - 输出通道为 cond_channels
    - 若空间尺寸不匹配，用 trilinear 插值对齐到 target_spatial
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,
        down: int = 2,
        norm_groups: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        ch = int(base_channels)
        layers = [
            nn.Conv3d(int(in_channels), ch, kernel_size=3, padding=1),
            GroupNorm(ch, num_groups=norm_groups),
            Swish(),
        ]
        for _ in range(int(down)):
            layers += [
                Downsample(ch, ch),
                Block(ch, ch, norm_groups=norm_groups, dropout=dropout),
            ]
        layers += [
            nn.Conv3d(ch, int(out_channels), kernel_size=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, mri: torch.Tensor, target_spatial: Tuple[int, int, int]) -> torch.Tensor:
        cond = self.net(mri)
        if cond.shape[2:] != tuple(target_spatial):
            cond = torch.nn.functional.interpolate(cond, size=target_spatial, mode="trilinear", align_corners=False)
        return cond


def _looks_like_binary_mask(x: torch.Tensor) -> bool:
    """
    用于“猜测 conti_x 的后半段是不是 missing mask”的启发式判断：
    - 值域接近 [0,1]
    - 且接近整数（0/1）
    """
    if x is None or x.numel() == 0:
        return False
    x_min = float(x.min().detach().cpu())
    x_max = float(x.max().detach().cpu())
    if x_min < -0.05 or x_max > 1.05:
        return False
    diff = (x - x.round()).abs().max().detach().cpu().item()
    return diff < 1e-3


# -------------------------
# UNet 主体：MRI concat + Table gated cross-attn
# -------------------------
class UNet(nn.Module):
    """
    forward 签名保持与 DiffusionDDPM wrapper 兼容：
        forward(x_t, mri, cate_x=None, conti_x=None, t)

    Conditioning 组成：
      - MRI -> BSFE（替代 MRICondEncoder）
          * 使用 BSFE 的所有输出：
              1) pyramid[0] -> mri_cond，与 x_t 在通道维 concat
              2) pyramid 多尺度 -> 在 UNet 各分辨率层做 gated additive injection
              3) tokens + global -> 作为 cross-attn context + gate 条件（与 table tokens 拼接）
      - table（可选）-> TableTokenizer -> tokens/global -> cross-attn（缺失掩码 mask: 1=missing）
    """
    def __init__(
        self,
        latent_dim: int = 4,
        cond_channels: int = 4,
        mri_in_channels: int = 1,
        mri_down: int = 2,

        inner_channel: int = 32,
        norm_groups: int = 16,
        channel_mults=(2, 4, 8, 16),
        attn_res=(40,),
        res_blocks: int = 1,
        dropout: float = 0.1,
        image_size: int = 40,
        time_dim: int = 128,
        mri_base_channels: int = 16,

        # --- table conditioning ---
        table_num_features: int = 17,
        table_token_dim: int = 128,
        table_cross_attn_heads: int = 4,
        table_cross_attn_res: Optional[Tuple[int, ...]] = None,
        table_has_missing_mask_in_conti: bool = False,
        use_table_cross_attn: bool = True,

        # --- BSFE conditioning ---
        use_bsfe: bool = True,
        bsfe_dir: Optional[str] = None,
        freeze_bsfe: bool = True,

        # BSFE architecture (must match the checkpoint you trained)
        bsfe_pre_down_hidden: int = 32,
        bsfe_base_channels: int = 32,
        bsfe_channel_mults: Tuple[int, ...] = (2, 4, 8, 16),
        bsfe_num_res_blocks: int = 3,
        bsfe_norm_groups: int = 32,
        bsfe_dropout: float = 0.1,
        bsfe_attn_stages: Tuple[int, ...] = (2, 3),
        bsfe_attn_heads: int = 4,
        bsfe_attn_head_dim: int = 32,
        bsfe_n_tokens: int = 32,
        bsfe_token_dim: int = 128,
        bsfe_global_dim: int = 256,

        # --- dual prediction head (v + eps) ---
        dual_pred: bool = False,
        dual_out_order: str = "v_eps",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.cond_channels = int(cond_channels)

        # Dual-head output: [v, eps] (or [eps, v]) along channel dim.
        # This only changes the output head channel count; the actual ordering
        # is interpreted inside DiffusionDDPM.
        self.dual_pred = bool(dual_pred)
        self.dual_out_order = str(dual_out_order)
        if self.dual_out_order not in ("v_eps", "eps_v"):
            raise ValueError(f"dual_out_order must be 'v_eps' or 'eps_v', got: {self.dual_out_order}")

        # -------------------------
        # Table modules
        # -------------------------
        self.table_num_features = int(table_num_features)
        self.table_token_dim = int(table_token_dim)
        self.table_has_missing_mask_in_conti = bool(table_has_missing_mask_in_conti)

        self.use_table_cross_attn = bool(use_table_cross_attn) and (self.table_num_features > 0)
        self.table_tokenizer = TableTokenizer(self.table_num_features, token_dim=self.table_token_dim) if self.use_table_cross_attn else None
        self.table_cross_attn_res = tuple(table_cross_attn_res) if table_cross_attn_res is not None else tuple(attn_res)

        # -------------------------
        # BSFE (replaces MRICondEncoder)
        # -------------------------
        self.use_bsfe = bool(use_bsfe)
        self.freeze_bsfe = bool(freeze_bsfe)

        # fallback MRICondEncoder (only used when use_bsfe=False)
        self.mri_encoder = MRICondEncoder(
            in_channels=int(mri_in_channels),
            out_channels=int(cond_channels),
            base_channels=int(mri_base_channels),
            down=int(mri_down),
            norm_groups=int(norm_groups),
            dropout=float(dropout),
        )

        self.bsfe = None
        self.bsfe_loaded = False
        self.bsfe_dir = bsfe_dir or os.environ.get("BSFE_DIR", "/zjs/MRI2PET/MRI2PET/result/bsfe/bsfe_only_best.pth")

        if self.use_bsfe:
            # Lazy import to keep this file standalone in different repos
            try:
                from model.BSFE import BSFEConfig, BrainStructuralFeatureExtractor
            except Exception:
                try:
                    from model.bsfe_latent import BSFEConfig, BrainStructuralFeatureExtractor
                except Exception as e:
                    raise ImportError(
                        "Cannot import BSFEConfig/BrainStructuralFeatureExtractor. "
                        "Make sure you have model/BSFE.py (or model/bsfe_latent.py) in your project."
                    ) from e

            # Make BSFE tokens align to UNet context dim (table_token_dim)
            cfg = BSFEConfig(
                in_channels=int(mri_in_channels),
                pre_down_hidden=int(bsfe_pre_down_hidden),
                base_channels=int(bsfe_base_channels),
                channel_mults=tuple(int(x) for x in bsfe_channel_mults),
                num_res_blocks=int(bsfe_num_res_blocks),
                norm_groups=int(bsfe_norm_groups),
                dropout=float(bsfe_dropout),
                attn_stages=tuple(int(x) for x in bsfe_attn_stages),
                attn_heads=int(bsfe_attn_heads),
                attn_head_dim=int(bsfe_attn_head_dim),
                n_tokens=int(bsfe_n_tokens),
                token_dim=int(bsfe_token_dim),
                global_dim=int(bsfe_global_dim),
                out_context_dim=int(self.table_token_dim),
            )
            self.bsfe = BrainStructuralFeatureExtractor(cfg)

            # Load checkpoint (best-effort, strict=False)
            self._load_bsfe_checkpoint(self.bsfe_dir)

            if self.freeze_bsfe:
                for p in self.bsfe.parameters():
                    p.requires_grad_(False)
                self.bsfe.eval()

        # -------------------------
        # Time embedding
        # -------------------------
        self.time_mlp = nn.Sequential(
            TimeEmbedding(int(time_dim)),
            nn.Linear(int(time_dim), int(time_dim) * 4),
            Swish(),
            nn.Linear(int(time_dim) * 4, int(time_dim)),
        )
        self.time_dim = int(time_dim)

        # UNet input: [x_t, mri_cond] concat
        in_channel = self.latent_dim + self.cond_channels
        out_channel = self.latent_dim * (2 if self.dual_pred else 1)

        # Use cross-attn if either (table enabled) or (BSFE enabled, for bsfe tokens/global injection)
        self.use_ctx_cross_attn = bool(self.use_table_cross_attn or self.use_bsfe)
        self.table_cross_attn_heads = int(table_cross_attn_heads)

        # -------------------------
        # BSFE projections (all outputs used)
        # -------------------------
        if self.use_bsfe:
            # mri_cond from pyramid[0]
            # NOTE: bsfe pyramid stage channels = base_channels * channel_mults[i]
            bsfe_stage_channels = [int(bsfe_base_channels) * int(m) for m in tuple(bsfe_channel_mults)]
            if len(bsfe_stage_channels) < len(channel_mults):
                bsfe_stage_channels = bsfe_stage_channels + [bsfe_stage_channels[-1]] * (len(channel_mults) - len(bsfe_stage_channels))

            self.bsfe_inp_proj = nn.Conv3d(int(bsfe_stage_channels[0]), int(self.cond_channels), kernel_size=1)

            # multi-scale injections: map each BSFE pyramid level -> UNet stage channels (per resolution)
            unet_stage_channels = [int(inner_channel) * int(m) for m in channel_mults]
            self.bsfe_pyr_proj = nn.ModuleList([
                nn.Conv3d(int(bsfe_stage_channels[i]), int(unet_stage_channels[i]), kernel_size=1)
                for i in range(len(unet_stage_channels))
            ])
            self.bsfe_pyr_gate = nn.ModuleList([
                nn.Linear(int(bsfe_global_dim), int(unet_stage_channels[i]))
                for i in range(len(unet_stage_channels))
            ])
            for lin in self.bsfe_pyr_gate:
                nn.init.constant_(lin.bias, -2.0)

            # tokens/global to context dim (table_token_dim)
            self.bsfe_global_to_token = nn.Linear(int(bsfe_global_dim), int(self.table_token_dim))
            self.bsfe_global_to_gate  = nn.Linear(int(bsfe_global_dim), int(self.table_token_dim))

            # fuse (table_global, bsfe_global, bsfe_token_mean) -> ctx_global for GCA gate
            self.cond_fuse = nn.Sequential(
                nn.Linear(int(self.table_token_dim) * 3, int(self.table_token_dim) * 2),
                Swish(),
                nn.Linear(int(self.table_token_dim) * 2, int(self.table_token_dim)),
                nn.LayerNorm(int(self.table_token_dim)),
            )
        else:
            self.bsfe_inp_proj = None
            self.bsfe_pyr_proj = None
            self.bsfe_pyr_gate = None
            self.bsfe_global_to_token = None
            self.bsfe_global_to_gate = None
            self.cond_fuse = None

        # -------------------------
        # UNet multi-scale structure (same as baseline)
        # -------------------------
        num_mults = len(channel_mults)
        now_res = int(image_size)

        self.input_conv = nn.Conv3d(in_channel, int(inner_channel), kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        pre_ch = int(inner_channel)

        # Down path
        for i in range(num_mults):
            ch = int(inner_channel) * int(channel_mults[i])

            use_attn = (now_res in set(attn_res))
            use_cross = self.use_ctx_cross_attn and (now_res in set(self.table_cross_attn_res))

            blocks = nn.ModuleList()
            for j in range(int(res_blocks)):
                in_ch = pre_ch if j == 0 else ch
                blocks.append(
                    ResnetBlockWithAttn(
                        in_ch, ch,
                        norm_groups=int(norm_groups),
                        dropout=float(dropout),
                        time_emb_dim=self.time_dim,
                        with_self_attn=use_attn,
                        with_cross_attn=use_cross,
                        table_token_dim=self.table_token_dim,
                        cross_attn_heads=self.table_cross_attn_heads,
                    )
                )
                pre_ch = ch
            self.down_blocks.append(blocks)

            if i != num_mults - 1:
                self.downsamples.append(Downsample(pre_ch, pre_ch))
                now_res //= 2
            else:
                self.downsamples.append(nn.Identity())

        # Bottleneck
        self.mid1 = ResnetBlockWithAttn(
            pre_ch, pre_ch,
            norm_groups=int(norm_groups),
            dropout=float(dropout),
            time_emb_dim=self.time_dim,
            with_self_attn=True,
            with_cross_attn=self.use_ctx_cross_attn,
            table_token_dim=self.table_token_dim,
            cross_attn_heads=self.table_cross_attn_heads,
        )
        self.mid2 = ResnetBlockWithAttn(
            pre_ch, pre_ch,
            norm_groups=int(norm_groups),
            dropout=float(dropout),
            time_emb_dim=self.time_dim,
            with_self_attn=False,
            with_cross_attn=self.use_ctx_cross_attn,
            table_token_dim=self.table_token_dim,
            cross_attn_heads=self.table_cross_attn_heads,
        )

        # Up path
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in reversed(range(num_mults)):
            ch = int(inner_channel) * int(channel_mults[i])

            use_attn = (now_res in set(attn_res))
            use_cross = self.use_ctx_cross_attn and (now_res in set(self.table_cross_attn_res))

            blocks = nn.ModuleList()
            for j in range(int(res_blocks)):
                in_ch = pre_ch + ch if j == 0 else ch + ch
                blocks.append(
                    ResnetBlockWithAttn(
                        in_ch, ch,
                        norm_groups=int(norm_groups),
                        dropout=float(dropout),
                        time_emb_dim=self.time_dim,
                        with_self_attn=use_attn,
                        with_cross_attn=use_cross,
                        table_token_dim=self.table_token_dim,
                        cross_attn_heads=self.table_cross_attn_heads,
                    )
                )
                pre_ch = ch
            self.up_blocks.append(blocks)

            if i != 0:
                self.upsamples.append(Upsample(pre_ch, pre_ch))
                now_res *= 2
            else:
                self.upsamples.append(nn.Identity())

        # Output head
        self.final_norm = GroupNorm(pre_ch, num_groups=int(norm_groups))
        self.final_act = Swish()
        self.final_conv = nn.Conv3d(pre_ch, out_channel, kernel_size=3, padding=1)

    # -------------------------
    # BSFE checkpoint loader
    # -------------------------
    def _load_bsfe_checkpoint(self, ckpt_path: Optional[str]) -> None:
        if (ckpt_path is None) or (not isinstance(ckpt_path, str)) or (ckpt_path.strip() == ""):
            return
        if not os.path.exists(ckpt_path):
            # do not hard-fail in case user wants to train end-to-end on a different machine
            return

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            return

        # common checkpoint layouts
        state = None
        if isinstance(ckpt, dict):
            for k in ["bsfe", "model", "state_dict", "net", "ema", "weights"]:
                if k in ckpt:
                    candidate = ckpt[k]
                    if isinstance(candidate, dict) and "state_dict" in candidate and isinstance(candidate["state_dict"], dict):
                        state = candidate["state_dict"]
                        break
                    if isinstance(candidate, dict):
                        state = candidate
                        break
        if state is None and isinstance(ckpt, dict):
            state = ckpt
        if state is None:
            return

        # strip possible 'module.' prefix
        if all(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}

        missing, unexpected = self.bsfe.load_state_dict(state, strict=False)
        self.bsfe_loaded = True

        # print only on rank0 if DDP is used
        if os.environ.get("RANK", "0") in ("0", "None", ""):
            msg = f"[BSFE] loaded ckpt: {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}"
            print(msg)

    # -------------------------
    # Table parsing helper
    # -------------------------
    def _split_table_and_mask(self, cate_x, conti_x):
        if conti_x is None or (not self.use_table_cross_attn):
            return None, None

        if cate_x is not None:
            values = conti_x
            miss = cate_x
            return values, miss

        if self.table_has_missing_mask_in_conti and conti_x.ndim == 2 and conti_x.shape[1] == 2 * self.table_num_features:
            values = conti_x[:, :self.table_num_features]
            miss = conti_x[:, self.table_num_features:]
            if _looks_like_binary_mask(miss):
                return values, miss

        values = conti_x
        miss = torch.zeros_like(values)
        return values, miss

    # -------------------------
    # BSFE multi-scale injection helper
    # -------------------------
    def _inject_bsfe_pyramid(self, x: torch.Tensor, stage_idx: int, bsfe_pyr, bsfe_global) -> torch.Tensor:
        if (not self.use_bsfe) or (self.bsfe_pyr_proj is None) or (bsfe_pyr is None) or (bsfe_global is None):
            return x
        if len(bsfe_pyr) == 0:
            return x

        i = int(stage_idx)
        i = max(0, min(i, len(self.bsfe_pyr_proj) - 1))
        src_i = max(0, min(i, len(bsfe_pyr) - 1))

        feat = bsfe_pyr[src_i]
        if feat.shape[2:] != x.shape[2:]:
            feat = F.interpolate(feat, size=x.shape[2:], mode="trilinear", align_corners=False)

        inj = self.bsfe_pyr_proj[i](feat)
        gate = torch.sigmoid(self.bsfe_pyr_gate[i](bsfe_global)).view(x.shape[0], inj.shape[1], 1, 1, 1)
        inj = inj.to(dtype=x.dtype)
        gate = gate.to(dtype=x.dtype)
        return x + gate * inj

    def forward(
        self,
        x: torch.Tensor,
        mri: torch.Tensor,
        cate_x=None,
        conti_x=None,
        time: Optional[torch.Tensor] = None
    ):
        # 1) time embedding
        t_emb = self.time_mlp(time) if time is not None else None

        # 2) BSFE forward (or fallback MRICondEncoder)
        bsfe_pyr = bsfe_tokens = bsfe_global = None
        if self.use_bsfe and (self.bsfe is not None):
            if self.freeze_bsfe:
                self.bsfe.eval()
                with torch.no_grad():
                    # run BSFE in full precision for stability, then cast to x.dtype later
                    with torch.cuda.amp.autocast(enabled=False):
                        out = self.bsfe(mri.float())
            else:
                out = self.bsfe(mri)

            bsfe_pyr = out.get("pyramid", None)
            bsfe_tokens = out.get("tokens", None)
            bsfe_global = out.get("global", None)

        # 3) MRI cond (concat)
        if (self.use_bsfe and bsfe_pyr is not None and len(bsfe_pyr) > 0 and self.bsfe_inp_proj is not None):
            mri_cond = self.bsfe_inp_proj(bsfe_pyr[0])
            if mri_cond.shape[2:] != x.shape[2:]:
                mri_cond = F.interpolate(mri_cond, size=x.shape[2:], mode="trilinear", align_corners=False)
            mri_cond = mri_cond.to(dtype=x.dtype)
        else:
            mri_cond = self.mri_encoder(mri, target_spatial=x.shape[2:]).to(dtype=x.dtype)

        x = torch.cat([x, mri_cond], dim=1)

        # 4) table tokens (optional)
        table_tokens = table_kpm = table_global = None
        if self.use_table_cross_attn and (conti_x is not None):
            values, miss = self._split_table_and_mask(cate_x, conti_x)
            if values is not None:
                table_tokens, table_kpm, table_global = self.table_tokenizer(values, miss)
                if table_kpm is not None:
                    table_kpm = table_kpm.to(device=x.device)

        # 5) build context tokens/global for cross-attn (table + bsfe)
        ctx_tokens = None
        ctx_kpm = None
        ctx_global = None

        if self.use_ctx_cross_attn:
            parts = []
            masks = []

            B = x.shape[0]

            # table tokens
            if table_tokens is not None:
                parts.append(table_tokens)
                masks.append(table_kpm.to(torch.bool) if table_kpm is not None else torch.zeros(B, table_tokens.shape[1], device=x.device, dtype=torch.bool))

            # bsfe tokens + global token
            if self.use_bsfe and (bsfe_tokens is not None) and (bsfe_global is not None):
                # tokens are already projected to table_token_dim if BSFEConfig.out_context_dim is set
                bt = bsfe_tokens
                if bt.shape[-1] != self.table_token_dim:
                    raise ValueError(
                        f"BSFE tokens dim={bt.shape[-1]} does not match table_token_dim={self.table_token_dim}. "
                        "Fix: set BSFEConfig.out_context_dim = table_token_dim (recommended), or make them equal."
                    )
                bg_tok = self.bsfe_global_to_token(bsfe_global).unsqueeze(1)  # (B,1,Dt)
                parts.append(bt)
                parts.append(bg_tok)

                masks.append(torch.zeros(B, bt.shape[1], device=x.device, dtype=torch.bool))
                masks.append(torch.zeros(B, 1, device=x.device, dtype=torch.bool))

                # ctx_global for GCA gate
                tg = table_global if table_global is not None else torch.zeros(B, self.table_token_dim, device=x.device, dtype=bt.dtype)
                bg = self.bsfe_global_to_gate(bsfe_global)
                bm = bt.mean(dim=1)
                ctx_global = self.cond_fuse(torch.cat([tg, bg, bm], dim=-1)).to(dtype=x.dtype)

            # final concat
            if len(parts) > 0:
                ctx_tokens = torch.cat(parts, dim=1).to(dtype=x.dtype)
                ctx_kpm = torch.cat(masks, dim=1)

        # 6) UNet down
        skips = []
        x = self.input_conv(x)

        for si, (blocks, down) in enumerate(zip(self.down_blocks, self.downsamples)):
            for blk in blocks:
                x = blk(x, t_emb, ctx_tokens, ctx_kpm, ctx_global)
                # BSFE multi-scale injection at this resolution
                x = self._inject_bsfe_pyramid(x, si, bsfe_pyr, bsfe_global)
                skips.append(x)
            x = down(x)

        # 7) bottleneck
        deep_idx = len(self.down_blocks) - 1
        x = self.mid1(x, t_emb, ctx_tokens, ctx_kpm, ctx_global)
        x = self._inject_bsfe_pyramid(x, deep_idx, bsfe_pyr, bsfe_global)
        x = self.mid2(x, t_emb, ctx_tokens, ctx_kpm, ctx_global)
        x = self._inject_bsfe_pyramid(x, deep_idx, bsfe_pyr, bsfe_global)

        # 8) UNet up
        # up_blocks are ordered from low-res -> high-res, matching reversed(range(num_mults))
        num_mults = len(self.down_blocks)
        for ui, (blocks, up) in enumerate(zip(self.up_blocks, self.upsamples)):
            # map ui -> stage index (low-res first)
            stage_i = (num_mults - 1) - ui
            for blk in blocks:
                skip = skips.pop()
                x = blk(torch.cat([x, skip], dim=1), t_emb, ctx_tokens, ctx_kpm, ctx_global)
                x = self._inject_bsfe_pyramid(x, stage_i, bsfe_pyr, bsfe_global)
            x = up(x)

        # 9) output head
        x = self.final_act(self.final_norm(x))
        return self.final_conv(x)
