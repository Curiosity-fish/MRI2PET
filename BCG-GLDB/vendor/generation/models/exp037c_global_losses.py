"""Global-specific auxiliary losses for Exp037C.

Total Exp037C objective:
    L = L_diff
        + lambda_gca * L_GCA
        + lambda_gms * L_GMS
        + lambda_gr  * L_GR

Definitions:
- GCA: global cosine alignment between coarse Global PET prediction and pooled z0.
- GMS: two-scale SmoothL1 supervision on G3/G2 coarse PET predictions.
- GR: off-diagonal spatial relation consistency at the G3 scale.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _pool_like(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool3d(target, output_size=pred.shape[-3:])


def global_context_alignment_loss(
    global_pred_g3: torch.Tensor,
    z0: torch.Tensor,
) -> torch.Tensor:
    """Cosine alignment of the entire low-resolution PET latent pattern."""
    target_g3 = _pool_like(z0, global_pred_g3)
    pred_flat = global_pred_g3.float().flatten(1)
    target_flat = target_g3.float().flatten(1)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8)
    return (1.0 - cosine).mean()


def global_multiscale_loss(
    global_pred_g3: torch.Tensor,
    global_pred_g2: torch.Tensor,
    z0: torch.Tensor,
    *,
    g3_weight: float = 0.7,
    g2_weight: float = 0.3,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-scale coarse PET supervision for the Global pyramid."""
    target_g3 = _pool_like(z0, global_pred_g3)
    target_g2 = _pool_like(z0, global_pred_g2)

    loss_g3 = F.smooth_l1_loss(
        global_pred_g3.float(), target_g3.float(), beta=float(beta)
    )
    loss_g2 = F.smooth_l1_loss(
        global_pred_g2.float(), target_g2.float(), beta=float(beta)
    )
    total = float(g3_weight) * loss_g3 + float(g2_weight) * loss_g2
    return total, loss_g3, loss_g2


def global_relation_loss(
    global_pred_g3: torch.Tensor,
    z0: torch.Tensor,
) -> torch.Tensor:
    """Match long-range spatial relation matrices at the coarsest Global scale.

    Each spatial location is represented by its PET latent channel vector.
    Cosine relations between all pairs of spatial tokens are compared. The
    diagonal is excluded because it is trivially one after normalization.
    """
    target = _pool_like(z0, global_pred_g3)

    pred_tokens = global_pred_g3.float().flatten(2).transpose(1, 2)  # [B,N,C]
    target_tokens = target.float().flatten(2).transpose(1, 2)

    pred_tokens = F.normalize(pred_tokens, dim=-1, eps=1e-8)
    target_tokens = F.normalize(target_tokens, dim=-1, eps=1e-8)

    pred_rel = pred_tokens @ pred_tokens.transpose(1, 2)
    target_rel = target_tokens @ target_tokens.transpose(1, 2)

    n = pred_rel.shape[-1]
    if n <= 1:
        return pred_rel.sum() * 0.0

    eye = torch.eye(n, device=pred_rel.device, dtype=torch.bool).unsqueeze(0)
    diff = (pred_rel - target_rel).abs().masked_fill(eye, 0.0)
    denom = pred_rel.shape[0] * n * (n - 1)
    return diff.sum() / max(int(denom), 1)


def compute_global_losses(
    aux: Dict[str, torch.Tensor],
    z0: torch.Tensor,
    *,
    gca_weight: float = 0.05,
    gms_weight: float = 0.10,
    gr_weight: float = 0.02,
    g3_scale_weight: float = 0.7,
    g2_scale_weight: float = 0.3,
    smooth_l1_beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    gca = global_context_alignment_loss(aux["global_pred_g3"], z0)
    gms, g3, g2 = global_multiscale_loss(
        aux["global_pred_g3"],
        aux["global_pred_g2"],
        z0,
        g3_weight=g3_scale_weight,
        g2_weight=g2_scale_weight,
        beta=smooth_l1_beta,
    )
    gr = global_relation_loss(aux["global_pred_g3"], z0)

    weighted = (
        float(gca_weight) * gca
        + float(gms_weight) * gms
        + float(gr_weight) * gr
    )

    return {
        "gca": gca,
        "gms": gms,
        "gms_g3": g3,
        "gms_g2": g2,
        "gr": gr,
        "global_weighted": weighted,
    }
