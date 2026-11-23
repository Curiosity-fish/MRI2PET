from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1.0 + logvar - mu.square() - logvar.exp())).mean()


def gradient_matching_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for dimension in (-3, -2, -1):
        pred_diff = torch.diff(prediction, dim=dimension)
        target_diff = torch.diff(target, dim=dimension)
        losses.append(F.l1_loss(pred_diff, target_diff))
    return torch.stack(losses).mean()


def patch_seam_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    grid_size: tuple[int, int, int],
) -> torch.Tensor:
    """Match target boundary jumps instead of forcing anatomy to be constant at seams."""
    if prediction.shape != target.shape or prediction.ndim != 6:
        raise ValueError("prediction and target must have equal [B,P,C,D,H,W] shapes")
    gz, gy, gx = [int(v) for v in grid_size]
    if prediction.shape[1] != gz * gy * gx:
        raise ValueError("Patch count does not match grid_size")
    terms = []
    index = lambda z, y, x: z * gy * gx + y * gx + x
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                current = index(z, y, x)
                if z + 1 < gz:
                    neighbor = index(z + 1, y, x)
                    pred_jump = prediction[:, neighbor, :, 0] - prediction[:, current, :, -1]
                    target_jump = target[:, neighbor, :, 0] - target[:, current, :, -1]
                    terms.append(F.l1_loss(pred_jump, target_jump))
                if y + 1 < gy:
                    neighbor = index(z, y + 1, x)
                    pred_jump = prediction[:, neighbor, :, :, 0] - prediction[:, current, :, :, -1]
                    target_jump = target[:, neighbor, :, :, 0] - target[:, current, :, :, -1]
                    terms.append(F.l1_loss(pred_jump, target_jump))
                if x + 1 < gx:
                    neighbor = index(z, y, x + 1)
                    pred_jump = prediction[:, neighbor, :, :, :, 0] - prediction[:, current, :, :, :, -1]
                    target_jump = target[:, neighbor, :, :, :, 0] - target[:, current, :, :, :, -1]
                    terms.append(F.l1_loss(pred_jump, target_jump))
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())

