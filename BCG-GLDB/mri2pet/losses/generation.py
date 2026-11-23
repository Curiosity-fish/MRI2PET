"""Equations (3.4), (3.30)–(3.47) and the four Parc98 constraints."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bridge_sigma(k: torch.Tensor, steps: int, sigma_max: float) -> torch.Tensor:
    m = k.float() / float(steps)
    return (4.0 * float(sigma_max) ** 2 * m * (1.0 - m)).clamp_min(0).sqrt()


def bridge_state(z_pet, z_mri, k, steps: int, sigma_max: float, noise=None):
    """z_0 is PET; z_T is MRI. The prediction target is z_k-z_PET."""
    if z_pet.shape != z_mri.shape:
        raise ValueError("The bridge endpoints must have the same shape")
    if noise is None:
        noise = torch.randn_like(z_pet)
    m = (k.float() / float(steps)).reshape(-1, 1, 1, 1, 1)
    sigma = bridge_sigma(k, steps, sigma_max).reshape(-1, 1, 1, 1, 1)
    return (1.0 - m) * z_pet + m * z_mri + sigma * noise


def reverse_step(
    z_k, predicted_residual, k: int, steps: int, sigma_max: float, eta: float = 1.0, noise=None
):
    """Thesis posterior: mean=z_k-r_hat/k; last step has zero variance."""
    if not 1 <= int(k) <= int(steps):
        raise ValueError("Require 1 <= k <= T")
    if z_k.shape != predicted_residual.shape:
        raise ValueError("Bridge state and residual shape differ")
    mean = z_k - predicted_residual / float(k)
    if k == 1:
        return mean
    m_k = k / float(steps)
    m_previous = (k - 1) / float(steps)
    variance = 4.0 * float(sigma_max) ** 2 * m_previous * (m_k - m_previous) / m_k
    if noise is None:
        noise = torch.randn_like(z_k)
    return mean + float(eta) * variance**0.5 * noise


def normalized_snr_weights(
    z_pet, k, steps: int, sigma_max: float, offset: float = 1.0, gamma: float = 0.5
):
    m = k.float() / float(steps)
    signal = (1.0 - m).square() * z_pet.detach().float().square().flatten(1).mean(1)
    noise = bridge_sigma(k, steps, sigma_max).float().square().clamp_min(1e-8)
    snr = signal / noise
    weights = (float(offset) + snr).pow(-float(gamma))
    return weights / weights.mean().clamp_min(1e-6)


def masked_image_l1(prediction, target, brain_mask):
    mask = brain_mask.to(dtype=prediction.dtype)
    per_sample = ((prediction - target).abs() * mask).flatten(1).sum(1)
    return (per_sample / mask.flatten(1).sum(1).clamp_min(1)).mean()


# These IDs refer specifically to the Parc98 atlas identified by this SHA-256.
# See configs/parc98_hemisphere_pairs.json for the source labels and names.
PARC98_ATLAS_SHA256 = 'e4ffea9635be911321a4af48f1555bd3ce985e7643c977f455c536d7d65d6a36'
PARC98_HOMOLOGOUS_PAIRS = ((1, 18), (2, 19), (3, 20), (4, 21), (5, 22), (6, 23), (7, 24), (8, 25), (9, 26), (13, 27), (14, 28), (16, 29), (17, 30), (31, 65), (32, 66), (33, 67), (34, 68), (35, 69), (36, 70), (37, 71), (38, 72), (39, 73), (40, 74), (41, 75), (42, 76), (43, 77), (44, 78), (45, 79), (46, 80), (47, 81), (48, 82), (49, 83), (50, 84), (51, 85), (52, 86), (53, 87), (54, 88), (55, 89), (56, 90), (57, 91), (58, 92), (59, 93), (60, 94), (61, 95), (62, 96), (63, 97), (64, 98))


def parc98_losses(prediction, target, labelmap):
    """RPA, RBR, RVC, and HAC over the 47 true homologous pairs."""
    if prediction.shape != target.shape or prediction.shape[1] != 1:
        raise ValueError("PET tensors must both have shape [B,1,D,H,W]")
    if tuple(labelmap.shape) != tuple(prediction.shape[-3:]):
        raise ValueError("Parc98 labelmap is not aligned to PET output")
    pred_means, true_means, rpa_terms, rvc_terms = [], [], [], []
    for label in range(1, 99):
        mask = (labelmap == label).to(dtype=prediction.dtype)[None, None]
        count = mask.sum().clamp_min(1)
        pm = (prediction * mask).flatten(1).sum(1) / count
        tm = (target * mask).flatten(1).sum(1) / count
        pred_means.append(pm)
        true_means.append(tm)
        rpa_terms.append(((prediction - target).abs() * mask).flatten(1).sum(1) / count)
        # Norm / sqrt(N) is the population standard deviation and has a finite
        # zero gradient for a constant region, unlike sqrt(sum(x**2)).
        ps = torch.linalg.vector_norm(((prediction - pm[:, None, None, None, None]) * mask).flatten(1), dim=1) / count.sqrt()
        ts = torch.linalg.vector_norm(((target - tm[:, None, None, None, None]) * mask).flatten(1), dim=1) / count.sqrt()
        rvc_terms.append((ps - ts).abs())
    pm = torch.stack(pred_means, dim=1)
    tm = torch.stack(true_means, dim=1)
    rpa = torch.stack(rpa_terms, dim=1).mean()
    rbr = ((pm[:, :-1] - pm[:, 1:]) - (tm[:, :-1] - tm[:, 1:])).abs().mean()
    rvc = torch.stack(rvc_terms, dim=1).mean()
    left = torch.tensor([a - 1 for a, _ in PARC98_HOMOLOGOUS_PAIRS], device=pm.device)
    right = torch.tensor([b - 1 for _, b in PARC98_HOMOLOGOUS_PAIRS], device=pm.device)
    p_left, p_right = pm.index_select(1, left), pm.index_select(1, right)
    t_left, t_right = tm.index_select(1, left), tm.index_select(1, right)
    ph = (p_left - p_right) / (p_left.abs() + p_right.abs() + 1e-6)
    th = (t_left - t_right) / (t_left.abs() + t_right.abs() + 1e-6)
    hac = (ph - th).abs().mean()
    return {"rpa": rpa, "rbr": rbr, "rvc": rvc, "hac": hac}


def validate_parc98_labelmap(labelmap):
    present = set(torch.unique(labelmap).detach().cpu().tolist())
    missing = [label for label in range(1, 99) if label not in present]
    if missing:
        raise ValueError(f"Parc98 is missing labels: {missing}")


def generation_loss(
    predicted_residual,
    true_residual,
    z_pet,
    k,
    steps,
    sigma_max,
    reconstructed_pet,
    true_pet,
    brain_mask,
    labelmap,
    use_roi=True,
):
    per_sample = (
        F.smooth_l1_loss(predicted_residual.float(), true_residual.float(), reduction="none")
        .flatten(1)
        .mean(1)
    )
    weights = normalized_snr_weights(z_pet, k, steps, sigma_max)
    bridge = (weights * per_sample).mean()
    image = masked_image_l1(reconstructed_pet, true_pet, brain_mask)
    if use_roi:
        regional = parc98_losses(reconstructed_pet, true_pet, labelmap)
        roi = (
            regional["rpa"] + 0.5 * regional["rbr"] + 0.2 * regional["rvc"] + 0.2 * regional["hac"]
        )
    else:
        roi = image.new_zeros(())
        regional = {name: roi for name in ("rpa", "rbr", "rvc", "hac")}
    total = bridge + image + roi
    return {"total": total, "bridge": bridge, "image": image, "roi": roi, **regional}
