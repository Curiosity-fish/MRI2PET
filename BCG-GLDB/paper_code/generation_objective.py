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


def reverse_step(z_k, predicted_residual, k: int, steps: int,
                 sigma_max: float, eta: float = 1.0, noise=None):
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


def normalized_snr_weights(z_pet, k, steps: int, sigma_max: float,
                           offset: float = 1.0, gamma: float = 0.5):
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


def parc98_losses(prediction, target, labelmap):
    """RPA, RBR, RVC, HAC. Labels 1/2,3/4,...,97/98 are homologous pairs."""
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
        ps = (((prediction - pm[:, None, None, None, None]).square() * mask)
              .flatten(1).sum(1) / count).clamp_min(0).sqrt()
        ts = (((target - tm[:, None, None, None, None]).square() * mask)
              .flatten(1).sum(1) / count).clamp_min(0).sqrt()
        rvc_terms.append((ps - ts).abs())
    pm = torch.stack(pred_means, dim=1)
    tm = torch.stack(true_means, dim=1)
    rpa = torch.stack(rpa_terms, dim=1).mean()
    rbr = ((pm[:, :-1] - pm[:, 1:]) - (tm[:, :-1] - tm[:, 1:])).abs().mean()
    rvc = torch.stack(rvc_terms, dim=1).mean()
    ph = (pm[:, 0::2] - pm[:, 1::2]) / (pm[:, 0::2].abs() + pm[:, 1::2].abs() + 1e-6)
    th = (tm[:, 0::2] - tm[:, 1::2]) / (tm[:, 0::2].abs() + tm[:, 1::2].abs() + 1e-6)
    hac = (ph - th).abs().mean()
    return {"rpa": rpa, "rbr": rbr, "rvc": rvc, "hac": hac}


def validate_parc98_labelmap(labelmap):
    present = set(torch.unique(labelmap).detach().cpu().tolist())
    missing = [label for label in range(1, 99) if label not in present]
    if missing:
        raise ValueError(f"Parc98 is missing labels: {missing}")


def generation_loss(predicted_residual, true_residual, z_pet, k, steps,
                    sigma_max, reconstructed_pet, true_pet, brain_mask, labelmap,
                    use_roi=True):
    per_sample = F.smooth_l1_loss(predicted_residual.float(), true_residual.float(),
                                  reduction="none").flatten(1).mean(1)
    weights = normalized_snr_weights(z_pet, k, steps, sigma_max)
    bridge = (weights * per_sample).mean()
    image = masked_image_l1(reconstructed_pet, true_pet, brain_mask)
    if use_roi:
        regional = parc98_losses(reconstructed_pet, true_pet, labelmap)
        roi = regional["rpa"] + 0.5 * regional["rbr"] + 0.2 * regional["rvc"] + 0.2 * regional["hac"]
    else:
        roi = image.new_zeros(())
        regional = {name: roi for name in ("rpa", "rbr", "rvc", "hac")}
    total = 0.5 * bridge + image + roi
    return {"total": total, "bridge": bridge, "image": image, "roi": roi, **regional}
