from __future__ import annotations

from typing import Optional

import torch


def ensure_mask(mask: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference)
    mask = mask.float()
    if mask.shape != reference.shape:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match reference {tuple(reference.shape)}")
    return mask


def masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        return value.mean()
    weight = ensure_mask(mask, value)
    return (value * weight).sum() / weight.sum().clamp_min(eps)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return masked_mean((prediction - target) ** 2, mask=mask)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return masked_mean(torch.abs(prediction - target), mask=mask)


def _low_frequency_amplitude(volume: torch.Tensor, crop_ratio: float = 0.125) -> torch.Tensor:
    freq = torch.fft.rfftn(volume, dim=(-3, -2, -1), norm="ortho")
    amplitude = torch.abs(freq)
    d = max(1, int(amplitude.shape[-3] * crop_ratio))
    h = max(1, int(amplitude.shape[-2] * crop_ratio))
    w = max(1, int(amplitude.shape[-1] * crop_ratio))
    return amplitude[..., :d, :h, :w]


def low_frequency_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    crop_ratio: float = 0.125,
) -> torch.Tensor:
    pred_amp = _low_frequency_amplitude(prediction, crop_ratio=crop_ratio)
    tgt_amp = _low_frequency_amplitude(target, crop_ratio=crop_ratio)
    return torch.mean(torch.abs(pred_amp - tgt_amp))
