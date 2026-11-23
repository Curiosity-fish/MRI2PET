"""Frequency-domain PET fidelity terms used by Exp030.

Spatial losses (L1/MSE/SSIM/gradient/local-contrast) supervise the image almost
entirely in the space domain.  `frequency_l1_loss` adds explicit supervision on
the 3D FFT magnitude spectrum so that high-frequency (texture / edge) error that
is nearly invisible to spatial L1 gets a direct gradient signal.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

_FFT_WEIGHT_CACHE: dict = {}


def _fft_radius_weight(
    shape: Tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    boost: float,
) -> torch.Tensor:
    """Normalised frequency radius weight (1 + boost * r_hat) over an rfftn grid."""

    key = (shape, str(device), str(dtype), float(boost))
    cached = _FFT_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    # ``shape`` is the rfftn *output* shape (depth, height, rfft_width) where
    # the last axis is already the half-spectrum (length W//2+1).  Recover the
    # original width so rfftfreq yields exactly rfft_width bins (using
    # rfftfreq(rfft_width) would give only rfft_width//2+1 bins and break the
    # broadcast with the magnitude grid).
    depth, height, rfft_width = shape
    d_freq = torch.fft.fftfreq(depth, device=device, dtype=dtype)
    h_freq = torch.fft.fftfreq(height, device=device, dtype=dtype)
    w_freq = torch.fft.rfftfreq(2 * (rfft_width - 1), device=device, dtype=dtype)
    radius = (
        d_freq[:, None, None] ** 2
        + h_freq[None, :, None] ** 2
        + w_freq[None, None, :] ** 2
    ).sqrt()
    radius = radius / radius.max().clamp_min(1e-6)
    weight = 1.0 + float(boost) * radius
    _FFT_WEIGHT_CACHE[key] = weight
    return weight


def frequency_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    log_scale: bool = True,
    high_freq_boost: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L1 between 3D FFT magnitudes of prediction and target.

    log_scale=True compresses the huge low-frequency magnitude range so that the
    loss is not dominated by the DC/low-frequency bins.  high_freq_boost weights
    higher frequencies by (1 + boost * normalised_radius).
    """

    prediction = prediction.float()
    target = target.float()
    # Compute magnitude via real/imag squares instead of complex .abs(): the
    # complex-abs CUDA kernel is JIT-compiled and this env lacks a working
    # libnvrtc-builtins, which raised ``nvrtc: error: failed to open
    # libnvrtc-builtins.so.13.0``.
    def _magnitude(spectrum: torch.Tensor) -> torch.Tensor:
        return (spectrum.real.square() + spectrum.imag.square()).sqrt()

    pred_magnitude = _magnitude(torch.fft.rfftn(prediction))
    target_magnitude = _magnitude(torch.fft.rfftn(target))
    if log_scale:
        pred_magnitude = torch.log1p(pred_magnitude)
        target_magnitude = torch.log1p(target_magnitude)
    weight = _fft_radius_weight(
        pred_magnitude.shape[2:], pred_magnitude.device, pred_magnitude.dtype, high_freq_boost
    )
    return (weight * (pred_magnitude - target_magnitude).abs()).mean()


__all__ = ["frequency_l1_loss"]
