from __future__ import annotations

from dataclasses import dataclass

import torch


def extract(values: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    gathered = values.gather(0, t)
    return gathered.view([t.shape[0]] + [1] * (len(x_shape) - 1))


@dataclass(frozen=True)
class VPCoefficients:
    alpha: torch.Tensor
    sigma: torch.Tensor


class LatentVPDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        self.timesteps = max(int(timesteps), 2)
        betas = torch.linspace(float(beta_start), float(beta_end), self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffers = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bar": alpha_bar,
            "sqrt_alpha_bar": torch.sqrt(alpha_bar.clamp_min(1e-8)),
            "sqrt_one_minus_alpha_bar": torch.sqrt((1.0 - alpha_bar).clamp_min(1e-8)),
        }

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "LatentVPDiffusion":
        for key, value in list(self.register_buffers.items()):
            self.register_buffers[key] = value.to(device=device, dtype=dtype or value.dtype)
        return self

    @property
    def alpha_bar(self) -> torch.Tensor:
        return self.register_buffers["alpha_bar"]

    def coeffs(self, t: torch.Tensor, x_shape: tuple[int, ...]) -> VPCoefficients:
        alpha = extract(self.register_buffers["sqrt_alpha_bar"], t, x_shape)
        sigma = extract(self.register_buffers["sqrt_one_minus_alpha_bar"], t, x_shape)
        return VPCoefficients(alpha=alpha, sigma=sigma)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(z0)
        coeffs = self.coeffs(t, z0.shape)
        zt = coeffs.alpha * z0 + coeffs.sigma * noise
        v = coeffs.alpha * noise - coeffs.sigma * z0
        return zt, v

    def predict_z0_from_v(self, zt: torch.Tensor, t: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        coeffs = self.coeffs(t, zt.shape)
        return coeffs.alpha * zt - coeffs.sigma * v_pred

    def predict_noise_from_v(self, zt: torch.Tensor, t: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        coeffs = self.coeffs(t, zt.shape)
        return coeffs.sigma * zt + coeffs.alpha * v_pred

