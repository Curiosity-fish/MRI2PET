from __future__ import annotations

import torch


class DiffusionSchedule:
    def __init__(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
    ) -> None:
        self.timesteps = int(timesteps)
        betas = torch.linspace(float(beta_start), float(beta_end), self.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_bars = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        shape = (timesteps.shape[0],) + (1,) * (target.ndim - 1)
        return values.gather(0, timesteps).reshape(shape).to(dtype=target.dtype)

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        clean_scale = self._extract(self.sqrt_alpha_bars, timesteps, clean)
        noise_scale = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, clean)
        return clean_scale * clean + noise_scale * noise

    def predict_clean(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        clean_scale = self._extract(self.sqrt_alpha_bars, timesteps, noisy)
        noise_scale = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, noisy)
        return (noisy - noise_scale * predicted_noise) / clean_scale.clamp_min(1e-6)

