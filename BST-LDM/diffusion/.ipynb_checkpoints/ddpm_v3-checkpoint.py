
# ddpm_v3_dual.py
# DiffusionDDPM with:
# - cosine noise schedule
# - dual objective: eps-pred + v-pred (combined loss)
# - P2 / SNR loss weighting
# - CFG (classifier-free guidance) support in sampling
# - DDIM sampling for fast final-generation validation

from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Literal

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule (Nichol & Dhariwal, 2021).
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999)


def extract(v: torch.Tensor, t: torch.Tensor, x_shape):
    out = v.gather(0, t)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)).to(t.device, torch.float32)


class DiffusionDDPM(nn.Module):
    """
    Dual-objective diffusion:
      - UNet outputs (eps_pred, v_pred)
      - loss = w_eps * MSE(eps_pred, eps) + w_v * MSE(v_pred, v), optionally P2-weighted by SNR
    """
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule: Literal["cosine"] = "cosine",
        cosine_s: float = 0.008,
        p2_gamma: float = 1.0,
        p2_k: float = 1.0,
        loss_weight_eps: float = 1.0,
        loss_weight_v: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.T = int(timesteps)
        self.schedule = schedule

        if schedule == "cosine":
            betas = cosine_beta_schedule(self.T, s=cosine_s)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # posterior variance q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        # loss weighting (P2)
        self.p2_gamma = float(p2_gamma)
        self.p2_k = float(p2_k)
        self.loss_weight_eps = float(loss_weight_eps)
        self.loss_weight_v = float(loss_weight_v)

    # -------------------------
    # Forward process q(x_t | x_0)
    # -------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

    # v target per Stable Diffusion convention:
    # v = alpha_t * eps - sigma_t * x0, with alpha_t = sqrt(alpha_bar), sigma_t = sqrt(1 - alpha_bar)
    def v_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return alpha * noise - sigma * x0

    def snr_weight(self, t: torch.Tensor, x_shape) -> torch.Tensor:
        # SNR = alpha_bar / (1 - alpha_bar)
        alpha_bar = extract(self.alphas_cumprod, t, x_shape)
        snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
        # P2 weight: (k + snr)^(-gamma)
        w = (self.p2_k + snr) ** (-self.p2_gamma)
        return w

    def forward(self, x0: torch.Tensor, mri: Optional[torch.Tensor], cate_x=None, conti_x=None) -> Dict[str, torch.Tensor]:
        """
        Returns dict with total loss + per-objective losses.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        eps_pred, v_pred = self.model(x_t, mri, cate_x, conti_x, t)

        eps_tgt = noise
        v_tgt = self.v_target(x0, t, noise)

        # per-sample weighting
        w = self.snr_weight(t, x0.shape)  # (B,1,1,1,1)

        eps_loss = F.mse_loss(eps_pred, eps_tgt, reduction="none")
        v_loss = F.mse_loss(v_pred, v_tgt, reduction="none")

        # mean over non-batch dims then apply weights
        while w.ndim < eps_loss.ndim:
            w = w
        eps_loss = (eps_loss.mean(dim=tuple(range(1, eps_loss.ndim))) * w.view(B)).mean()
        v_loss = (v_loss.mean(dim=tuple(range(1, v_loss.ndim))) * w.view(B)).mean()

        total = self.loss_weight_eps * eps_loss + self.loss_weight_v * v_loss
        return {"loss": total, "loss_eps": eps_loss, "loss_v": v_loss}

    # -------------------------
    # Helpers: convert between eps / v / x0
    # -------------------------
    @torch.no_grad()
    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sigma * eps) / (alpha + 1e-8)

    @torch.no_grad()
    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # x0 = alpha * x_t - sigma * v  (since alpha^2 + sigma^2 = 1)
        return alpha * x_t - sigma * v

    @torch.no_grad()
    def predict_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # eps = sigma * x_t + alpha * v
        return sigma * x_t + alpha * v

    # -------------------------
    # Reverse process p(x_{t-1}|x_t)
    # -------------------------
    @torch.no_grad()
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        mri: Optional[torch.Tensor],
        cate_x,
        conti_x,
        t: torch.Tensor,
        clip_x0: bool = True,
        cfg_scale: float = 1.0,
        use_ensemble_x0: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: mean, var, x0_pred
        """
        if cfg_scale != 1.0:
            eps_u, v_u = self.model(x_t, None, None, None, t)
            eps_c, v_c = self.model(x_t, mri, cate_x, conti_x, t)
            eps = eps_u + cfg_scale * (eps_c - eps_u)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            eps, v = self.model(x_t, mri, cate_x, conti_x, t)

        x0_eps = self.predict_x0_from_eps(x_t, t, eps)
        x0_v = self.predict_x0_from_v(x_t, t, v)
        x0_pred = (x0_eps + x0_v) * 0.5 if use_ensemble_x0 else x0_v

        if clip_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean, var, x0_pred

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        mri: Optional[torch.Tensor],
        cate_x,
        conti_x,
        t_int: int,
        clip_x0: bool = True,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        t = torch.full((B,), t_int, device=x_t.device, dtype=torch.long)
        mean, var, x0_pred = self.p_mean_variance(x_t, mri, cate_x, conti_x, t, clip_x0=clip_x0, cfg_scale=cfg_scale)
        if t_int == 0:
            return x0_pred
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_ddpm(
        self,
        shape: Tuple[int, ...],
        mri: Optional[torch.Tensor],
        cate_x=None,
        conti_x=None,
        clip_x0: bool = True,
        cfg_scale: float = 1.0,
        progress: bool = True,
    ) -> torch.Tensor:
        device = mri.device if mri is not None else next(self.parameters()).device
        x = torch.randn(shape, device=device)
        rng = range(self.T - 1, -1, -1)
        if progress:
            rng = tqdm(rng, total=self.T, desc="DDPM sampling")
        for t in rng:
            x = self.p_sample(x, mri, cate_x, conti_x, t_int=t, clip_x0=clip_x0, cfg_scale=cfg_scale)
        return x

    # -------------------------
    # DDIM sampling (fast)
    # -------------------------
    @torch.no_grad()
    def sample_ddim(
        self,
        shape: Tuple[int, ...],
        mri: Optional[torch.Tensor],
        cate_x=None,
        conti_x=None,
        steps: int = 50,
        eta: float = 0.0,
        clip_x0: bool = True,
        cfg_scale: float = 1.0,
        progress: bool = True,
        use_ensemble_x0: bool = True,
    ) -> torch.Tensor:
        device = mri.device if mri is not None else next(self.parameters()).device
        x = torch.randn(shape, device=device)

        # choose timesteps
        steps = int(steps)
        times = torch.linspace(self.T - 1, 0, steps, device=device).long()

        if progress:
            times_iter = tqdm(range(len(times)), desc="DDIM sampling", total=len(times))
        else:
            times_iter = range(len(times))

        for i in times_iter:
            t = times[i]
            t_batch = torch.full((shape[0],), t.item(), device=device, dtype=torch.long)

            # predict eps/v with CFG
            if cfg_scale != 1.0:
                eps_u, v_u = self.model(x, None, None, None, t_batch)
                eps_c, v_c = self.model(x, mri, cate_x, conti_x, t_batch)
                eps = eps_u + cfg_scale * (eps_c - eps_u)
                v = v_u + cfg_scale * (v_c - v_u)
            else:
                eps, v = self.model(x, mri, cate_x, conti_x, t_batch)

            x0_eps = self.predict_x0_from_eps(x, t_batch, eps)
            x0_v = self.predict_x0_from_v(x, t_batch, v)
            x0 = (x0_eps + x0_v) * 0.5 if use_ensemble_x0 else x0_v
            if clip_x0:
                x0 = x0.clamp(-1.0, 1.0)

            # derive eps for update (prefer v-derived for consistency)
            eps_for_update = self.predict_eps_from_v(x, t_batch, v)

            if i == len(times) - 1:
                x = x0
                continue

            t_prev = times[i + 1]
            t_prev_batch = torch.full((shape[0],), t_prev.item(), device=device, dtype=torch.long)

            a_t = extract(self.alphas_cumprod, t_batch, x.shape)
            a_prev = extract(self.alphas_cumprod, t_prev_batch, x.shape)

            # DDIM sigma
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0.0)
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            # direction pointing to x_t
            dir_xt = torch.sqrt(1 - a_prev - sigma**2) * eps_for_update
            x = torch.sqrt(a_prev) * x0 + dir_xt + sigma * noise

        return x