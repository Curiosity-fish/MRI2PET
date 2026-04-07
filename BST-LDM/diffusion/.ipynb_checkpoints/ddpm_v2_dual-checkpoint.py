
# diffusion/ddpm.py (v2)
# Supports:
# - cosine / linear beta schedule
# - eps- or v-parameterization
# - SNR / P2 loss weighting
# - DDPM ancestral sampling
# - DDIM sampling
# - Classifier-free guidance (CFG) over TABLE condition (drops table for uncond branch)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def extract(v: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """v: (T,), t: (B,) -> (B,1,1,1,1) broadcastable"""
    out = v.gather(0, t)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)).to(t.device, torch.float32)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule -> betas in (0,1)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999)

def snr_from_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    return alpha_bar / (1.0 - alpha_bar + 1e-8)

def p2_weight(snr: torch.Tensor, gamma: float = 0.5, k: float = 1.0) -> torch.Tensor:
    return (k + snr) ** (-gamma)

def snr_weight(snr: torch.Tensor) -> torch.Tensor:
    return snr / (snr + 1.0)

def v_target(x0: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0"""
    return torch.sqrt(alpha_bar) * eps - torch.sqrt(1.0 - alpha_bar) * x0

def x0_from_v(xt: torch.Tensor, v: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """x0 = sqrt(alpha_bar)*xt - sqrt(1-alpha_bar)*v"""
    return torch.sqrt(alpha_bar) * xt - torch.sqrt(1.0 - alpha_bar) * v

def eps_from_v(xt: torch.Tensor, v: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """eps = sqrt(1-alpha_bar)*xt + sqrt(alpha_bar)*v"""
    return torch.sqrt(1.0 - alpha_bar) * xt + torch.sqrt(alpha_bar) * v

class DiffusionDDPM(nn.Module):
    """
    Conditional DDPM/DDIM for MRI->PET latent denoising, supporting:
      - pred_type: 'eps' or 'v'
      - loss_weighting: 'none' | 'snr' | 'p2'
      - schedule: 'linear' | 'cosine'
      - CFG over TABLE: uncond branch = same MRI, but table=None
    """
    def __init__(
        self,
        model: nn.Module,
        T: int = 1000,
        beta: Tuple[float, float] = (1e-4, 2e-2),
        schedule: str = "cosine",
        pred_type: str = "v",
        loss_weighting: str = "p2",
        p2_gamma: float = 0.5,
        p2_k: float = 1.0,
        loss_weight_eps: float = 1.0,
        loss_weight_v: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.T = int(T)
        self.pred_type = str(pred_type).lower()
        assert self.pred_type in ("eps", "v")
        self.loss_weighting = str(loss_weighting).lower()
        assert self.loss_weighting in ("none", "snr", "p2")
        self.p2_gamma = float(p2_gamma)
        self.p2_k = float(p2_k)
        self.loss_weight_eps = float(loss_weight_eps)
        self.loss_weight_v = float(loss_weight_v)

        if str(schedule).lower() == "cosine":
            betas = cosine_beta_schedule(self.T)
        else:
            betas = torch.linspace(beta[0], beta[1], self.T, dtype=torch.float32)

        self.register_buffer("betas", betas)

        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)

        alpha_bar = torch.cumprod(alphas, dim=0)  # (T,)
        self.register_buffer("alphas_cumprod", alpha_bar)

        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bar[:-1]], dim=0)
        self.register_buffer("alphas_cumprod_prev", alpha_bar_prev)

        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

        posterior_mean_coef1 = betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)


    # -------------------------
    # dual-output helpers
    # -------------------------
    def _split_model_out(self, model_out):
        # model may return:
        #  - tensor (legacy)
        #  - (eps_pred, v_pred)
        if isinstance(model_out, (tuple, list)) and len(model_out) == 2:
            return model_out[0], model_out[1]
        return (model_out if self.pred_type == "eps" else None), (model_out if self.pred_type == "v" else None)

    def _pick_for_pred_type(self, model_out):
        eps_pred, v_pred = self._split_model_out(model_out)
        return eps_pred if self.pred_type == "eps" else v_pred

    def _cfg_combine(self, out_u, out_c, cfg_scale: float):
        # support tensor or (eps, v)
        if isinstance(out_u, (tuple, list)) and isinstance(out_c, (tuple, list)):
            return (out_u[0] + cfg_scale * (out_c[0] - out_u[0]),
                    out_u[1] + cfg_scale * (out_c[1] - out_u[1]))
        return out_u + cfg_scale * (out_c - out_u)
        # -------------------------
        # q(x_t|x0)
        # -------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_alpha_bar, t, x0.shape) * x0 + extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * noise

    # -------------------------
    # prediction helpers
    # -------------------------
    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, model_out) -> torch.Tensor:
        a = extract(self.alphas_cumprod, t, x_t.shape)
        out = self._pick_for_pred_type(model_out)
        if self.pred_type == "eps":
            return (x_t - torch.sqrt(1.0 - a) * out) / (torch.sqrt(a) + 1e-8)
        else:
            return x0_from_v(x_t, out, a)

    @torch.no_grad()
    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, model_out) -> torch.Tensor:
        a = extract(self.alphas_cumprod, t, x_t.shape)
        out = self._pick_for_pred_type(model_out)
        if self.pred_type == "eps":
            return out
        else:
            return eps_from_v(x_t, out, a)

    # -------------------------
    # training loss
    # -------------------------
    def forward(self, x0: torch.Tensor, mri: torch.Tensor, cate_x=None, conti_x=None, cond_drop_prob: float = 0.0) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        # classifier-free: drop TABLE only
        if cond_drop_prob and conti_x is not None:
            drop_mask = (torch.rand((B,), device=x0.device) < float(cond_drop_prob))
            if drop_mask.any():
                conti_x = conti_x.clone()
                conti_x[drop_mask] = 0.0
                # also disable cross-attn path by setting to None for those samples is hard in batch,
                # so we use zeros as "uncond table" representation (works well with standardization).

        # model prediction (can be tensor or (eps_pred, v_pred))
        model_out = self.model(x_t, mri, cate_x, conti_x, t)
        eps_pred, v_pred = self._split_model_out(model_out)

        alpha_bar = extract(self.alphas_cumprod, t, x0.shape)

        # targets
        target_eps = noise
        target_v = v_target(x0, noise, alpha_bar)

        # per-sample mse for each head (B,)
        loss_eps = 0.0
        loss_v = 0.0
        if eps_pred is not None:
            le = (eps_pred - target_eps) ** 2
            loss_eps = le.mean(dim=list(range(1, le.ndim)))
        if v_pred is not None:
            lv = (v_pred - target_v) ** 2
            loss_v = lv.mean(dim=list(range(1, lv.ndim)))

        # weighting by SNR (apply to both)
        if self.loss_weighting != "none":
            snr = snr_from_alpha_bar(alpha_bar.view(B))
            if self.loss_weighting == "snr":
                w = snr_weight(snr)
            else:
                w = p2_weight(snr, gamma=self.p2_gamma, k=self.p2_k)
            if isinstance(loss_eps, torch.Tensor):
                loss_eps = loss_eps * w
            if isinstance(loss_v, torch.Tensor):
                loss_v = loss_v * w

        loss = self.loss_weight_eps * loss_eps + self.loss_weight_v * loss_v
        return loss.mean()

    # -------------------------
    # DDPM sampling step
    # -------------------------
    @torch.no_grad()
    def p_mean_variance(self, x_t, mri, cate_x, conti_x, t, clip_x0: bool = True, cfg_scale: float = 1.0):
        # cfg over TABLE: uncond = table zeros (or None if you prefer)
        if cfg_scale != 1.0 and conti_x is not None:
            out_uncond = self.model(x_t, mri, None, torch.zeros_like(conti_x), t)
            out_cond   = self.model(x_t, mri, cate_x, conti_x, t)
            model_out  = self._cfg_combine(out_uncond, out_cond, cfg_scale)
        else:
            model_out = self.model(x_t, mri, cate_x, conti_x, t)

        x0_pred = self.predict_x0(x_t, t, model_out)
        if clip_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var  = extract(self.posterior_variance, t, x_t.shape)
        return mean, var, x0_pred

    @torch.no_grad()
    def p_sample(self, x_t, mri, cate_x=None, conti_x=None, t_int: int = 0, clip_x0: bool = True, cfg_scale: float = 1.0):
        B = x_t.shape[0]
        t = torch.full((B,), int(t_int), device=x_t.device, dtype=torch.long)

        mean, var, x0_pred = self.p_mean_variance(x_t, mri, cate_x, conti_x, t, clip_x0=clip_x0, cfg_scale=cfg_scale)
        if t_int == 0:
            return x0_pred

        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_ddpm(self, shape, mri, cate_x=None, conti_x=None, clip_x0: bool = True, cfg_scale: float = 1.0, progress: bool = True):
        x = torch.randn(shape, device=mri.device)
        rng = range(self.T - 1, -1, -1)
        if progress:
            rng = tqdm(rng, total=self.T, desc="DDPM sampling")
        for t in rng:
            x = self.p_sample(x, mri, cate_x, conti_x, t_int=t, clip_x0=clip_x0, cfg_scale=cfg_scale)
        return x

    # -------------------------
    # DDIM sampling
    # -------------------------
    @torch.no_grad()
    def sample_ddim(
        self,
        shape,
        mri,
        cate_x=None,
        conti_x=None,
        steps: int = 50,
        eta: float = 0.0,
        clip_x0: bool = True,
        cfg_scale: float = 1.0,
        progress: bool = True,
    ):
        device = mri.device
        x = torch.randn(shape, device=device)
        steps = int(steps)
        times = torch.linspace(self.T - 1, 0, steps, device=device).long()

        it = list(times.tolist())
        if progress:
            it = tqdm(it, total=len(it), desc="DDIM sampling")

        for i, t_int in enumerate(it):
            t = torch.full((shape[0],), int(t_int), device=device, dtype=torch.long)

            # cfg over TABLE
            if cfg_scale != 1.0 and conti_x is not None:
                out_uncond = self.model(x, mri, None, torch.zeros_like(conti_x), t)
                out_cond   = self.model(x, mri, cate_x, conti_x, t)
                model_out  = self._cfg_combine(out_uncond, out_cond, cfg_scale)
            else:
                model_out = self.model(x, mri, cate_x, conti_x, t)

            alpha_bar_t = extract(self.alphas_cumprod, t, x.shape)
            x0 = self.predict_x0(x, t, model_out)
            if clip_x0:
                x0 = x0.clamp(-1.0, 1.0)

            eps = self.predict_eps(x, t, model_out)

            # next timestep
            if i == len(times) - 1:
                x = x0
                continue

            t_prev_int = int(it[i + 1])
            t_prev = torch.full((shape[0],), t_prev_int, device=device, dtype=torch.long)
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)

            if eta > 0.0:
                sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            else:
                sigma = 0.0

            c1 = torch.sqrt(alpha_bar_prev)
            c2 = torch.sqrt(1 - alpha_bar_prev - sigma * sigma)
            noise = torch.randn_like(x) if eta > 0.0 else 0.0
            x = c1 * x0 + c2 * eps + sigma * noise

        return x