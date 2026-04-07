# diffusion/ddpm_table.py
# Conditional DDPM/DDIM for MRI->PET latent denoising.
#
# Drop-in replacement for your current diffusion/ddpm_table.py.
#
# Key additions
# - Optional dual-target training: optimize BOTH eps-target and v-target.
# - Supports dual-head denoiser output: 2*C channels where C is x0 channel count.
#   Default order: [v, eps] along channel dim.
#
# Notes
# - Sampling remains governed by `pred_type` (default: 'v').
# - The second head (e.g., eps head) is primarily used as an auxiliary loss; sampler can remain v-param.

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm


def extract(v: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """v: (T,), t: (B,) -> broadcastable to x_shape."""
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


def _looks_like_binary_mask(x: torch.Tensor) -> bool:
    """Heuristic check: values approximately in {0,1}."""
    if x is None or x.numel() == 0:
        return False
    # allow tiny numerical noise
    x_min = float(x.min().detach().cpu())
    x_max = float(x.max().detach().cpu())
    if x_min < -0.05 or x_max > 1.05:
        return False
    diff = (x - x.round()).abs().max().detach().cpu().item()
    return diff < 1e-3


def _infer_concat_half(conti_x: torch.Tensor) -> Optional[int]:
    """If conti_x is likely concat([values, missing_mask]), return half dim; else None."""
    if conti_x is None or conti_x.ndim != 2:
        return None
    D = int(conti_x.shape[1])
    if D % 2 != 0:
        return None
    half = D // 2
    mask = conti_x[:, half:]
    if _looks_like_binary_mask(mask):
        return half
    return None


def _make_uncond_table_like(conti_x: torch.Tensor) -> torch.Tensor:
    """Build an unconditional "table" representation.

    - values -> 0
    - missing_mask -> 1 (so attention can ignore all tokens)
    """
    z = torch.zeros_like(conti_x)
    half = _infer_concat_half(conti_x)
    if half is not None:
        z[:, half:] = 1.0
    return z


def _drop_table_inplace(conti_x: torch.Tensor, drop_mask: torch.Tensor) -> torch.Tensor:
    """Apply classifier-free condition dropping on conti_x (in a clone)."""
    conti_x = conti_x.clone()
    half = _infer_concat_half(conti_x)
    if half is not None:
        conti_x[drop_mask, :half] = 0.0
        conti_x[drop_mask, half:] = 1.0
    else:
        conti_x[drop_mask] = 0.0
    return conti_x


class DiffusionDDPM(nn.Module):
    """Conditional DDPM/DDIM for MRI->PET latent denoising.

    Core options
    - pred_type: 'eps' or 'v'
    - loss_weighting: 'none' | 'snr' | 'p2'
    - schedule: 'linear' | 'cosine'

    Table conditioning
    - Classifier-free guidance (CFG) over TABLE:
        uncond branch keeps MRI, but uses a synthetic "uncond table".

    Dual-target training (optional)
    - If use_dual_target_loss=True, we compute:
        L = loss_v_weight * L_v + loss_eps_weight * L_eps
      where each L_* is an SNR-weighted per-sample MSE (if enabled).

    Dual-head output (optional)
    - If model outputs 2*C channels, interpreted as two heads.
      Default order: [v, eps] (configurable via dual_out_order).
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
        # --- dual-target training ---
        use_dual_target_loss: bool = True,
        loss_v_weight: float = 1.0,
        loss_eps_weight: float = 1.0,
        dual_out_order: str = "v_eps",  # 'v_eps' or 'eps_v'
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

        self.use_dual_target_loss = bool(use_dual_target_loss)
        self.loss_v_weight = float(loss_v_weight)
        self.loss_eps_weight = float(loss_eps_weight)

        self.dual_out_order = str(dual_out_order).lower()
        assert self.dual_out_order in ("v_eps", "eps_v")

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
    # q(x_t|x0)
    # -------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_alpha_bar, t, x0.shape) * x0 + extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * noise

    # -------------------------
    # model output parsing
    # -------------------------
    def _split_dual_heads(self, model_out: torch.Tensor, C: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (v_head, eps_head). If not dual, returns (None, None)."""
        if model_out.ndim < 2:
            return None, None
        if int(model_out.shape[1]) != 2 * int(C):
            return None, None
        if self.dual_out_order == "v_eps":
            v = model_out[:, :C]
            eps = model_out[:, C:]
        else:
            eps = model_out[:, :C]
            v = model_out[:, C:]
        return v, eps

    def _select_main_pred(self, model_out: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """Pick the head that matches self.pred_type, or return model_out if single-head."""
        C = int(x_shape[1])
        v_head, eps_head = self._split_dual_heads(model_out, C)
        if v_head is None:
            return model_out
        return v_head if self.pred_type == "v" else eps_head

    # -------------------------
    # prediction helpers
    # -------------------------
    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        a = extract(self.alphas_cumprod, t, x_t.shape)
        main = self._select_main_pred(model_out, x_t.shape)
        if self.pred_type == "eps":
            return (x_t - torch.sqrt(1.0 - a) * main) / (torch.sqrt(a) + 1e-8)
        else:
            return x0_from_v(x_t, main, a)

    @torch.no_grad()
    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        a = extract(self.alphas_cumprod, t, x_t.shape)
        main = self._select_main_pred(model_out, x_t.shape)
        if self.pred_type == "eps":
            return main
        else:
            return eps_from_v(x_t, main, a)

    # -------------------------
    # loss helpers
    # -------------------------
    def _weight_per_sample(self, loss_b: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        """loss_b: (B,), alpha_bar: (B,1,1,1,1) -> weighted (B,)"""
        if self.loss_weighting == "none":
            return loss_b
        B = int(loss_b.shape[0])
        snr = snr_from_alpha_bar(alpha_bar.view(B))
        if self.loss_weighting == "snr":
            w = snr_weight(snr)
        else:
            w = p2_weight(snr, gamma=self.p2_gamma, k=self.p2_k)
        return loss_b * w

    # -------------------------
    # training loss
    # -------------------------
    def forward(self, x0: torch.Tensor, mri: torch.Tensor, cate_x=None, conti_x=None, cond_drop_prob: float = 0.0) -> torch.Tensor:
        B = int(x0.shape[0])
        C = int(x0.shape[1])

        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        # classifier-free: drop TABLE only
        if cond_drop_prob and conti_x is not None:
            drop_mask = (torch.rand((B,), device=x0.device) < float(cond_drop_prob))
            if drop_mask.any():
                conti_x = _drop_table_inplace(conti_x, drop_mask)

        model_out = self.model(x_t, mri, cate_x, conti_x, t)

        alpha_bar = extract(self.alphas_cumprod, t, x0.shape)

        # ---- default (single-target) behavior ----
        if not self.use_dual_target_loss:
            pred = self._select_main_pred(model_out, x0.shape)
            if self.pred_type == "eps":
                target = noise
            else:
                target = v_target(x0, noise, alpha_bar)

            loss_b = (pred - target).pow(2).mean(dim=tuple(range(1, pred.ndim)))  # (B,)
            loss_b = self._weight_per_sample(loss_b, alpha_bar)
            return loss_b.mean()

        # ---- dual-target training ----
        # targets
        tgt_eps = noise
        tgt_v = v_target(x0, noise, alpha_bar)

        # predictions
        v_head, eps_head = self._split_dual_heads(model_out, C)
        if v_head is not None:
            pred_v = v_head
            pred_eps = eps_head
        else:
            # single-head fallback (still gives you a well-defined dual loss)
            if self.pred_type == "v":
                pred_v = model_out
                pred_eps = eps_from_v(x_t, pred_v, alpha_bar)
            else:
                pred_eps = model_out
                # v = sqrt(a)*eps - sqrt(1-a)*x0 (using GT x0). This makes L_v a scaled form of L_eps.
                pred_v = torch.sqrt(alpha_bar) * pred_eps - torch.sqrt(1.0 - alpha_bar) * x0

        loss_v_b = (pred_v - tgt_v).pow(2).mean(dim=tuple(range(1, pred_v.ndim)))  # (B,)
        loss_eps_b = (pred_eps - tgt_eps).pow(2).mean(dim=tuple(range(1, pred_eps.ndim)))  # (B,)

        loss_v_b = self._weight_per_sample(loss_v_b, alpha_bar)
        loss_eps_b = self._weight_per_sample(loss_eps_b, alpha_bar)

        total_b = self.loss_v_weight * loss_v_b + self.loss_eps_weight * loss_eps_b
        return total_b.mean()

    # -------------------------
    # DDPM sampling step
    # -------------------------
    @torch.no_grad()
    def p_mean_variance(self, x_t, mri, cate_x, conti_x, t, clip_x0: bool = True, cfg_scale: float = 1.0):
        # cfg over TABLE
        if cfg_scale != 1.0 and conti_x is not None:
            conti_uncond = _make_uncond_table_like(conti_x)
            out_uncond = self.model(x_t, mri, None, conti_uncond, t)
            out_cond = self.model(x_t, mri, cate_x, conti_x, t)
            model_out = out_uncond + cfg_scale * (out_cond - out_uncond)
        else:
            model_out = self.model(x_t, mri, cate_x, conti_x, t)

        x0_pred = self.predict_x0(x_t, t, model_out)
        if clip_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
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
                conti_uncond = _make_uncond_table_like(conti_x)
                out_uncond = self.model(x, mri, None, conti_uncond, t)
                out_cond = self.model(x, mri, cate_x, conti_x, t)
                model_out = out_uncond + cfg_scale * (out_cond - out_uncond)
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
