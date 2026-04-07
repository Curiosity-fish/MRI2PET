# diffusion.py (DDPM)
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def extract(v: torch.Tensor, t: torch.Tensor, x_shape):
    """
    v: (T,) buffer
    t: (B,) long
    return: (B,1,1,1,1) broadcastable to x
    """
    out = v.gather(0, t)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)).to(t.device, torch.float32)

class DiffusionDDPM(nn.Module):
    """
    DDPM with epsilon-prediction training and ancestral sampling.
    """
    def __init__(self, model: nn.Module, beta=(1e-4, 2e-2), T=1000):
        super().__init__()
        self.model = model
        self.T = int(T)

        # betas: (T,)
        betas = torch.linspace(beta[0], beta[1], self.T, dtype=torch.float32)
        self.register_buffer("betas", betas)

        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)

        alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # useful sqrt terms
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # posterior variance q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

        # posterior mean coefficients:
        # mean = coef1 * x0 + coef2 * x_t
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    # -------------------------
    # Forward process q(x_t|x0)
    # -------------------------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    # -------------------------
    # Training loss (DDPM L_simple)
    # -------------------------
    def forward(self, x0, mri, cate_x=None, conti_x=None):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        # model predicts epsilon
        eps_theta = self.model(x_t, mri, cate_x, conti_x, t)
        return F.mse_loss(eps_theta, noise)

    # -------------------------
    # Reverse process p(x_{t-1}|x_t)
    # -------------------------
    @torch.no_grad()
    def predict_x0_from_eps(self, x_t, t, eps):
        # x0 = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)
        return (
            x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps
        ) / (extract(self.sqrt_alphas_cumprod, t, x_t.shape) + 1e-8)

    @torch.no_grad()
    def p_mean_variance(self, x_t, mri, cate_x, conti_x, t, clip_x0=True):
        eps_theta = self.model(x_t, mri, cate_x, conti_x, t)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_theta)

        if clip_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean, var, x0_pred

    @torch.no_grad()
    def p_sample(self, x_t, mri, cate_x=None, conti_x=None, t_int=None, clip_x0=True):
        """
        sample x_{t-1} from p(x_{t-1}|x_t)
        """
        if t_int is None:
            raise ValueError("t_int must be provided (python int).")
        B = x_t.shape[0]
        t = torch.full((B,), t_int, device=x_t.device, dtype=torch.long)

        mean, var, x0_pred = self.p_mean_variance(x_t, mri, cate_x, conti_x, t, clip_x0=clip_x0)

        if t_int == 0:
            return x0_pred  # final

        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, shape, mri, cate_x=None, conti_x=None, clip_x0=True, progress=True):
        """
        DDPM ancestral sampling: start from N(0,1) and iterate t=T-1..0
        shape: (B,C,H,W,D)
        mri: conditioning tensor
        """
        x = torch.randn(shape, device=mri.device)
        rng = range(self.T - 1, -1, -1)
        if progress:
            rng = tqdm(rng, total=self.T, desc="DDPM sampling")

        for t in rng:
            x = self.p_sample(x, mri, cate_x, conti_x, t_int=t, clip_x0=clip_x0)
        return x
