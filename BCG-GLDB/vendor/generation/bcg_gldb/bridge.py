"""Scaled Brownian-bridge schedule and discrete stochastic transition.

The sampling-time convention used by BCG-GLDB is source oriented: ``t=0`` is
the MRI latent endpoint and ``t=1`` is the predicted PET latent endpoint.  It
is the time-reversed convention of the original BBDM implementation, while
retaining the same Brownian-bridge marginal and Markov transition.
"""

from __future__ import annotations

import torch


def bridge_variance(t: torch.Tensor, sigma_max: float) -> torch.Tensor:
    """Return the scaled BBDM marginal variance at normalized time ``t``.

    BBDM uses a variance proportional to ``m_t (1 - m_t)``.  Here the scale is
    parameterized by the maximum standard deviation so that
    ``Var(t=0.5) = sigma_max ** 2`` and both endpoint variances are exactly 0.
    """

    t = t.clamp(0.0, 1.0)
    return 4.0 * float(sigma_max) ** 2 * t * (1.0 - t)


def bridge_sigma(t: torch.Tensor, sigma_max: float) -> torch.Tensor:
    """Return the standard deviation of the scaled Brownian bridge."""

    return bridge_variance(t, sigma_max).clamp_min(0.0).sqrt()


def bridge_transition(
    x_t: torch.Tensor,
    t: torch.Tensor,
    next_t: torch.Tensor,
    z_pet: torch.Tensor,
    sigma_max: float,
    *,
    eta: float = 1.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample ``q(z_next | z_t, z_pet)`` for a scaled Brownian bridge.

    ``z_pet`` is the endpoint predicted at the current step.  ``eta=1`` gives
    the strict stochastic transition; ``eta=0`` is retained only as an
    explicit deterministic diagnostic.  The final transition to ``next_t=1``
    has zero variance and therefore reaches the predicted PET endpoint.
    """

    if x_t.shape != z_pet.shape:
        raise ValueError(
            f"x_t and z_pet must have identical shapes, got {x_t.shape} and {z_pet.shape}"
        )
    if t.ndim != 1 or next_t.ndim != 1 or t.shape != next_t.shape:
        raise ValueError("t and next_t must be one-dimensional tensors with equal shape")
    if t.shape[0] != x_t.shape[0]:
        raise ValueError("one timestep value is required for each batch element")
    if torch.any(next_t <= t) or torch.any(t < 0) or torch.any(next_t > 1):
        raise ValueError("bridge_transition requires 0 <= t < next_t <= 1")

    expand = (-1,) + (1,) * (x_t.ndim - 1)
    m_t = t.reshape(expand)
    m_next = next_t.reshape(expand)
    var_t = bridge_variance(t, sigma_max).reshape(expand)
    var_next = bridge_variance(next_t, sigma_max).reshape(expand)

    # BBDM forward conditional transition, written in the MRI->PET time
    # convention.  This is algebraically equivalent to reversing the time
    # index of the canonical BBDM posterior sampler.
    source_ratio = (1.0 - m_next) / (1.0 - m_t).clamp_min(1e-8)
    endpoint_ratio = m_next - source_ratio * m_t
    mean = source_ratio * x_t + endpoint_ratio * z_pet
    transition_variance = (var_next - source_ratio.square() * var_t).clamp_min(0.0)

    if float(eta) == 0.0 or not bool(torch.any(transition_variance > 0)):
        return mean
    if noise is None:
        noise = torch.randn_like(x_t)
    elif noise.shape != x_t.shape:
        raise ValueError(f"noise must have shape {x_t.shape}, got {noise.shape}")
    return mean + float(eta) * transition_variance.sqrt() * noise
