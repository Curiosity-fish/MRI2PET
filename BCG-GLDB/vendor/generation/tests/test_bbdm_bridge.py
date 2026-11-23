"""CPU-only checks for the strict stochastic BBDM bridge schedule."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bcg_gldb.bridge import bridge_transition, bridge_variance


def main() -> None:
    sigma_max = 0.25
    times = torch.tensor([0.0, 0.5, 1.0])
    variance = bridge_variance(times, sigma_max)
    expected = torch.tensor([0.0, sigma_max**2, 0.0])
    assert torch.allclose(variance, expected, atol=1e-7), (variance, expected)

    x_t = torch.zeros(2, 1, 3, 3, 3)
    z_pet = torch.ones_like(x_t)
    t = torch.full((2,), 0.2)
    next_t = torch.full((2,), 0.4)
    deterministic = bridge_transition(
        x_t, t, next_t, z_pet, sigma_max, eta=0.0
    )
    stochastic = bridge_transition(
        x_t,
        t,
        next_t,
        z_pet,
        sigma_max,
        eta=1.0,
        noise=torch.ones_like(x_t),
    )
    assert not torch.allclose(stochastic, deterministic)

    final = bridge_transition(
        stochastic,
        torch.full((2,), 0.9),
        torch.ones(2),
        z_pet,
        sigma_max,
        eta=1.0,
        noise=torch.randn_like(x_t),
    )
    assert torch.allclose(final, z_pet, atol=1e-7)
    print("bbdm_bridge_ok endpoints=exact interior=stochastic steps=10")


if __name__ == "__main__":
    main()
