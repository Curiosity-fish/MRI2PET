from __future__ import annotations

from typing import Dict, Optional

import torch

from losses.diffusion_losses import ensure_mask


def summarize_tensor(name: str, tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    tensor = tensor.float()
    if mask is not None:
        weight = ensure_mask(mask, tensor)
        values = tensor[weight > 0]
    else:
        values = tensor.reshape(-1)
    if values.numel() == 0:
        return {
            f"{name}_min": float("nan"),
            f"{name}_max": float("nan"),
            f"{name}_mean": float("nan"),
            f"{name}_std": float("nan"),
        }
    return {
        f"{name}_min": float(values.min().item()),
        f"{name}_max": float(values.max().item()),
        f"{name}_mean": float(values.mean().item()),
        f"{name}_std": float(values.std().item()),
    }
