
# ema.py
from __future__ import annotations
import copy
import torch
from torch import nn

class EMA:
    """Exponential Moving Average (EMA) for model weights.S

    Usage:
        ema = EMA(model, decay=0.9999)
        ...
        ema.update(model)  # after optimizer step
        ...
        ema.ema_model.eval()  # use for validation / sampling
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: torch.device | None = None):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        for k, v in esd.items():
            if k not in msd:
                continue
            src = msd[k].detach()
            if not torch.is_floating_point(src):
                esd[k].copy_(src)
            else:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "ema_model": self.ema_model.state_dict()}

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        self.ema_model.load_state_dict(state["ema_model"], strict=True)
