from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from . import resnet


DEPTH_TO_CONSTRUCTOR = {
    10: resnet.resnet10,
    18: resnet.resnet18,
    34: resnet.resnet34,
    50: resnet.resnet50,
    101: resnet.resnet101,
    152: resnet.resnet152,
    200: resnet.resnet200,
}


class MedicalNetClassifier(nn.Module):
    def __init__(
        self,
        model_depth: int = 18,
        n_classes: int = 3,
        in_channels: int = 1,
        shortcut_type: str = "A",
        dropout: float = 0.0,
        head_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if model_depth not in DEPTH_TO_CONSTRUCTOR:
            raise ValueError(f"Unsupported MedicalNet depth: {model_depth}")

        self.backbone = DEPTH_TO_CONSTRUCTOR[model_depth](
            shortcut_type=shortcut_type,
            in_channels=in_channels,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        hidden_dims = list(head_hidden_dims or [])
        dims = [self.backbone.out_channels] + hidden_dims + [n_classes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.classifier = nn.Sequential(*layers)
        self.num_classes = n_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.classifier(x)


def _unwrap_checkpoint_state_dict(checkpoint: dict[str, Any] | torch.Tensor) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format for MedicalNet weights.")


def load_medicalnet_pretrained(
    model: MedicalNetClassifier,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = _unwrap_checkpoint_state_dict(checkpoint)

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("backbone."):
            cleaned[new_key] = value
            continue
        if new_key.startswith("conv_seg"):
            continue
        cleaned[f"backbone.{new_key}"] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if verbose:
        print(f"Loaded MedicalNet weights from {checkpoint_path}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return missing, unexpected


def create_medicalnet_classifier(
    model_depth: int = 18,
    n_classes: int = 3,
    in_channels: int = 1,
    shortcut_type: str = "A",
    dropout: float = 0.0,
    head_hidden_dims: list[int] | None = None,
    checkpoint_path: str | Path | None = None,
) -> MedicalNetClassifier:
    model = MedicalNetClassifier(
        model_depth=model_depth,
        n_classes=n_classes,
        in_channels=in_channels,
        shortcut_type=shortcut_type,
        dropout=dropout,
        head_hidden_dims=head_hidden_dims,
    )
    if checkpoint_path:
        load_medicalnet_pretrained(model, checkpoint_path)
    return model
