"""Chapter 4 classifiers and the seven named comparison configurations."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from . import _vendor  # noqa: F401
from medicalnet50.classifier import MedicalNetClassifier
from medicalnet50.resnet import resnet50
from exp.exp037B_v4.exp037B_v4_unet3d import ClinicalTableEncoder
from models.exp037c_global_local_vit_unet3d import GlobalViT3D


TABLE_TYPES = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2] + [4] * 14


def initialize_medicalnet50(model, checkpoint_path):
    """Load a Med3D ResNet-50 backbone and adapt its first convolution to 1/2 inputs."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(source, dict):
        raise ValueError("MedicalNet checkpoint must contain a tensor state dictionary")
    backbone = model.model.backbone if isinstance(model, IgnoreTable) else model.backbone
    target = backbone.state_dict()
    loaded = {}
    for name, value in source.items():
        name = name.removeprefix("module.").removeprefix("backbone.")
        if name not in target or not isinstance(value, torch.Tensor):
            continue
        if name == "conv1.weight" and value.ndim == 5 and value.shape[1] != target[name].shape[1]:
            value = value.mean(dim=1, keepdim=True).repeat(1, target[name].shape[1], 1, 1, 1)
            value = value / target[name].shape[1]
        if value.shape == target[name].shape:
            loaded[name] = value
    essential = ("conv1.weight", "layer1.0.conv1.weight", "layer4.2.conv3.weight")
    absent = [name for name in essential if name not in loaded]
    if absent:
        raise ValueError(f"The supplied checkpoint is not a compatible Med3D ResNet-50: {absent}")
    backbone.load_state_dict(loaded, strict=False)
    return len(loaded)


class CappedBiCA(nn.Module):
    """Two-way image/table attention with fixed residual scales and smooth caps."""

    def __init__(self, channels, token_dim=128, heads=4,
                 image_scale=0.10, table_scale=0.05,
                 image_cap=0.02, table_cap=0.10):
        super().__init__()
        self.channels = channels
        self.token_dim = token_dim
        self.image_scale = image_scale
        self.table_scale = table_scale
        self.image_cap = image_cap
        self.table_cap = table_cap
        self.image_projection = nn.Conv3d(channels, token_dim, 1)
        self.image_norm = nn.LayerNorm(token_dim)
        self.table_norm = nn.LayerNorm(token_dim)
        self.image_attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.table_attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.output_projection = nn.Conv3d(token_dim, channels, 1)
        nn.init.normal_(self.output_projection.weight, std=5e-3)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def cap(reference, delta, max_ratio):
        reference_norm = reference.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        ratio = delta.float().flatten(1).norm(dim=1) / reference_norm
        x = ratio / max_ratio
        factor = torch.where(x > 1e-6, torch.tanh(x) / x.clamp_min(1e-6), torch.ones_like(x))
        return delta * factor.reshape(-1, *([1] * (delta.ndim - 1))).to(delta.dtype)

    def forward(self, image, tokens, key_padding_mask, observed, reliability):
        batch, _, depth, height, width = image.shape
        projected = self.image_projection(image).flatten(2).transpose(1, 2)
        table_kv = self.table_norm(tokens)
        from_table = self.image_attention(self.image_norm(projected), table_kv, table_kv,
                                          key_padding_mask=key_padding_mask, need_weights=False)[0]
        image_kv = self.image_norm(projected)
        from_image = self.table_attention(self.table_norm(tokens), image_kv, image_kv,
                                          need_weights=False)[0]
        image_delta = self.output_projection(from_table.transpose(1, 2).reshape(
            batch, self.token_dim, depth, height, width))
        image_delta = self.image_scale * image_delta * reliability[:, :, None, None, None]
        image_next = image + self.cap(image, image_delta, self.image_cap)
        table_delta = self.table_scale * from_image * observed[:, :, None].to(from_image.dtype)
        table_next = tokens + self.cap(tokens, table_delta, self.table_cap)
        return image_next, table_next


class TGLM(nn.Module):
    """ResNet-50 + f3/f4 BiCA + global visual branch + 25 clinical fields."""

    def __init__(self, in_channels=2, use_table=True, use_global=True):
        super().__init__()
        self.use_table = bool(use_table)
        self.use_global = bool(use_global)
        self.backbone = resnet50(shortcut_type="B", in_channels=in_channels)
        self.table_encoder = ClinicalTableEncoder(25, TABLE_TYPES, token_dim=128,
                                                  depth=2, heads=4, ff_mult=4)
        self.bica3 = CappedBiCA(1024)
        self.bica4 = CappedBiCA(2048)
        self.global_vit = GlobalViT3D(2048, token_dim=256, depth=3, heads=8,
                                      bottleneck_shape=(5, 6, 5))
        self.global_projection = nn.Conv3d(2048, 2048, 1)
        self.global_gate = nn.Sequential(nn.Conv3d(2048, 2048, 1), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(1024 + 2048 + 2048 + 128, 512),
                                  nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3))

    def forward(self, image, table, missing):
        backbone = self.backbone
        x = backbone.relu(backbone.bn1(backbone.conv1(image)))
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        f3 = backbone.layer3(x)
        f4 = backbone.layer4(f3)
        table_vector = torch.zeros(image.shape[0], 128, device=image.device, dtype=f4.dtype)
        if self.use_table:
            state = self.table_encoder(table, missing)
            tokens = state["tokens"]
            f3, tokens = self.bica3(f3, tokens, state["key_padding_mask"],
                                    state["effective_observed"], state["reliability"])
            f4, tokens = self.bica4(f4, tokens, state["key_padding_mask"],
                                    state["effective_observed"], state["reliability"])
            observed = state["effective_observed"].to(tokens.dtype)[:, :, None]
            pooled = (tokens * observed).sum(1) / observed.sum(1).clamp_min(1)
            table_vector = self.table_encoder.subject_mlp(torch.cat([
                pooled, state["reliability"].to(pooled.dtype)], dim=1))
        global_vector = torch.zeros(image.shape[0], 2048, device=image.device, dtype=f4.dtype)
        if self.use_global:
            global_feature = self.global_vit(F.adaptive_avg_pool3d(f4, (5, 6, 5)))
            global_feature = F.interpolate(global_feature, size=f4.shape[-3:],
                                           mode="trilinear", align_corners=False)
            f4 = f4 + self.global_gate(f4) * self.global_projection(global_feature)
            global_vector = F.adaptive_avg_pool3d(global_feature, 1).flatten(1)
        z3 = F.adaptive_avg_pool3d(f3, 1).flatten(1)
        z4 = F.adaptive_avg_pool3d(f4, 1).flatten(1)
        return self.head(torch.cat([z3, z4, global_vector, table_vector], dim=1))


class ClinicalConcatenationResNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.backbone = MedicalNetClassifier(model_depth=50, in_channels=in_channels,
                                             shortcut_type="B")
        self.table = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 128))
        self.head = nn.Linear(self.backbone.backbone.out_channels + 128, 3)

    def forward(self, image, table, missing):
        visual = self.backbone.forward_features(image)
        clinical = self.table(torch.cat([table, missing], dim=1))
        return self.head(torch.cat([visual, clinical], dim=1))


class IgnoreTable(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, table, missing):
        return self.model(image)


def build_classifier(name: str, in_channels: int):
    """MedicalNet50 is the identical Baseline row in both comparison and ablation."""
    key = name.lower().replace("-", "_")
    if key in {"medicalnet50_3d", "baseline"}:
        return IgnoreTable(MedicalNetClassifier(model_depth=50, in_channels=in_channels,
                                                shortcut_type="B"))
    if key in {"tglm", "ours"}:
        return TGLM(in_channels, use_table=True, use_global=True)
    if key == "tglm_no_table":
        return TGLM(in_channels, use_table=False, use_global=True)
    if key == "table_branch":
        return TGLM(in_channels, use_table=True, use_global=False)
    if key == "global_branch":
        return TGLM(in_channels, use_table=False, use_global=True)
    if key == "early_fusion_3d_resnet":
        if in_channels != 2:
            raise ValueError("Early fusion requires MRI+PET input")
        return IgnoreTable(MedicalNetClassifier(model_depth=18, in_channels=2,
                                                shortcut_type="B"))
    if key == "clinical_concatenation_resnet":
        return ClinicalConcatenationResNet(in_channels)
    if key == "3d_densenet121":
        from monai.networks.nets import DenseNet121
        return IgnoreTable(DenseNet121(spatial_dims=3, in_channels=in_channels,
                                       out_channels=3))
    if key == "3d_vit_small":
        from monai.networks.nets import ViT
        return IgnoreTable(ViT(in_channels=in_channels, img_size=(160, 192, 160),
                               patch_size=(16, 16, 16), hidden_size=384, mlp_dim=1536,
                               num_layers=8, num_heads=6, classification=True, num_classes=3))
    if key == "3d_swin_t_encoder":
        from monai.networks.nets.swin_unetr import SwinTransformer
        backbone = SwinTransformer(in_chans=in_channels, embed_dim=48,
                                   window_size=(7, 7, 7), patch_size=(2, 2, 2),
                                   depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), spatial_dims=3)

        class SwinClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Linear(768, 3)

            def forward(self, image, table, missing):
                feature = self.backbone(image)[-1]
                return self.head(F.adaptive_avg_pool3d(feature, 1).flatten(1))

        return SwinClassifier()
    raise ValueError(f"Unknown thesis classification configuration: {name}")
