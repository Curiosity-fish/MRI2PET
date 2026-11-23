from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = REPO_ROOT.parent / "MRI2PET_V2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from datasets.paired_t1_fdg_dataset import PairedT1FDGDataset
from engine.trainer_t1_fdg import append_log, now_str, prepare_experiment_dir, reset_log, save_pickle
from utils.psnr import calculate_psnr_3d
from utils.ssim import ssim3D

DEFAULT_IMAGE_SIZE = (160, 196, 160)


def _load_v2_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DVAE_MODULE = _load_v2_module(V2_ROOT / "model" / "DVAE.py", "mri2pet_v2_dvae")
BSFE_MODULE = _load_v2_module(V2_ROOT / "model" / "BSFE.py", "mri2pet_v2_bsfe")
BSFE_HEAD_MODULE = _load_v2_module(V2_ROOT / "model" / "bsfe_head.py", "mri2pet_v2_bsfe_head")

VAE = DVAE_MODULE.VAE
BSFEConfig = BSFE_MODULE.BSFEConfig
BrainStructuralFeatureExtractor = BSFE_MODULE.BrainStructuralFeatureExtractor
BSFEPetLatentHead = BSFE_HEAD_MODULE.BSFEPetLatentHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp011b BSFE on the current MRI2PET protocol.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["experiment"]["config_path"] = str(config_path)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_device(device: torch.device) -> dict:
    info = {
        "device": str(device),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(device)
    else:
        info["gpu_name"] = "CPU"
    return info


def build_loader(config: dict, split: str, fold: int | None) -> DataLoader:
    dataset = PairedT1FDGDataset(
        pairs_csv=config["data"]["pairs_csv"],
        split_csv=config["data"]["split_csv"],
        split=split,
        fold=fold,
        spatial_size=tuple(config["data"].get("image_size", DEFAULT_IMAGE_SIZE)),
        max_samples=config["data"].get(f"{split}_max_samples"),
        normalize_inputs=bool(config["data"].get("normalize_inputs", False)),
        normalize_targets=bool(config["data"].get("normalize_targets", False)),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=(split == "train"),
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=bool(config["training"]["pin_memory"]),
        drop_last=(split == "train"),
    )


def img_grad_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (
        F.l1_loss(x_hat.diff(dim=2), x.diff(dim=2))
        + F.l1_loss(x_hat.diff(dim=3), x.diff(dim=3))
        + F.l1_loss(x_hat.diff(dim=4), x.diff(dim=4))
    ) / 3.0


class BSFEPetLatentHeadWithLatent(BSFEPetLatentHead):
    def forward(self, pyramid, global_vec, tokens=None):
        n_levels = len(self.lateral)
        level_low = n_levels - 1
        x = self.lateral[level_low](pyramid[level_low])
        x = self.mid1(x)
        x = self._apply_film(x, idx=level_low, global_vec=global_vec, tokens=tokens)
        x = self._bottleneck_attn(x, tokens)
        x = self.mid2(x)
        x = self._apply_film(x, idx=level_low, global_vec=global_vec, tokens=tokens)

        bi = 0
        for level in range(n_levels - 2, -1, -1):
            skip = self.lateral[level](pyramid[level])
            x = self.up_layers[bi](x)
            x = self._match_spatial_local(x, skip.shape[-3:])
            x = torch.cat([x, skip], dim=1)
            x = self.decode_blocks[bi](x)
            x = self._apply_film(x, idx=level, global_vec=global_vec, tokens=tokens)
            bi += 1

        x = self.out_refine(x)
        latent_hat = self.latent_out(x)
        for j in range(self.extra_upsamples):
            x = self.extra_up[j](x)
            x = self.extra_refine[j](x)
        pet_hat = self.pet_out(x)
        if tuple(pet_hat.shape[-3:]) != tuple(self.pet_target_spatial):
            pet_hat = F.interpolate(pet_hat, size=self.pet_target_spatial, mode="trilinear", align_corners=False)
        return {"latent_prediction": latent_hat, "pet_prediction": pet_hat}

    def forward_latent_only(self, pyramid, global_vec, tokens=None):
        """Decode to the latent-resolution prediction only (skip the expensive
        full-resolution PET path: extra_up/extra_refine/pet_out).

        Used during phase-1 (BSFE re-training) where the full-resolution head
        path exceeds VRAM with trainable weights (cuDNN 3D-conv workspace
        explosion at ~160^3 on Blackwell/RTX 5090). The returned latent
        prediction is 1ch at the latent (F1) spatial size; the UNet projects it
        via cond_proj (1 -> latent_channels). Gradients still flow back to BSFE
        and the head through the pyramid-decode / out_refine / latent_out path.
        """
        n_levels = len(self.lateral)
        level_low = n_levels - 1
        x = self.lateral[level_low](pyramid[level_low])
        x = self.mid1(x)
        x = self._apply_film(x, idx=level_low, global_vec=global_vec, tokens=tokens)
        x = self._bottleneck_attn(x, tokens)
        x = self.mid2(x)
        x = self._apply_film(x, idx=level_low, global_vec=global_vec, tokens=tokens)

        bi = 0
        for level in range(n_levels - 2, -1, -1):
            skip = self.lateral[level](pyramid[level])
            x = self.up_layers[bi](x)
            x = self._match_spatial_local(x, skip.shape[-3:])
            x = torch.cat([x, skip], dim=1)
            x = self.decode_blocks[bi](x)
            x = self._apply_film(x, idx=level, global_vec=global_vec, tokens=tokens)
            bi += 1

        x = self.out_refine(x)
        latent_hat = self.latent_out(x)
        return {"latent_prediction": latent_hat}

    @staticmethod
    def _match_spatial_local(x: torch.Tensor, ref_spatial):
        rd, rh, rw = ref_spatial
        xd, xh, xw = x.shape[-3:]
        if xd > rd:
            s = (xd - rd) // 2
            x = x[:, :, s:s + rd, :, :]
        if xh > rh:
            s = (xh - rh) // 2
            x = x[:, :, :, s:s + rh, :]
        if xw > rw:
            s = (xw - rw) // 2
            x = x[:, :, :, :, s:s + rw]
        pd = max(rd - x.shape[-3], 0)
        ph = max(rh - x.shape[-2], 0)
        pw = max(rw - x.shape[-1], 0)
        if pd or ph or pw:
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2))
        return x


def ensure_csv_header(path: Path, header: list[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        return
    first_line = path.read_text(encoding="utf-8").splitlines()[:1]
    if first_line != [",".join(header)]:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_csv_row(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def load_frozen_dvae(config: dict, device: torch.device) -> torch.nn.Module:
    model = VAE(image_channels=1, latent_dim=int(config["dvae"]["latent_dim"])).to(device)
    checkpoint = torch.load(config["dvae"]["checkpoint_path"], map_location=device)
    state = checkpoint.get("generator_state_dict", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def infer_latent_shape(dvae: torch.nn.Module, spatial_size: tuple[int, int, int], device: torch.device) -> tuple[int, int, int]:
    with torch.no_grad():
        dummy = torch.zeros((1, 1, *spatial_size), device=device)
        mu, _ = dvae.encoder(dummy)
    return tuple(int(v) for v in mu.shape[-3:])


def build_models(config: dict, latent_shape: tuple[int, int, int], device: torch.device):
    model_cfg = config["model"]
    bsfe_cfg = BSFEConfig(
        input_shape=tuple(config["data"]["image_size"]),
        latent_shape=latent_shape,
        in_channels=1,
        pre_down_hidden=int(model_cfg["pre_down_hidden"]),
        base_channels=int(model_cfg["bsfe_base_channels"]),
        channel_mults=tuple(model_cfg["bsfe_channel_mults"]),
        num_res_blocks=int(model_cfg["bsfe_num_res_blocks"]),
        norm_groups=int(model_cfg["bsfe_norm_groups"]),
        dropout=float(model_cfg["bsfe_dropout"]),
        attn_stages=tuple(model_cfg["bsfe_attn_stages"]),
        attn_heads=int(model_cfg["bsfe_attn_heads"]),
        attn_head_dim=int(model_cfg["bsfe_attn_head_dim"]),
        n_tokens=int(model_cfg["bsfe_n_tokens"]),
        token_dim=int(model_cfg["bsfe_token_dim"]),
        global_dim=int(model_cfg["bsfe_global_dim"]),
        out_context_dim=model_cfg.get("bsfe_out_context_dim"),
    )
    bsfe = BrainStructuralFeatureExtractor(bsfe_cfg).to(device)

    base = int(model_cfg["bsfe_base_channels"])
    in_channels_list = [base * int(m) for m in tuple(model_cfg["bsfe_channel_mults"])]
    head = BSFEPetLatentHeadWithLatent(
        in_channels_list=in_channels_list,
        global_dim=int(model_cfg["bsfe_global_dim"]),
        latent_dim=int(config["dvae"]["latent_dim"]),
        decoder_channels=in_channels_list,
        n_fuse_blocks=int(model_cfg.get("head_blocks", 2)),
        out_blocks=int(model_cfg.get("head_out_blocks", 1)),
        extra_upsamples=2,
        extra_up_channels=list(model_cfg.get("head_extra_up_channels", [64, 32])),
        pet_out_channels=1,
        pet_target_spatial=tuple(config["data"]["image_size"]),
        norm_groups=int(model_cfg["bsfe_norm_groups"]),
        dropout=float(model_cfg["bsfe_dropout"]),
        use_global_film=bool(model_cfg.get("head_use_global_film", True)),
        token_dim=int(model_cfg["bsfe_token_dim"]),
        use_token_film=bool(model_cfg.get("head_use_token_film", True)),
        use_token_cross_attn=bool(model_cfg.get("head_use_token_cross_attn", True)),
        attn_heads=int(model_cfg.get("head_attn_heads", 4)),
    ).to(device)
    return bsfe, head


def compute_losses(outputs: dict, pet: torch.Tensor, latent_target: torch.Tensor, dvae: torch.nn.Module, config: dict):
    latent_hat = outputs["latent_prediction"]
    pet_hat = outputs["pet_prediction"]
    loss_cfg = config["loss"]

    latent_l1 = F.l1_loss(latent_hat, latent_target)
    latent_grad = img_grad_loss(latent_hat, latent_target)
    pet_l1 = F.l1_loss(pet_hat, pet)
    pet_grad = img_grad_loss(pet_hat, pet)

    with torch.no_grad():
        mu_hat, _ = dvae.encoder(pet_hat)
    latent_cycle = F.l1_loss(mu_hat, latent_target)

    total = (
        float(loss_cfg.get("w_latent_l1", 1.0)) * latent_l1
        + float(loss_cfg.get("w_latent_grad", 0.0)) * latent_grad
        + float(loss_cfg.get("w_pet_l1", 1.0)) * pet_l1
        + float(loss_cfg.get("w_pet_grad", 0.0)) * pet_grad
        + float(loss_cfg.get("w_latent_cycle", 0.5)) * latent_cycle
    )

    pet_ssim = pet.new_zeros(())
    if float(loss_cfg.get("w_pet_ssim", 0.0)) > 0:
        pet_ssim = ssim3D(pet.float(), pet_hat.float())
        total = total + float(loss_cfg["w_pet_ssim"]) * (1.0 - pet_ssim)

    details = {
        "loss_total": float(total.detach().item()),
        "loss_latent_l1": float(latent_l1.detach().item()),
        "loss_latent_grad": float(latent_grad.detach().item()),
        "loss_pet_l1": float(pet_l1.detach().item()),
        "loss_pet_grad": float(pet_grad.detach().item()),
        "loss_latent_cycle": float(latent_cycle.detach().item()),
        "pet_ssim": float(pet_ssim.detach().item()),
    }
    return total, details


def train_epoch(bsfe, head, dvae, loader, optimizer, scaler, device, epoch, total_epochs, config, best_ssim):
    bsfe.train()
    head.train()
    sums = {
        "loss_total": 0.0,
        "loss_latent_l1": 0.0,
        "loss_latent_grad": 0.0,
        "loss_pet_l1": 0.0,
        "loss_pet_grad": 0.0,
        "loss_latent_cycle": 0.0,
    }
    total_items = 0
    progress = tqdm(loader, desc=f"Exp011b Epoch {epoch}/{total_epochs}", leave=True)
    amp_enabled = bool(config["training"].get("amp", True))
    for batch in progress:
        mri = batch["input"].to(device)
        pet = batch["target"].to(device)
        with torch.no_grad():
            latent_target, _ = dvae.encoder(pet)
        with autocast(enabled=amp_enabled):
            features = bsfe(mri)
            outputs = head(features["pyramid"], features["global"], features["tokens"])
            loss, details = compute_losses(outputs, pet, latent_target, dvae, config)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = int(mri.shape[0])
        total_items += bs
        for key in sums:
            sums[key] += details[key] * bs
        progress.set_postfix(
            train_loss=f"{details['loss_total']:.4f}",
            latent_l1=f"{details['loss_latent_l1']:.4f}",
            pet_l1=f"{details['loss_pet_l1']:.4f}",
            best_ssim=f"{best_ssim:.4f}" if np.isfinite(best_ssim) else "-inf",
        )
    return {k: v / max(total_items, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(bsfe, head, dvae, loader, device, config):
    bsfe.eval()
    head.eval()
    totals = {"loss": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0, "latent_l1": 0.0, "latent_cycle": 0.0}
    total_items = 0
    amp_enabled = bool(config["training"].get("amp", True))
    psnr_max_value = float(config["evaluation"].get("psnr_max_value", 2.5))
    for batch in loader:
        mri = batch["input"].to(device)
        pet = batch["target"].to(device)
        latent_target, _ = dvae.encoder(pet)
        with autocast(enabled=amp_enabled):
            features = bsfe(mri)
            outputs = head(features["pyramid"], features["global"], features["tokens"])
            loss, details = compute_losses(outputs, pet, latent_target, dvae, config)
        pet_hat = outputs["pet_prediction"].float()
        bs = int(mri.shape[0])
        total_items += bs
        totals["loss"] += float(loss.item()) * bs
        totals["mae"] += float(F.l1_loss(pet_hat, pet.float()).item()) * bs
        totals["psnr"] += float(calculate_psnr_3d(pet.float(), pet_hat, max_value=psnr_max_value)) * bs
        totals["ssim"] += float(ssim3D(pet.float(), pet_hat)) * bs
        totals["latent_l1"] += details["loss_latent_l1"] * bs
        totals["latent_cycle"] += details["loss_latent_cycle"] * bs
    return {k: v / max(total_items, 1) for k, v in totals.items()}


def run_fold(config: dict, paths, device: torch.device, fold: int):
    dvae = load_frozen_dvae(config, device)
    latent_shape = infer_latent_shape(dvae, tuple(config["data"]["image_size"]), device)
    bsfe, head = build_models(config, latent_shape, device)
    optimizer = optim.AdamW(
        list(bsfe.parameters()) + list(head.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scaler = GradScaler(enabled=bool(config["training"].get("amp", True)))

    train_loader = build_loader(config, "train", fold)
    val_loader = build_loader(config, "val", fold)
    final_val_loader = build_loader(config, "final_val", None)

    total_epochs = int(config["training"]["epochs"])
    history = []
    best_record = None
    best_score = float("-inf")
    best_ckpt = paths.checkpoints / f"fold{fold}_best.pt"
    last_ckpt = paths.checkpoints / f"fold{fold}_last.pt"
    curve_csv = paths.root / "validation.csv"
    ensure_csv_header(curve_csv, ["epoch", "val_loss", "val_mae", "val_psnr", "val_ssim", "val_latent_l1", "val_latent_cycle"])

    append_log(paths.train_log, f"Fold {fold} start: {now_str()}")
    append_log(paths.train_log, f"Frozen DVAE checkpoint: {config['dvae']['checkpoint_path']}")
    append_log(paths.train_log, f"Derived latent shape: {latent_shape}")
    for epoch in range(1, total_epochs + 1):
        train_metrics = train_epoch(bsfe, head, dvae, train_loader, optimizer, scaler, device, epoch, total_epochs, config, best_score)
        val_metrics = evaluate(bsfe, head, dvae, val_loader, device, config)
        row = {"fold": fold, "epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        append_log(paths.train_log, str(row))
        append_csv_row(curve_csv, [epoch, val_metrics["loss"], val_metrics["mae"], val_metrics["psnr"], val_metrics["ssim"], val_metrics["latent_l1"], val_metrics["latent_cycle"]])

        checkpoint = {
            "bsfe": bsfe.state_dict(),
            "head": head.state_dict(),
            "epoch": epoch,
            "metrics": row,
            "latent_shape": latent_shape,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, last_ckpt)
        if val_metrics["ssim"] > best_score:
            best_score = val_metrics["ssim"]
            best_record = row.copy()
            torch.save(checkpoint, best_ckpt)
            torch.save(bsfe.state_dict(), paths.checkpoints / "bsfe_only_best.pt")

        print(
            f"Exp011b Fold {fold} Epoch {epoch}/{total_epochs} "
            f"train={train_metrics['loss_total']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} val_psnr={val_metrics['psnr']:.4f} "
            f"val_ssim={val_metrics['ssim']:.6f} best_ssim={best_score:.6f}"
        )

    checkpoint = torch.load(best_ckpt, map_location=device)
    bsfe.load_state_dict(checkpoint["bsfe"])
    head.load_state_dict(checkpoint["head"])
    final_metrics = evaluate(bsfe, head, dvae, final_val_loader, device, config)
    append_log(paths.train_log, f"Fold {fold} best epoch: {best_record['epoch']}")
    append_log(paths.train_log, f"Fold {fold} best checkpoint: {best_ckpt}")
    append_log(paths.train_log, f"Fold {fold} final validation metrics: {final_metrics}")
    append_log(paths.train_log, f"Fold {fold} end: {now_str()}")
    return history, {"best_record": best_record, "final_metrics": final_metrics, "latent_shape": latent_shape}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))
    paths = prepare_experiment_dir(config["experiment"]["output_dir"], config)
    reset_log(paths.train_log)
    append_log(paths.train_log, f"Experiment start: {config['experiment']['name']}")
    append_log(paths.train_log, f"Start time: {now_str()}")
    append_log(paths.train_log, f"Config path: {config['experiment']['config_path']}")
    append_log(paths.train_log, f"Pairs CSV: {config['data']['pairs_csv']}")
    append_log(paths.train_log, f"Split CSV: {config['data']['split_csv']}")
    append_log(paths.train_log, f"Table CSV (for next stage): {config['data'].get('table_csv', '')}")
    device = torch.device(config["training"]["device"])
    append_log(paths.train_log, f"Device info: {describe_device(device)}")
    append_log(paths.train_log, f"BSFE config: {config['model']}")
    append_log(paths.train_log, f"Loss config: {config['loss']}")
    append_log(paths.train_log, f"Training config: {config['training']}")

    fold = int(config["data"]["fold"])
    history, summary = run_fold(config, paths, device, fold)
    save_pickle(paths.training_history_pkl, {fold: history})
    pd.DataFrame(
        [{
            "fold": fold,
            "best_epoch": int(summary["best_record"]["epoch"]),
            "best_val_ssim": float(summary["best_record"]["val_ssim"]),
            "best_val_mae": float(summary["best_record"]["val_mae"]),
            "best_val_psnr": float(summary["best_record"]["val_psnr"]),
            "final_val_mae": float(summary["final_metrics"]["mae"]),
            "final_val_psnr": float(summary["final_metrics"]["psnr"]),
            "final_val_ssim": float(summary["final_metrics"]["ssim"]),
        }]
    ).to_csv(paths.metrics_csv, index=False)
    torch.save(summary["final_metrics"], paths.predictions / "fold0_final_metrics.pt")
    shutil.copy2(paths.checkpoints / f"fold{fold}_best.pt", paths.checkpoints / "best.pt")
    shutil.copy2(paths.checkpoints / f"fold{fold}_last.pt", paths.checkpoints / "last.pt")

    readme = [
        f"# {config['experiment']['name']}",
        "",
        f"- 实验目的: {config['experiment']['description']}",
        f"- 配置文件: {config['experiment']['config_path']}",
        f"- 数据索引文件: {config['data']['pairs_csv']}",
        f"- 数据划分文件: {config['data']['split_csv']}",
        f"- DVAE checkpoint: {config['dvae']['checkpoint_path']}",
        f"- 潜变量形状: {summary['latent_shape']}",
        f"- 训练轮数: {config['training']['epochs']}",
        (
            "- 主要结果: "
            f"best_epoch={int(summary['best_record']['epoch'])}, "
            f"best_val_mae={summary['best_record']['val_mae']:.6f}, "
            f"best_val_psnr={summary['best_record']['val_psnr']:.6f}, "
            f"best_val_ssim={summary['best_record']['val_ssim']:.6f}"
        ),
        (
            "- 最终验证结果: "
            f"MAE={summary['final_metrics']['mae']:.6f}, "
            f"PSNR={summary['final_metrics']['psnr']:.6f}, "
            f"SSIM={summary['final_metrics']['ssim']:.6f}"
        ),
        "- 备注: 本阶段冻结 Exp011a DVAE，用 BSFE + reconstruction head 学习 MRI 到 PET latent / PET 空间的一致结构表征，为第三阶段 latent diffusion 提供结构先验 backbone。",
    ]
    paths.readme.write_text("\n".join(readme), encoding="utf-8")
    append_log(paths.train_log, f"End time: {now_str()}")


if __name__ == "__main__":
    main()
