# train_bsfe_pretrain.py
# Pretrain BSFE by learning MRI -> PET reconstruction targets (NO VAE).
# Saves best checkpoint by validation loss.

import os
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from dataloader.mri_pet_table_loader import MRI2PETTableDataset, safe_collate

from model.BSFE import BSFEConfig, BrainStructuralFeatureExtractor
from model.bsfe_head import BSFEPetLatentHead  # assume head outputs PET-space tensor now

try:
    from MRI2PET_old.utils.ssim import ssim3D
except Exception:
    ssim3D = None

try:
    from utils.psnr import calculate_psnr_3d
except Exception:
    calculate_psnr_3d = None


# -------------------------
# DDP helpers
# -------------------------
def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world():
    return dist.get_world_size() if is_dist() else 1


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def ddp_cleanup():
    if is_dist():
        dist.destroy_process_group()


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= get_world()
    return y


# -------------------------
# misc
# -------------------------
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def pick(batch, keys):
    for k in keys:
        if k in batch:
            return batch[k]
    return None


def to_tensor(x, device):
    if hasattr(x, "as_tensor"):  # MONAI MetaTensor
        x = x.as_tensor()
    return x.to(device, non_blocking=True).float()


def img_grad_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """L1 loss of finite differences along D/H/W."""
    def diff(t, dim):
        return t.diff(dim=dim)
    loss = 0.0
    loss = loss + F.l1_loss(diff(x_hat, 2), diff(x, 2))
    loss = loss + F.l1_loss(diff(x_hat, 3), diff(x, 3))
    loss = loss + F.l1_loss(diff(x_hat, 4), diff(x, 4))
    return loss / 3.0


# -------------------------
# data
# -------------------------
@torch.no_grad()
def build_table_stats_ds(config):
    train_ds = MRI2PETTableDataset(
        data_path=config.train_data,
        table_csv=config.table_csv,
        desired_shape=config.desired_shape,
        table_cols=getattr(config, "table_cols", None),
        add_missing_mask=bool(getattr(config, "add_missing_mask", True)),
        standardize_table=bool(getattr(config, "standardize_table", True)),
        allow_unmatched_table=bool(getattr(config, "allow_unmatched_table", False)),
        strict_channel_check=bool(getattr(config, "strict_channel_check", True)),
        match_policy=getattr(config, "match_policy", "exact"),
        max_date_delta_days=int(getattr(config, "max_date_delta_days", 180)),
    )
    val_ds = MRI2PETTableDataset(
        data_path=config.test_data,
        table_csv=config.table_csv,
        desired_shape=config.desired_shape,
        table_cols=getattr(config, "table_cols", None),
        add_missing_mask=bool(getattr(config, "add_missing_mask", True)),
        standardize_table=bool(getattr(config, "standardize_table", True)),
        tabular_stats=train_ds.tab_stats,
        allow_unmatched_table=bool(getattr(config, "allow_unmatched_table", False)),
        strict_channel_check=bool(getattr(config, "strict_channel_check", True)),
        match_policy=getattr(config, "match_policy", "exact"),
        max_date_delta_days=int(getattr(config, "max_date_delta_days", 180)),
    )
    return train_ds, val_ds


def build_loaders(config, train_ds, val_ds):
    if is_dist():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.batch_size),
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=int(getattr(config, "numworker", 4)),
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(getattr(config, "val_batch_size", config.batch_size)),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(getattr(config, "numworker", 4)),
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate,
    )
    return train_loader, val_loader, train_sampler


# -------------------------
# models (NO VAE)
# -------------------------
def build_models(config, device):
    bsfe_cfg = BSFEConfig(
        in_channels=1,
        pre_down_hidden=int(getattr(config, "pre_down_hidden", 8)) if hasattr(config, "pre_down_hidden") else 8,
        base_channels=int(config.bsfe_base_channels),
        channel_mults=tuple(config.bsfe_channel_mults),
        num_res_blocks=int(config.bsfe_num_res_blocks),
        norm_groups=int(config.bsfe_norm_groups),
        dropout=float(config.bsfe_dropout),
        attn_stages=tuple(config.bsfe_attn_stages),
        attn_heads=int(config.bsfe_attn_heads),
        attn_head_dim=int(config.bsfe_attn_head_dim),
        n_tokens=int(config.bsfe_n_tokens),
        token_dim=int(config.bsfe_token_dim),
        global_dim=int(config.bsfe_global_dim),
        out_context_dim=getattr(config, "bsfe_out_context_dim", None),
    )
    bsfe = BrainStructuralFeatureExtractor(bsfe_cfg).to(device)

    ch0 = int(config.bsfe_base_channels)
    in_ch_list = [ch0 * m for m in tuple(config.bsfe_channel_mults)]

    head_token_dim = int(getattr(config, "bsfe_out_context_dim", None) or config.bsfe_token_dim)

    # Keep backward-compatible: only pass args that exist in your head implementation via getattr defaults.
    head = BSFEPetLatentHead(
        in_channels_list=in_ch_list,
        global_dim=int(config.bsfe_global_dim),
        latent_dim=int(getattr(config, "latent_dim", 1)),  # kept name; if head outputs PET, it may ignore this

        n_fuse_blocks=int(getattr(config, "head_blocks", 2)),
        norm_groups=int(getattr(config, "bsfe_norm_groups", 16)),
        dropout=float(getattr(config, "bsfe_dropout", 0.0)),
        use_global_film=bool(getattr(config, "head_use_global_film", True)),
        decoder_channels=in_ch_list,

        token_dim=head_token_dim,
        use_token_film=bool(getattr(config, "head_use_token_film", True)),

        # If your head supports these new fields, config can provide them; otherwise it will still run if head ignores.
        extra_upsamples=int(getattr(config, "head_extra_upsamples", 2)),
        extra_up_channels=getattr(config, "head_extra_up_channels", None),
        pet_out_channels=int(getattr(config, "pet_out_channels", 1)),
        pet_target_spatial=tuple(getattr(config, "pet_target_spatial", (96, 160, 160))),
    ).to(device)

    return bsfe, head


# -------------------------
# validate (NO VAE)
# -------------------------
@torch.no_grad()
def validate(config, bsfe, head, val_loader, device, amp_enabled=True):
    """
    Validation reports:
      - loss (reconstruction objective)
      - PET-space MAE / PSNR / SSIM between pred PET and real PET
    """
    bsfe.eval()
    head.eval()

    loss_sum = 0.0
    mae_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    n = 0

    psnr_max_value = float(getattr(config, "psnr_max_value", 2.0))

    def _psnr_fallback(x, y, max_value=1.0, eps=1e-8):
        mse = torch.mean((x - y) ** 2).clamp(min=eps)
        return 20.0 * torch.log10(torch.tensor(max_value, device=x.device)) - 10.0 * torch.log10(mse)

    w_l1 = float(getattr(config, "w_pet_l1", 1.0))
    w_grad = float(getattr(config, "w_pet_grad", 0.0))
    w_ssim = float(getattr(config, "w_pet_ssim", 0.0))

    for batch in tqdm(val_loader, desc="Val", leave=False):
        if batch is None:
            continue
        mri = to_tensor(pick(batch, ["mri", "MRI", "image", "x", "input"]), device)
        pet = to_tensor(pick(batch, ["pet", "PET", "label", "y"]), device)
        B = mri.size(0)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = bsfe(mri)
            pet_hat = head(out["pyramid"], out["global"], out["tokens"])

            l1 = F.l1_loss(pet_hat, pet)
            loss = w_l1 * l1
            if w_grad > 0:
                loss = loss + w_grad * img_grad_loss(pet_hat, pet)

        # SSIM in fp32 (often more stable)
        if w_ssim > 0 and ssim3D is not None:
            with torch.cuda.amp.autocast(enabled=False):
                ssim_val = ssim3D(pet.float(), pet_hat.float())
            loss = loss + w_ssim * (1.0 - ssim_val)

        loss_sum += float(loss.item()) * B
        n += B

        # metrics (fp32)
        pet_f = pet.float()
        pet_hat_f = pet_hat.float()

        mae_sum += float(F.l1_loss(pet_hat_f, pet_f).item()) * B

        if calculate_psnr_3d is not None:
            psnr_val = calculate_psnr_3d(pet_f, pet_hat_f, max_value=psnr_max_value)
            psnr_val = psnr_val.item() if torch.is_tensor(psnr_val) else float(psnr_val)
        else:
            psnr_val = float(_psnr_fallback(pet_f, pet_hat_f, max_value=psnr_max_value).item())
        psnr_sum += psnr_val * B

        if ssim3D is not None:
            ssim_val = ssim3D(pet_f, pet_hat_f)
            ssim_val = ssim_val.item() if torch.is_tensor(ssim_val) else float(ssim_val)
        else:
            ssim_val = 0.0
        ssim_sum += ssim_val * B

    loss_avg = loss_sum / max(n, 1)
    mae_avg = mae_sum / max(n, 1)
    psnr_avg = psnr_sum / max(n, 1)
    ssim_avg = ssim_sum / max(n, 1)

    # DDP reduce
    loss_avg = float(reduce_mean(torch.tensor(loss_avg, device=device)).item())
    mae_avg = float(reduce_mean(torch.tensor(mae_avg, device=device)).item())
    psnr_avg = float(reduce_mean(torch.tensor(psnr_avg, device=device)).item())
    ssim_avg = float(reduce_mean(torch.tensor(ssim_avg, device=device)).item())

    return {"loss": loss_avg, "mae": mae_avg, "psnr": psnr_avg, "ssim": ssim_avg}


def save_ckpt(path, epoch, bsfe, head, optim, scaler, best_val):
    payload = {
        "epoch": epoch,
        "bsfe": bsfe.state_dict(),
        "head": head.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val": best_val,
    }
    torch.save(payload, path)


# -------------------------
# train (NO VAE)
# -------------------------
def train(config):
    ddp_setup()
    rank = get_rank()
    device = config.device

    safe_mkdir(config.save_dir)
    if rank == 0:
        with open(os.path.join(config.save_dir, "config_dump.json"), "w", encoding="utf-8") as f:
            json.dump(
                {k: getattr(config, k) for k in dir(config) if not k.startswith("_")},
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    train_ds, val_ds = build_table_stats_ds(config)
    train_loader, val_loader, train_sampler = build_loaders(config, train_ds, val_ds)

    bsfe, head = build_models(config, device)

    if is_dist():
        bsfe = DDP(bsfe, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)
        head = DDP(head, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    params = list(bsfe.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(config, "amp", True)))

    best_val = float("inf")

    w_l1 = float(getattr(config, "w_pet_l1", 1.0))
    w_grad = float(getattr(config, "w_pet_grad", 0.0))
    w_ssim = float(getattr(config, "w_pet_ssim", 0.0))

    for epoch in range(1, int(config.epochs) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n[Epoch {epoch}/{config.epochs}]")

        bsfe.train()
        head.train()

        running = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc="Train", disable=(rank != 0))
        for batch in pbar:
            if batch is None:
                continue

            mri = to_tensor(pick(batch, ["mri", "MRI", "image", "x", "input"]), device)
            pet = to_tensor(pick(batch, ["pet", "PET", "label", "y"]), device)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(getattr(config, "amp", True))):
                out = bsfe(mri)
                pet_hat = head(out["pyramid"], out["global"], out["tokens"])

                l1 = F.l1_loss(pet_hat, pet)
                loss = w_l1 * l1
                if w_grad > 0:
                    loss = loss + w_grad * img_grad_loss(pet_hat, pet)

            if w_ssim > 0 and ssim3D is not None:
                # keep grad, but compute in fp32 for stability
                with torch.cuda.amp.autocast(enabled=False):
                    ssim_val = ssim3D(pet.float(), pet_hat.float())
                loss = loss + w_ssim * (1.0 - ssim_val)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += float(loss.item()) * mri.size(0)
            seen += mri.size(0)

            if rank == 0:
                pbar.set_postfix(loss=f"{running/max(seen,1):.4f}", l1=f"{l1.item():.4f}")

        # validation
        if (epoch % int(getattr(config, "val_interval", 1)) == 0) or (epoch == int(config.epochs)):
            val_metrics = validate(
                config,
                bsfe.module if isinstance(bsfe, DDP) else bsfe,
                head.module if isinstance(head, DDP) else head,
                val_loader,
                device=device,
                amp_enabled=bool(getattr(config, "amp", True)),
            )
            val_loss = val_metrics["loss"]
        else:
            val_metrics = None
            val_loss = None

        if rank == 0:
            train_loss = running / max(seen, 1)
            if val_loss is not None:
                print(
                    f"TrainLoss={train_loss:.6f}  ValLoss={val_loss:.6f}  "
                    f"PET_MAE={val_metrics.get('mae',0.0):.6f}  "
                    f"PET_PSNR={val_metrics.get('psnr',0.0):.4f}  "
                    f"PET_SSIM={val_metrics.get('ssim',0.0):.6f}"
                )
            else:
                print(f"TrainLoss={train_loss:.6f}")

            # save last
            if (epoch % int(getattr(config, "save_interval", 1)) == 0) or (epoch == int(config.epochs)):
                ckpt_last = os.path.join(config.save_dir, "bsfe_last.pth")
                save_ckpt(
                    ckpt_last,
                    epoch,
                    bsfe.module if isinstance(bsfe, DDP) else bsfe,
                    head.module if isinstance(head, DDP) else head,
                    optim,
                    scaler,
                    best_val,
                )

            # save best
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_SSIM = val_metrics.get('ssim',0.0)
                best_PSNR = val_metrics.get('psnr',0.0)
                best_MAE = val_metrics.get('mae', 0.0)
                ckpt_best = os.path.join(config.save_dir, "bsfe_best.pth")
                save_ckpt(
                    ckpt_best,
                    epoch,
                    bsfe.module if isinstance(bsfe, DDP) else bsfe,
                    head.module if isinstance(head, DDP) else head,
                    optim,
                    scaler,
                    best_val,
                )
                torch.save(
                    (bsfe.module if isinstance(bsfe, DDP) else bsfe).state_dict(),
                    os.path.join(config.save_dir, "bsfe_only_best.pth"),
                )
                print(f"[Saved] best @ epoch {epoch}: {ckpt_best}")

            print(
                f"BSET_EPOCH={best_epoch} "
                f"PET_MAE={best_MAE:.6f}  "
                f"PET_PSNR={best_PSNR:.4f}  "
                f"PET_SSIM={best_SSIM:.6f}"
            )

    ddp_cleanup()


if __name__ == "__main__":
    try:
        import bsfe_pretrain_config as config
    except Exception as e:
        raise ImportError("Cannot import bsfe_pretrain_config.py") from e

    train(config)
