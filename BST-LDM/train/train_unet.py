# train_unet_vae.py
# Baseline training: UNet (no BSFE, no table conditioning), MRI encoded and concatenated with PET latent/noisy input.
# Parameters are fully driven by an imported config module (same style as your current experiments).
#
# Update (2026-01-22):
# - Validate *every epoch* with diffusion training loss on the test set, save best (min loss) as ddpm_unet_best_loss.pth
# - Run sampling-based validation every N epochs (default N=10) using DDIM (default 100 steps),
#   compute PSNR/SSIM/MAE, save best (max SSIM) as ddpm_unet_best_ssim.pth

from __future__ import annotations
import os
import csv
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataloader.mri_pet_table_loader import MRI2PETTableDataset, safe_collate
from model.unet import UNet
from model.DVAE import VAE
from ema import EMA
from diffusion.ddpm_dual import DiffusionDDPM
from utils.ssim import ssim3D
from utils.psnr import calculate_psnr_3d

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0

def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1

def _all_reduce_tensor(x: torch.Tensor) -> torch.Tensor:
    if not _is_dist():
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def _ensure_csv_header(path: str, header: list[str], rank: int):
    if rank != 0:
        return
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)

def pick(batch: Dict[str, Any], keys):
    for k in keys:
        if k in batch:
            return batch[k]
    return None

def to_tensor(x, device):
    if hasattr(x, "as_tensor"):
        x = x.as_tensor()
    return x.to(device, non_blocking=True).float()

def get_optional_table(batch, device):
    """
    Returns (cate_x, conti_x).

    Convention used by UNet-GCA:
      - If batch provides table values + missing mask separately:
            conti_x := table values   (B,F)
            cate_x  := table_mask     (B,F)  1=missing, 0=observed
      - Else if only a single 'table' vector is provided:
            conti_x := table (possibly concat([values, mask]) if dataloader uses add_missing_mask)
            cate_x  := None
    """
    # New path: explicit missing mask field
    table = pick(batch, ["table", "table_x", "tab", "tab_x"])
    table_mask = pick(batch, ["table_mask", "missing_mask", "miss_mask", "mask"])
    if table is not None:
        conti = table
        cate = table_mask  # can be None
    else:
        # Legacy path for codebases that split cate/conti by design
        cate = pick(batch, ["cate_x", "cate", "cat", "cate_feat"])
        conti = pick(batch, ["conti_x", "conti", "cont", "cont_feat"])

    cate = cate.to(device, non_blocking=True).float() if cate is not None else None
    conti = conti.to(device, non_blocking=True).float() if conti is not None else None
    return cate, conti

@torch.no_grad()
def get_x0_from_pet(pet, pet_latent_encoder=None):
    if pet_latent_encoder is None:
        return pet
    pet_latent_encoder.eval()
    if hasattr(pet_latent_encoder, "encode"):
        out = pet_latent_encoder.encode(pet)
        if isinstance(out, (tuple, list)):
            return out[-1]
        return out
    if hasattr(pet_latent_encoder, "encoder"):
        out, _ = pet_latent_encoder.encoder(pet)
        if isinstance(out, (tuple, list)):
            return out[-1] if len(out) >= 3 else out[0]
        return out
    raise AttributeError("pet_latent_encoder must have encode() or encoder().")

@torch.no_grad()
def validate_ddpm_loss(
    ddpm_core, model_eval, val_loader, device, amp=True,
    pet_latent_encoder=None, max_batches: int = -1
):
    """Compute diffusion training loss on validation/test set (lower is better)."""
    ddpm_core.eval()
    model_eval.eval()

    loss_sum = 0.0
    n_samples = 0

    # swap model to evaluate EMA/current weights without rebuilding ddpm
    orig_model = ddpm_core.model
    ddpm_core.model = model_eval

    it = tqdm(val_loader, leave=True) if _rank() == 0 else val_loader
    for bi, batch in enumerate(it):
        if batch is None:
            continue
        if int(max_batches) >= 0 and bi >= int(max_batches):
            break

        mri = pick(batch, ["mri", "MRI", "image", "x", "input"])
        pet = pick(batch, ["pet", "PET", "label", "y"])
        if mri is None or pet is None:
            raise KeyError(f"Cannot find MRI/PET in batch keys: {list(batch.keys())}")

        mri = to_tensor(mri, device)
        pet = to_tensor(pet, device)
        cate_x, conti_x = get_optional_table(batch, device)

        x0_lat = get_x0_from_pet(pet, pet_latent_encoder)
        B = int(x0_lat.size(0))

        with autocast(enabled=amp):
            # IMPORTANT: validation should NOT drop conditions
            loss = ddpm_core(x0_lat, mri, cate_x, conti_x, cond_drop_prob=0.0)

        loss_sum += float(loss.item()) * B
        n_samples += B

    ddpm_core.model = orig_model

    stats = torch.tensor([loss_sum, n_samples], device=device, dtype=torch.float64)
    stats = _all_reduce_tensor(stats)
    loss_sum_g, n_g = stats.tolist()
    loss_avg = loss_sum_g / max(n_g, 1.0)
    return float(loss_avg)

@torch.no_grad()
def validate_final_generation(
    ddpm_core, model_eval, val_loader, device, amp=True,
    psnr_max_value=1.0, clip_x0=True, pet_latent_encoder=None,
    steps=1, eta=0.0, cfg_scale=1.0, max_batches: int = -1
):
    if (calculate_psnr_3d is None) or (ssim3D is None):
        raise RuntimeError("utils.psnr.calculate_psnr_3d and utils.ssim.ssim3D are required for validation.")

    ddpm_core.eval()
    model_eval.eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    mae_sum = 0.0
    n_samples = 0

    orig_model = ddpm_core.model
    ddpm_core.model = model_eval

    it = tqdm(val_loader, leave=True) if _rank() == 0 else val_loader
    for bi, batch in enumerate(it):
        if batch is None:
            continue
        if int(max_batches) >= 0 and bi >= int(max_batches):
            break

        mri = pick(batch, ["mri", "MRI", "image", "x", "input"])
        pet = pick(batch, ["pet", "PET", "label", "y"])
        if mri is None or pet is None:
            raise KeyError(f"Cannot find MRI/PET in batch keys: {list(batch.keys())}")

        mri = to_tensor(mri, device)
        pet = to_tensor(pet, device)
        cate_x, conti_x = get_optional_table(batch, device)

        x0_lat = get_x0_from_pet(pet, pet_latent_encoder)
        shape = tuple(x0_lat.shape)

        with autocast(enabled=amp):
            x_gen_lat = ddpm_core.sample_ddim(
                shape=shape, mri=mri, cate_x=cate_x, conti_x=conti_x,
                steps=int(steps), eta=float(eta), clip_x0=clip_x0, cfg_scale=float(cfg_scale), progress=False
            )

        if pet_latent_encoder is not None and hasattr(pet_latent_encoder, "decoder"):
            x_gen = pet_latent_encoder.decoder(x_gen_lat)
        else:
            x_gen = x_gen_lat

        x0_f = pet.float()
        xg_f = x_gen.float()

        psnr_val = calculate_psnr_3d(x0_f, xg_f, max_value=psnr_max_value)
        ssim_val = ssim3D(x0_f, xg_f)
        mae_val = F.l1_loss(x0_f, xg_f)

        psnr_val = psnr_val.item() if torch.is_tensor(psnr_val) else float(psnr_val)
        ssim_val = ssim_val.item() if torch.is_tensor(ssim_val) else float(ssim_val)
        mae_val  = mae_val.item()  if torch.is_tensor(mae_val)  else float(mae_val)

        B = x0_f.size(0)
        psnr_sum += psnr_val * B
        ssim_sum += ssim_val * B
        mae_sum  += mae_val  * B
        n_samples += B

    ddpm_core.model = orig_model

    stats = torch.tensor([psnr_sum, ssim_sum, mae_sum, n_samples], device=device, dtype=torch.float64)
    stats = _all_reduce_tensor(stats)
    psnr_sum_g, ssim_sum_g, mae_sum_g, n_g = stats.tolist()

    psnr_avg = psnr_sum_g / max(n_g, 1.0)
    ssim_avg = ssim_sum_g / max(n_g, 1.0)
    mae_avg  = mae_sum_g  / max(n_g, 1.0)
    return float(psnr_avg), float(ssim_avg), float(mae_avg)

def train_DDPM(config):
    rank = _rank()
    world_size = _world_size()
    device = config.device

    amp_enabled = bool(getattr(config, "amp", True))
    psnr_max_value = float(getattr(config, "psnr_max_value", 1.0))
    clip_x0 = bool(getattr(config, "clip_x0", True))

    schedule = getattr(config, "schedule", "cosine")
    pred_type = getattr(config, "pred_type", "v")
    loss_weighting = getattr(config, "loss_weighting", "p2")
    p2_gamma = float(getattr(config, "p2_gamma", 0.5))
    p2_k = float(getattr(config, "p2_k", 1.0))
    cond_drop_prob = float(getattr(config, "cond_drop_prob", 0.0))

    cfg_scale = float(getattr(config, "cfg_scale", 1.0))

    use_ema = bool(getattr(config, "use_ema", True))
    ema_decay = float(getattr(config, "ema_decay", 0.9999))

    # how often to run sampling-based validation
    sample_val_interval = int(getattr(config, "sample_val_interval", 10))

    train_ds = MRI2PETTableDataset(
        data_path=config.train_data,
        table_csv=getattr(config, "table_csv", None),
        desired_shape=tuple(getattr(config, "desired_shape", (160, 160, 96))),
        table_cols=getattr(config, "table_cols", None),
        add_missing_mask=bool(getattr(config, "add_missing_mask", True)),
        standardize_table=bool(getattr(config, "standardize_table", True)),
        allow_unmatched_table=bool(getattr(config, "allow_unmatched_table", False)),
        strict_channel_check=bool(getattr(config, "strict_channel_check", True))
    )
    val_ds = MRI2PETTableDataset(
        data_path=config.val_data,
        table_csv=getattr(config, "table_csv", None),
        desired_shape=tuple(getattr(config, "desired_shape", (160, 160, 96))),
        table_cols=getattr(config, "table_cols", None),
        add_missing_mask=bool(getattr(config, "add_missing_mask", True)),
        standardize_table=bool(getattr(config, "standardize_table", True)),
        tabular_stats=train_ds.tab_stats,
        allow_unmatched_table=bool(getattr(config, "allow_unmatched_table", False)),
        strict_channel_check=bool(getattr(config, "strict_channel_check", True))
    )

    if rank == 0:
        print(f"[Data] train={len(train_ds)}  val={len(val_ds)}")

    # -------------------------
    # Table feature dimension (WITHOUT missing-mask)
    # train_ds.tab_stats.columns is the canonical source of selected table columns.
    # If table_csv is None, tab_stats may be None and table_num_features will be 0 (table conditioning disabled).
    # -------------------------
    add_missing_mask = bool(getattr(config, "add_missing_mask", True))
    table_num_features = int(len(train_ds.tab_stats.columns)) if getattr(train_ds, "tab_stats", None) is not None else 0

    # -------------------------
    # Dual-target training (eps + v)
    # -------------------------
    dual_pred = bool(getattr(config, "dual_pred", True))
    dual_out_order = str(getattr(config, "dual_out_order", "v_eps"))
    use_dual_target_loss = bool(getattr(config, "use_dual_target_loss", True))
    loss_v_weight = float(getattr(config, "loss_v_weight", 1.0))
    loss_eps_weight = float(getattr(config, "loss_eps_weight", 1.0))

    unet = UNet(
        latent_dim=int(getattr(config, "latent_dim", 4)),
        cond_channels=int(getattr(config, "cond_channels", int(getattr(config, "latent_dim", 4)))),
        mri_in_channels=int(getattr(config, "mri_in_channels", 1)),
        mri_down=int(getattr(config, "mri_down", 2)),
        inner_channel=int(getattr(config, "inner_channel", 32)),
        norm_groups=int(getattr(config, "norm_groups", 16)),
        channel_mults=tuple(getattr(config, "channel_mults", (2, 4, 8, 16))),
        attn_res=tuple(getattr(config, "attn_res", (int(getattr(config, "image_size", 40)),))),
        res_blocks=int(getattr(config, "res_blocks", 1)),
        dropout=float(getattr(config, "dropout", 0.1)),
        image_size=int(getattr(config, "image_size", 40)),
        time_dim=int(getattr(config, "time_dim", 128)),
        mri_base_channels=int(getattr(config, "mri_base_channels", 16)),
        # --- Gated Cross-Attention table conditioning ---
        table_num_features=table_num_features,
        table_token_dim=int(getattr(config, "table_token_dim", 128)),
        table_cross_attn_heads=int(getattr(config, "table_cross_attn_heads", 4)),
        table_cross_attn_res=tuple(getattr(config, "table_cross_attn_res", getattr(config, "attn_res", (int(getattr(config, "image_size", 40)),)))),
        table_has_missing_mask_in_conti=add_missing_mask,
        use_table_cross_attn=bool(getattr(config, "use_table_cross_attn", True)),
        # --- BSFE conditioning (MRI guidance) ---
        use_bsfe=bool(getattr(config, "use_bsfe", True)),
        bsfe_dir=str(getattr(config, "bsfe_dir", "/zjs/MRI2PET/MRI2PET/result/bsfe/bsfe_only_best.pth")),
        freeze_bsfe=bool(getattr(config, "freeze_bsfe", True)),
        bsfe_pre_down_hidden=int(getattr(config, "bsfe_pre_down_hidden", 32)),
        bsfe_base_channels=int(getattr(config, "bsfe_base_channels", 32)),
        bsfe_channel_mults=tuple(getattr(config, "bsfe_channel_mults", (2, 4, 8, 16))),
        bsfe_num_res_blocks=int(getattr(config, "bsfe_num_res_blocks", 3)),
        bsfe_norm_groups=int(getattr(config, "bsfe_norm_groups", 32)),
        bsfe_dropout=float(getattr(config, "bsfe_dropout", 0.1)),
        bsfe_attn_stages=tuple(getattr(config, "bsfe_attn_stages", (2, 3))),
        bsfe_attn_heads=int(getattr(config, "bsfe_attn_heads", 4)),
        bsfe_attn_head_dim=int(getattr(config, "bsfe_attn_head_dim", 32)),
        bsfe_n_tokens=int(getattr(config, "bsfe_n_tokens", 32)),
        bsfe_token_dim=int(getattr(config, "bsfe_token_dim", 128)),
        bsfe_global_dim=int(getattr(config, "bsfe_global_dim", 256)),

        # --- dual prediction head (v + eps) ---
        dual_pred=dual_pred,
        dual_out_order=dual_out_order,
    ).to(device)

    ddpm = DiffusionDDPM(
        unet,
        T=int(getattr(config, "T", 1000)),
        beta=(float(getattr(config, "beta_start", 1e-4)), float(getattr(config, "beta_end", 2e-2))),
        schedule=schedule,
        pred_type=pred_type,
        loss_weighting=loss_weighting,
        p2_gamma=p2_gamma,
        p2_k=p2_k,
        dual_out_order=dual_out_order,
        use_dual_target_loss=use_dual_target_loss,
        loss_v_weight=loss_v_weight,
        loss_eps_weight=loss_eps_weight,
    ).to(device)

    ema = EMA(unet, decay=ema_decay, device=device) if use_ema else None

    if _is_dist():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        ddpm = torch.nn.parallel.DistributedDataParallel(
            ddpm,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,  # 关键
        )
    lr = float(getattr(config, "learning_rate", 1e-4))
    opt = optim.AdamW([p for p in ddpm.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.999))
    scaler = GradScaler(enabled=amp_enabled)

    pet_latent_encoder = None
    if bool(getattr(config, "use_pet_latent_encoder", False)):
        pet_latent_encoder = VAE(latent_dim=int(getattr(config, "latent_dim", 4))).to(device)
        pet_latent_encoder.load_state_dict(torch.load(config.vae_dir, map_location=device))
        pet_latent_encoder.eval()
        for p in pet_latent_encoder.parameters():
            p.requires_grad_(False)

    os.makedirs(config.save_dir, exist_ok=True)
    train_csv = os.path.join(config.save_dir, "train_ddpm.csv")
    val_loss_csv = os.path.join(config.save_dir, "val_ddpm_loss.csv")
    val_sample_csv = os.path.join(config.save_dir, "val_ddpm_sample.csv")

    _ensure_csv_header(train_csv,      ["Epoch", "TrainLoss"], rank)
    _ensure_csv_header(val_loss_csv,   ["Epoch", "ValLoss"], rank)
    _ensure_csv_header(val_sample_csv, ["Epoch", "PSNR", "SSIM", "MAE", "DDIMSteps"], rank)

    train_sampler = None
    val_sampler = None
    if _is_dist():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(getattr(config, "batch_size", 1)),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(getattr(config, "numworker", 4)),
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(getattr(config, "val_batch_size", int(getattr(config, "batch_size", 1)))),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(getattr(config, "numworker", 4)),
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate
    )

    ddpm_core = ddpm.module if hasattr(ddpm, "module") else ddpm

    best_val_loss = float("inf")
    best_ssim = -1e9

    epochs = int(getattr(config, "epochs", 100))
    # DDIM sampling params
    val_ddim_steps = int(getattr(config, "val_ddim_steps", 100))
    ddim_eta = float(getattr(config, "ddim_eta", 0.0))
    val_max_batches = int(getattr(config, "val_max_batches", -1))

    for epoch in range(epochs):
        ddpm.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        loss_sum = 0.0
        n_batches = 0

        it = tqdm(train_loader, leave=True) if rank == 0 else train_loader
        for batch in it:
            if batch is None:
                continue

            mri = pick(batch, ["mri", "MRI", "image", "x", "input"])
            pet = pick(batch, ["pet", "PET", "label", "y"])
            if mri is None or pet is None:
                raise KeyError(f"Cannot find MRI/PET in batch keys: {list(batch.keys())}")

            mri = to_tensor(mri, device)
            pet = to_tensor(pet, device)
            cate_x, conti_x = get_optional_table(batch, device)

            x0_lat = get_x0_from_pet(pet, pet_latent_encoder)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                loss = ddpm(x0_lat, mri, cate_x, conti_x, cond_drop_prob=cond_drop_prob)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if ema is not None:
                ema.update(unet)

            loss_sum += float(loss.item())
            n_batches += 1

            if rank == 0:
                it.set_description(f"Epoch {epoch+1}/{epochs} Loss {loss.item():.4f}")

        train_loss = loss_sum / max(n_batches, 1)
        if rank == 0:
            with open(train_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, train_loss])
            print(f"[Train] Epoch {epoch+1:03d}  loss={train_loss:.6f}")

        # -------------------------
        # 1) every-epoch validation loss (best=min)
        # -------------------------
        '''
        model_eval = ema.ema_model if ema is not None else ddpm_core.model
        val_loss = validate_ddpm_loss(
            ddpm_core, model_eval, val_loader, device,
            amp=amp_enabled, pet_latent_encoder=pet_latent_encoder,
            max_batches=val_max_batches
        )

        if rank == 0:
            with open(val_loss_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, val_loss])
            print(f"[ValLoss] Epoch {epoch+1:03d}  loss={val_loss:.6f}  (best={best_val_loss:.6f})")

            payload = {"ddpm": ddpm_core.state_dict()}
            if ema is not None:
                payload["ema"] = ema.state_dict()

            # always save last
            torch.save(payload, os.path.join(config.save_dir, "ddpm_unet_last.pth"))

            # best by val loss
            if (val_loss == val_loss) and (val_loss < best_val_loss):  # val_loss==val_loss filters NaN
                best_val_loss = val_loss
                torch.save(payload, os.path.join(config.save_dir, "ddpm_unet_best_loss.pth"))
                print(f"[BestLoss] loss={best_val_loss:.6f} @ epoch {epoch+1}")

        if _is_dist():
            dist.barrier()
        '''

        # -------------------------
        # 2) sampling-based validation every N epochs (best=max SSIM)
        # -------------------------
        #do_sample_val = (sample_val_interval > 0) and ((epoch + 1) % sample_val_interval == 0)
        #if do_sample_val:
        model_eval = ema.ema_model if ema is not None else ddpm_core.model
        psnr_avg, ssim_avg, mae_avg = validate_final_generation(
            ddpm_core, model_eval, val_loader, device,
            amp=amp_enabled, psnr_max_value=psnr_max_value, clip_x0=clip_x0,
            pet_latent_encoder=pet_latent_encoder,
            steps=val_ddim_steps, eta=ddim_eta, cfg_scale=cfg_scale,
            max_batches=val_max_batches
        )

        if rank == 0:
            with open(val_sample_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, psnr_avg, ssim_avg, mae_avg, val_ddim_steps])
            print(f"[ValSample] Epoch {epoch+1:03d}  PSNR={psnr_avg:.4f}  SSIM={ssim_avg:.6f}  MAE={mae_avg:.6f}  (DDIM steps={val_ddim_steps})")

            payload = {"ddpm": ddpm_core.state_dict()}
            if ema is not None:
                payload["ema"] = ema.state_dict()

            # best by SSIM
            if (ssim_avg == ssim_avg) and (ssim_avg > best_ssim):  # filter NaN
                best_ssim = ssim_avg
                torch.save(payload, os.path.join(config.save_dir, "ddpm_unet_best_ssim.pth"))
                print(f"[BestSSIM] SSIM={best_ssim:.6f} @ epoch {epoch+1}")

        if _is_dist():
            dist.barrier()