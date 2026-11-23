# train_vae.py
import os
import csv
import torch
import torch.distributed as dist
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from dataloader.mri_pet_loader import MRI2PET_dataset
from utils.ssim import ssim3D
from utils.psnr import calculate_psnr_3d
from model.DVAE import VAE, vae_loss, Discriminator3DPatch


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _is_dist() else 0


def _world_size():
    return dist.get_world_size() if _is_dist() else 1


def _all_reduce_tensor(t: torch.Tensor):
    if _is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _ensure_csv_header(path: str, header: list, rank: int):
    if rank != 0:
        return
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return

    try:
        with open(path, "r", newline="") as f:
            first = f.readline().strip()
        if first != ",".join(header):
            bak = path + ".bak"
            os.replace(path, bak)
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)
    except Exception:
        bak = path + ".bak"
        try:
            os.replace(path, bak)
        except Exception:
            pass
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def _get_pet_from_batch(batch: dict):
    for k in ["pet", "PET", "target", "label", "image"]:
        if k in batch:
            return batch[k]
    raise KeyError(f"Cannot find PET tensor in batch keys: {list(batch.keys())}")


def _to_plain_tensor(x, device):
    if hasattr(x, "as_tensor"):  # MONAI MetaTensor
        x = x.as_tensor()
    x = x.to(device, non_blocking=True)
    return x.float()


def _linear_warmup(epoch: int, max_val: float, warmup_epochs: int):
    if warmup_epochs <= 0:
        return max_val
    t = min(1.0, float(epoch + 1) / float(warmup_epochs))
    return max_val * t


def _normalize_pet_01(PET: torch.Tensor, clip_max: float):
    """
    将 PET 映射到 [0,1]，用于匹配 decoder Sigmoid() 输出。
    - 负值当噪声：clamp(min=0)
    - clip 到 clip_max
    - / clip_max
    """
    PET = PET.clamp(min=0.0)
    PET = PET.clamp(max=clip_max) / (clip_max + 1e-8)
    return PET


def _set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)


# -------------------------
# GAN losses (Hinge)
# -------------------------
def d_hinge_loss(d_real, d_fake):
    # real -> want large positive; fake -> want large negative
    loss_real = F.relu(1.0 - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    return loss_real + loss_fake


def g_hinge_adv_loss(d_fake):
    # generator wants D(fake) large positive
    return (-d_fake).mean()


def train_DVAE(config):
    rank = _rank()
    world_size = _world_size()
    device = config.device

    # -------------------------
    # Config (safe getattr)
    # -------------------------
    amp_enabled = getattr(config, "amp", True)

    recon_type = getattr(config, "recon_type", "l1")
    ssim_weight = getattr(config, "ssim_weight", 0.3)
    psnr_weight = getattr(config, "psnr_weight", 0.01)

    beta_max = getattr(config, "beta_max", 1e-2)
    beta_warmup_epochs = getattr(config, "beta_warmup_epochs", 20)

    # --- PET normalize (关键) ---
    pet_normalize = getattr(config, "pet_normalize", True)
    pet_clip_max = float(getattr(config, "pet_clip_max", 2.5))  # 你之前样本 max≈2.216，先用 2.5
    psnr_max_value = 1.0 if pet_normalize else float(getattr(config, "psnr_max_value", 1.0))

    model_name = getattr(config, "model_name", "VAE_GAN")

    # --- GAN switches ---
    use_gan = getattr(config, "use_gan", True)

    # 对抗项权重：务必从小开始（3D更需要小）
    lambda_adv_max = float(getattr(config, "lambda_adv_max", 1e-3))     # 1e-4~1e-3 推荐起步
    adv_warmup_epochs = int(getattr(config, "adv_warmup_epochs", 10))   # 前10个epoch线性升到max
    d_steps = int(getattr(config, "d_steps", 1))                        # 每步训练D的次数，1~2即可

    # Discriminator config
    d_base_channels = int(getattr(config, "d_base_channels", 32))
    d_layers = int(getattr(config, "d_layers", 4))

    # lr config（常见：D 稍大一点）
    lr_g = float(getattr(config, "learning_rate", 1e-4))
    lr_d = float(getattr(config, "lr_d", lr_g))

    # -------------------------
    # Model (Generator / Discriminator)
    # -------------------------
    G = VAE(latent_dim=config.latent_dim).to(device)

    D = None
    if use_gan:
        D = Discriminator3DPatch(in_channels=1, base_channels=d_base_channels, num_layers=d_layers).to(device)

    if _is_dist():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        G = torch.nn.parallel.DistributedDataParallel(
            G, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
        if use_gan:
            D = torch.nn.parallel.DistributedDataParallel(
                D, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )

    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999)) if use_gan else None

    scaler = GradScaler(enabled=amp_enabled)

    os.makedirs(config.save_dir, exist_ok=True)

    best_ssim = -1e9
    best_epoch = -1

    loss_csv_path = os.path.join(config.save_dir, "loss_curve.csv")
    val_csv_path = os.path.join(config.save_dir, "validation.csv")

    # headers
    _ensure_csv_header(
        loss_csv_path,
        ["Epoch", "beta", "lambda_adv", "train_total", "train_recon", "train_kl", "train_ssim", "train_advG", "train_lossD", "clip_max"],
        rank
    )
    _ensure_csv_header(
        val_csv_path,
        ["Epoch", "PSNR", "SSIM", "MAE", "Best_SSIM", "Best_Epoch", "clip_max"],
        rank
    )

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = MRI2PET_dataset(config.train_data, (160, 160, 96))
    val_dataset = MRI2PET_dataset(config.test_data, (160, 160, 96))

    train_sampler = None
    val_sampler = None
    if _is_dist():
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.numworker,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.numworker,
        pin_memory=True,
        drop_last=True
    )

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(config.epochs):
        if _is_dist():
            train_sampler.set_epoch(epoch)

        G.train()
        if use_gan:
            D.train()

        beta = _linear_warmup(epoch, max_val=beta_max, warmup_epochs=beta_warmup_epochs)
        lambda_adv = _linear_warmup(epoch, max_val=lambda_adv_max, warmup_epochs=adv_warmup_epochs) if use_gan else 0.0

        # accumulators
        total_sum = recon_sum = kl_sum = ssim_sum = 0.0
        advg_sum = lossd_sum = 0.0
        n_samples = 0

        iterable = tqdm(train_loader, desc=f"Train E{epoch+1}/{config.epochs}", leave=True) if rank == 0 else train_loader

        for step, batch in enumerate(iterable):
            PET = _to_plain_tensor(_get_pet_from_batch(batch), device)

            #PET = _normalize_pet_01(PET_raw, pet_clip_max) if pet_normalize else PET_raw

            bs = PET.size(0)

            # =========================
            # (1) Train Discriminator
            # =========================
            loss_D_val = 0.0
            if use_gan:
                _set_requires_grad(D, True)
                _set_requires_grad(G, False)

                for _ in range(d_steps):
                    opt_D.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        with autocast(enabled=amp_enabled):
                            PET_hat, _, _, _ = G(PET)

                    with autocast(enabled=amp_enabled):
                        d_real = D(PET)
                        d_fake = D(PET_hat.detach())
                        loss_D = d_hinge_loss(d_real, d_fake)

                    scaler.scale(loss_D).backward()
                    scaler.step(opt_D)
                    scaler.update()

                    loss_D_val += float(loss_D.detach().item())

                loss_D_val /= max(d_steps, 1)

            # =========================
            # (2) Train Generator (VAE)
            # =========================
            _set_requires_grad(G, True)
            if use_gan:
                _set_requires_grad(D, False)

            opt_G.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                PET_hat, mu, logvar, z = G(PET)
                base_loss, recon, kl = vae_loss(PET_hat, PET, mu, logvar, beta=beta, recon_type=recon_type)

            # SSIM 建议 float32
            ssim_val = ssim3D(PET.float(), PET_hat.float())
            ssim_loss = (1.0 - ssim_val)
            
            psnr_val = calculate_psnr_3d(PET.float(), PET_hat.float(), max_value=psnr_max_value)
            psnr_loss = (3.0 - psnr_val)
            

            adv_g = torch.tensor(0.0, device=device)
            if use_gan and lambda_adv > 0:
                with autocast(enabled=amp_enabled):
                    d_fake_for_g = D(PET_hat)
                    adv_g = g_hinge_adv_loss(d_fake_for_g)

            loss_G = base_loss + ssim_weight * ssim_loss + psnr_weight * psnr_loss + (lambda_adv * adv_g if use_gan else 0.0)

            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            # logging accum
            total_sum += float(loss_G.detach().item()) * bs
            recon_sum += float(recon.detach().item()) * bs
            kl_sum += float(kl.detach().item()) * bs
            ssim_sum += float(ssim_val.detach().item()) * bs
            advg_sum += float(adv_g.detach().item()) * bs
            lossd_sum += float(loss_D_val) * bs
            n_samples += bs

            if rank == 0 and epoch == 0 and step == 0:
                print("Batch keys:", list(batch.keys()))
                print("PET range:", PET.min().item(), PET.max().item(), PET.mean().item(), PET.std().item())
                print("PET(norm) range:" if pet_normalize else "PET range:",
                      PET.min().item(), PET.max().item(), PET.mean().item(), PET.std().item())
                print("PET_hat range:", PET_hat.float().min().item(), PET_hat.float().max().item(),
                      PET_hat.float().mean().item(), PET_hat.float().std().item())
                if pet_normalize:
                    sat_ratio = (PET_raw > pet_clip_max).float().mean().item()
                    print(f"Saturation ratio (PET_raw > clip_max={pet_clip_max}): {sat_ratio:.6f}")

            if rank == 0:
                iterable.set_postfix(
                    loss=float(loss_G.item()),
                    recon=float(recon.item()),
                    kl=float(kl.item()),
                    ssim=float(ssim_val.item()),
                    advG=float(adv_g.item()) if use_gan else 0.0,
                    lossD=float(loss_D_val) if use_gan else 0.0,
                    beta=beta,
                    lam_adv=lambda_adv,
                    clip=pet_clip_max if pet_normalize else -1,
                )

        # -------------------------
        # all-reduce train stats
        # -------------------------
        stats = torch.tensor(
            [total_sum, recon_sum, kl_sum, ssim_sum, advg_sum, lossd_sum, n_samples],
            device=device, dtype=torch.float64
        )
        stats = _all_reduce_tensor(stats)
        total_sum_g, recon_sum_g, kl_sum_g, ssim_sum_g, advg_sum_g, lossd_sum_g, n_samples_g = stats.tolist()

        train_total = total_sum_g / max(n_samples_g, 1.0)
        train_recon = recon_sum_g / max(n_samples_g, 1.0)
        train_kl = kl_sum_g / max(n_samples_g, 1.0)
        train_ssim = ssim_sum_g / max(n_samples_g, 1.0)
        train_advG = advg_sum_g / max(n_samples_g, 1.0)
        train_lossD = lossd_sum_g / max(n_samples_g, 1.0)

        # -------------------------
        # save (rank0)
        # -------------------------
        if rank == 0:
            G_state = G.module.state_dict() if hasattr(G, "module") else G.state_dict()
            torch.save(G_state, os.path.join(config.save_dir, "model.pth"))
            if use_gan:
                D_state = D.module.state_dict() if hasattr(D, "module") else D.state_dict()
                torch.save(D_state, os.path.join(config.save_dir, "disc.pth"))

            with open(loss_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, beta, lambda_adv, train_total, train_recon, train_kl, train_ssim,
                    train_advG, train_lossD, pet_clip_max if pet_normalize else ""
                ])

        if _is_dist():
            dist.barrier()

        # -------------------------
        # Validation
        # -------------------------
        G.eval()
        if _is_dist():
            val_sampler.set_epoch(epoch)

        psnr_sum = 0.0
        ssim_sum = 0.0
        mae_sum = 0.0
        val_samples = 0

        iterable = tqdm(val_loader, desc=f"Val   E{epoch+1}/{config.epochs}", leave=True) if rank == 0 else val_loader

        with torch.no_grad():
            for batch in iterable:
                PET = _to_plain_tensor(_get_pet_from_batch(batch), device)
                #PET = _normalize_pet_01(PET_raw, pet_clip_max) if pet_normalize else PET_raw

                with autocast(enabled=amp_enabled):
                    PET_hat, mu, logvar, z = G(PET)

                PET_f = PET.float()
                PET_hat_f = PET_hat.float()

                bs = PET.size(0)

                psnr_val = calculate_psnr_3d(PET_f, PET_hat_f, max_value=psnr_max_value)
                ssim_val = ssim3D(PET_f, PET_hat_f)
                mae_val = F.l1_loss(PET_f, PET_hat_f).item()

                psnr_val = psnr_val.item() if torch.is_tensor(psnr_val) else float(psnr_val)
                ssim_val = ssim_val.item() if torch.is_tensor(ssim_val) else float(ssim_val)

                psnr_sum += psnr_val * bs
                ssim_sum += ssim_val * bs
                mae_sum += mae_val * bs
                val_samples += bs

                if rank == 0:
                    iterable.set_postfix(psnr=psnr_val, ssim=ssim_val, mae=mae_val, clip=pet_clip_max if pet_normalize else -1)

        vstats = torch.tensor([psnr_sum, ssim_sum, mae_sum, val_samples], device=device, dtype=torch.float64)
        vstats = _all_reduce_tensor(vstats)
        psnr_sum_g, ssim_sum_g, mae_sum_g, val_samples_g = vstats.tolist()

        psnr_avg = psnr_sum_g / max(val_samples_g, 1.0)
        ssim_avg = ssim_sum_g / max(val_samples_g, 1.0)
        mae_avg = mae_sum_g / max(val_samples_g, 1.0)

        # -------------------------
        # best save (rank0)
        # -------------------------
        if rank == 0:
            if ssim_avg > best_ssim:
                best_ssim = ssim_avg
                best_epoch = epoch + 1
                G_state = G.module.state_dict() if hasattr(G, "module") else G.state_dict()
                torch.save(G_state, os.path.join(config.save_dir, "best_model.pth"))

            with open(val_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, psnr_avg, ssim_avg, mae_avg, best_ssim, best_epoch, pet_clip_max if pet_normalize else ""])

            print(
                model_name,
                f"[Epoch {epoch+1:03d}/{config.epochs}] "
                f"beta={beta:.6g} lambda_adv={lambda_adv:.6g} | "
                f"train loss={train_total:.6f}, recon={train_recon:.6f}, kl={train_kl:.6f}, ssim={train_ssim:.6f}, "
                f"advG={train_advG:.6f}, lossD={train_lossD:.6f} | "
                f"val PSNR={psnr_avg:.4f}, SSIM={ssim_avg:.6f}, MAE={mae_avg:.6f} | "
                f"best SSIM={best_ssim:.6f} (epoch {best_epoch})"
            )

        if _is_dist():
            dist.barrier()
