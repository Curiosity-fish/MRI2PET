from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util

_dataset_spec = importlib.util.spec_from_file_location('exp032j_paired_t1_fdg_table_dataset', REPO_ROOT / 'datasets' / 'paired_t1_fdg_table_dataset.py')
_dataset_mod = importlib.util.module_from_spec(_dataset_spec)
assert _dataset_spec.loader is not None
_dataset_spec.loader.exec_module(_dataset_mod)
PairedT1FDGTableDataset = _dataset_mod.PairedT1FDGTableDataset

from models.exp026.utils.patch_codec import build_patch_grid
from models.exp032.data import load_labelmap, resolve_path
from models.exp032.evaluation import parc98_roi_metrics, patch_boundary_metrics
from models.exp032e.utils import load_autoencoder
from models.exp032f.vp_diffusion import LatentVPDiffusion
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp032i_table_fusion_unet import Exp032ITableFusionUNet3D
def infer_feature_type_ids(table_cols: list[str]) -> tuple[list[int], dict[int, str]]:
    label_map = {0: 'demographic', 1: 'genetic', 2: 'cognitive', 3: 'functional_psych', 4: 'other'}
    type_ids = []
    for name in table_cols:
        lower = str(name).lower()
        if any(key in lower for key in ['age', 'gender', 'sex', 'educ', 'race', 'eth', 'marry', 'hand']):
            type_ids.append(0)
        elif 'apoe' in lower:
            type_ids.append(1)
        elif any(key in lower for key in ['mmse', 'adas', 'cdr', 'ravlt', 'moca', 'digit', 'trail', 'clock', 'boston', 'animal', 'lm']):
            type_ids.append(2)
        elif any(key in lower for key in ['faq', 'ecog', 'gds', 'npi']):
            type_ids.append(3)
        else:
            type_ids.append(4)
    return type_ids, label_map
from train.train_exp032g_mri_control_v import boundary_band_mae, decode_local, ensure_csv, local_inputs


def parse_args():
    p = argparse.ArgumentParser(description='Exp032J Sampling-Aligned Training Benchmark')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--smoke-test', action='store_true')
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['experiment']['config_path'] = str(path)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv(path: Path, row: list[object]) -> None:
    with path.open('a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)


class Exp032JTableDataset(Dataset):
    def __init__(self, base: PairedT1FDGTableDataset, parc98: torch.Tensor) -> None:
        self.base = base
        self.parc98 = parc98.float().contiguous()
        self.table_cols = list(base.table_cols)
        self.tabular_stats = base.tabular_stats

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        return {
            'mri': sample['input'],
            'pet': sample['target'],
            'brain_mask': sample['brain_mask'],
            'table': sample['table'],
            'table_mask': sample['table_mask'],
            'parc98': self.parc98,
            'subject_id': sample['subject_id'],
            'sample_id': sample['sample_id'],
        }


def build_dataset(config: dict, split: str, train_dataset: Exp032JTableDataset | None = None) -> Exp032JTableDataset:
    data = config['data']
    target_shape = tuple(int(v) for v in data['image_size'])
    base = PairedT1FDGTableDataset(
        pairs_csv=resolve_path(REPO_ROOT, data['pairs_csv']),
        split_csv=resolve_path(REPO_ROOT, data['split_csv']),
        table_csv=resolve_path(REPO_ROOT, data['table_csv']),
        split=split,
        fold=data.get('fold'),
        spatial_size=target_shape,
        tabular_stats=(train_dataset.tabular_stats if train_dataset is not None else None),
        table_cols=(train_dataset.table_cols if train_dataset is not None else data.get('table_cols')),
        max_samples=data.get(f'{split}_max_samples'),
        include_sample_ids=data.get(f'{split}_sample_ids'),
        standardize_table=True,
        add_missing_mask=True,
        cache_dir=data.get('cache_dir'),
        max_table_delta_days=data.get('max_table_delta_days'),
    )
    parc98 = load_labelmap(resolve_path(REPO_ROOT, data['parc98_labelmap_path']), target_shape)
    return Exp032JTableDataset(base, parc98)


def build_loaders(config: dict):
    train_dataset = build_dataset(config, 'train')
    train_loader = DataLoader(train_dataset, batch_size=int(config['training']['batch_size']), shuffle=True, num_workers=int(config['training'].get('num_workers', 0)), pin_memory=bool(config['training'].get('pin_memory', False)))
    val_splits = list(config.get('evaluation', {}).get('val_splits', []))
    val_sets = [build_dataset(config, split, train_dataset=train_dataset) for split in val_splits]
    val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    val_loader = DataLoader(val_ds, batch_size=int(config['training']['batch_size']), shuffle=False, num_workers=int(config['training'].get('num_workers', 0)), pin_memory=bool(config['training'].get('pin_memory', False)))
    small_n = int(config.get('evaluation', {}).get('small_val_subjects', 4))
    small_subset = Subset(val_ds, list(range(min(len(val_ds), small_n))))
    small_val_loader = DataLoader(small_subset, batch_size=int(config['training']['batch_size']), shuffle=False, num_workers=0)
    return train_dataset, train_loader, val_loader, small_val_loader


def build_model(cfg: dict, latent_channels: int, table_num_features: int, feature_type_ids: list[int]) -> Exp032ITableFusionUNet3D:
    m = cfg['model']
    g = cfg['global']
    t = cfg['table']
    return Exp032ITableFusionUNet3D(
        table_mode=str(t['mode']),
        table_num_features=table_num_features,
        feature_type_ids=feature_type_ids,
        table_dim=int(t.get('token_dim', 64)),
        table_heads=int(t.get('num_heads', 4)),
        table_gamma_scale=float(t.get('gamma_scale', 0.5)),
        table_beta_scale=float(t.get('beta_scale', 0.5)),
        routing_temperature=float(t.get('routing_temperature', 1.0)),
        in_channels=latent_channels * 2,
        out_channels=latent_channels,
        base_channels=int(m['base_channels']),
        channel_mults=tuple(m.get('channel_mults', [1, 2, 4])),
        dropout=float(m.get('dropout', 0.0)),
        num_groups=int(m.get('norm_groups', 8)),
        use_attention=bool(m.get('use_attention', False)),
        attention_level=int(m.get('attention_level', 1)),
        num_heads=int(m.get('num_heads', 4)),
        mri_base_channels=int(cfg['control'].get('mri_base_channels', 8)),
        gamma_scale=float(cfg['control'].get('gamma_scale', 0.5)),
        beta_scale=float(cfg['control'].get('beta_scale', 0.5)),
        strategy=str(g['strategy']),
        global_dim=int(g.get('dim', 64)),
        global_depth=int(g.get('depth', 2)),
        vit_heads=int(g.get('num_heads', 4)),
        vit_mlp_ratio=float(g.get('mlp_ratio', 2.0)),
    )


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in module.parameters()), sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_optimizer(model: Exp032ITableFusionUNet3D, cfg: dict):
    base_params, cond_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(('pet_token.','mri_token.','pet_pos','mri_pos','table_fusion.','global_backbone.','to_global.','feedback.','pos.','scale_emb.','h_to_m.','m_to_l.','mri_encoder.','control.')):
            cond_params.append(p)
        else:
            base_params.append(p)
    groups = []
    if base_params:
        groups.append({'params': base_params, 'lr': float(cfg['training']['base_finetune_lr'])})
    if cond_params:
        groups.append({'params': cond_params, 'lr': float(cfg['training']['cond_lr'])})
    return torch.optim.AdamW(groups, weight_decay=float(cfg['training'].get('weight_decay', 0.0)))


def strategy_name(cfg: dict) -> str:
    return str(cfg['training_strategy']['name']).lower()


def objective_name(cfg: dict) -> str:
    return {'onestepv':'one_step_v','noisepathv10':'path_v','samplingawarediff':'sampling_aware_v'}[strategy_name(cfg)]


def best_objective_ckpt_name(cfg: dict) -> str:
    return {'onestepv':'best_val_v.pt','noisepathv10':'best_val_path_v.pt','samplingawarediff':'best_val_v_loss.pt'}[strategy_name(cfg)]


def best_generation_ckpt_name(cfg: dict) -> str:
    return 'best_full_sampling_ssim.pt' if strategy_name(cfg) == 'samplingawarediff' else 'best_full_generation_ssim.pt'


def ddim_sample(model, diffusion, z_shape, mri_patch, table, table_mask, device):
    z = torch.randn(z_shape, device=device)
    cond_zero = torch.zeros(z_shape, device=device)
    use_amp = device.type == 'cuda'
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for step in reversed(range(diffusion.timesteps)):
            t = torch.full((z_shape[0],), step, device=device, dtype=torch.long)
            v = model(z, cond_zero, t, mri_patch, table, table_mask)
            z0 = diffusion.predict_z0_from_v(z, t, v)
            if step == 0:
                z = z0
            else:
                t_prev = torch.full((z_shape[0],), step - 1, device=device, dtype=torch.long)
                coeffs_prev = diffusion.coeffs(t_prev, z.shape)
                eps = diffusion.predict_noise_from_v(z, t, v)
                z = coeffs_prev.alpha * z0 + coeffs_prev.sigma * eps
            if device.type == 'cuda' and step % 5 == 0:
                torch.cuda.empty_cache()
    return z.float()


def build_noise_path_state(z0: torch.Tensor, endpoint_noise: torch.Tensor, t: torch.Tensor, steps: int) -> torch.Tensor:
    s = (t.float() / float(steps)).view(-1, *([1] * (z0.ndim - 1)))
    return (1.0 - s) * z0 + s * endpoint_noise


def sample_high_t_biased(batch_size: int, timesteps: int, device: torch.device) -> torch.Tensor:
    terminal = timesteps - 1
    xs = []
    for _ in range(batch_size):
        r = random.random()
        if r < 0.2:
            xs.append(random.randint(0, max(0, int(0.5 * terminal))))
        elif r < 0.5:
            xs.append(random.randint(max(0, int(0.5 * terminal)), max(0, int(0.8 * terminal))))
        elif r < 0.8:
            xs.append(random.randint(max(0, int(0.8 * terminal)), max(0, terminal - 1)))
        else:
            xs.append(terminal)
    return torch.tensor(xs, device=device, dtype=torch.long)


def diffusion_rollout_losses(diffusion, z_t, t, v_pred, z0, eps, cfg):
    t_prev = torch.clamp(t - 1, min=0)
    coeffs_prev = diffusion.coeffs(t_prev, z_t.shape)
    z0_hat = diffusion.predict_z0_from_v(z_t, t, v_pred)
    eps_hat = diffusion.predict_noise_from_v(z_t, t, v_pred)
    z_prev_hat = coeffs_prev.alpha * z0_hat + coeffs_prev.sigma * eps_hat
    z_prev_gt = coeffs_prev.alpha * z0 + coeffs_prev.sigma * eps
    roll = F.smooth_l1_loss(z_prev_hat, z_prev_gt)
    norm_t = t.float() / float(max(diffusion.timesteps - 1, 1))
    high_mask = (norm_t >= float(cfg['sampling_aware']['high_t_x0']['threshold'])).float().view(-1, *([1] * (z0.ndim - 1)))
    x0_aux = (F.smooth_l1_loss(z0_hat, z0, reduction='none') * high_mask).mean()
    return roll, x0_aux, z0_hat


def train_objective(model, diffusion, z0, mri_patch, table, table_mask, cfg, device):
    strategy = strategy_name(cfg)
    cond = torch.zeros_like(z0)
    if strategy == 'onestepv':
        z_noise = torch.randn_like(z0)
        t = torch.full((z0.shape[0],), diffusion.timesteps - 1, device=device, dtype=torch.long)
        v_target = z_noise - z0
        v_pred = model(z_noise, cond, t, mri_patch, table, table_mask)
        loss = F.mse_loss(v_pred, v_target)
        z0_pred = z_noise - v_pred
        return loss, {'v_loss': float(loss.item()), 'latent_x0_mae': float((z0_pred - z0).abs().mean().item()), 'latent_x0_cosine': float(F.cosine_similarity(z0_pred.flatten(1), z0.flatten(1), dim=1).mean().item())}
    if strategy == 'noisepathv10':
        steps = int(cfg['transport']['steps'])
        t = torch.randint(1, steps + 1, (z0.shape[0],), device=device)
        endpoint_noise = torch.randn_like(z0)
        z_t = build_noise_path_state(z0, endpoint_noise, t, steps)
        v_target = endpoint_noise - z0
        v_pred = model(z_t, cond, t, mri_patch, table, table_mask)
        loss = F.mse_loss(v_pred, v_target)
        z0_pred = endpoint_noise - v_pred
        return loss, {'path_v_loss': float(loss.item()), 'latent_x0_mae': float((z0_pred - z0).abs().mean().item()), 'latent_x0_cosine': float(F.cosine_similarity(z0_pred.flatten(1), z0.flatten(1), dim=1).mean().item()), 'timestep_hist_json': json.dumps({int(i): int((t == i).sum().item()) for i in range(1, steps + 1)})}
    t = sample_high_t_biased(z0.shape[0], diffusion.timesteps, device)
    eps = torch.randn_like(z0)
    z_t, v_target = diffusion.q_sample(z0, t, noise=eps)
    v_pred = model(z_t, cond, t, mri_patch, table, table_mask)
    v_loss = F.mse_loss(v_pred, v_target)
    roll, x0_aux, z0_hat = diffusion_rollout_losses(diffusion, z_t, t, v_pred, z0, eps, cfg)
    total = v_loss + float(cfg['sampling_aware']['rollout_consistency']['weight']) * roll + float(cfg['sampling_aware']['high_t_x0']['weight']) * x0_aux
    return total, {'v_loss': float(v_loss.item()), 'roll_loss': float(roll.item()), 'high_t_x0_loss': float(x0_aux.item()), 'latent_x0_mae': float((z0_hat - z0).abs().mean().item()), 'latent_x0_cosine': float(F.cosine_similarity(z0_hat.flatten(1), z0.flatten(1), dim=1).mean().item()), 'timestep_hist_json': json.dumps({int(i): int((t == i).sum().item()) for i in range(diffusion.timesteps)})}


def sample_prediction(model, diffusion, z0, mri_patch, table, table_mask, cfg, device):
    strategy = strategy_name(cfg)
    cond = torch.zeros_like(z0)
    if strategy == 'onestepv':
        z_noise = torch.randn_like(z0)
        t = torch.full((z0.shape[0],), diffusion.timesteps - 1, device=device, dtype=torch.long)
        v_pred = model(z_noise, cond, t, mri_patch, table, table_mask)
        return z_noise - v_pred
    if strategy == 'noisepathv10':
        steps = int(cfg['transport']['steps'])
        z = torch.randn_like(z0)
        dt = 1.0 / float(steps)
        for step in range(steps, 0, -1):
            t = torch.full((z0.shape[0],), step, device=device, dtype=torch.long)
            v_pred = model(z, cond, t, mri_patch, table, table_mask)
            z = z - dt * v_pred
        return z
    return ddim_sample(model, diffusion, z0.shape, mri_patch, table, table_mask, device)


def aggregate_metric_dicts(vals: list[dict[str, float]]) -> dict[str, float]:
    keys = vals[0].keys()
    out = {k: float(np.mean([v[k] for v in vals])) for k in keys}
    for k in keys:
        out[f'{k}_std'] = float(np.std([v[k] for v in vals]))
    return out


def evaluate_generation(model, ae_model, diffusion, loader, stats, cfg, device, coords, seeds=(42,123,2026), eval_modes=('normal',)):
    image_patch_coords, patch_size, latent_patch_coords, latent_patch_size = coords
    image_size = tuple(cfg['data']['image_size'])
    latent_full_size = tuple(int(v) // 4 for v in image_size)
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            pet, _, z0, _, mri_patch = local_inputs(batch, ae_model, stats, cfg, device, image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)
            table = batch['table'].to(device).float()
            table_mask = batch['table_mask'].to(device).float()
            for mode in eval_modes:
                if mode == 'mri_off':
                    mri_cur = torch.zeros_like(mri_patch)
                elif mode == 'mri_shuffle':
                    mri_cur = torch.roll(mri_patch.view(pet.shape[0], len(latent_patch_coords), 1, *patch_size), 1, 0).reshape_as(mri_patch)
                else:
                    mri_cur = mri_patch
                vals = []
                for seed in seeds:
                    torch.manual_seed(int(seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed))
                    z_pred = sample_prediction(model, diffusion, z0, mri_cur, table, table_mask, cfg, device)
                    recon, _ = decode_local(ae_model, z_pred, pet.shape, ae_model.latent_channels, latent_patch_coords, latent_patch_size, latent_full_size, stats)
                    metrics = compute_image_metrics_detailed(recon, pet, mask=batch['brain_mask'].to(device).float(), data_range=2.0)
                    roi = parc98_roi_metrics(recon, pet, batch['parc98'].to(device))
                    boundary = patch_boundary_metrics(recon, pet, patch_size)
                    band = boundary_band_mae(recon, pet, patch_size, band=int(cfg['evaluation'].get('boundary_band', 1)))
                    vals.append({**{k: float(metrics[k]) for k in ['ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae','brain_rmse']}, **{k: float(roi[k]) for k in ['roi_mae','roi_rmse','roi_pearson','roi_spearman']}, **{k: float(boundary[k]) for k in ['boundary_mae','boundary_gradient_error']}, 'boundary_band_mae': float(band)})
                agg = aggregate_metric_dicts(vals)
                agg['eval_mode'] = mode
                agg['seed_count'] = len(seeds)
                rows.append(agg)
    return rows


def evaluate_generation_on_device(model, ae_model, diffusion, loader, stats, cfg, device, coords, seeds=(42,123,2026), eval_modes=('normal',)):
    if device.type == next(model.parameters()).device.type:
        return evaluate_generation(model, ae_model, diffusion, loader, stats, cfg, device, coords, seeds=seeds, eval_modes=eval_modes)
    model_copy = copy.deepcopy(model).to(device).eval()
    ae_copy = copy.deepcopy(ae_model).to(device).eval()
    for p in ae_copy.parameters():
        p.requires_grad_(False)
    diffusion_copy = LatentVPDiffusion(timesteps=diffusion.timesteps, beta_start=float(diffusion.register_buffers['betas'][0].item()), beta_end=float(diffusion.register_buffers['betas'][-1].item())).to(device)
    return evaluate_generation(model_copy, ae_copy, diffusion_copy, loader, stats, cfg, device, coords, seeds=seeds, eval_modes=eval_modes)


def append_val_rows(val_csv: Path, epoch: int, rows: list[dict[str, float]]) -> None:
    fields = ['epoch','eval_mode','seed_count','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae','brain_rmse','roi_mae','roi_rmse','roi_pearson','roi_spearman','boundary_mae','boundary_gradient_error','boundary_band_mae','ssim_std','psnr_std','mae_std','rmse_std','brain_ssim_std','brain_psnr_std','brain_mae_std','brain_rmse_std','roi_mae_std','roi_rmse_std','roi_pearson_std','roi_spearman_std','boundary_mae_std','boundary_gradient_error_std','boundary_band_mae_std']
    for r in rows:
        append_csv(val_csv, [epoch] + [r.get(k, 0.0) for k in fields[1:]])


def save_checkpoint(path: Path, epoch: int, cfg: dict, model, optimizer, scheduler, payload: dict) -> None:
    torch.save({'epoch': epoch, 'config': cfg, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'payload': payload}, path)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg['experiment'].get('seed', 42)))
    device = torch.device(cfg['training'].get('device', 'cuda'))
    out_dir = REPO_ROOT / cfg['experiment']['output_dir']
    for sub in ['checkpoints','logs','evaluation']:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    image_size = tuple(cfg['data']['image_size'])
    patch_size = tuple(cfg['data']['patch_size'])
    image_patch_coords = build_patch_grid(image_size, patch_size)
    latent_patch_size = tuple(int(v) // 4 for v in patch_size)
    latent_full_size = tuple(int(v) // 4 for v in image_size)
    latent_patch_coords = build_patch_grid(latent_full_size, latent_patch_size)
    coords = (image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)

    train_dataset, train_loader, val_loader, small_val_loader = build_loaders(cfg)
    table_num_features = len(train_dataset.table_cols)
    feature_type_ids, _ = infer_feature_type_ids(train_dataset.table_cols)
    ae_model = load_autoencoder('ae', REPO_ROOT / cfg['ae']['checkpoint_path'], device)
    for p in ae_model.parameters():
        p.requires_grad_(False)
    model = build_model(cfg, ae_model.latent_channels, table_num_features, feature_type_ids).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    warm = torch.load(REPO_ROOT / cfg['experiment']['common_init_checkpoint'], map_location=device)
    model.load_backbone_state(warm['model_state_dict'])
    diffusion = LatentVPDiffusion(timesteps=int(cfg['diffusion']['timesteps']), beta_start=float(cfg['diffusion'].get('beta_start', 1e-4)), beta_end=float(cfg['diffusion'].get('beta_end', 2e-2))).to(device)
    stats = torch.load(REPO_ROOT / cfg['experiment']['latent_stats_path'], map_location='cpu')
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg['training']['epochs']))

    ckpt_path = REPO_ROOT / cfg['experiment']['common_init_checkpoint']
    with open(ckpt_path, 'rb') as f:
        init_hash = hashlib.sha256(f.read()).hexdigest()

    batch = next(iter(train_loader))
    pet, _, z0, _, mri_patch = local_inputs(batch, ae_model, stats, cfg, device, image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)
    table = batch['table'].to(device).float(); table_mask = batch['table_mask'].to(device).float()
    audit = {'experiment': cfg['experiment']['name'], 'strategy': strategy_name(cfg), 'standard_diffusion_T': diffusion.timesteps, 'common_init_checkpoint': str(ckpt_path.resolve()), 'common_init_sha256': init_hash, 'ae_checkpoint': str((REPO_ROOT / cfg['ae']['checkpoint_path']).resolve()), 'latent_stats_path': str((REPO_ROOT / cfg['experiment']['latent_stats_path']).resolve()), 'latent_patch_shape': list(z0.shape), 'trainable_params': count_params(model)[1]}
    print(json.dumps(audit, indent=2))

    if args.smoke_test:
        model.train()
        loss, details = train_objective(model, diffusion, z0, mri_patch, table, table_mask, cfg, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sample_device = device
        smoke_model = model
        smoke_ae = ae_model
        smoke_diff = diffusion
        if strategy_name(cfg) == 'samplingawarediff':
            sample_device = torch.device('cpu')
            smoke_model = copy.deepcopy(model).to(sample_device).eval()
            smoke_ae = copy.deepcopy(ae_model).to(sample_device).eval()
            smoke_diff = LatentVPDiffusion(timesteps=diffusion.timesteps, beta_start=float(cfg['diffusion']['beta_start']), beta_end=float(cfg['diffusion']['beta_end'])).to(sample_device)
            pet = pet.cpu(); z0 = z0.cpu(); mri_patch = mri_patch.cpu(); table = table.cpu(); table_mask = table_mask.cpu()
        sample = sample_prediction(smoke_model, smoke_diff, z0, mri_patch, table, table_mask, cfg, sample_device)
        recon, _ = decode_local(smoke_ae, sample, pet.shape, smoke_ae.latent_channels, latent_patch_coords, latent_patch_size, latent_full_size, stats)
        smoke_csv = out_dir / 'logs' / 'smoke_train_log.csv'
        ensure_csv(smoke_csv, ['ok','loss'])
        append_csv(smoke_csv, [1, float(loss.item())])
        print(json.dumps({'loss': float(loss.item()), 'details': details, 'sample_shape': list(sample.shape), 'recon_shape': list(recon.shape), 'ae_grad_count': sum(1 for p in ae_model.parameters() if p.grad is not None)}, indent=2))
        return

    train_csv = out_dir / 'logs' / 'train_log.csv'
    val_csv = out_dir / 'logs' / 'val_log.csv'
    ensure_csv(train_csv, ['epoch','train_loss','val_loss','objective_name','lr_base','lr_cond','epoch_time_sec','trainable_params','peak_gpu_mem_mb','details_json'])
    ensure_csv(val_csv, ['epoch','eval_mode','seed_count','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae','brain_rmse','roi_mae','roi_rmse','roi_pearson','roi_spearman','boundary_mae','boundary_gradient_error','boundary_band_mae','ssim_std','psnr_std','mae_std','rmse_std','brain_ssim_std','brain_psnr_std','brain_mae_std','brain_rmse_std','roi_mae_std','roi_rmse_std','roi_pearson_std','roi_spearman_std','boundary_mae_std','boundary_gradient_error_std','boundary_band_mae_std'])

    best_obj = float('inf')
    best_ssim = -float('inf')
    sampling_interval = int(cfg.get('evaluation', {}).get('sampling_interval', 10))
    for epoch in range(1, int(cfg['training']['epochs']) + 1):
        start = time.time()
        model.train()
        loss_sum = 0.0
        seen = 0
        detail_bag = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        for batch in train_loader:
            pet, _, z0, _, mri_patch = local_inputs(batch, ae_model, stats, cfg, device, image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)
            table = batch['table'].to(device).float(); table_mask = batch['table_mask'].to(device).float()
            loss, details = train_objective(model, diffusion, z0, mri_patch, table, table_mask, cfg, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = int(z0.shape[0]); seen += n; loss_sum += float(loss.item()) * n
            for k, v in details.items():
                detail_bag.setdefault(k, []).append(v)
        scheduler.step()
        train_loss = loss_sum / max(seen, 1)
        detail_mean = {k: float(np.mean(v)) for k, v in detail_bag.items() if k != 'timestep_hist_json'}
        if 'timestep_hist_json' in detail_bag:
            detail_mean['timestep_hist_json'] = detail_bag['timestep_hist_json'][-1]

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        val_detail_bag = {}
        with torch.inference_mode():
            for batch in val_loader:
                pet, _, z0, _, mri_patch = local_inputs(batch, ae_model, stats, cfg, device, image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)
                table = batch['table'].to(device).float(); table_mask = batch['table_mask'].to(device).float()
                loss, details = train_objective(model, diffusion, z0, mri_patch, table, table_mask, cfg, device)
                n = int(z0.shape[0]); val_seen += n; val_loss_sum += float(loss.item()) * n
                for k, v in details.items():
                    val_detail_bag.setdefault(k, []).append(v)
        val_loss = val_loss_sum / max(val_seen, 1)
        val_detail_mean = {f'val_{k}': float(np.mean(v)) for k, v in val_detail_bag.items() if k != 'timestep_hist_json'}

        final_epoch = epoch == int(cfg['training']['epochs'])
        eval_modes = ('normal',) if not final_epoch else ('normal','mri_off','table_off','mri_shuffle','table_shuffle','global_off')
        eval_seeds = (42,) if not final_epoch else (42,123,2026)
        if strategy_name(cfg) == 'samplingawarediff' and (not final_epoch) and (epoch % sampling_interval != 0):
            rows = [{'eval_mode':'normal','seed_count':0,'ssim':float('nan'),'psnr':float('nan'),'mae':float('nan'),'rmse':float('nan'),'brain_ssim':float('nan'),'brain_psnr':float('nan'),'brain_mae':float('nan'),'brain_rmse':float('nan'),'roi_mae':float('nan'),'roi_rmse':float('nan'),'roi_pearson':float('nan'),'roi_spearman':float('nan'),'boundary_mae':float('nan'),'boundary_gradient_error':float('nan'),'boundary_band_mae':float('nan')}]
        else:
            eval_loader = val_loader if final_epoch else (small_val_loader if strategy_name(cfg) == 'samplingawarediff' else val_loader)
            eval_device = torch.device('cpu') if strategy_name(cfg) == 'samplingawarediff' else device
            rows = evaluate_generation_on_device(model, ae_model, diffusion, eval_loader, stats, cfg, eval_device, coords, seeds=eval_seeds, eval_modes=eval_modes)
        normal = [r for r in rows if r['eval_mode'] == 'normal'][0]

        peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        details_json = json.dumps({**detail_mean, **val_detail_mean})
        append_csv(train_csv, [epoch, train_loss, val_loss, objective_name(cfg), optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr'], time.time()-start, count_params(model)[1], peak_gpu_mem_mb, details_json])
        append_val_rows(val_csv, epoch, rows)
        payload = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'normal_ssim': float(normal['ssim']) if not math.isnan(float(normal['ssim'])) else -1.0, 'audit': audit, 'details': {**detail_mean, **val_detail_mean}}
        save_checkpoint(out_dir / 'checkpoints' / 'last.pt', epoch, cfg, model, optimizer, scheduler, payload)
        if val_loss < best_obj:
            best_obj = val_loss
            save_checkpoint(out_dir / 'checkpoints' / best_objective_ckpt_name(cfg), epoch, cfg, model, optimizer, scheduler, payload)
        if not math.isnan(float(normal['ssim'])) and float(normal['ssim']) > best_ssim:
            best_ssim = float(normal['ssim'])
            save_checkpoint(out_dir / 'checkpoints' / best_generation_ckpt_name(cfg), epoch, cfg, model, optimizer, scheduler, payload)


if __name__ == '__main__':
    main()
