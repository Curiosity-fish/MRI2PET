from __future__ import annotations

import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp026.training.trainer import Exp026Trainer
from exp026.utils.checkpoints import load_checkpoint, save_checkpoint
from exp026.utils.config import STAGE_NAMES
from exp025s.models.exp025s_model import Exp025SMRI2PET
from exp025s.training.stages import (
    configure_trainable_modules,
    set_stage_mode,
    trainable_parameter_names,
)
from metrics.image_metrics import compute_image_metrics_detailed
from utils.ssim import SSIM3D


class Exp025STrainer(Exp026Trainer):
    """Seven-phase trainer whose deployable path never consumes target PET."""

    def __init__(
        self,
        model: Exp025SMRI2PET,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: dict,
        device: torch.device,
    ) -> None:
        super().__init__(model, train_loader, val_loader, config, device)
        self.model: Exp025SMRI2PET
        self.ssim_downsample = max(
            1, int(config["training"].get("ssim_downsample", 2))
        )
        self.ssim_loss = SSIM3D(
            window_size=int(config["training"].get("ssim_window_size", 7)),
            data_range=float(config["training"].get("metric_data_range", 2.0)),
        ).to(device)
        self.metric_data_range = float(
            config["training"].get("metric_data_range", 2.0)
        )

    @staticmethod
    def _masked_mean(
        value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        return (value * mask).sum() / mask.sum().clamp_min(eps)

    @classmethod
    def _masked_l1(
        cls, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return cls._masked_mean((prediction - target).abs(), mask)

    @classmethod
    def _gradient_loss(
        cls, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        terms = []
        for dimension in (-3, -2, -1):
            pred_diff = torch.diff(prediction, dim=dimension)
            target_diff = torch.diff(target, dim=dimension)
            left = [slice(None)] * mask.ndim
            right = [slice(None)] * mask.ndim
            left[dimension] = slice(1, None)
            right[dimension] = slice(None, -1)
            edge_mask = mask[tuple(left)] * mask[tuple(right)]
            terms.append(cls._masked_l1(pred_diff, target_diff, edge_mask))
        return torch.stack(terms).mean()

    @staticmethod
    def _stage_flags(stage: str) -> tuple[bool, bool, bool, bool]:
        if stage == "mri_patch_prior":
            return False, False, False, False
        if stage == "local_control":
            return False, False, True, False
        if stage == "table_mri_prior":
            return True, False, True, False
        if stage == "global_mamba":
            return True, True, True, False
        if stage in ("joint_finetune", "decoder_calibration"):
            return True, True, True, True
        raise ValueError(f"Stage {stage} has no deterministic image path")

    def _brain_mask(self, batch: dict, target: torch.Tensor) -> torch.Tensor:
        mask = batch.get("brain_mask")
        if mask is None:
            return torch.ones_like(target)
        return mask.to(self.device, non_blocking=True).float()

    def _image_objective(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        brain_mask: torch.Tensor,
        refined_latent: torch.Tensor,
        target_latent: torch.Tensor,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction32 = prediction.float()
        target32 = target.float()
        mask32 = brain_mask.float()
        l1 = F.l1_loss(prediction32, target32)
        mse = F.mse_loss(prediction32, target32)
        brain_l1 = self._masked_l1(prediction32, target32, mask32)
        brain_mse = self._masked_mean((prediction32 - target32).square(), mask32)
        gradient = self._gradient_loss(prediction32, target32, mask32)
        latent = F.smooth_l1_loss(refined_latent.float(), target_latent.float())

        ssim_prediction = prediction32
        ssim_target = target32
        if self.ssim_downsample > 1:
            factor = self.ssim_downsample
            ssim_prediction = F.avg_pool3d(
                ssim_prediction, kernel_size=factor, stride=factor
            )
            ssim_target = F.avg_pool3d(
                ssim_target, kernel_size=factor, stride=factor
            )
        ssim = self.ssim_loss(ssim_prediction, ssim_target)
        ssim_term = 1.0 - ssim

        loss = (
            self._weight(stage_config, "l1", 1.0) * l1
            + self._weight(stage_config, "mse", 0.5) * mse
            + self._weight(stage_config, "brain_l1", 1.0) * brain_l1
            + self._weight(stage_config, "brain_mse", 0.25) * brain_mse
            + self._weight(stage_config, "ssim", 0.5) * ssim_term
            + self._weight(stage_config, "gradient", 0.1) * gradient
            + self._weight(stage_config, "latent", 0.25) * latent
        )
        return loss, {
            "loss": loss.detach(),
            "l1": l1.detach(),
            "mse": mse.detach(),
            "brain_l1": brain_l1.detach(),
            "brain_mse": brain_mse.detach(),
            "ssim_loss": ssim_term.detach(),
            "gradient": gradient.detach(),
            "latent": latent.detach(),
            "coarse_latent_weight": torch.sigmoid(
                self.model.coarse_gate_logit.detach()
            ),
            "direct_residual_scale": self.model.direct_residual_scale.detach(),
        }

    def _direct_step(
        self,
        stage: str,
        batch: dict,
        stage_config: dict,
        include_official_metrics: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mri, pet, table, table_mask = self._batch(batch)
        brain_mask = self._brain_mask(batch, pet)
        use_table, use_global, use_local, use_image = self._stage_flags(stage)
        prediction, auxiliary = self.model.predict_direct(
            mri,
            table,
            table_mask,
            self.patch_batch_size,
            use_table=use_table,
            use_global=use_global,
            use_local_refiner=use_local,
            use_image_refiner=use_image,
            clamp_output=False,
        )
        with torch.no_grad():
            target_latent = self.model.encode_pet_patches(
                pet,
                self.patch_batch_size,
                sample=False,
                encoder_grad=False,
            )
        loss, metrics = self._image_objective(
            prediction,
            pet,
            brain_mask,
            auxiliary["refined_latent"],
            target_latent,
            stage_config,
        )
        if include_official_metrics:
            official = compute_image_metrics_detailed(
                prediction.detach().float().clamp(-1.0, 1.0),
                pet.detach().float(),
                mask=brain_mask.detach().float(),
                data_range=self.metric_data_range,
            )
            for name, value in official.items():
                metrics[name] = prediction.new_tensor(float(value))
            metrics["selection_score"] = prediction.new_tensor(
                0.65 * (1.0 - official["ssim"])
                + 0.25 * (1.0 - official["brain_ssim"])
                + 0.10 * official["rmse"]
            )
        return loss, metrics

    def _train_step(
        self,
        stage: str,
        batch: dict,
        stage_config: dict,
        scaler: torch.cuda.amp.GradScaler,
    ) -> dict[str, torch.Tensor]:
        if stage == "patch_vae":
            return super()._train_step(stage, batch, stage_config, scaler)
        with torch.cuda.amp.autocast(enabled=self.amp):
            loss, metrics = self._direct_step(
                stage, batch, stage_config, include_official_metrics=False
            )
        scaler.scale(loss).backward()
        return metrics

    def _step(
        self, stage: str, batch: dict, stage_config: dict
    ) -> tuple[torch.Tensor, dict]:
        if stage == "patch_vae":
            return super()._step(stage, batch, stage_config)
        return self._direct_step(
            stage, batch, stage_config, include_official_metrics=True
        )

    def _run_epoch(
        self,
        stage: str,
        stage_config: dict,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        scaler: torch.cuda.amp.GradScaler,
        epoch: int,
        training: bool,
    ) -> dict[str, float]:
        set_stage_mode(self.model, stage, training)
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        progress = tqdm(
            loader,
            desc=f"exp025s {stage} {epoch} {'train' if training else 'val'}",
            leave=False,
        )
        rng_context = nullcontext() if training else self._validation_rng()
        with rng_context:
            for batch in progress:
                if batch is None:
                    continue
                batch_size = int(batch["image"].shape[0])
                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    metrics = self._train_step(
                        stage, batch, stage_config, scaler
                    )
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=self.amp):
                            _, metrics = self._step(stage, batch, stage_config)
                if training and optimizer is not None:
                    if self.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [
                                parameter
                                for parameter in self.model.parameters()
                                if parameter.requires_grad
                            ],
                            self.gradient_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                seen += batch_size
                for name, value in metrics.items():
                    totals[name] += float(value.item()) * batch_size
                progress.set_postfix(loss=float(metrics["loss"].item()))
        if seen == 0:
            raise RuntimeError(f"No valid batches were produced for stage {stage}")
        return {name: value / seen for name, value in totals.items()}

    @staticmethod
    def _read_best_payload(path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_best_payload(
        path: Path,
        stage: str,
        epoch: int,
        score: Optional[float],
        metrics: Optional[dict],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stage": stage,
                    "epoch": int(epoch),
                    "score": float(score) if score is not None else None,
                    "metrics": metrics,
                },
                handle,
                indent=2,
            )
        temporary.replace(path)

    def run(self, only_stage: Optional[str] = None) -> None:
        if only_stage is not None and only_stage not in STAGE_NAMES:
            raise ValueError(f"Unknown stage: {only_stage}")
        for stage in STAGE_NAMES:
            if only_stage is not None and stage != only_stage:
                continue
            stage_config = self.config["stages"][stage]
            epochs = int(stage_config["epochs"])
            if not bool(stage_config.get("enabled", True)) or epochs == 0:
                print(
                    f"[exp025s] skip {stage}: "
                    f"enabled={stage_config.get('enabled', True)}, epochs={epochs}"
                )
                continue

            groups = configure_trainable_modules(self.model, stage)
            parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            if not parameters:
                raise RuntimeError(f"Stage {stage} has no trainable parameters")
            optimizer = torch.optim.AdamW(
                parameters,
                lr=float(stage_config["learning_rate"]),
                weight_decay=float(
                    self.config["training"].get("weight_decay", 0.0)
                ),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(
                    self.config["training"].get("lr_plateau_factor", 0.5)
                ),
                patience=int(
                    self.config["training"].get("lr_plateau_patience", 8)
                ),
                min_lr=float(self.config["training"].get("min_learning_rate", 1e-7)),
            )
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
            start_epoch = 1
            resume = stage_config.get("resume")
            if resume:
                checkpoint = load_checkpoint(
                    resume,
                    self.model,
                    optimizer=optimizer,
                    map_location=self.device,
                    strict=True,
                )
                if checkpoint.get("stage") != stage:
                    raise ValueError(
                        f"Resume checkpoint stage={checkpoint.get('stage')} "
                        f"does not match {stage}"
                    )
                start_epoch = int(checkpoint["epoch"]) + 1

            stage_dir = self.output_root / stage
            checkpoints = stage_dir / "checkpoints"
            best_path = checkpoints / "best.pt"
            best_metric_path = stage_dir / "best_metric.json"
            previous = self._read_best_payload(best_metric_path) if resume else {}
            best_score = float(previous.get("score", float("inf")))
            best_ssim = float(previous.get("best_ssim", float("-inf")))
            best_psnr = float(previous.get("best_psnr", float("-inf")))
            if not resume:
                self._write_best_payload(
                    best_metric_path, stage, 0, None, None
                )

            print(
                f"[exp025s] stage={stage}, epochs={epochs}, "
                f"trainable_groups={groups}, "
                f"trainable_tensors={len(trainable_parameter_names(self.model))}"
            )
            for epoch in range(start_epoch, epochs + 1):
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                train_metrics = self._run_epoch(
                    stage,
                    stage_config,
                    self.train_loader,
                    optimizer,
                    scaler,
                    epoch,
                    True,
                )
                train_peak_mib = (
                    torch.cuda.max_memory_reserved(self.device) / (1024**2)
                    if self.device.type == "cuda"
                    else 0.0
                )
                val_metrics = None
                interval = max(
                    1, int(self.config["training"].get("val_interval", 1))
                )
                if self.val_loader is not None and (
                    epoch % interval == 0 or epoch == epochs
                ):
                    val_metrics = self._run_epoch(
                        stage,
                        stage_config,
                        self.val_loader,
                        None,
                        scaler,
                        epoch,
                        False,
                    )
                score = None
                if val_metrics is not None:
                    score = float(
                        val_metrics.get("selection_score", val_metrics["loss"])
                    )
                elif self.val_loader is None:
                    score = float(train_metrics["loss"])
                record = {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train_peak_reserved_mib": float(train_peak_mib),
                    "train": train_metrics,
                    "val": val_metrics,
                }
                self._append_metrics(stage, record)

                if score is not None:
                    scheduler.step(score)
                if score is not None and score < best_score:
                    best_score = score
                    save_checkpoint(
                        best_path,
                        self.model,
                        optimizer,
                        stage,
                        epoch,
                        self.config,
                        record,
                    )
                    payload_metrics = val_metrics or train_metrics
                    self._write_best_payload(
                        best_metric_path,
                        stage,
                        epoch,
                        score,
                        payload_metrics,
                    )
                if val_metrics is not None and "ssim" in val_metrics:
                    if val_metrics["ssim"] > best_ssim:
                        best_ssim = float(val_metrics["ssim"])
                        save_checkpoint(
                            checkpoints / "best_ssim.pt",
                            self.model,
                            optimizer,
                            stage,
                            epoch,
                            self.config,
                            record,
                        )
                    if val_metrics["psnr"] > best_psnr:
                        best_psnr = float(val_metrics["psnr"])
                        save_checkpoint(
                            checkpoints / "best_psnr.pt",
                            self.model,
                            optimizer,
                            stage,
                            epoch,
                            self.config,
                            record,
                        )
                save_checkpoint(
                    checkpoints / "last.pt",
                    self.model,
                    optimizer,
                    stage,
                    epoch,
                    self.config,
                    record,
                )
                print(
                    f"[exp025s] {stage} epoch={epoch} "
                    f"peak_reserved_mib={train_peak_mib:.1f} "
                    f"train={train_metrics} val={val_metrics}"
                )

            if not best_path.exists():
                raise FileNotFoundError(
                    f"Stage {stage} finished without a best checkpoint: {best_path}"
                )
            selected = load_checkpoint(
                best_path,
                self.model,
                optimizer=None,
                map_location=self.device,
                strict=True,
            )
            print(
                f"[exp025s] stage={stage} selected best.pt "
                f"from epoch={selected['epoch']} for the next stage"
            )
