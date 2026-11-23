from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp026.utils.checkpoints import load_checkpoint, save_checkpoint
from exp026.utils.config import STAGE_NAMES
from exp025s.training.trainer import Exp025STrainer
from exp025t.models.exp025t_model import Exp025TMRI2PET
from exp025t.training.stages import (
    configure_trainable_modules,
    set_stage_mode,
    trainable_parameter_names,
)
from metrics.image_metrics import compute_image_metrics_detailed


class Exp025TTrainer(Exp025STrainer):
    """Full-resolution structural calibration with an epoch-zero safeguard."""

    def __init__(
        self,
        model: Exp025TMRI2PET,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: dict,
        device: torch.device,
    ) -> None:
        super().__init__(model, train_loader, val_loader, config, device)
        self.model: Exp025TMRI2PET
        self.multiscale_factor = max(
            1, int(config["training"].get("multiscale_ssim_factor", 2))
        )
        self.selection_weights = config["training"].get(
            "selection_weights",
            {
                "ssim": 0.55,
                "brain_ssim": 0.25,
                "rmse": 0.10,
                "brain_rmse": 0.10,
            },
        )

    def _configure_trainable_modules(self, stage: str) -> list[str]:
        return configure_trainable_modules(self.model, stage)

    def _set_stage_mode(self, stage: str, training: bool) -> None:
        set_stage_mode(self.model, stage, training)

    def _image_objective(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        brain_mask: torch.Tensor,
        refined_latent: torch.Tensor,
        target_latent: Optional[torch.Tensor],
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction32 = prediction.float()
        target32 = target.float()
        mask32 = brain_mask.float()
        l1 = F.l1_loss(prediction32, target32)
        mse = F.mse_loss(prediction32, target32)
        brain_l1 = self._masked_l1(prediction32, target32, mask32)
        brain_mse = self._masked_mean(
            (prediction32 - target32).square(), mask32
        )
        gradient = self._gradient_loss(prediction32, target32, mask32)

        ssim_map = self.ssim_loss(
            prediction32, target32, return_map=True
        )
        ssim = ssim_map.mean()
        brain_ssim = self._masked_mean(ssim_map, mask32)
        ssim_term = 1.0 - ssim
        brain_ssim_term = 1.0 - brain_ssim

        multiscale_ssim_term = prediction32.new_zeros(())
        if self.multiscale_factor > 1:
            factor = self.multiscale_factor
            pooled_prediction = F.avg_pool3d(
                prediction32, kernel_size=factor, stride=factor
            )
            pooled_target = F.avg_pool3d(
                target32, kernel_size=factor, stride=factor
            )
            multiscale_ssim_term = 1.0 - self.ssim_loss(
                pooled_prediction, pooled_target
            )

        if target_latent is None:
            latent = prediction32.new_zeros(())
        else:
            latent = F.smooth_l1_loss(
                refined_latent.float(), target_latent.float()
            )
        range_penalty = F.relu(prediction32.abs() - 1.0).square().mean()

        loss = (
            self._weight(stage_config, "l1", 1.0) * l1
            + self._weight(stage_config, "mse", 0.5) * mse
            + self._weight(stage_config, "brain_l1", 1.0) * brain_l1
            + self._weight(stage_config, "brain_mse", 0.25) * brain_mse
            + self._weight(stage_config, "ssim", 0.5) * ssim_term
            + self._weight(stage_config, "brain_ssim", 0.5)
            * brain_ssim_term
            + self._weight(stage_config, "multiscale_ssim", 0.25)
            * multiscale_ssim_term
            + self._weight(stage_config, "gradient", 0.1) * gradient
            + self._weight(stage_config, "latent", 0.0) * latent
            + self._weight(stage_config, "range", 0.1) * range_penalty
        )
        return loss, {
            "loss": loss.detach(),
            "l1": l1.detach(),
            "mse": mse.detach(),
            "brain_l1": brain_l1.detach(),
            "brain_mse": brain_mse.detach(),
            "ssim_loss": ssim_term.detach(),
            "brain_ssim_loss": brain_ssim_term.detach(),
            "multiscale_ssim_loss": multiscale_ssim_term.detach(),
            "gradient": gradient.detach(),
            "latent": latent.detach(),
            "range": range_penalty.detach(),
            "smooth_weight": self.model.smooth_weight.detach(),
            "atlas_weight": self.model.atlas_weight.detach(),
            "coarse_latent_weight": torch.sigmoid(
                self.model.coarse_gate_logit.detach()
            ),
            "direct_residual_scale": self.model.direct_residual_scale.detach(),
        }

    def _selection_score(self, metrics: dict[str, float]) -> float:
        weights = self.selection_weights
        return float(
            float(weights.get("ssim", 0.55)) * (1.0 - metrics["ssim"])
            + float(weights.get("brain_ssim", 0.25))
            * (1.0 - metrics["brain_ssim"])
            + float(weights.get("rmse", 0.10)) * metrics["rmse"]
            + float(weights.get("brain_rmse", 0.10))
            * metrics["brain_rmse"]
        )

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

        target_latent = None
        if self._weight(stage_config, "latent", 0.0) > 0:
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
                self._selection_score(official)
            )
        return loss, metrics

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
        self._set_stage_mode(stage, training)
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        progress = tqdm(
            loader,
            desc=f"exp025t {stage} {epoch} {'train' if training else 'val'}",
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

    def _save_baseline(
        self,
        stage: str,
        stage_config: dict,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        checkpoints: Path,
        best_metric_path: Path,
    ) -> tuple[float, float, float]:
        if self.val_loader is None:
            raise RuntimeError("Exp025t requires a validation loader for baseline guard")
        val_metrics = self._run_epoch(
            stage,
            stage_config,
            self.val_loader,
            None,
            scaler,
            0,
            False,
        )
        score = float(val_metrics["selection_score"])
        record = {
            "epoch": 0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_peak_reserved_mib": 0.0,
            "train": None,
            "val": val_metrics,
            "baseline_guard": True,
        }
        self._append_metrics(stage, record)
        for name in ("best.pt", "best_ssim.pt", "best_psnr.pt", "last.pt"):
            save_checkpoint(
                checkpoints / name,
                self.model,
                optimizer,
                stage,
                0,
                self.config,
                record,
            )
        self._write_best_payload(
            best_metric_path, stage, 0, score, val_metrics
        )
        print(
            f"[exp025t] stage={stage} epoch=0 guard "
            f"ssim={val_metrics['ssim']:.6f} "
            f"psnr={val_metrics['psnr']:.6f} "
            f"brain_ssim={val_metrics['brain_ssim']:.6f}"
        )
        return score, float(val_metrics["ssim"]), float(val_metrics["psnr"])

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
                    f"[exp025t] skip {stage}: "
                    f"enabled={stage_config.get('enabled', True)}, epochs={epochs}"
                )
                continue

            groups = self._configure_trainable_modules(stage)
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
                    self.config["training"].get("lr_plateau_patience", 6)
                ),
                min_lr=float(
                    self.config["training"].get("min_learning_rate", 1e-7)
                ),
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
            checkpoints.mkdir(parents=True, exist_ok=True)
            best_path = checkpoints / "best.pt"
            best_metric_path = stage_dir / "best_metric.json"

            print(
                f"[exp025t] stage={stage}, epochs={epochs}, "
                f"trainable_groups={groups}, "
                f"trainable_tensors={len(trainable_parameter_names(self.model))}"
            )
            if resume:
                previous = self._read_best_payload(best_metric_path)
                best_score = float(previous.get("score", float("inf")))
                previous_metrics = previous.get("metrics") or {}
                best_ssim = float(previous_metrics.get("ssim", float("-inf")))
                best_psnr = float(previous_metrics.get("psnr", float("-inf")))
            else:
                best_score, best_ssim, best_psnr = self._save_baseline(
                    stage,
                    stage_config,
                    optimizer,
                    scaler,
                    checkpoints,
                    best_metric_path,
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
                score = (
                    float(val_metrics["selection_score"])
                    if val_metrics is not None
                    else float(train_metrics["loss"])
                )
                record = {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train_peak_reserved_mib": float(train_peak_mib),
                    "train": train_metrics,
                    "val": val_metrics,
                }
                self._append_metrics(stage, record)
                scheduler.step(score)

                if score < best_score:
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
                    self._write_best_payload(
                        best_metric_path,
                        stage,
                        epoch,
                        score,
                        val_metrics or train_metrics,
                    )
                if val_metrics is not None and val_metrics["ssim"] > best_ssim:
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
                if val_metrics is not None and val_metrics["psnr"] > best_psnr:
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
                    f"[exp025t] {stage} epoch={epoch} "
                    f"peak_reserved_mib={train_peak_mib:.1f} "
                    f"train={train_metrics} val={val_metrics}"
                )

            selected = load_checkpoint(
                best_path,
                self.model,
                optimizer=None,
                map_location=self.device,
                strict=True,
            )
            print(
                f"[exp025t] stage={stage} selected best.pt "
                f"from epoch={selected['epoch']} for the next stage"
            )
