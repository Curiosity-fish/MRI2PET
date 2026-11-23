"""Exp031_HMRD main model.

Hierarchical Metabolic Residual Diffusion
==========================================

    PET = PET_low + PET_residual

The model does *not* ask the diffusion to generate the whole PET.  Instead:

  * GlobalMetabolicPrior3D predicts the coarse metabolic PET (low-frequency
    component) from MRI multi-scale features + missing-aware Table tokens, and
    also predicts the 98 Parc98 ROI mean uptake values.
  * A *separate* residual Patch-DVAE (``residual_vae``) encodes the residual
    Y_res = Y - Y_low into the same patch-latent space used by the stable
    Patch-Latent Global-Local Diffusion (Exp024/025/026 backbone).
  * The residual diffusion (Local Patch Denoiser + Global Bi-Mamba + MRI
    control + Table condition) is conditioned on the coarse PET latent pyramid,
    the 98 Parc98 metabolic tokens and the soft Patch-to-Region anatomical
    routing bias.
  * Final PET = clamp(coarse_pet + pred_residual_pet, -1, 1).

The model is fully independent: it only reuses the stable building blocks
(PatchDVAE3D, MRIControlEncoder3D, JointTableMRIPrior3D, GlobalBiMamba3D,
DiffusionSchedule) and never loads Exp027/028/029/030 checkpoints.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from exp026.models.exp026_model import Exp026MRI2PET, _ceil_half
from exp026.models.patch_vae import PatchDVAE3D
from exp026.training.diffusion import DiffusionSchedule

from exp031.models.anatomical_routing import get_or_build_routing_matrix
from exp031.models.decomposition import decompose_pet
from exp031.models.denoiser_hmrd import HMRDLocalControlDenoiser3D
from exp031.models.metabolic_prior import GlobalMetabolicPrior3D
from exp031.models.region_routing_attn import RegionRoutingAttention3D
from exp031.models.roi_tokens import Parc98MetabolicTokens


class Exp031MRI2PET(Exp026MRI2PET):
    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        table_num_features: int,
        feature_type_ids: Optional[Sequence[int]] = None,
        image_channels: int = 1,
        latent_channels: int = 4,
        vae_base_channels: int = 16,
        mri_channels: tuple[int, int, int, int] = (8, 16, 32, 64),
        local_channels: tuple[int, int, int] = (32, 64, 96),
        time_dim: int = 128,
        table_token_dim: int = 128,
        table_token_grid: tuple[int, int, int] = (4, 4, 4),
        table_depth: int = 2,
        table_heads: int = 4,
        num_feature_types: int = 1,
        global_token_dim: int = 128,
        mamba_depth: int = 2,
        mamba_state_dim: int = 16,
        mamba_conv_width: int = 4,
        mamba_expand: int = 2,
        mamba_backend: str = "mamba",
        allow_test_backend: bool = False,
        norm_groups: int = 8,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        *,
        num_regions: int = 98,
        region_token_dim: int = 128,
        routing_num_heads: int = 4,
        routing_labelmap_path: Optional[str] = None,
        routing_cache_path: Optional[str] = None,
        diffusion_timesteps: int = 1000,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 2e-2,
        metabolic_prior_channels: Optional[tuple[int, int, int]] = None,
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            table_num_features=table_num_features,
            feature_type_ids=feature_type_ids,
            image_channels=image_channels,
            latent_channels=latent_channels,
            vae_base_channels=vae_base_channels,
            mri_channels=mri_channels,
            local_channels=local_channels,
            time_dim=time_dim,
            table_token_dim=table_token_dim,
            table_token_grid=table_token_grid,
            table_depth=table_depth,
            table_heads=table_heads,
            num_feature_types=num_feature_types,
            global_token_dim=global_token_dim,
            mamba_depth=mamba_depth,
            mamba_state_dim=mamba_state_dim,
            mamba_conv_width=mamba_conv_width,
            mamba_expand=mamba_expand,
            mamba_backend=mamba_backend,
            allow_test_backend=allow_test_backend,
            norm_groups=norm_groups,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        control_channels = tuple(int(v) for v in mri_channels[1:])
        self.num_regions = int(num_regions)
        self.region_token_dim = int(region_token_dim)

        # Replace the local denoiser with the HMRD variant (coarse + region conds).
        self.local_denoiser = HMRDLocalControlDenoiser3D(
            latent_channels=latent_channels,
            mri_channels=control_channels,
            local_channels=local_channels,
            time_dim=time_dim,
            table_token_dim=table_token_dim,
            global_token_dim=global_token_dim,
            region_token_dim=region_token_dim,
            grid_size=self.grid_size,
            norm_groups=norm_groups,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        # ---- Exp031 modules (from-scratch, independent instances) ----
        self.residual_vae = PatchDVAE3D(
            image_channels=image_channels,
            latent_channels=latent_channels,
            base_channels=vae_base_channels,
            norm_groups=norm_groups,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        mri_prior_channels = (
            metabolic_prior_channels
            if metabolic_prior_channels is not None
            else control_channels
        )
        self.metabolic_prior = GlobalMetabolicPrior3D(
            mri_channels=tuple(int(v) for v in mri_prior_channels),
            fuse_channels=int(local_channels[0]),
            region_token_dim=region_token_dim,
            num_regions=self.num_regions,
            table_token_dim=table_token_dim,
            norm_groups=norm_groups,
            dropout=dropout,
        )
        self.roi_tokens = Parc98MetabolicTokens(
            num_regions=self.num_regions,
            token_dim=region_token_dim,
            table_token_dim=table_token_dim,
        )
        self.routing_attn = RegionRoutingAttention3D(
            latent_channels=latent_channels,
            mri_channels=control_channels[0],
            token_dim=region_token_dim,
            num_regions=self.num_regions,
            num_heads=routing_num_heads,
        )

        # Diffusion schedule used by the DDIM sampler during validation/eval.
        self._schedule_cache: dict[str, DiffusionSchedule] = {}
        self.diffusion_timesteps = int(diffusion_timesteps)
        self._diffusion_beta_start = float(diffusion_beta_start)
        self._diffusion_beta_end = float(diffusion_beta_end)
        self.schedule = DiffusionSchedule(
            timesteps=self.diffusion_timesteps,
            beta_start=self._diffusion_beta_start,
            beta_end=self._diffusion_beta_end,
            device=torch.device("cpu"),
        )
        self._schedule_cache[str(torch.device("cpu"))] = self.schedule

        # Patch-to-Region anatomical routing matrix [num_patches, num_regions].
        routing = None
        if routing_labelmap_path is not None:
            routing = get_or_build_routing_matrix(
                routing_labelmap_path,
                self.image_size,
                self.patch_size,
                cache_path=routing_cache_path,
                num_regions=self.num_regions,
                normalize_rows=True,
            )
        if routing is None:
            routing = torch.zeros((self.num_patches, self.num_regions))
        self.register_buffer("routing_matrix", routing.contiguous(), persistent=False)

    def _schedule_for(self, device: torch.device) -> DiffusionSchedule:
        """Return a DiffusionSchedule living on ``device`` (lazily built and cached)."""
        device = torch.device(device)
        key = str(device)
        cached = self._schedule_cache.get(key)
        if cached is None:
            cached = DiffusionSchedule(
                timesteps=self.diffusion_timesteps,
                beta_start=self._diffusion_beta_start,
                beta_end=self._diffusion_beta_end,
                device=device,
            )
            self._schedule_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Residual Patch-DVAE helpers (independent of self.patch_vae)
    # ------------------------------------------------------------------
    def encode_residual_patches(
        self,
        residual_pet: torch.Tensor,
        patch_batch_size: int,
        sample: bool = False,
        encoder_grad: bool = False,
    ) -> torch.Tensor:
        """Encode Y_res (image domain) into residual latent patches [B,P,C,d,h,w]."""
        patches = self.extract_image_patches(residual_pet)
        chunks = []
        grad_context = nullcontext() if encoder_grad else torch.no_grad()
        with grad_context:
            for start in range(0, self.num_patches, int(patch_batch_size)):
                patch = patches[:, start : start + int(patch_batch_size)]
                batch, count = patch.shape[:2]
                mu, logvar = self.residual_vae.encoder(
                    patch.reshape(batch * count, patch.shape[2], *patch.shape[-3:])
                )
                latent = self.residual_vae.reparameterize(mu, logvar) if sample else mu
                chunks.append(latent.reshape(batch, count, *latent.shape[1:]))
        return torch.cat(chunks, dim=1)

    def decode_residual_latent_patches(
        self,
        latent_patches: torch.Tensor,
        patch_batch_size: int,
    ) -> torch.Tensor:
        """Decode residual latent patches back to image-domain residual patches."""
        if latent_patches.ndim != 6:
            raise ValueError(f"Expected [B,P,C,D,H,W] latent patches, got {tuple(latent_patches.shape)}")
        chunks = []
        for start in range(0, self.num_patches, int(patch_batch_size)):
            latent = latent_patches[:, start : start + int(patch_batch_size)]
            batch, count = latent.shape[:2]
            decoded = self.residual_vae.decoder(
                latent.reshape(batch * count, latent.shape[2], *latent.shape[-3:])
            )
            chunks.append(decoded.reshape(batch, count, *decoded.shape[1:]))
        return torch.cat(chunks, dim=1)

    # ------------------------------------------------------------------
    # HMRD context construction
    # ------------------------------------------------------------------
    def build_hmrd_context(
        self,
        mri: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        *,
        use_table: bool,
        use_global: bool,
        inject_table_locally: bool,
        use_metabolic_prior: bool,
        use_roi_tokens: bool,
        use_anatomical_routing: bool,
        patch_batch_size: int,
    ) -> dict[str, object]:
        controls = self.mri_control(mri)
        control_patches = self.build_control_patch_pyramid(controls)
        mri_prior_pyramid = None
        table_context = None
        prior_patches = None
        if use_table:
            if table is None:
                raise ValueError("HMRD table conditioning requires table values")
            mri_prior_pyramid = self.mri_prior_heads(control_patches)
            table_context = self.table_interaction(
                control_patches["high"],
                table,
                table_mask,
                base_prior_patches=mri_prior_pyramid["high"],
            )
            if inject_table_locally:
                prior_patches = self.build_table_prior_pyramid(
                    table_context["prior_patches"], mri_prior_pyramid
                )
        table_tokens = table_context["table_tokens"] if table_context is not None else None
        subject_token = table_context["subject_token"] if table_context is not None else None

        metabolic = None
        coarse_pet = None
        coarse_low_small = None
        region_uptake = None
        coarse_prior = None
        if use_metabolic_prior:
            metabolic = self.metabolic_prior(
                controls,
                table_tokens=table_tokens,
                subject_token=subject_token,
            )
            coarse_low_small = metabolic["coarse_low_small"]
            coarse_pet = metabolic["coarse_pet"]
            region_uptake = metabolic["region_uptake"]
            coarse_patches = self.encode_residual_patches(
                coarse_pet, patch_batch_size, sample=False, encoder_grad=False
            )
            coarse_prior = self.build_latent_patch_pyramid(coarse_patches)

        roi_tokens = None
        if use_roi_tokens and region_uptake is not None:
            roi_tokens = self.roi_tokens(
                region_uptake,
                table_tokens=table_tokens,
                subject_token=subject_token,
            )

        return {
            "controls": controls,
            "control_patches": control_patches,
            "mri_prior_pyramid": mri_prior_pyramid,
            "table_context": table_context,
            "prior_patches": prior_patches,
            "table_tokens": table_tokens,
            "subject_token": subject_token,
            "metabolic": metabolic,
            "coarse_pet": coarse_pet,
            "coarse_low_small": coarse_low_small,
            "region_uptake": region_uptake,
            "coarse_prior": coarse_prior,
            "roi_tokens": roi_tokens,
            "use_table": bool(use_table),
            "use_global": bool(use_global),
            "inject_table_locally": bool(inject_table_locally),
            "use_metabolic_prior": bool(use_metabolic_prior),
            "use_roi_tokens": bool(use_roi_tokens),
            "use_anatomical_routing": bool(use_anatomical_routing),
        }

    def _hmrd_aux_tokens(
        self,
        noisy_latent_patches: torch.Tensor,
        timesteps: torch.Tensor,
        context: dict[str, object],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Global Mamba tokens + per-patch region tokens for a denoise call."""
        global_tokens = None
        if context["use_global"]:
            time_embedding = self.local_denoiser.embed_time(timesteps)
            global_tokens, _ = self.global_mamba(
                noisy_latent_patches,
                context["control_patches"]["high"],
                time_embedding,
                table_tokens=context["table_tokens"],
                table_key_padding_mask=(
                    context["table_context"]["table_key_padding_mask"]
                    if context["table_context"] is not None
                    else None
                ),
                reliability=(
                    context["table_context"]["reliability"]
                    if context["table_context"] is not None
                    else None
                ),
            )
        region_token = None
        if context["use_roi_tokens"] and context["use_anatomical_routing"]:
            region_token = self.routing_attn(
                noisy_latent_patches,
                context["control_patches"]["high"],
                context["roi_tokens"],
                self.routing_matrix.to(device=noisy_latent_patches.device),
            )
        return global_tokens, region_token

    def denoise_hmrd_patch_chunk(
        self,
        noisy_latent_patches: torch.Tensor,
        timesteps: torch.Tensor,
        context: dict[str, object],
        global_tokens: Optional[torch.Tensor],
        region_token: Optional[torch.Tensor],
        start: int,
        end: int,
    ) -> torch.Tensor:
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > self.num_patches:
            raise ValueError(f"Invalid patch range [{start},{end})")
        batch = noisy_latent_patches.shape[0]
        count = end - start
        control_patches = context["control_patches"]
        prior_patches = context["prior_patches"]
        table_context = context["table_context"]
        inject_table_locally = bool(context["inject_table_locally"])

        local_control = {
            level: self._flatten_selected(value, start, end)
            for level, value in control_patches.items()
        }
        local_prior = (
            {
                level: self._flatten_selected(value, start, end)
                for level, value in prior_patches.items()
            }
            if prior_patches is not None
            else None
        )
        coarse_prior = context["coarse_prior"]
        local_coarse = (
            {
                level: self._flatten_selected(value, start, end)
                for level, value in coarse_prior.items()
            }
            if coarse_prior is not None
            else None
        )
        coords = self.grid_coords[start:end][None].expand(batch, -1, -1).reshape(-1, 3)
        time_chunk = timesteps.repeat_interleave(count)
        subject = None
        if table_context is not None and inject_table_locally:
            subject = table_context["subject_token"].repeat_interleave(count, dim=0)
        global_chunk = (
            self._flatten_selected(global_tokens, start, end)
            if global_tokens is not None
            else None
        )
        region_chunk = (
            self._flatten_selected(region_token, start, end)
            if region_token is not None
            else None
        )
        noisy_chunk = self._flatten_selected(noisy_latent_patches, start, end)
        prediction = self.local_denoiser(
            noisy_chunk,
            time_chunk,
            coords,
            local_control,
            table_prior=local_prior,
            subject_token=subject,
            global_token=global_chunk,
            coarse_prior=local_coarse,
            region_token=region_chunk,
        )
        return prediction.reshape(batch, count, *prediction.shape[1:])

    def denoise_hmrd_patches(
        self,
        noisy_latent_patches: torch.Tensor,
        timesteps: torch.Tensor,
        context: dict[str, object],
        patch_batch_size: int,
    ) -> torch.Tensor:
        global_tokens, region_token = self._hmrd_aux_tokens(
            noisy_latent_patches, timesteps, context
        )
        predictions = []
        for start in range(0, self.num_patches, int(patch_batch_size)):
            end = min(start + int(patch_batch_size), self.num_patches)
            predictions.append(
                self.denoise_hmrd_patch_chunk(
                    noisy_latent_patches,
                    timesteps,
                    context,
                    global_tokens,
                    region_token,
                    start,
                    end,
                )
            )
        return torch.cat(predictions, dim=1)

    # ------------------------------------------------------------------
    # DDIM sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_residual_latent_ddim(
        self,
        context: dict[str, object],
        patch_batch_size: int,
        *,
        ddim_steps: int = 50,
        eta: float = 0.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """DDIM sample the residual latent from pure noise.

        Returns [B, P, C, d, h, w] residual latent patches.
        """
        batch = context["control_patches"]["high"].shape[0]
        latent_shape = (
            batch,
            self.num_patches,
            self.latent_channels,
            *self.latent_patch_size,
        )
        if seed is not None:
            generator = torch.Generator(device=context["control_patches"]["high"].device)
            generator.manual_seed(int(seed))
        else:
            generator = None
        if generator is not None:
            noisy = torch.randn(latent_shape, device=context["control_patches"]["high"].device, generator=generator)
        else:
            noisy = torch.randn(latent_shape, device=context["control_patches"]["high"].device)
        steps = max(2, int(ddim_steps))
        timestep_list = torch.linspace(
            self.diffusion_timesteps - 1, 0, steps + 1, dtype=torch.long
        ).tolist()
        timestep_list = [int(value) for value in timestep_list]
        schedule = self._schedule_for(noisy.device)
        for index in range(steps):
            t = timestep_list[index]
            t_prev = timestep_list[index + 1]
            t_tensor = torch.full((batch,), t, device=noisy.device, dtype=torch.long)
            predicted_noise = self.denoise_hmrd_patches(
                noisy, t_tensor, context, patch_batch_size
            )
            alpha_bar_t = schedule.sqrt_alpha_bars[t].to(noisy.device).square()
            alpha_bar_prev = schedule.sqrt_alpha_bars[t_prev].to(noisy.device).square()
            predicted_clean = (
                noisy - (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8) * predicted_noise
            ) / alpha_bar_t.sqrt().clamp_min(1e-6)
            if t_prev <= 0:
                noisy = predicted_clean
                continue
            sigma = (
                eta
                * ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)).sqrt()
                * (1.0 - alpha_bar_t / alpha_bar_prev).sqrt()
            )
            direction = (1.0 - alpha_bar_prev - sigma.square()).clamp_min(0.0).sqrt() * predicted_noise
            if generator is not None:
                noise = torch.randn_like(noisy, generator=generator)
            else:
                noise = torch.randn_like(noisy)
            noisy = alpha_bar_prev.sqrt() * predicted_clean + direction + sigma * noise
        return noisy

    # ------------------------------------------------------------------
    # Full HMRD prediction (validation / evaluation)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_hmrd(
        self,
        mri: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        patch_batch_size: int,
        *,
        use_table: bool,
        use_global: bool,
        use_metabolic_prior: bool,
        use_roi_tokens: bool,
        use_anatomical_routing: bool,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        validation_seed: int = 2601,
        clamp_output: bool = True,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        timesteps = torch.zeros((mri.shape[0],), device=mri.device, dtype=torch.long)
        context = self.build_hmrd_context(
            mri,
            table,
            table_mask,
            timesteps,
            use_table=use_table,
            use_global=use_global,
            inject_table_locally=use_table,
            use_metabolic_prior=use_metabolic_prior,
            use_roi_tokens=use_roi_tokens,
            use_anatomical_routing=use_anatomical_routing,
            patch_batch_size=patch_batch_size,
        )
        if context["use_metabolic_prior"]:
            coarse_pet = context["coarse_pet"]
        else:
            coarse_pet = torch.zeros_like(mri)
        residual_latent = self.sample_residual_latent_ddim(
            context,
            patch_batch_size,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            seed=validation_seed,
        )
        residual_patches = self.decode_residual_latent_patches(
            residual_latent, patch_batch_size
        )
        pred_residual_pet = self.stitch_image_patches(residual_patches)
        pred_pet = coarse_pet + pred_residual_pet
        if clamp_output:
            pred_pet = pred_pet.clamp(-1.0, 1.0)
        auxiliary = {
            "coarse_pet": coarse_pet,
            "coarse_low_small": context["coarse_low_small"],
            "region_uptake": context["region_uptake"],
            "pred_residual_pet": pred_residual_pet,
            "residual_latent": residual_latent,
            "context": context,
        }
        return pred_pet, auxiliary

    # ------------------------------------------------------------------
    # Training helper: one HMRD forward (no sampling) for diffusion training
    # ------------------------------------------------------------------
    def forward_hmrd_diffusion(
        self,
        clean_residual_latent: torch.Tensor,
        timesteps: torch.Tensor,
        mri: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        patch_batch_size: int,
        *,
        use_table: bool,
        use_global: bool,
        use_metabolic_prior: bool,
        use_roi_tokens: bool,
        use_anatomical_routing: bool,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object], torch.Tensor]:
        """Return (noisy, predicted_noise, predicted_clean, context, noise)."""
        if noise is None:
            noise = torch.randn_like(clean_residual_latent)
        schedule = self._schedule_for(clean_residual_latent.device)
        noisy = schedule.q_sample(clean_residual_latent, timesteps, noise)
        context = self.build_hmrd_context(
            mri,
            table,
            table_mask,
            timesteps,
            use_table=use_table,
            use_global=use_global,
            inject_table_locally=use_table,
            use_metabolic_prior=use_metabolic_prior,
            use_roi_tokens=use_roi_tokens,
            use_anatomical_routing=use_anatomical_routing,
            patch_batch_size=patch_batch_size,
        )
        predicted_noise = self.denoise_hmrd_patches(
            noisy, timesteps, context, patch_batch_size
        )
        predicted_clean = schedule.predict_clean(
            noisy, timesteps, predicted_noise
        )
        return noisy, predicted_noise, predicted_clean, context, noise


def build_exp031_model(
    config: dict,
    table_num_features: int,
    feature_type_ids: Optional[Sequence[int]] = None,
    routing_labelmap_path: Optional[str] = None,
    routing_cache_path: Optional[str] = None,
) -> Exp031MRI2PET:
    data = config["data"]
    model = config["model"]
    mamba = model["global_mamba"]
    table = model["table"]
    hmrd = model.get("hmrd", {})
    diffusion = config["diffusion"]
    return Exp031MRI2PET(
        image_size=tuple(data["image_size"]),
        patch_size=tuple(data["patch_size"]),
        table_num_features=table_num_features,
        feature_type_ids=feature_type_ids,
        image_channels=int(model.get("image_channels", 1)),
        latent_channels=int(model.get("latent_channels", 4)),
        vae_base_channels=int(model.get("vae_base_channels", 16)),
        mri_channels=tuple(model.get("mri_channels", [8, 16, 32, 64])),
        local_channels=tuple(model.get("local_channels", [32, 64, 96])),
        time_dim=int(model.get("time_dim", 128)),
        table_token_dim=int(table.get("token_dim", 128)),
        table_token_grid=tuple(table.get("token_grid", [4, 4, 4])),
        table_depth=int(table.get("depth", 2)),
        table_heads=int(table.get("heads", 4)),
        num_feature_types=int(table.get("num_feature_types", 1)),
        global_token_dim=int(mamba.get("token_dim", 128)),
        mamba_depth=int(mamba.get("depth", 2)),
        mamba_state_dim=int(mamba.get("state_dim", 16)),
        mamba_conv_width=int(mamba.get("conv_width", 4)),
        mamba_expand=int(mamba.get("expand", 2)),
        mamba_backend=str(mamba.get("backend", "mamba")),
        allow_test_backend=bool(mamba.get("allow_test_backend", False)),
        norm_groups=int(model.get("norm_groups", 8)),
        dropout=float(model.get("dropout", 0.0)),
        gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True)),
        num_regions=int(hmrd.get("num_regions", 98)),
        region_token_dim=int(hmrd.get("region_token_dim", 128)),
        routing_num_heads=int(hmrd.get("routing_num_heads", 4)),
        routing_labelmap_path=routing_labelmap_path,
        routing_cache_path=routing_cache_path,
        diffusion_timesteps=int(diffusion.get("timesteps", 1000)),
        diffusion_beta_start=float(diffusion.get("beta_start", 1e-4)),
        diffusion_beta_end=float(diffusion.get("beta_end", 2e-2)),
        metabolic_prior_channels=(
            tuple(int(v) for v in hmrd["metabolic_prior_channels"])
            if hmrd.get("metabolic_prior_channels") is not None
            else None
        ),
    )


__all__ = ["Exp031MRI2PET", "build_exp031_model"]
