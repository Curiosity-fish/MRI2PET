from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"Function {name} not found in {path}")


class TrainingContractTest(unittest.TestCase):
    def test_patch_prior_contract_has_no_legacy_full_prior_key(self) -> None:
        model_source = (ROOT / "models" / "exp026_model.py").read_text(encoding="utf-8")
        table_source = (ROOT / "models" / "table_interaction.py").read_text(encoding="utf-8")
        trainer_source = (ROOT / "training" / "trainer.py").read_text(encoding="utf-8")
        legacy_key = "prior" + "_full"
        self.assertNotIn(legacy_key, model_source)
        self.assertNotIn(legacy_key, table_source)
        self.assertNotIn(legacy_key, trainer_source)
        self.assertIn("prior_patches", model_source)
        self.assertIn("prior_patches", table_source)
        self.assertIn("prior_patches", trainer_source)

    def test_diffusion_state_remains_patch_latent(self) -> None:
        source = function_source(ROOT / "training" / "trainer.py", "_diffusion_step")
        self.assertIn("encode_pet_patches", source)
        self.assertIn("q_sample(clean", source)
        self.assertNotIn("stitch_latent_patches", source)

    def test_mri_patch_prior_stage_trains_encoder_and_prior_heads(self) -> None:
        source = function_source(ROOT / "training" / "stages.py", "configure_trainable_modules")
        branch = source.split('stage == "mri_patch_prior"', 1)[1].split(
            'stage == "local_control"', 1
        )[0]
        self.assertIn("_requires_grad(model.mri_control, True)", branch)
        self.assertIn("_requires_grad(model.mri_prior_heads, True)", branch)

    def test_local_stage_keeps_mri_prior_head_calibrated(self) -> None:
        stages = function_source(ROOT / "training" / "stages.py", "configure_trainable_modules")
        branch = stages.split('stage == "local_control"', 1)[1].split(
            'stage == "table_mri_prior"', 1
        )[0]
        self.assertIn("_requires_grad(model.mri_control, True)", branch)
        self.assertIn("_requires_grad(model.mri_prior_heads, True)", branch)
        trainer = function_source(
            ROOT / "training" / "trainer.py", "_streaming_diffusion_train_step"
        )
        self.assertIn("_mri_prior_consistency", trainer)
        self.assertIn('"mri_prior"', trainer)

    def test_mri_patch_prior_has_three_scale_supervision(self) -> None:
        source = function_source(ROOT / "training" / "trainer.py", "_mri_patch_prior_step")
        self.assertIn("build_control_patch_pyramid", source)
        self.assertIn("build_latent_patch_pyramid", source)
        for level in ('"high"', '"middle"', '"low"'):
            self.assertIn(level, source)

    def test_local_denoiser_has_exactly_two_downsampling_layers(self) -> None:
        source = (ROOT / "models" / "local_denoiser.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("self.down1 = Downsample3D"), 1)
        self.assertEqual(source.count("self.down2 = Downsample3D"), 1)
        self.assertNotIn("self.down3 = Downsample3D", source)

    def test_patch_vae_backpropagates_one_chunk_at_a_time(self) -> None:
        source = function_source(ROOT / "training" / "trainer.py", "_patch_vae_train_step")
        self.assertIn("for start in range", source)
        self.assertIn("backward()", source)
        self.assertNotIn("torch.cat(reconstructions", source)

    def test_diffusion_backpropagates_one_chunk_at_a_time(self) -> None:
        source = function_source(
            ROOT / "training" / "trainer.py", "_streaming_diffusion_train_step"
        )
        self.assertIn("prepare_denoise_context", source)
        self.assertIn("denoise_patch_chunk", source)
        self.assertIn("retain_graph=end < self.model.num_patches", source)
        self.assertIn("seam_surrogate", source)
        self.assertNotIn("predictions.append", source)
        run_epoch = function_source(ROOT / "training" / "trainer.py", "_run_epoch")
        self.assertNotIn(".backward(", run_epoch)

    def test_decoder_calibration_backpropagates_by_chunk(self) -> None:
        source = function_source(ROOT / "training" / "trainer.py", "_decoder_train_step")
        self.assertIn("decode_latent_chunk", source)
        self.assertIn("backward()", source)
        self.assertNotIn("decode_latent_patches(", source)

    def test_patch_vae_has_no_duplicate_full_volume_l1(self) -> None:
        trainer = (ROOT / "training" / "trainer.py").read_text(encoding="utf-8")
        config = (ROOT / "configs" / "t1mri_fdgpet_exp026.yaml").read_text(
            encoding="utf-8"
        )
        legacy_loss = "full" + "_reconstruction"
        self.assertNotIn(legacy_loss, trainer)
        self.assertNotIn(legacy_loss, config)

    def test_decoder_forward_has_no_internal_no_grad(self) -> None:
        wrapper = function_source(
            ROOT / "models" / "exp026_model.py", "decode_latent_patches"
        )
        chunk = function_source(ROOT / "models" / "exp026_model.py", "decode_latent_chunk")
        self.assertNotIn("no_grad", wrapper.replace("without no_grad", ""))
        self.assertNotIn("no_grad", chunk)
        self.assertIn("decode_latent_chunk", wrapper)
        self.assertIn("self.patch_vae.decoder", chunk)

    def test_joint_stage_decodes_predicted_latent(self) -> None:
        source = function_source(ROOT / "training" / "trainer.py", "_diffusion_step")
        self.assertIn("decode_latent_patches", source)
        self.assertIn("full_pet_loss", source)

    def test_decoder_stage_opens_only_decoder_after_global_freeze(self) -> None:
        source = function_source(ROOT / "training" / "stages.py", "configure_trainable_modules")
        self.assertIn("_requires_grad(model, False)", source)
        decoder_branch = source.split('stage == "decoder_calibration"', 1)[1]
        self.assertIn("_requires_grad(model.patch_vae.decoder, True)", decoder_branch)
        self.assertNotIn("_requires_grad(model.patch_vae.encoder, True)", decoder_branch)

    def test_global_branch_is_mamba_by_default(self) -> None:
        yaml_text = (ROOT / "configs" / "t1mri_fdgpet_exp026.yaml").read_text(encoding="utf-8")
        self.assertIn("backend: mamba", yaml_text)
        source = (ROOT / "models" / "global_mamba.py").read_text(encoding="utf-8")
        self.assertIn("from mamba_ssm import Mamba", source)

    def test_each_stage_selects_best_checkpoint_for_next_stage(self) -> None:
        source = function_source(ROOT / "training" / "trainer.py", "run")
        self.assertIn('best_path = checkpoints / "best.pt"', source)
        self.assertIn("selected = load_checkpoint", source)
        self.assertIn("best_path,", source)
        self.assertIn("optimizer=None", source)

    def test_best_score_survives_same_stage_resume(self) -> None:
        trainer = (ROOT / "training" / "trainer.py").read_text(encoding="utf-8")
        self.assertIn("_read_best_metric", trainer)
        self.assertIn("_write_best_metric", trainer)
        self.assertIn('best_metric_path = stage_dir / "best_metric.json"', trainer)
        self.assertIn("if resume:", trainer)
        self.assertIn("self._write_best_metric(best_metric_path, stage, 0, None)", trainer)

    def test_validation_selection_is_deterministic(self) -> None:
        trainer = (ROOT / "training" / "trainer.py").read_text(encoding="utf-8")
        config = (ROOT / "configs" / "t1mri_fdgpet_exp026.yaml").read_text(encoding="utf-8")
        self.assertIn("_validation_rng", trainer)
        self.assertIn("validation_seed: 2601", config)
        self.assertIn("epoch == epochs", trainer)


if __name__ == "__main__":
    unittest.main()
