import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri2pet.models.generation import PaperBCGGLDB, SharedPatchAE, TimeGatedBiCA
from mri2pet.models.unet_blocks import TimeResBlock3D
from mri2pet.losses.generation import (
    bridge_state,
    reverse_step,
    generation_loss,
    parc98_losses,
)
from mri2pet.data.classification import TABLE_COLS, fit_clinical_stats
from mri2pet.metrics.classification import classify_metrics, five_model_summary
from mri2pet.models.classification import CappedBiCA
from mri2pet.data.paired_mri_pet_clinical import PairedT1FDGTableDataset


class ProtocolTests(unittest.TestCase):
    def test_time_residual_block_follows_equation(self):
        block = TimeResBlock3D(4, 8, 16, 4, 0.0).eval()
        x = torch.randn(2, 4, 4, 4, 4)
        time = torch.randn(2, 16)
        h = torch.nn.functional.silu(block.norm1(block.conv1(x)) + block.time(time)[:, :, None, None, None])
        expected = block.skip(x) + block.conv2(torch.nn.functional.silu(block.norm2(h)))
        self.assertTrue(torch.allclose(block(x, time), expected))

    def test_generation_attention_queries_and_time_update(self):
        block = TimeGatedBiCA(8, token_dim=8, time_dim=8, heads=2, token_grid=(2, 2, 2)).eval()
        image, table, time = torch.randn(1, 8, 4, 4, 4), torch.randn(1, 3, 8), torch.randn(1, 8)
        observed_queries, updates = [], []
        hook1 = block.image_reads_table.register_forward_pre_hook(lambda module, args: observed_queries.append(args[0].detach().clone()))
        hook2 = block.table_reads_image.register_forward_hook(lambda module, args, output: updates.append(output[0].detach().clone()))
        with torch.no_grad():
            _, updated = block(image, table, torch.zeros(1, 3, dtype=torch.bool), time)
        hook1.remove()
        hook2.remove()
        self.assertTrue(torch.allclose(observed_queries[0], observed_queries[1]))
        self.assertTrue(torch.allclose(updated, table + block.time_to_table(time)[:, None, :] + updates[0]))

    def test_patch_ae_grid_and_single_l1_equivalence(self):
        autoencoder = SharedPatchAE(
            image_shape=(32, 32, 32), patch_shape=(8, 8, 8), base_channels=8
        )
        pet = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            latent = autoencoder.encode_volume(pet)
            reconstructed = autoencoder.decode_volume(latent)
        self.assertEqual(tuple(latent.shape), (1, 4, 8, 8, 8))
        self.assertEqual(tuple(reconstructed.shape), tuple(pet.shape))
        patch_error = (reconstructed - pet).abs().reshape(1, 1, 4, 8, 4, 8, 4, 8).mean()
        self.assertAlmostEqual(
            float(patch_error), float((reconstructed - pet).abs().mean()), places=6
        )

    def test_bridge_endpoints_and_stochastic_reverse(self):
        pet = torch.zeros(2, 4, 8, 8, 8)
        mri = torch.ones_like(pet)
        zeros = torch.zeros_like(pet)
        initial = bridge_state(pet, mri, torch.tensor([0, 0]), 10, 0.25, zeros)
        final = bridge_state(pet, mri, torch.tensor([10, 10]), 10, 0.25, zeros)
        self.assertTrue(torch.equal(initial, pet))
        self.assertTrue(torch.equal(final, mri))
        middle = reverse_step(mri, torch.ones_like(mri), 10, 10, 0.25, 1.0, torch.ones_like(mri))
        deterministic = reverse_step(mri, torch.ones_like(mri), 10, 10, 0.25, 0.0, zeros)
        self.assertFalse(torch.allclose(middle, deterministic))
        self.assertTrue(torch.equal(reverse_step(mri, mri, 1, 10, 0.25), pet))

    def test_seven_clinical_and_zero_conv_locations(self):
        model = PaperBCGGLDB(
            [0] * 25,
            channels=(8, 16, 32, 64),
            mri_channels=(8, 16, 32, 64),
            time_dim=32,
            token_dim=32,
            global_token_dim=32,
            bottleneck_shape=(1, 1, 1),
        )
        self.assertEqual(tuple(model.bica.keys()), model.STAGES)
        self.assertEqual(tuple(model.zero_conv.keys()), model.STAGES)
        for layer in model.zero_conv.values():
            self.assertEqual(torch.count_nonzero(layer.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(layer.bias).item(), 0)
        with torch.no_grad():
            output = model(
                torch.randn(1, 4, 8, 8, 8),
                torch.tensor([0.5]),
                torch.randn(1, 1, 32, 32, 32),
                torch.randn(1, 25),
                torch.zeros(1, 25),
            )
        self.assertEqual(tuple(output.shape), (1, 4, 8, 8, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_three_term_loss_and_98_region_pairs(self):
        prediction = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
        target = torch.randn_like(prediction)
        labels = (torch.arange(512).reshape(8, 8, 8) % 98 + 1).long()
        latent = torch.zeros(2, 4, 8, 8, 8)
        loss = generation_loss(
            latent,
            torch.ones_like(latent),
            latent,
            torch.tensor([1, 10]),
            10,
            0.25,
            prediction,
            target,
            torch.ones_like(prediction),
            labels,
        )
        self.assertTrue(
            torch.allclose(loss["total"], loss["bridge"] + loss["image"] + loss["roi"])
        )
        loss["total"].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_region_loss_constant_image_has_finite_gradient(self):
        labels = (torch.arange(512).reshape(8, 8, 8) % 98 + 1).long()
        prediction = torch.zeros(1, 1, 8, 8, 8, requires_grad=True)
        loss = sum(parc98_losses(prediction, torch.ones_like(prediction), labels).values())
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_frozen_decoder_passes_image_loss_gradient(self):
        ae = SharedPatchAE(image_shape=(32, 32, 32), patch_shape=(8, 8, 8), base_channels=8)
        ae.freeze_for_bridge()
        latent = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        ae.decode_volume(latent).abs().mean().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertTrue(all(p.grad is None for p in ae.parameters()))

    def test_clinical_statistics_use_only_training_fold(self):
        train = pd.DataFrame(
            {
                "subject_id": ["a", "b"],
                "sample_id": ["a1", "b1"],
                "table_date_key": ["2026-01-01", "2026-01-01"],
            }
        )
        table = pd.DataFrame({"PTID": ["a", "b", "held_out"], "DATE_KEY": ["2026-01-01"] * 3})
        for name in TABLE_COLS:
            table[name] = [1.0, 3.0, 10000.0]
        stats = fit_clinical_stats(train, table)
        self.assertEqual(stats["fit_sample_ids"], ["a1", "b1"])
        self.assertTrue(np.allclose(stats["mean"], 2.0))

    def test_classification_metrics_and_five_model_summary(self):
        labels = [0, 1, 2, 0, 1, 2]
        probability = np.eye(3)[labels]
        result = classify_metrics(labels, probability)
        self.assertEqual(result["macro_f1"], 1.0)
        summary = five_model_summary([result] * 5)
        self.assertEqual(summary["macro_auc"]["mean"], 1.0)
        self.assertEqual(summary["macro_auc"]["std"], 0.0)

    def test_classification_attention_output_normalization_and_caps(self):
        model = CappedBiCA(8, token_dim=8, heads=2)
        image = torch.randn(2, 8, 2, 2, 2)
        tokens = torch.randn(2, 3, 8)
        observed = torch.ones(2, 3, dtype=torch.bool)
        seen = []
        hooks = [layer.register_forward_hook(lambda *args: seen.append(True)) for layer in
                 (model.image_output_norm, model.table_output_norm)]
        image_new, table_new = model(image, tokens, ~observed, observed, torch.ones(2, 1))
        for hook in hooks:
            hook.remove()
        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.all((image_new-image).flatten(1).norm(dim=1) <= 0.02001*image.flatten(1).norm(dim=1)))
        self.assertTrue(torch.all((table_new-tokens).flatten(1).norm(dim=1) <= 0.10001*tokens.flatten(1).norm(dim=1)))

    def test_static_clinical_row_survives_date_window(self):
        dataset = object.__new__(PairedT1FDGTableDataset)
        dataset.max_table_delta_days = 90
        dataset.table_cols = ["feature"]
        dataset.table_by_ptid = {"subject": [pd.Series({"DATE_KEY": "__static__", "feature": 7.0})]}
        dataset.table_index = {}
        dataset.tabular_stats = {
            "medians": np.array([0.0], dtype=np.float32),
            "means": np.array([0.0], dtype=np.float32),
            "stds": np.array([1.0], dtype=np.float32),
        }
        dataset.standardize_table = True
        values, missing = dataset._get_table_tensors(
            pd.Series({"subject_id": "subject", "t1_date": "2026-01-01"})
        )
        self.assertEqual(float(values[0]), 7.0)
        self.assertEqual(float(missing[0]), 0.0)

    def test_all_missing_clinical_fields_are_finite(self):
        model = PaperBCGGLDB(
            [0] * 25,
            channels=(8, 16, 32, 64),
            mri_channels=(8, 16, 32, 64),
            time_dim=32,
            token_dim=32,
            global_token_dim=32,
            bottleneck_shape=(1, 1, 1),
        ).eval()
        inputs = (
            torch.randn(1, 4, 8, 8, 8),
            torch.tensor([0.5]),
            torch.randn(1, 1, 32, 32, 32),
            torch.randn(1, 25),
            torch.ones(1, 25),
        )
        with torch.no_grad():
            output = model(*inputs)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
