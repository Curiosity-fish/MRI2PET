import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_code.generation_model import PaperBCGGLDB, SharedPatchAE
from paper_code.generation_objective import (
    bridge_state, reverse_step, generation_loss,
)
from paper_code.classification_data import TABLE_COLS, fit_clinical_stats
from paper_code.classification_metrics import classify_metrics, five_model_summary
from paper_code.classification_model import build_classifier, initialize_medicalnet50
from datasets.paired_t1_fdg_table_dataset import PairedT1FDGTableDataset


class ProtocolTests(unittest.TestCase):
    def test_patch_ae_grid_and_single_l1_equivalence(self):
        autoencoder = SharedPatchAE(image_shape=(32, 32, 32), patch_shape=(8, 8, 8),
                                     base_channels=8)
        pet = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            latent = autoencoder.encode_volume(pet)
            reconstructed = autoencoder.decode_volume(latent)
        self.assertEqual(tuple(latent.shape), (1, 4, 8, 8, 8))
        self.assertEqual(tuple(reconstructed.shape), tuple(pet.shape))
        patch_error = (reconstructed - pet).abs().reshape(1, 1, 4, 8, 4, 8, 4, 8).mean()
        self.assertAlmostEqual(float(patch_error), float((reconstructed - pet).abs().mean()), places=6)

    def test_bridge_endpoints_and_stochastic_reverse(self):
        pet = torch.zeros(2, 4, 8, 8, 8)
        mri = torch.ones_like(pet)
        zeros = torch.zeros_like(pet)
        initial = bridge_state(pet, mri, torch.tensor([0, 0]), 10, .25, zeros)
        final = bridge_state(pet, mri, torch.tensor([10, 10]), 10, .25, zeros)
        self.assertTrue(torch.equal(initial, pet))
        self.assertTrue(torch.equal(final, mri))
        middle = reverse_step(mri, torch.ones_like(mri), 10, 10, .25, 1.0, torch.ones_like(mri))
        deterministic = reverse_step(mri, torch.ones_like(mri), 10, 10, .25, 0.0, zeros)
        self.assertFalse(torch.allclose(middle, deterministic))
        self.assertTrue(torch.equal(reverse_step(mri, mri, 1, 10, .25), pet))

    def test_seven_clinical_and_zero_conv_locations(self):
        model = PaperBCGGLDB([0] * 25, channels=(8, 16, 32, 64),
                              mri_channels=(8, 16, 32, 64), time_dim=32,
                              token_dim=32, global_token_dim=32,
                              bottleneck_shape=(1, 1, 1))
        self.assertEqual(tuple(model.bica.keys()), model.STAGES)
        self.assertEqual(tuple(model.zero_conv.keys()), model.STAGES)
        for layer in model.zero_conv.values():
            self.assertEqual(torch.count_nonzero(layer.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(layer.bias).item(), 0)
        with torch.no_grad():
            output = model(torch.randn(1, 4, 8, 8, 8), torch.tensor([.5]),
                           torch.randn(1, 1, 32, 32, 32), torch.randn(1, 25),
                           torch.zeros(1, 25))
        self.assertEqual(tuple(output.shape), (1, 4, 8, 8, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_three_term_loss_and_98_region_pairs(self):
        prediction = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
        target = torch.randn_like(prediction)
        labels = (torch.arange(512).reshape(8, 8, 8) % 98 + 1).long()
        latent = torch.zeros(2, 4, 8, 8, 8)
        loss = generation_loss(latent, torch.ones_like(latent), latent,
                               torch.tensor([1, 10]), 10, .25,
                               prediction, target, torch.ones_like(prediction), labels)
        self.assertTrue(torch.allclose(loss["total"], .5 * loss["bridge"] + loss["image"] + loss["roi"]))
        loss["total"].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_clinical_statistics_use_only_training_fold(self):
        train = pd.DataFrame({"subject_id": ["a", "b"], "sample_id": ["a1", "b1"],
                              "table_date_key": ["2026-01-01", "2026-01-01"]})
        table = pd.DataFrame({"PTID": ["a", "b", "held_out"],
                              "DATE_KEY": ["2026-01-01"] * 3})
        for name in TABLE_COLS:
            table[name] = [1., 3., 10000.]
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

    def test_medicalnet_baseline_alias(self):
        baseline = build_classifier("baseline", 2)
        comparison = build_classifier("medicalnet50_3d", 2)
        self.assertEqual(type(baseline), type(comparison))
        self.assertEqual([tuple(p.shape) for p in baseline.parameters()],
                         [tuple(p.shape) for p in comparison.parameters()])

    def test_med3d_checkpoint_adapts_one_input_channel_to_two(self):
        model = build_classifier("medicalnet50_3d", 2)
        target = model.model.backbone.state_dict()
        source = {name: target[name].clone() for name in (
            "layer1.0.conv1.weight", "layer4.2.conv3.weight")}
        source["conv1.weight"] = target["conv1.weight"][:, :1].clone()
        with patch("paper_code.classification_model.torch.load", return_value={"state_dict": source}):
            loaded = initialize_medicalnet50(model, "example.pth")
        self.assertEqual(loaded, 3)
        self.assertEqual(model.model.backbone.conv1.weight.shape[1], 2)

    def test_static_clinical_row_survives_date_window(self):
        dataset = object.__new__(PairedT1FDGTableDataset)
        dataset.max_table_delta_days = 90
        dataset.table_cols = ["feature"]
        dataset.table_by_ptid = {"subject": [pd.Series({"DATE_KEY": "__static__", "feature": 7.0})]}
        dataset.table_index = {}
        dataset.tabular_stats = {"medians": np.array([0.], dtype=np.float32),
                                 "means": np.array([0.], dtype=np.float32),
                                 "stds": np.array([1.], dtype=np.float32)}
        dataset.standardize_table = True
        values, missing = dataset._get_table_tensors(pd.Series(
            {"subject_id": "subject", "t1_date": "2026-01-01"}))
        self.assertEqual(float(values[0]), 7.0)
        self.assertEqual(float(missing[0]), 0.0)

    def test_all_missing_clinical_fields_do_not_inject_image_features(self):
        model = PaperBCGGLDB([0] * 25, channels=(8, 16, 32, 64),
                              mri_channels=(8, 16, 32, 64), time_dim=32,
                              token_dim=32, global_token_dim=32,
                              bottleneck_shape=(1, 1, 1)).eval()
        inputs = (torch.randn(1, 4, 8, 8, 8), torch.tensor([.5]),
                  torch.randn(1, 1, 32, 32, 32), torch.randn(1, 25),
                  torch.ones(1, 25))
        with torch.no_grad():
            with_clinical = model(*inputs)
            model.use_clinical = False
            without_clinical = model(*inputs)
        self.assertTrue(torch.allclose(with_clinical, without_clinical, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
