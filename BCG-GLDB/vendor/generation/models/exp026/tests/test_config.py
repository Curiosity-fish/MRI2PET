from __future__ import annotations

import copy
import unittest
from pathlib import Path

from exp026.utils.config import STAGE_NAMES, load_config, validate_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "t1mri_fdgpet_exp026.yaml"
SMOKE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "t1mri_fdgpet_exp026_smoke.yaml"
)


class ConfigTest(unittest.TestCase):
    def test_all_seven_stage_epochs_are_independent(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(tuple(config["stages"]), STAGE_NAMES)
        self.assertEqual(
            STAGE_NAMES,
            (
                "patch_vae",
                "mri_patch_prior",
                "local_control",
                "table_mri_prior",
                "global_mamba",
                "joint_finetune",
                "decoder_calibration",
            ),
        )
        for index, name in enumerate(STAGE_NAMES):
            changed = copy.deepcopy(config)
            changed["stages"][name]["epochs"] = index
            validate_config(changed)
            self.assertEqual(changed["stages"][name]["epochs"], index)

    def test_smoke_config_runs_one_epoch_for_all_seven_stages(self) -> None:
        config = load_config(SMOKE_CONFIG_PATH)
        self.assertEqual(tuple(config["stages"]), STAGE_NAMES)
        for name in STAGE_NAMES:
            self.assertTrue(config["stages"][name]["enabled"])
            self.assertEqual(config["stages"][name]["epochs"], 1)

    def test_review_fix_losses_are_enabled(self) -> None:
        config = load_config(CONFIG_PATH)
        patch_loss = config["stages"]["patch_vae"]["loss"]
        local_loss = config["stages"]["local_control"]["loss"]
        legacy_loss = "full" + "_reconstruction"
        self.assertNotIn(legacy_loss, patch_loss)
        self.assertGreater(float(patch_loss["gradient"]), 0.0)
        self.assertGreater(float(local_loss["mri_prior"]), 0.0)

    def test_default_geometry_has_64_patches(self) -> None:
        config = load_config(CONFIG_PATH)
        image = config["data"]["image_size"]
        patch = config["data"]["patch_size"]
        count = (image[0] // patch[0]) * (image[1] // patch[1]) * (image[2] // patch[2])
        self.assertEqual(count, 64)

    def test_negative_epoch_is_rejected(self) -> None:
        config = load_config(CONFIG_PATH)
        config["stages"]["global_mamba"]["epochs"] = -1
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
