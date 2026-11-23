"""Runtime compatibility entry point for Exp030.

Exp027's trainer resolves `configure_trainable_modules` / `set_stage_mode` from
its own module globals at call time.  Exp030 swaps those bindings for the Exp030
freeze policy (which additionally unlocks `performance_refiner` during
joint_finetune) so the inherited 13-stage trainer loop trains the new module
without copying the whole run loop.
"""

from __future__ import annotations

import exp027.runtime
import exp027.training.trainer as exp027_trainer_module
from exp030.training import stages as exp030_stages


def install_runtime_patches() -> None:
    exp027.runtime.install_runtime_patches()
    exp027_trainer_module.configure_trainable_modules = exp030_stages.configure_trainable_modules
    exp027_trainer_module.set_stage_mode = exp030_stages.set_stage_mode


__all__ = ["install_runtime_patches"]
