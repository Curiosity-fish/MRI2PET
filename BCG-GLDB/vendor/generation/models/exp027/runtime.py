"""Compatibility entry point for Exp027.

All Exp027 objectives are implemented directly in the trainer.  This function is
kept so older launch commands remain valid, but it intentionally performs no
monkey patching.
"""

from __future__ import annotations


def install_runtime_patches() -> None:
    return None
