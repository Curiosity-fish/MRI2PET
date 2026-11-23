"""Backward-compatible aliases for the canonical soft Parc98 losses."""

from exp027.losses.soft_region import (
    regional_metabolic_graph_loss_soft as regional_metabolic_graph_loss,
    soft_patch_to_region_alignment_loss as patch_to_region_alignment_loss,
)

__all__ = ["patch_to_region_alignment_loss", "regional_metabolic_graph_loss"]
