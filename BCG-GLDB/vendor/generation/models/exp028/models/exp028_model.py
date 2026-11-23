"""Exp028 model construction.

The neural backbone remains checkpoint-compatible with Exp027.  Exp028's
innovation is deliberately placed in the objective and model-selection path so
that the generated PET stays close to the established Exp027 solution while
being selected for structural fidelity.
"""

from exp027.models.exp027_model import Exp027MRI2PET, build_exp027_model


Exp028MRI2PET = Exp027MRI2PET


def build_exp028_model(*args, **kwargs):
    return build_exp027_model(*args, **kwargs)


__all__ = ["Exp028MRI2PET", "build_exp028_model"]
