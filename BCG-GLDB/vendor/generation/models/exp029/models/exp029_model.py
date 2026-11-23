"""Exp029 keeps the Exp027-compatible neural backbone."""

from exp027.models.exp027_model import Exp027MRI2PET, build_exp027_model


Exp029MRI2PET = Exp027MRI2PET


def build_exp029_model(*args, **kwargs):
    return build_exp027_model(*args, **kwargs)


__all__ = ["Exp029MRI2PET", "build_exp029_model"]
