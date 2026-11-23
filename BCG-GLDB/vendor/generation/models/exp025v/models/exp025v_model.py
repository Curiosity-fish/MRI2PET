from exp025u.models.exp025u_model import (
    Exp025UMRI2PET,
    build_exp025u_model,
)

Exp025VMRI2PET = Exp025UMRI2PET
build_exp025v_model = build_exp025u_model

__all__ = ["Exp025VMRI2PET", "build_exp025v_model"]
