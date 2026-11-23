"""BCG-GLDB public network name.

The implementation is exposed under the thesis method name.  The legacy
implementation remains internal only to preserve checkpoint compatibility.
"""

from exp.exp037E.exp037e_global_bsfe_unet3d import Exp037EGlobalBSFEUNet3D


class BCGGLDBNetwork(Exp037EGlobalBSFEUNet3D):
    """Brain-Structural and Clinical-Condition Guided Global–Local LDB."""

