"""
Carve: Coherent Morphological Loss Erosion Optimizer.
Reversing Sharpness-Aware Minimization for fast training and validation loss collapse.
"""

from .anti_sam import AntiSAM
from .carve import Carve

# Backward compatibility alias
IsoAntiSAM = Carve

__all__ = ["Carve", "AntiSAM", "IsoAntiSAM"]
__version__ = "0.1.0"
