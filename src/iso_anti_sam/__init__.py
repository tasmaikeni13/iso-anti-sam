"""
IsoAntiSAM: Isochoric and Coherent Anti-Sharpness-Aware Minimization
A mathematically verified optimizer solving the sharp-needle generalization catastrophe.
"""

from .anti_sam import AntiSAM
from .iso_anti_sam import IsoAntiSAM

__all__ = ["AntiSAM", "IsoAntiSAM"]
__version__ = "0.1.0"
