"""
CURE: Concept Unlearning via Orthogonal Representation Editing

A lightweight implementation for Stable Diffusion v1.4 concept removal.
"""

from .cure import CURE
from .spectral import compute_svd, spectral_expansion, build_projector, SPECTRAL_MODES
from .attention import get_cross_attention_layers, get_projection_matrices, apply_weight_update

__version__ = "0.1.0"
__all__ = [
    "CURE",
    "compute_svd",
    "spectral_expansion",
    "build_projector",
    "SPECTRAL_MODES",
    "get_cross_attention_layers",
    "get_projection_matrices",
    "apply_weight_update",
]
