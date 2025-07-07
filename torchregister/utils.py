"""
Utility functions for image I/O and transformations.

This module aggregates functions from submodules io, processing, and transforms.
"""

from .io import (
    load_image,
    save_image,
    sitk_to_torch,
    torch_to_sitk,
)
from .processing import (
    gaussian_blur,
    normalize_image,
    resample_image,
)
from .transforms import (
    invert_affine_transform,
    to_homogeneous,
    torch_affine_to_sitk_transform,
)

__all__ = [
    # From io.py
    "load_image",
    "save_image",
    "sitk_to_torch",
    "torch_to_sitk",
    # From processing.py
    "gaussian_blur",
    "normalize_image",
    "resample_image",
    # From transforms.py
    "torch_affine_to_sitk_transform",
    "invert_affine_transform",
    "to_homogeneous",
]
