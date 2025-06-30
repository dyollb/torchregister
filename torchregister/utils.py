"""
Utility functions for image I/O and transformations.

This module provides functions for converting between SimpleITK and PyTorch,
creating coordinate grids, and applying transformations.
"""

# Import from io.py
# Import from conversion.py
from .conversion import (
    sitk_displacement_to_torch_deformation,
    sitk_transform_to_torch_affine,
    torch_affine_to_sitk_transform,
    torch_deformation_to_sitk_field,
    torch_deformation_to_sitk_transform,
)
from .io import (
    load_image,
    normalize_image,
    resample_image,
    save_image,
    sitk_to_torch,
    torch_to_sitk,
)

# Import from transforms.py
from .transforms import (
    apply_deformation,
    apply_transform,
    compose_transforms,
    compute_gradient,
    compute_target_registration_error,
    create_grid,
    create_identity_transform,
)

__all__ = [
    # From io.py
    "load_image",
    "normalize_image",
    "resample_image",
    "save_image",
    "sitk_to_torch",
    "torch_to_sitk",
    # From transforms.py
    "apply_deformation",
    "apply_transform",
    "compose_transforms",
    "compute_gradient",
    "compute_target_registration_error",
    "create_grid",
    "create_identity_transform",
    # From conversion.py
    "sitk_displacement_to_torch_deformation",
    "sitk_transform_to_torch_affine",
    "torch_affine_to_sitk_transform",
    "torch_deformation_to_sitk_field",
    "torch_deformation_to_sitk_transform",
]
