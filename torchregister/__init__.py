"""
TorchRegister: Multi-scale affine and deformable registration using PyTorch

A comprehensive package for image registration with differentiable loss functions
and GPU acceleration.

Key Features:
- Multi-modal image registration (T1, T2, FLAIR, etc.)
- Differentiable similarity metrics
- GPU acceleration
- SimpleITK integration

Quick Example:
    >>> import torchregister
    >>> import torch
    >>>
    >>> # Single modality registration
    >>> fixed = torch.rand(1, 1, 64, 64, 64)  # [B, C, D, H, W]
    >>> moving = torch.rand(1, 1, 64, 64, 64)
    >>>
    >>> reg = torchregister.RDMMRegistration()
    >>> deformation, registered = reg.register(fixed, moving)
    >>>
    >>> # Multi-modal registration (T1 + T2)
    >>> t1_fixed = torch.rand(64, 64, 64)    # [D, H, W]
    >>> t2_fixed = torch.rand(64, 64, 64)
    >>> multi_fixed = torch.stack([t1_fixed, t2_fixed], dim=0).unsqueeze(0)  # [1, 2, D, H, W]
    >>>
    >>> # Register multi-modal images
    >>> deformation, registered = reg.register(multi_fixed, multi_moving)

For detailed dimension conventions, see docs/IMAGE_DIMENSIONS.md
"""

__version__ = "0.1.0"
__author__ = "TorchRegister Contributors"

from .affine import AffineRegistration
from .metrics import LNCC, MSE, NCC, CombinedLoss, Dice, MattesMI
from .rdmm import RDMMRegistration
from .utils import (
    load_image,
    resample_image,
    save_image,
    sitk_displacement_to_torch_deformation,
    sitk_transform_to_torch_affine,
    torch_affine_to_sitk_transform,
    torch_deformation_to_sitk_field,
    torch_deformation_to_sitk_transform,
)

__all__ = [
    "AffineRegistration",
    "RDMMRegistration",
    "NCC",
    "LNCC",
    "MSE",
    "MattesMI",
    "Dice",
    "CombinedLoss",
    "load_image",
    "save_image",
    "resample_image",
    "torch_affine_to_sitk_transform",
    "torch_deformation_to_sitk_transform",
    "torch_deformation_to_sitk_field",
    "sitk_transform_to_torch_affine",
    "sitk_displacement_to_torch_deformation",
]
