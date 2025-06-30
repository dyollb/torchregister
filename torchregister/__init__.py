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

# Import submodules to make them available as torchregister.submodule
from . import affine, base, conversion, io, metrics, rdmm, transforms, utils

# Only expose the most essential classes/functions at the top level
from .affine import AffineRegistration
from .rdmm import RDMMRegistration

__all__ = [
    # Essential registration classes (top-level access)
    "AffineRegistration",
    "RDMMRegistration",
    # Submodules (for organized access: torchregister.metrics.NCC, etc.)
    "affine",
    "base",
    "conversion",
    "io",
    "metrics",
    "rdmm",
    "transforms",
    "utils",
]
