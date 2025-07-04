"""
Base registration class for both affine and deformable registration.

This module provides a base class with common functionality for multi-scale
registration methods.
"""

from abc import abstractmethod
from typing import Any

import SimpleITK as sitk
import torch
import torch.nn.functional as F

from .io import sitk_to_torch
from .metrics import RegistrationLoss
from .processing import gaussian_blur


class BaseRegistration:
    """
    Base class for multi-scale image registration.

    Provides common functionality for both affine and deformable registration
    methods, including image pyramid creation, tensor handling, and metric setup.
    """

    def __init__(
        self,
        similarity_metric: RegistrationLoss,
        interp_mode: str = "bilinear",
        shrink_factors: list[int] | None = None,
        smoothing_sigmas: list[float] | None = None,
        num_iterations: list[int] | None = None,
        learning_rate: float = 0.01,
        regularization_weight: float = 0.0,
        device: torch.device | None = None,
    ):
        """
        Initialize base registration parameters.

        Args:
            similarity_metric: RegistrationLoss instance for computing similarity
            interp_mode: Mode used in grid_sample (e.g., "bilinear", "nearest")
            shrink_factors: List of downsample factors per scale (e.g., [8, 4, 2, 1])
            smoothing_sigmas: List of Gaussian smoothing sigmas in pixel units per scale
            num_iterations: List of iterations per scale (finest to coarsest)
            learning_rate: Optimizer learning rate
            regularization_weight: Weight for regularization term
            device: PyTorch device
        """
        self.interp_mode = interp_mode
        self.shrink_factors = shrink_factors or [4, 2, 1]
        self.smoothing_sigmas = smoothing_sigmas or [2.0, 1.0, 0.0]
        self.num_scales = len(self.shrink_factors)
        self.num_iterations = num_iterations or [100, 100, 100]
        self.learning_rate = learning_rate
        self.regularization_weight = regularization_weight
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize similarity metric
        if not isinstance(similarity_metric, RegistrationLoss):
            raise TypeError(
                f"similarity_metric must be an instance of RegistrationLoss, "
                f"got {type(similarity_metric).__name__}. "
                f"Use NCC(), MSE(), LNCC(), or another RegistrationLoss subclass instead."
            )
        self.loss_fn = similarity_metric

        # Validate that shrink_factors and smoothing_sigmas have the same length
        if len(self.shrink_factors) != len(self.smoothing_sigmas):
            raise ValueError(
                f"shrink_factors and smoothing_sigmas must have the same length. "
                f"Got {len(self.shrink_factors)} and {len(self.smoothing_sigmas)} respectively."
            )

    def _create_pyramid(self, image: torch.Tensor) -> list[torch.Tensor]:
        """
        Create image pyramid for multi-scale registration.

        Args:
            image: Input image tensor

        Returns:
            List of image tensors from coarse to fine resolution
        """
        pyramid = []

        for shrink_factor, sigma in zip(
            self.shrink_factors, self.smoothing_sigmas, strict=False
        ):
            current = image

            # Apply smoothing if sigma > 0
            if sigma > 0:
                kernel_size = int(6 * sigma + 1)  # 6-sigma rule for better coverage
                if kernel_size % 2 == 0:
                    kernel_size += 1
                current = gaussian_blur(current, kernel_size, sigma)

            # Downsample to the target shrink factor
            if shrink_factor > 1:
                if len(current.shape) == 4:  # 2D: [B, C, H, W]
                    current = F.avg_pool2d(
                        current, kernel_size=shrink_factor, stride=shrink_factor
                    )
                else:  # 3D: [B, C, D, H, W]
                    current = F.avg_pool3d(
                        current, kernel_size=shrink_factor, stride=shrink_factor
                    )

            pyramid.append(current)

        return pyramid

    def _prepare_input_tensors(
        self,
        fixed_image: sitk.Image | torch.Tensor,
        moving_image: sitk.Image | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Convert input images to PyTorch tensors with appropriate dimensions.

        Args:
            fixed_image: Fixed/reference image
            moving_image: Moving image to be registered

        Returns:
            Tuple of (fixed_tensor, moving_tensor, ndim)
        """
        # Convert SimpleITK images to PyTorch tensors
        if isinstance(fixed_image, sitk.Image):
            fixed = sitk_to_torch(fixed_image).to(self.device).detach()
        else:
            fixed = fixed_image.to(self.device).detach()

        if isinstance(moving_image, sitk.Image):
            moving = sitk_to_torch(moving_image).to(self.device).detach()
        else:
            moving = moving_image.to(self.device).detach()

        # Ensure 4D/5D tensors (add batch and channel dims if needed)
        if len(fixed.shape) == 2:  # 2D image
            fixed = fixed.unsqueeze(0).unsqueeze(0)
            moving = moving.unsqueeze(0).unsqueeze(0)
            ndim = 2
        elif len(fixed.shape) == 3:  # 3D image
            fixed = fixed.unsqueeze(0).unsqueeze(0)
            moving = moving.unsqueeze(0).unsqueeze(0)
            ndim = 3
        elif len(fixed.shape) == 4:  # Already has batch/channel
            ndim = 2
        else:  # 5D: [B, C, D, H, W]
            ndim = 3

        return fixed, moving, ndim

    @abstractmethod
    def _regularization_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute regularization loss.

        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def register(
        self,
        fixed_image: sitk.Image | torch.Tensor,
        moving_image: sitk.Image | torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, torch.Tensor]:
        """
        Perform registration.

        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        fixed_image: sitk.Image | torch.Tensor,
        moving_image: sitk.Image | torch.Tensor,
        transform: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Evaluate registration quality.

        To be implemented by subclasses.
        """
        pass
