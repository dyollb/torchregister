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
from .metrics import LNCC, MSE, NCC, BaseLoss


class BaseRegistration:
    """
    Base class for multi-scale image registration.

    Provides common functionality for both affine and deformable registration
    methods, including image pyramid creation, tensor handling, and metric setup.
    """

    def __init__(
        self,
        similarity_metric: BaseLoss | str = "ncc",
        interp_mode: str = "bilinear",
        num_scales: int = 3,
        num_iterations: list[int] | None = None,
        learning_rate: float = 0.01,
        regularization_weight: float = 0.0,
        device: torch.device | None = None,
    ):
        """
        Initialize base registration parameters.

        Args:
            similarity_metric: Either a BaseLoss instance or a string identifier ("ncc", "mse", "lncc")
            interp_mode: Mode used in grid_sample (e.g., "bilinear", "nearest")
            num_scales: Number of pyramid scales for multi-scale registration
            num_iterations: List of iterations per scale (finest to coarsest)
            learning_rate: Optimizer learning rate
            regularization_weight: Weight for regularization term
            device: PyTorch device
        """
        self.interp_mode = interp_mode
        self.num_scales = num_scales
        self.num_iterations = num_iterations or [100, 200, 300]
        self.learning_rate = learning_rate
        self.regularization_weight = regularization_weight
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize similarity metric
        if isinstance(similarity_metric, BaseLoss):
            self.loss_fn = similarity_metric
        elif isinstance(similarity_metric, str):
            if similarity_metric.lower() == "ncc":
                self.loss_fn = NCC()
            elif similarity_metric.lower() == "mse":
                self.loss_fn = MSE()
            elif similarity_metric.lower() == "lncc":
                self.loss_fn = LNCC()
            else:
                raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
        else:
            raise TypeError("similarity_metric must be a BaseLoss instance or a string")

    def _create_pyramid(
        self, image: torch.Tensor, num_scales: int
    ) -> list[torch.Tensor]:
        """
        Create image pyramid for multi-scale registration.

        Args:
            image: Input image tensor
            num_scales: Number of scales in the pyramid

        Returns:
            List of image tensors from coarse to fine resolution
        """
        pyramid = [image]

        current = image
        for _i in range(num_scales - 1):
            # Downsample by factor of 2
            if len(current.shape) == 4:  # 2D: [B, C, H, W]
                current = F.avg_pool2d(current, kernel_size=2, stride=2)
            else:  # 3D: [B, C, D, H, W]
                current = F.avg_pool3d(current, kernel_size=2, stride=2)
            pyramid.append(current)

        # Return pyramid from coarse to fine
        return pyramid[::-1]

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
