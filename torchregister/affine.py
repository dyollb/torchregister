"""
Affine registration implementation using PyTorch.

This module provides differentiable affine registration with various
optimization strategies and multi-scale approaches.
"""

from typing import Any

import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .base import BaseRegistration
from .metrics import MSE, NCC, RegistrationLoss


class AffineTransform(nn.Module):
    """
    Learnable affine transformation matrix.

    Parametrizes 2D/3D affine transformations using a transformation matrix
    that can be optimized via gradient descent.
    """

    def __init__(
        self,
        ndim: int = 3,
        init_translation: torch.Tensor | None = None,
        init_rotation: torch.Tensor | None = None,
        init_zoom: torch.Tensor | None = None,
        init_shear: torch.Tensor | None = None,
    ):
        """
        Args:
            ndim: Number of spatial dimensions (2 or 3)
            init_identity: Whether to initialize as identity transform
        """
        super().__init__()

        if ndim not in (2, 3):
            raise ValueError(f"Unsupported ndim: {ndim}")

        translation = (
            torch.zeros(ndim)
            if init_translation is None
            else init_translation.detach().clone()
        )
        rotation = (
            torch.eye(ndim) if init_rotation is None else init_rotation.detach().clone()
        )
        zoom = torch.ones(ndim) if init_zoom is None else init_zoom.detach().clone()
        shear = torch.zeros(ndim) if init_shear is None else init_shear.detach().clone()
        self.params = nn.ParameterList(
            [
                nn.Parameter(p, requires_grad=True)
                for p in [translation, rotation, zoom, shear]
            ]
        )

    def forward(
        self,
        image: torch.Tensor,
        sample_mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        Apply affine transformation to image

        Args:
            image: [B, C, H, W] or [B, C, D, H, W]
            sample_mode: Interpolation mode ('bilinear', 'nearest', etc.)
            padding_mode: Padding mode for out-of-bounds pixels ('zeros', 'border', etc.)
            align_corners: Whether to align corners in grid sampling

        Returns:
            Transformed image
        """
        # Create affine grid
        batch_size = image.shape[0]
        matrix = self._compose_affine().expand(*[batch_size, -1, -1])
        shape = image.shape[2:]
        grid = F.affine_grid(
            matrix, [1, len(shape), *shape], align_corners=align_corners
        )

        # Apply transformation
        return F.grid_sample(
            image,
            grid,
            mode=sample_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    def _compose_affine(self) -> torch.Tensor:
        translation, rotation, zoom, shear = self.params
        linear = torch.diag(zoom)
        if len(zoom) == 3:
            linear[0, 1:] = shear[:2]
            linear[1, 2] = shear[2]
        else:
            linear[0, 1] = shear[0]
        linear = rotation @ linear
        return torch.cat([linear, translation.unsqueeze(-1)], dim=-1)

    def get_affine(self, with_grad: bool = False) -> torch.Tensor:
        """Get the current transformation matrix."""
        affine = self._compose_affine()
        return affine if with_grad else affine.detach()

    @property
    def ndim(self) -> int:
        """Get the number of spatial dimensions."""
        return len(self.params[0])


class AffineRegistration(BaseRegistration):
    """
    Multi-scale affine registration using gradient-based optimization.

    Supports both 2D and 3D registration with various similarity metrics
    and optimization strategies.
    """

    def __init__(
        self,
        similarity_metric: RegistrationLoss,
        shrink_factors: list[int] | None = None,
        smoothing_sigmas: list[float] | None = None,
        num_iterations: list[int] | None = None,
        learning_rate: float = 0.01,
        regularization_weight: float = 0.0,
        device: torch.device | None = None,
    ):
        """
        Args:
            similarity_metric: RegistrationLoss instance for computing similarity
            shrink_factors: List of downsample factors per scale (e.g., [8, 4, 2, 1])
            smoothing_sigmas: List of Gaussian smoothing sigmas in pixel units per scale
            num_iterations: Iterations per scale
            learning_rate: Optimizer learning rate
            regularization_weight: Weight for regularization term
            device: PyTorch device
        """
        super().__init__(
            similarity_metric=similarity_metric,
            shrink_factors=shrink_factors,
            smoothing_sigmas=smoothing_sigmas,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            regularization_weight=regularization_weight,
            device=device,
        )

    def _regularization_loss(self, transform: AffineTransform) -> torch.Tensor:
        """Compute regularization loss to prevent large deformations."""
        matrix = transform.get_affine(with_grad=True)

        # L2 regularization on deviation from identity
        if transform.ndim == 2:
            identity = torch.eye(2, 3, device=matrix.device)
        else:
            identity = torch.eye(3, 4, device=matrix.device)

        return torch.norm(matrix - identity) ** 2  # type: ignore[no-any-return]

    def _register_single_scale(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        transform: AffineTransform,
        iterations: int,
    ) -> None:
        optimizer = optim.Adam(transform.parameters(), lr=self.learning_rate)
        progress_bar = tqdm(range(iterations), disable=False)
        for self.iter in progress_bar:
            optimizer.zero_grad()
            moved = transform(moving)
            loss = self.loss_fn(moved, fixed)
            progress_bar.set_description(
                f"Shape: {[*fixed.shape]}; Dissimiliarity: {loss.item()}"
            )
            loss.backward()
            optimizer.step()

    def register(
        self,
        fixed_image: sitk.Image | torch.Tensor,
        moving_image: sitk.Image | torch.Tensor,
        initial_transform: AffineTransform | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform affine registration.

        Args:
            fixed_image: Fixed/reference image
                - Single-channel: [H, W], [D, H, W], [B, C, H, W], [B, C, D, H, W]
                - Multi-modal: [B, C, H, W] or [B, C, D, H, W] where C > 1
            moving_image: Moving image to be registered (same format as fixed)
            initial_transform: Initial transformation matrix (optional)

        Returns:
            Tuple of (transformation_matrix, registered_image)
            - transformation_matrix: [2, 3] for 2D or [3, 4] for 3D affine transform
            - registered_image: Same shape as input moving_image

        Note:
            For multi-modal images:
            - Stack modalities in channel dimension: torch.stack([t1, t2], dim=1)
            - All channels share the same spatial transformation
            - Similarity metric computed per-channel then averaged
        """
        # Convert and prepare the input tensors
        fixed, moving, ndim = self._prepare_input_tensors(fixed_image, moving_image)
        interp_mode = "trilinear" if ndim == 3 else "bilinear"
        moving_ = F.interpolate(
            moving, fixed.shape[2:], mode=interp_mode, align_corners=True
        )

        # Create image pyramids
        fixed_pyramid = self._create_pyramid(fixed)
        moving_pyramid = self._create_pyramid(moving_)

        # Initialize transformation
        transform = (
            AffineTransform(ndim=ndim).to(self.device)
            if initial_transform is None
            else initial_transform.to(self.device)
        )

        # Multi-scale registration
        for scale_idx in range(self.num_scales):
            print(f"Scale {scale_idx + 1}/{self.num_scales}")

            # Access pyramid from coarse to fine
            fixed_scale = fixed_pyramid[scale_idx]
            moving_scale = moving_pyramid[scale_idx]

            # Adjust number of iterations for this scale
            scale_iterations = self.num_iterations[
                min(scale_idx, len(self.num_iterations) - 1)
            ]

            # Register at this scale
            self._register_single_scale(
                fixed_scale, moving_scale, transform, scale_iterations
            )

        # Apply final transformation to original moving image
        registered = transform(moving)

        return transform.get_affine(), registered.squeeze()

    def evaluate(
        self,
        fixed_image: sitk.Image | torch.Tensor,
        moving_image: sitk.Image | torch.Tensor,
        transform: AffineTransform,
    ) -> dict[str, Any]:
        """
        Evaluate registration quality.

        Args:
            fixed_image: Fixed/reference image
            moving_image: Moving image
            transform_matrix: Transformation matrix

        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to tensors
        fixed, moving, _ = self._prepare_input_tensors(fixed_image, moving_image)

        # Apply transformation
        registered = transform(moving)

        # Compute metrics
        with torch.no_grad():
            ncc_metric = NCC()
            mse_metric = MSE()
            ncc_loss = ncc_metric(fixed, registered)
            mse_loss = mse_metric(fixed, registered)

        metrics = {
            "ncc": -ncc_loss.item(),  # Convert back to positive
            "mse": mse_loss.item(),
            "transformation_matrix": transform.get_affine().cpu().detach().numpy(),
        }

        return metrics
