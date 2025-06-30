"""
Affine registration implementation using PyTorch.

This module provides differentiable affine registration with various
optimization strategies and multi-scale approaches.
"""

from typing import Any

import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseRegistration
from .metrics import MSE, NCC, RegistrationLoss
from .transforms import apply_transform, create_grid


class AffineTransform(nn.Module):
    """
    Learnable affine transformation matrix.

    Parameterizes 2D/3D affine transformations using a transformation matrix
    that can be optimized via gradient descent.
    """

    def __init__(self, ndim: int = 3, init_identity: bool = True):
        """
        Args:
            ndim: Number of spatial dimensions (2 or 3)
            init_identity: Whether to initialize as identity transform
        """
        super().__init__()
        self.ndim = ndim

        if ndim == 2:
            # 2D affine: [2x3] matrix
            if init_identity:
                matrix = torch.eye(2, 3)
            else:
                matrix = torch.randn(2, 3) * 0.1
        elif ndim == 3:
            # 3D affine: [3x4] matrix
            if init_identity:
                matrix = torch.eye(3, 4)
            else:
                matrix = torch.randn(3, 4) * 0.1
        else:
            raise ValueError(f"Unsupported ndim: {ndim}")

        self.matrix = nn.Parameter(matrix)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transformation to coordinate grid.

        Args:
            grid: Coordinate grid [..., ndim]

        Returns:
            Transformed coordinates
        """
        # Add homogeneous coordinate
        if self.ndim == 2:
            ones = torch.ones(*grid.shape[:-1], 1, device=grid.device)
            grid_homo = torch.cat([grid, ones], dim=-1)  # [..., 3]
        else:  # ndim == 3
            ones = torch.ones(*grid.shape[:-1], 1, device=grid.device)
            grid_homo = torch.cat([grid, ones], dim=-1)  # [..., 4]

        # Apply transformation: grid_homo @ matrix.T
        transformed = torch.matmul(grid_homo, self.matrix.T)

        return transformed

    def get_matrix(self) -> torch.Tensor:
        """Get the current transformation matrix."""
        return self.matrix.clone()

    def set_matrix(self, matrix: torch.Tensor) -> None:
        """Set the transformation matrix."""
        self.matrix.data = matrix.clone()


class AffineRegistration(BaseRegistration):
    """
    Multi-scale affine registration using gradient-based optimization.

    Supports both 2D and 3D registration with various similarity metrics
    and optimization strategies.
    """

    def __init__(
        self,
        similarity_metric: RegistrationLoss,
        num_scales: int = 3,
        num_iterations: list[int] | None = None,
        learning_rate: float = 0.01,
        regularization_weight: float = 0.0,
        device: torch.device | None = None,
    ):
        """
        Args:
            similarity_metric: RegistrationLoss instance for computing similarity
            num_scales: Number of pyramid scales
            num_iterations: Iterations per scale (default: [100, 200, 300])
            learning_rate: Optimizer learning rate
            regularization_weight: Weight for regularization term
            device: PyTorch device
        """
        super().__init__(
            similarity_metric=similarity_metric,
            num_scales=num_scales,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            regularization_weight=regularization_weight,
            device=device,
        )

    def _regularization_loss(self, transform: AffineTransform) -> torch.Tensor:
        """Compute regularization loss to prevent large deformations."""
        matrix = transform.get_matrix()

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
        num_iterations: int,
    ) -> AffineTransform:
        """Register at a single scale."""
        transform.train()
        optimizer = optim.Adam(transform.parameters(), lr=self.learning_rate)

        # Create coordinate grid
        grid = create_grid(fixed.shape[2:], device=self.device)

        best_loss = float("inf")
        best_transform = None

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Apply transformation (create fresh grid each time to avoid autograd issues)
            transformed_grid = transform(grid.detach())

            # Sample moving image at transformed coordinates
            warped_moving = apply_transform(moving, transformed_grid)

            # Compute similarity loss
            sim_loss = self.loss_fn(fixed, warped_moving)

            # Add regularization
            reg_loss = self._regularization_loss(transform)
            total_loss = sim_loss + self.regularization_weight * reg_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Track best transformation
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_transform = AffineTransform(transform.ndim, init_identity=False)
                best_transform.set_matrix(transform.get_matrix().detach().clone())

            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss.item():.6f}")

        return best_transform or transform

    def register(
        self,
        fixed_image: sitk.Image | torch.Tensor,
        moving_image: sitk.Image | torch.Tensor,
        initial_transform: torch.Tensor | None = None,
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

        # Create image pyramids
        fixed_pyramid = self._create_pyramid(fixed, self.num_scales)
        moving_pyramid = self._create_pyramid(moving, self.num_scales)

        # Initialize transformation
        transform = AffineTransform(ndim=ndim, init_identity=True).to(self.device)

        if initial_transform is not None:
            transform.set_matrix(initial_transform)

        # Multi-scale registration
        for scale in range(self.num_scales):
            print(f"Scale {scale + 1}/{self.num_scales}")

            # Access pyramid from coarse to fine
            fixed_scale = fixed_pyramid[scale]
            moving_scale = moving_pyramid[scale]

            # Adjust number of iterations for this scale
            scale_iterations = self.num_iterations[
                min(scale, len(self.num_iterations) - 1)
            ]

            # Register at this scale
            transform = self._register_single_scale(
                fixed_scale, moving_scale, transform, scale_iterations
            )

            # If not the finest scale, upsample transformation
            if scale < self.num_scales - 1:
                matrix = transform.get_matrix()
                # Scale translation components by 2
                if ndim == 2:
                    matrix[:, 2] *= 2
                else:
                    matrix[:, 3] *= 2
                transform.set_matrix(matrix)

        # Apply final transformation to original moving image
        grid = create_grid(fixed.shape[2:], device=self.device)
        transformed_grid = transform(grid)
        registered = apply_transform(moving, transformed_grid)

        return transform.get_matrix(), registered.squeeze()

    def evaluate(
        self,
        fixed_image: sitk.Image | torch.Tensor,
        moving_image: sitk.Image | torch.Tensor,
        transform_matrix: torch.Tensor,
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
        transform = AffineTransform(ndim=len(fixed.shape) - 2, init_identity=False)
        transform.set_matrix(transform_matrix)

        grid = create_grid(fixed.shape[2:], device=self.device)
        transformed_grid = transform(grid)
        registered = apply_transform(moving, transformed_grid)

        # Compute metrics
        with torch.no_grad():
            ncc_metric = NCC()
            mse_metric = MSE()
            ncc_loss = ncc_metric(fixed, registered)
            mse_loss = mse_metric(fixed, registered)

        metrics = {
            "ncc": -ncc_loss.item(),  # Convert back to positive
            "mse": mse_loss.item(),
            "transformation_matrix": transform_matrix.cpu().detach().numpy(),
        }

        return metrics
