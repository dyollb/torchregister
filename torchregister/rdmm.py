"""
RDMM (Regularized Diffeomorphic Momentum Mapping) registration implementation.

This module provides differentiable deformable registration using the
diffeomorphic demon algorithm with momentum regularization.
"""

from collections.abc import Callable
from typing import Any

import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseRegistration
from .metrics import MSE, NCC, BaseLoss
from .transforms import apply_deformation, create_grid


class GaussianSmoothing(nn.Module):
    """
    Apply Gaussian smoothing to regularize deformation fields.
    """

    def __init__(self, channels: int, sigma: float, ndim: int = 3):
        """
        Args:
            channels: Number of channels
            sigma: Standard deviation for Gaussian kernel
            ndim: Number of spatial dimensions
        """
        super().__init__()
        self.ndim = ndim
        self.sigma = sigma

        # Create Gaussian kernel
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = self._create_gaussian_kernel(kernel_size, sigma)

        if ndim == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = kernel.repeat(channels, 1, 1, 1)
            self.conv: Callable = F.conv2d
        else:  # ndim == 3
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = kernel.repeat(channels, 1, 1, 1, 1)
            self.conv = F.conv3d

        self.register_buffer("weight", kernel)
        self.padding = kernel_size // 2

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2

        if self.ndim == 2:
            g_x, g_y = torch.meshgrid(coords, coords, indexing="ij")
            g = torch.exp(-(g_x**2 + g_y**2) / (2 * sigma**2))
        else:  # ndim == 3
            g_x, g_y, g_z = torch.meshgrid(coords, coords, coords, indexing="ij")
            g = torch.exp(-(g_x**2 + g_y**2 + g_z**2) / (2 * sigma**2))

        return g / g.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing."""
        return self.conv(x, self.weight, padding=self.padding, groups=x.shape[1])  # type: ignore[no-any-return]


class VelocityField(nn.Module):
    """
    Learnable velocity field for diffeomorphic registration.

    The velocity field is integrated to produce diffeomorphic deformations.
    """

    def __init__(self, shape: tuple[int, ...], ndim: int = 3):
        """
        Args:
            shape: Spatial shape of the velocity field
            ndim: Number of spatial dimensions
        """
        super().__init__()
        self.ndim = ndim
        self.shape = shape

        # Initialize velocity field with small random values
        if ndim == 2:
            velocity = torch.randn(1, 2, *shape) * 0.01
        else:  # ndim == 3
            velocity = torch.randn(1, 3, *shape) * 0.01

        self.velocity = nn.Parameter(velocity)

    def forward(self) -> torch.Tensor:
        """Get the current velocity field."""
        return self.velocity


class RDMMRegistration(BaseRegistration):
    """
    RDMM (Regularized Diffeomorphic Momentum Mapping) registration.

    Implements diffeomorphic deformable registration with momentum
    regularization for smooth, invertible transformations.
    """

    def __init__(
        self,
        similarity_metric: BaseLoss | str = "lncc",
        num_scales: int = 3,
        num_iterations: list[int] | None = None,
        learning_rate: float = 0.01,
        smoothing_sigma: float = 1.0,
        alpha: float = 1.0,  # Regularization weight
        num_integration_steps: int = 7,
        device: torch.device | None = None,
    ):
        """
        Args:
            similarity_metric: Similarity metric (BaseLoss instance or "ncc", "lncc", "mse")
            num_scales: Number of pyramid scales
            num_iterations: Iterations per scale
            learning_rate: Optimizer learning rate
            smoothing_sigma: Gaussian smoothing sigma for regularization
            alpha: Regularization weight
            num_integration_steps: Number of steps for velocity integration
            device: PyTorch device
        """
        super().__init__(
            similarity_metric=similarity_metric,
            num_scales=num_scales,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            regularization_weight=alpha,
            device=device,
        )

        self.smoothing_sigma = smoothing_sigma
        self.alpha = alpha
        self.num_integration_steps = num_integration_steps

    def _integrate_velocity(
        self, velocity: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """
        Integrate velocity field using scaling and squaring.

        Args:
            velocity: Velocity field [B, ndim, ...]
            num_steps: Number of integration steps

        Returns:
            Deformation field
        """
        # Scale velocity by number of steps
        v = velocity / (2.0**num_steps)

        # Create identity grid
        grid = create_grid(velocity.shape[2:], device=velocity.device)
        # Add batch dimension to grid
        grid = grid.unsqueeze(0).expand(v.shape[0], *grid.shape)

        # Initial deformation is the scaled velocity
        if len(velocity.shape) == 4:  # 2D
            deformation = v.permute(0, 2, 3, 1)  # [B, H, W, 2]
        else:  # 3D
            deformation = v.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

        # Scaling and squaring integration
        for _ in range(num_steps):
            # Current positions
            positions = grid + deformation

            # Sample deformation at current positions
            if len(velocity.shape) == 4:  # 2D
                sampled_def = F.grid_sample(
                    v,
                    positions,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                ).permute(0, 2, 3, 1)
            else:  # 3D
                sampled_def = F.grid_sample(
                    v,
                    positions,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                ).permute(0, 2, 3, 4, 1)

            # Update deformation: φ = φ + φ∘v
            deformation = deformation + sampled_def

        return deformation

    def _compute_jacobian_determinant(self, deformation: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian determinant for deformation field.

        Args:
            deformation: Deformation field [..., ndim]

        Returns:
            Jacobian determinant
        """
        if len(deformation.shape) == 4:  # 2D: [B, H, W, 2]
            # Compute gradients
            dy_dx = torch.gradient(deformation[..., 0], dim=1)[0]
            dy_dy = torch.gradient(deformation[..., 0], dim=2)[0]
            dx_dx = torch.gradient(deformation[..., 1], dim=1)[0]
            dx_dy = torch.gradient(deformation[..., 1], dim=2)[0]

            # Add identity
            dy_dx += 1.0
            dx_dy += 1.0

            # Compute determinant
            jac_det = dy_dx * dx_dy - dy_dy * dx_dx

        else:  # 3D: [B, D, H, W, 3]
            # Compute gradients for 3D
            gradients = []
            for i in range(3):
                grad_x = torch.gradient(deformation[..., i], dim=1)[0]
                grad_y = torch.gradient(deformation[..., i], dim=2)[0]
                grad_z = torch.gradient(deformation[..., i], dim=3)[0]
                gradients.append([grad_x, grad_y, grad_z])

            # Add identity to diagonal
            gradients[0][0] += 1.0
            gradients[1][1] += 1.0
            gradients[2][2] += 1.0

            # Compute 3x3 determinant
            jac_det = (
                gradients[0][0]
                * (
                    gradients[1][1] * gradients[2][2]
                    - gradients[1][2] * gradients[2][1]
                )
                - gradients[0][1]
                * (
                    gradients[1][0] * gradients[2][2]
                    - gradients[1][2] * gradients[2][0]
                )
                + gradients[0][2]
                * (
                    gradients[1][0] * gradients[2][1]
                    - gradients[1][1] * gradients[2][0]
                )
            )

        return jac_det

    def _regularization_loss(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss for velocity field.

        Args:
            velocity: Velocity field tensor

        Returns:
            Regularization loss value
        """
        # L2 regularization on velocity
        l2_reg = torch.norm(velocity) ** 2

        # Integrate velocity to get deformation
        deformation = self._integrate_velocity(velocity, self.num_integration_steps)

        # Jacobian determinant penalty (prevent folding)
        jac_det = self._compute_jacobian_determinant(deformation)
        jac_penalty = torch.mean((jac_det - 1) ** 2)

        # Negative Jacobian penalty (prevent folding)
        neg_jac_penalty = torch.mean(F.relu(-jac_det))

        return l2_reg + jac_penalty + 10.0 * neg_jac_penalty  # type: ignore[no-any-return]

    def _register_single_scale(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        velocity_field: VelocityField,
        smoother: GaussianSmoothing,
        num_iterations: int,
    ) -> VelocityField:
        """Register at a single scale."""
        velocity_field.train()
        optimizer = optim.Adam(velocity_field.parameters(), lr=self.learning_rate)

        best_loss = float("inf")
        best_velocity = None

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Get current velocity
            velocity = velocity_field()

            # Apply smoothing
            velocity_smooth = smoother(velocity)

            # Integrate to get deformation
            deformation = self._integrate_velocity(
                velocity_smooth, self.num_integration_steps
            )

            # Apply deformation to moving image
            warped_moving = apply_deformation(moving, deformation)

            # Compute similarity loss
            sim_loss = self.loss_fn(fixed, warped_moving)

            # Compute regularization loss
            reg_loss = self._regularization_loss(velocity_smooth)

            # Total loss
            total_loss = sim_loss + self.alpha * reg_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Track best velocity field
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_velocity = VelocityField(velocity_field.shape, velocity_field.ndim)
                best_velocity.velocity.data = velocity_field.velocity.data.clone()

            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss.item():.6f}")

        return best_velocity or velocity_field

    def register(
        self,
        fixed_image: sitk.Image | torch.Tensor,
        moving_image: sitk.Image | torch.Tensor,
        initial_velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform RDMM deformable registration.

        Args:
            fixed_image: Fixed/reference image
                - Single-channel: [H, W], [D, H, W], [B, C, H, W], [B, C, D, H, W]
                - Multi-modal: [B, C, H, W] or [B, C, D, H, W] where C > 1
            moving_image: Moving image to be registered (same format as fixed)
            initial_velocity: Initial velocity field (optional)

        Returns:
            Tuple of (deformation_field, registered_image)
            - deformation_field: [B, H, W, 2] for 2D or [B, D, H, W, 3] for 3D
            - registered_image: Same shape as input moving_image

        Note:
            For multi-modal images:
            - Stack modalities in channel dimension: torch.stack([t1, t2], dim=1)
            - All channels share the same spatial transformation
            - Similarity metric computed per-channel then averaged
        """
        # Convert and prepare input tensors
        fixed, moving, ndim = self._prepare_input_tensors(fixed_image, moving_image)

        # Create image pyramids
        fixed_pyramid = self._create_pyramid(fixed, self.num_scales)
        moving_pyramid = self._create_pyramid(moving, self.num_scales)

        # Initialize velocity field - use the coarsest level (index 0)
        coarsest_shape = fixed_pyramid[0].shape[2:]
        velocity_field = VelocityField(coarsest_shape, ndim).to(self.device)

        if initial_velocity is not None:
            velocity_field.velocity.data = initial_velocity

        # Multi-scale registration (coarse to fine)
        for scale in range(self.num_scales):
            print(f"Scale {scale + 1}/{self.num_scales}")

            # Access pyramid directly from coarse to fine
            fixed_scale = fixed_pyramid[scale]
            moving_scale = moving_pyramid[scale]

            # Create Gaussian smoother for current scale
            smoother = GaussianSmoothing(ndim, self.smoothing_sigma, ndim).to(
                self.device
            )

            # If not the first scale, upsample velocity field
            if scale > 0:
                current_shape = fixed_scale.shape[2:]
                if ndim == 2:
                    upsampled_velocity = (
                        F.interpolate(
                            velocity_field.velocity,
                            size=current_shape,
                            mode="bilinear",
                            align_corners=True,
                        )
                        * 2
                    )  # Scale by 2 for upsampling
                else:
                    upsampled_velocity = (
                        F.interpolate(
                            velocity_field.velocity,
                            size=current_shape,
                            mode="trilinear",
                            align_corners=True,
                        )
                        * 2
                    )

                velocity_field = VelocityField(current_shape, ndim).to(self.device)
                velocity_field.velocity.data = upsampled_velocity

            # Adjust number of iterations for this scale
            scale_iterations = self.num_iterations[
                min(scale, len(self.num_iterations) - 1)
            ]

            # Register at this scale
            velocity_field = self._register_single_scale(
                fixed_scale, moving_scale, velocity_field, smoother, scale_iterations
            )

        # Generate final deformation field
        final_velocity = velocity_field()
        smoother = GaussianSmoothing(ndim, self.smoothing_sigma, ndim).to(self.device)
        final_velocity_smooth = smoother(final_velocity)
        deformation_field = self._integrate_velocity(
            final_velocity_smooth, self.num_integration_steps
        )

        # Apply final deformation
        registered = apply_deformation(moving, deformation_field)

        return deformation_field.squeeze(), registered.squeeze()

    def evaluate(
        self,
        fixed_image: sitk.Image | torch.Tensor,
        moving_image: sitk.Image | torch.Tensor,
        deformation_field: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Evaluate registration quality.

        Args:
            fixed_image: Fixed/reference image
            moving_image: Moving image
            deformation_field: Deformation field

        Returns:
            Dictionary of evaluation metrics
        """
        # Convert and prepare input tensors
        fixed, moving, _ = self._prepare_input_tensors(fixed_image, moving_image)

        # Apply deformation
        registered = apply_deformation(moving, deformation_field)

        # Ensure deformation field has batch dimension for Jacobian computation
        if len(deformation_field.shape) == 3:  # [H, W, 2] -> [1, H, W, 2]
            deformation_batch = deformation_field.unsqueeze(0)
        elif len(deformation_field.shape) == 4:  # [D, H, W, 3] -> [1, D, H, W, 3]
            deformation_batch = deformation_field.unsqueeze(0)
        else:
            deformation_batch = deformation_field

        # Compute metrics
        with torch.no_grad():
            ncc_metric = NCC()
            mse_metric = MSE()
            ncc_loss = ncc_metric(fixed, registered)
            mse_loss = mse_metric(fixed, registered)

            # Compute Jacobian determinant statistics
            jac_det = self._compute_jacobian_determinant(deformation_batch)

        metrics = {
            "ncc": -ncc_loss.item(),
            "mse": mse_loss.item(),
            "jacobian_det_mean": jac_det.mean().item(),
            "jacobian_det_std": jac_det.std().item(),
            "negative_jacobian_ratio": (jac_det < 0).float().mean().item(),
            "deformation_magnitude": torch.norm(deformation_field).item(),
        }

        return metrics
