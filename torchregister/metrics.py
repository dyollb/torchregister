"""
Differentiable loss functions for image registration.

This module implements various similarity metrics commonly used in medical
image registration, all differentiable and GPU-accelerated using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegistrationLoss(nn.Module):
    """
    Base class for all registration loss functions.

    All loss functions should inherit from this class and implement the forward method.
    """

    def __init__(self, name: str = "RegistrationLoss") -> None:
        super().__init__()
        self.name = name

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between fixed and moving images.

        Args:
            fixed: Fixed image tensor
            moving: Moving image tensor

        Returns:
            Loss value (lower is better for optimization)
        """
        raise NotImplementedError("Subclasses must implement the forward method")

    def __call__(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """Make the loss callable."""
        return self.forward(fixed, moving)

    def __str__(self) -> str:
        """String representation of the loss."""
        return self.name

    def __repr__(self) -> str:
        """Detailed string representation of the loss."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class NCC(RegistrationLoss):
    """
    Normalized Cross-Correlation (NCC) loss.

    Computes the negative normalized cross-correlation between two images.
    Higher correlation means better alignment.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small value to avoid division by zero
        """
        super().__init__("ncc")
        self.eps = eps

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fixed: Fixed image tensor [B, C, H, W] or [B, C, D, H, W]
            moving: Moving image tensor [B, C, H, W] or [B, C, D, H, W]

        Returns:
            Negative NCC loss (lower is better)
        """
        # Flatten spatial dimensions
        fixed_flat = fixed.view(fixed.shape[0], fixed.shape[1], -1)
        moving_flat = moving.view(moving.shape[0], moving.shape[1], -1)

        # Compute means
        fixed_mean = fixed_flat.mean(dim=2, keepdim=True)
        moving_mean = moving_flat.mean(dim=2, keepdim=True)

        # Center the data
        fixed_centered = fixed_flat - fixed_mean
        moving_centered = moving_flat - moving_mean

        # Compute correlation
        numerator = (fixed_centered * moving_centered).sum(dim=2)

        # Compute standard deviations
        fixed_std = torch.sqrt((fixed_centered**2).sum(dim=2) + self.eps)
        moving_std = torch.sqrt((moving_centered**2).sum(dim=2) + self.eps)

        # Compute NCC
        ncc = numerator / (fixed_std * moving_std + self.eps)

        # Return negative NCC (we want to minimize)
        return -ncc.mean()


class LNCC(RegistrationLoss):
    """
    Local Normalized Cross-Correlation (LNCC) loss.

    Computes NCC in local windows across the image, providing better
    handling of local intensity variations.
    """

    def __init__(self, window_size: int = 9, eps: float = 1e-8):
        """
        Args:
            window_size: Size of the local window
            eps: Small value to avoid division by zero
        """
        super().__init__("lncc")
        self.window_size = window_size
        self.eps = eps

    def _create_window(self, channel: int, device: torch.device) -> torch.Tensor:
        """Create averaging window for local computations."""
        window = torch.ones(
            channel,
            1,
            self.window_size,
            self.window_size,
            device=device,
            dtype=torch.float32,
        )
        return window / (self.window_size**2)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fixed: Fixed image tensor [B, C, H, W]
            moving: Moving image tensor [B, C, H, W]

        Returns:
            Negative LNCC loss (lower is better)
        """
        B, C, H, W = fixed.shape
        padding = self.window_size // 2

        # Create averaging window
        window = self._create_window(C, fixed.device)

        # Compute local means
        fixed_mean = F.conv2d(fixed, window, padding=padding, groups=C)
        moving_mean = F.conv2d(moving, window, padding=padding, groups=C)

        # Compute local variances and covariance
        fixed_sq = F.conv2d(fixed * fixed, window, padding=padding, groups=C)
        moving_sq = F.conv2d(moving * moving, window, padding=padding, groups=C)
        fixed_moving = F.conv2d(fixed * moving, window, padding=padding, groups=C)

        # Compute local standard deviations
        fixed_var = fixed_sq - fixed_mean * fixed_mean
        moving_var = moving_sq - moving_mean * moving_mean
        covar = fixed_moving - fixed_mean * moving_mean

        # Compute LNCC
        fixed_std = torch.sqrt(fixed_var + self.eps)
        moving_std = torch.sqrt(moving_var + self.eps)

        lncc = covar / (fixed_std * moving_std + self.eps)

        # Return negative mean LNCC
        return -lncc.mean()


class MSE(RegistrationLoss):
    """
    Mean Squared Error (MSE) loss.

    Simple intensity-based similarity metric.
    """

    def __init__(self) -> None:
        super().__init__("mse")

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fixed: Fixed image tensor
            moving: Moving image tensor

        Returns:
            MSE loss
        """
        return F.mse_loss(fixed, moving)


class MattesMI(RegistrationLoss):
    """
    Mattes Mutual Information (MI) loss.

    Information-theoretic similarity metric that measures statistical
    dependence between image intensities.
    """

    def __init__(self, bins: int = 64, sigma: float = 1.0):
        """
        Args:
            bins: Number of histogram bins
            sigma: Gaussian kernel standard deviation for Parzen windowing
        """
        super().__init__("mattes_mi")
        self.bins = bins
        self.sigma = sigma

    def _compute_joint_histogram(
        self, fixed: torch.Tensor, moving: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint histogram using Parzen windowing."""
        # Normalize intensities to [0, 1]
        fixed_norm = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-8)
        moving_norm = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)

        # Scale to bin range
        fixed_scaled = fixed_norm * (self.bins - 1)
        moving_scaled = moving_norm * (self.bins - 1)

        # Create bin centers
        bin_centers = torch.linspace(0, self.bins - 1, self.bins, device=fixed.device)

        # Compute Parzen window contributions
        fixed_contrib = torch.exp(
            -0.5 * ((fixed_scaled.unsqueeze(-1) - bin_centers) / self.sigma) ** 2
        )
        moving_contrib = torch.exp(
            -0.5 * ((moving_scaled.unsqueeze(-1) - bin_centers) / self.sigma) ** 2
        )

        # Normalize contributions
        fixed_contrib = fixed_contrib / (fixed_contrib.sum(dim=-1, keepdim=True) + 1e-8)
        moving_contrib = moving_contrib / (
            moving_contrib.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Compute joint histogram
        joint_hist = torch.zeros(self.bins, self.bins, device=fixed.device)
        for i in range(self.bins):
            for j in range(self.bins):
                joint_hist[i, j] = (
                    fixed_contrib[..., i] * moving_contrib[..., j]
                ).sum()

        # Normalize
        joint_hist = joint_hist / (joint_hist.sum() + 1e-8)

        return joint_hist

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fixed: Fixed image tensor
            moving: Moving image tensor

        Returns:
            Negative mutual information (lower is better)
        """
        # Flatten images
        fixed_flat = fixed.view(-1)
        moving_flat = moving.view(-1)

        # Compute joint histogram
        joint_hist = self._compute_joint_histogram(fixed_flat, moving_flat)

        # Compute marginal histograms
        fixed_hist = joint_hist.sum(dim=1)
        moving_hist = joint_hist.sum(dim=0)

        # Compute mutual information
        mi = torch.tensor(0.0, device=fixed.device, dtype=fixed.dtype)
        for i in range(self.bins):
            for j in range(self.bins):
                if joint_hist[i, j] > 1e-8:
                    mi += joint_hist[i, j] * torch.log(
                        joint_hist[i, j] / (fixed_hist[i] * moving_hist[j] + 1e-8)
                        + 1e-8
                    )

        # Return negative MI (we want to minimize)
        return -mi


class Dice(RegistrationLoss):
    """
    Dice coefficient loss for segmentation overlap.

    Commonly used for evaluating registration quality when segmentations
    are available.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__("dice")
        self.smooth = smooth

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fixed: Fixed image tensor (typically target segmentation)
            moving: Moving image tensor (typically registered moving segmentation)

        Returns:
            Dice loss (1 - Dice coefficient)
        """
        # Flatten tensors
        fixed_flat = fixed.view(-1)
        moving_flat = moving.view(-1)

        # Compute intersection and union
        intersection = (fixed_flat * moving_flat).sum()
        union = fixed_flat.sum() + moving_flat.sum()

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss (1 - Dice coefficient)
        return torch.tensor(1.0, device=dice.device, dtype=dice.dtype) - dice


class CombinedLoss(RegistrationLoss):
    """
    Combines multiple loss functions with weights.

    Useful for multi-term registration objectives.
    """

    def __init__(self, losses: dict, weights: dict) -> None:
        """
        Args:
            losses: Dictionary of loss functions
            weights: Dictionary of weights for each loss
        """
        super().__init__("combined_loss")
        self.losses = nn.ModuleDict(losses)
        self.weights = weights

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted combination of losses.

        Args:
            fixed: Fixed image tensor
            moving: Moving image tensor
            **kwargs: Additional arguments for specific losses

        Returns:
            Combined loss value
        """
        total_loss = torch.tensor(0.0, device=fixed.device, dtype=fixed.dtype)

        for name, loss_fn in self.losses.items():
            if name in self.weights:
                loss_value = loss_fn(fixed, moving)
                total_loss += self.weights[name] * loss_value

        return total_loss
