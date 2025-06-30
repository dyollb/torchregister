"""
Image processing utilities for TorchRegister.

This module contains image processing functions including Gaussian filtering,
normalization, and resampling operations.
"""

from collections.abc import Sequence

import SimpleITK as sitk
import torch
import torch.nn.functional as F


def gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur to an image tensor using separable 1D filters.
    Works for both 2D [B, C, H, W] and 3D [B, C, D, H, W] tensors.

    Args:
        image: Input tensor of shape [B, C, H, W] or [B, C, D, H, W]
        kernel_size: Size of the Gaussian kernel (should be odd)
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Blurred image tensor of the same shape
    """
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=image.device)
    coords -= kernel_size // 2

    # Compute Gaussian values
    gaussian_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian_1d /= gaussian_1d.sum()

    num_channels = image.shape[1]
    padding = kernel_size // 2
    current = image

    if len(image.shape) == 4:  # 2D case: [B, C, H, W]
        # Apply 1D convolution along height (dim 2)
        kernel_h = gaussian_1d.view(1, 1, kernel_size, 1).expand(
            num_channels, 1, kernel_size, 1
        )
        current = F.conv2d(current, kernel_h, padding=(padding, 0), groups=num_channels)

        # Apply 1D convolution along width (dim 3)
        kernel_w = gaussian_1d.view(1, 1, 1, kernel_size).expand(
            num_channels, 1, 1, kernel_size
        )
        current = F.conv2d(current, kernel_w, padding=(0, padding), groups=num_channels)

    elif len(image.shape) == 5:  # 3D case: [B, C, D, H, W]
        # Apply 1D convolution along depth (dim 2)
        kernel_d = gaussian_1d.view(1, 1, kernel_size, 1, 1).expand(
            num_channels, 1, kernel_size, 1, 1
        )
        current = F.conv3d(
            current, kernel_d, padding=(padding, 0, 0), groups=num_channels
        )

        # Apply 1D convolution along height (dim 3)
        kernel_h = gaussian_1d.view(1, 1, 1, kernel_size, 1).expand(
            num_channels, 1, 1, kernel_size, 1
        )
        current = F.conv3d(
            current, kernel_h, padding=(0, padding, 0), groups=num_channels
        )

        # Apply 1D convolution along width (dim 4)
        kernel_w = gaussian_1d.view(1, 1, 1, 1, kernel_size).expand(
            num_channels, 1, 1, 1, kernel_size
        )
        current = F.conv3d(
            current, kernel_w, padding=(0, 0, padding), groups=num_channels
        )

    return current


def normalize_image(image: torch.Tensor, method: str = "minmax") -> torch.Tensor:
    """
    Normalize image intensities.

    Args:
        image: Input image tensor
        method: Normalization method ("minmax", "zscore")

    Returns:
        Normalized image
    """
    if method == "minmax":
        min_val = image.min()
        max_val = image.max()
        normalized = (image - min_val) / (max_val - min_val + 1e-8)
    elif method == "zscore":
        mean_val = image.mean()
        std_val = image.std()
        normalized = (image - mean_val) / (std_val + 1e-8)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return normalized


def resample_image(
    image: sitk.Image,
    new_spacing: Sequence[float] | None = None,
    new_size: Sequence[int] | None = None,
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """
    Resample SimpleITK image to new spacing or size.

    Args:
        image: Input SimpleITK image
        new_spacing: New voxel spacing
        new_size: New image size
        interpolator: SimpleITK interpolation method

    Returns:
        Resampled image
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    if new_spacing is not None:
        # Calculate new size based on spacing
        new_size_list = [
            int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
            for i in range(len(original_size))
        ]
        new_spacing_list = list(new_spacing)
    elif new_size is not None:
        # Calculate new spacing based on size
        new_spacing_list = [
            original_spacing[i] * original_size[i] / new_size[i]
            for i in range(len(original_size))
        ]
        new_size_list = list(new_size)
    else:
        raise ValueError("Either new_spacing or new_size must be provided")

    # Set up resampling
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing_list)
    resampler.SetSize(new_size_list)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)  # type: ignore[no-any-return]
