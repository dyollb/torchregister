"""
Utility functions for image I/O and conversion between SimpleITK and PyTorch.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def sitk_to_torch(image: sitk.Image) -> torch.Tensor:
    """
    Convert SimpleITK image to PyTorch tensor.

    Args:
        image: SimpleITK image

    Returns:
        PyTorch tensor with shape [H, W] for 2D or [D, H, W] for 3D

    Note:
        - SimpleITK uses (z, y, x) ordering, we convert to PyTorch's (D, H, W)
        - For registration, add batch and channel dimensions: [1, 1, D, H, W]
        - For multi-modal: stack modalities in channel dim: [1, C, D, H, W]
    """
    array = sitk.GetArrayFromImage(image)
    tensor = torch.from_numpy(array).float()

    # SimpleITK uses (z, y, x) ordering, reverse for consistency
    if len(tensor.shape) == 3:
        tensor = tensor.flip(dims=[0])  # Flip z-axis for consistency

    return tensor


def torch_to_sitk(
    tensor: torch.Tensor, reference_image: sitk.Image | None = None
) -> sitk.Image:
    """
    Convert PyTorch tensor to SimpleITK image.

    Args:
        tensor: PyTorch tensor
        reference_image: Reference image for spacing/origin/direction

    Returns:
        SimpleITK image
    """
    # Convert to numpy
    if tensor.requires_grad:
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor.cpu().numpy()

    # Ensure correct data type
    array = array.astype(np.float32)

    # Create SimpleITK image
    image = sitk.GetImageFromArray(array)

    # Copy metadata from reference image if provided
    if reference_image is not None:
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())

    return image


def load_image(filepath: str | Path) -> sitk.Image:
    """
    Load image from file using SimpleITK.

    Args:
        filepath: Path to image file

    Returns:
        SimpleITK image
    """
    try:
        image = sitk.ReadImage(filepath)
        return image
    except Exception as e:
        raise OSError(f"Failed to load image from {filepath}: {str(e)}") from e


def save_image(
    image: sitk.Image | torch.Tensor,
    filepath: str | Path,
    reference_image: sitk.Image | None = None,
) -> None:
    """
    Save image to file using SimpleITK.

    Args:
        image: Image to save (SimpleITK or PyTorch tensor)
        filepath: Output file path
        reference_image: Reference image for metadata (if image is tensor)
    """
    if isinstance(image, torch.Tensor):
        sitk_image = torch_to_sitk(image, reference_image)
    else:
        sitk_image = image

    try:
        sitk.WriteImage(sitk_image, filepath)
    except Exception as e:
        raise OSError(f"Failed to save image to {filepath}: {str(e)}") from e


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
