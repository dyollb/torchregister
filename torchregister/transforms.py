"""
Utility functions for creating and converting transformations.
"""

import SimpleITK as sitk
import torch


def torch_affine_to_sitk_transform(
    affine_matrix: torch.Tensor, fixed_image: sitk.Image, moving_image: sitk.Image
) -> sitk.AffineTransform:
    """
    Convert PyTorch affine transformation matrix to SimpleITK AffineTransform.

    Args:
        affine_matrix: Affine transformation matrix from PyTorch registration
            - 2D: [2, 3] matrix
            - 3D: [3, 4] matrix
        fixed_image: Fixed/reference image defining the target coordinate space
        moving_image: Moving image defining the source coordinate space

    Returns:
        SimpleITK AffineTransform in physical coordinates

    Note:
        PyTorch affine transforms work in normalized pixel coordinates [-1, 1],
        while SimpleITK works in physical world coordinates. This function
        performs the necessary coordinate system conversion accounting for:
        - Image spacing (pixel size)
        - Image origin (physical location of first pixel)
        - Image direction (orientation matrix)

        The transformation chain:
        PyTorch: normalized [-1,1] → pixel [0,size-1] → physical (mm)
        SimpleITK: physical (mm) → physical (mm)
    """
    # Convert to numpy
    affine = affine_matrix.detach().cpu()

    # Determine dimensionality
    if affine.shape not in ((2, 3), (3, 4)):
        raise ValueError(f"Unsupported affine matrix shape: {affine.shape}")

    ndim = affine.shape[0]
    fixed_to_normalized = image_to_normalized_affine(fixed_image)
    moving_to_normalized_inv = invert_affine_transform(
        image_to_normalized_affine(moving_image)
    )

    fixed_to_normalized = to_homogeneous(fixed_to_normalized).to(affine.dtype)
    affine = to_homogeneous(affine)
    moving_to_normalized_inv = to_homogeneous(moving_to_normalized_inv).to(affine.dtype)

    composed_matrix = moving_to_normalized_inv @ affine @ fixed_to_normalized
    composed_matrix = composed_matrix[:ndim, :]

    # create SimpleITK AffineTransform
    composed_linear = composed_matrix[:, :ndim]
    composed_translation = composed_matrix[:, ndim]

    composed_transform = sitk.AffineTransform(ndim)
    composed_transform.SetMatrix(composed_linear.flatten().numpy().astype(float))
    composed_transform.SetTranslation(composed_translation.numpy().astype(float))
    return composed_transform


def image_to_normalized_affine(image: sitk.Image) -> torch.Tensor:
    """Compute affine transform from physical coordinates to normalized pixel coordinates (-1, 1)."""
    size = torch.Tensor(image.GetSize())
    spacing = torch.Tensor(image.GetSpacing())
    origin = torch.Tensor(image.GetOrigin())
    direction = torch.Tensor(image.GetDirection())

    ndim = len(size)

    # reshape direction matrix
    direction = direction.reshape(ndim, ndim)
    inv_direction = direction.T

    # rotation + scale: 2 * direction^T / (spacing * (size - 1))
    scale = 2.0 / (spacing * (size - 1))
    linear = inv_direction * scale[None, :]

    # translation: -2 * direction^T * origin / (spacing * (size - 1)) - 1
    translation = -2.0 * (inv_direction @ origin) / (spacing * (size - 1)) - 1.0

    # combine affine matrix [ndim, ndim+1]
    return torch.cat([linear, translation[:, None]], dim=-1)


def to_homogeneous(affine: torch.Tensor) -> torch.Tensor:
    if affine.shape not in ((2, 3), (3, 4)):
        raise ValueError(f"Unsupported affine shape: {affine.shape}")

    ndim = affine.shape[0]
    return torch.cat(
        [affine, torch.tensor([[0.0] * ndim + [1.0]], device=affine.device)], dim=0
    )


def invert_affine_transform(transform: torch.Tensor) -> torch.Tensor:
    """Invert an affine transformation matrix."""
    # invert linear part
    linear_inv = torch.inverse(transform[:, :-1])

    # invert translation: -A^(-1) * t
    translation_inv = -linear_inv @ transform[:, -1]

    return torch.cat([linear_inv, translation_inv.unsqueeze(-1)], dim=-1)
