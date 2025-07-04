"""
Utility functions for creating and applying transformations.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def create_grid(
    shape: Sequence[int], device: torch.device | None = None
) -> torch.Tensor:
    """
    Create coordinate grid for given shape.

    Args:
        shape: Spatial dimensions (H, W) for 2D or (D, H, W) for 3D
        device: PyTorch device

    Returns:
        Coordinate grid tensor:
        - 2D: [H, W, 2] with coordinates [x, y] in range [-1, 1]
        - 3D: [D, H, W, 3] with coordinates [x, y, z] in range [-1, 1]

    Note:
        Follows PyTorch grid_sample convention where:
        - x corresponds to width dimension (last spatial dim)
        - y corresponds to height dimension (second-to-last spatial dim)
        - z corresponds to depth dimension (third-to-last spatial dim)
    """
    if device is None:
        device = torch.device("cpu")

    if len(shape) == 2:  # 2D
        H, W = shape
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)

        # Create meshgrid
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Stack coordinates [H, W, 2]
        grid = torch.stack([grid_x, grid_y], dim=-1)

    elif len(shape) == 3:  # 3D
        D, H, W = shape
        # Create coordinate grids
        z_coords = torch.linspace(-1, 1, D, device=device)
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)

        # Create meshgrid
        grid_z, grid_y, grid_x = torch.meshgrid(
            z_coords, y_coords, x_coords, indexing="ij"
        )

        # Stack coordinates [D, H, W, 3]
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return grid


def apply_transform(
    image: torch.Tensor, transformed_grid: torch.Tensor, interp_mode: str = "bilinear"
) -> torch.Tensor:
    """
    Apply transformation grid to image using grid sampling.

    Args:
        image: Input image tensor
            - 2D: [H, W], [B, C, H, W], or compatible shapes
            - 3D: [D, H, W], [B, C, D, H, W], or compatible shapes
        transformed_grid: Transformed coordinate grid
            - 2D: [H, W, 2], [B, H, W, 2], or compatible shapes
            - 3D: [D, H, W, 3], [B, D, H, W, 3], or compatible shapes

    Returns:
        Transformed image with same shape as input

    Note:
        - Grid coordinates should be in range [-1, 1] (PyTorch convention)
        - Grid last dimension: 2 for 2D (x, y), 3 for 3D (x, y, z)
        - For multi-modal images, same grid applied to all channels
    """
    # Handle input tensor that might not have batch/channel dimensions
    original_shape = image.shape

    if len(image.shape) == 2:  # [H, W] -> [1, 1, H, W]
        image = image.unsqueeze(0).unsqueeze(0)
        ndim = 2
    elif len(image.shape) == 3:  # [D, H, W] -> [1, 1, D, H, W]
        image = image.unsqueeze(0).unsqueeze(0)
        ndim = 3
    elif len(image.shape) == 4:  # 2D with batch/channel
        ndim = 2
    elif len(image.shape) == 5:  # 3D with batch/channel
        ndim = 3
    else:
        raise ValueError(f"Unsupported image shape: {original_shape}")

    # Validate grid dimensions
    if transformed_grid.shape[-1] != ndim:
        raise ValueError(
            f"Grid last dimension should be {ndim} for {ndim}D images, got {transformed_grid.shape[-1]}"
        )

    if ndim == 2:  # 2D
        # transformed_grid should be [B, H, W, 2] or [H, W, 2]
        if len(transformed_grid.shape) == 3:  # [H, W, 2] -> [1, H, W, 2]
            transformed_grid = transformed_grid.unsqueeze(0)  # Add batch dim
        elif len(transformed_grid.shape) == 5:  # [1, 1, H, W, 2] -> [1, H, W, 2]
            transformed_grid = transformed_grid.squeeze(1)  # Remove extra dim

        warped = F.grid_sample(
            image,
            transformed_grid,
            mode=interp_mode,
            padding_mode="border",
            align_corners=True,
        )

    else:  # 3D
        # transformed_grid should be [B, D, H, W, 3] or [D, H, W, 3]
        if len(transformed_grid.shape) == 4:  # [D, H, W, 3] -> [1, D, H, W, 3]
            transformed_grid = transformed_grid.unsqueeze(0)  # Add batch dim
        elif len(transformed_grid.shape) == 6:  # [1, 1, D, H, W, 3] -> [1, D, H, W, 3]
            transformed_grid = transformed_grid.squeeze(1)  # Remove extra dim

        warped = F.grid_sample(
            image,
            transformed_grid,
            mode=interp_mode,
            padding_mode="border",
            align_corners=True,
        )

    # Restore original shape if needed
    if len(original_shape) in (2, 3):
        warped = warped.squeeze(0).squeeze(0)

    return warped


def apply_deformation(
    image: torch.Tensor, deformation: torch.Tensor, interp_mode: str = "bilinear"
) -> torch.Tensor:
    """
    Apply deformation field to image.

    Args:
        image: Input image tensor
            - 2D: [H, W], [B, C, H, W], or compatible shapes
            - 3D: [D, H, W], [B, C, D, H, W], or compatible shapes
        deformation: Deformation field with displacement vectors
            - 2D: [H, W, 2], [B, H, W, 2] with [dx, dy] displacements
            - 3D: [D, H, W, 3], [B, D, H, W, 3] with [dx, dy, dz] displacements

    Returns:
        Deformed image with same shape as input

    Note:
        - Deformation vectors are added to identity grid before sampling
        - Displacements follow coordinate convention: [x, y] for 2D, [x, y, z] for 3D
        - For multi-modal images, same deformation applied to all channels
    """
    # Handle input tensors that might not have batch/channel dimensions
    original_image_shape = image.shape

    # Ensure proper dimensions for processing
    if len(image.shape) == 2:  # [H, W] -> [1, 1, H, W]
        image = image.unsqueeze(0).unsqueeze(0)
        ndim = 2
    elif len(image.shape) == 3:  # [D, H, W] -> [1, 1, D, H, W]
        image = image.unsqueeze(0).unsqueeze(0)
        ndim = 3
    elif len(image.shape) == 4:  # 2D with batch/channel
        ndim = 2
    elif len(image.shape) == 5:  # 3D with batch/channel
        ndim = 3
    else:
        raise ValueError(f"Unsupported image shape: {original_image_shape}")

    # Ensure deformation has batch dimension
    if len(deformation.shape) == 3:  # [H, W, 2] -> [1, H, W, 2]
        deformation = deformation.unsqueeze(0)
    elif len(deformation.shape) == 4:  # [D, H, W, 3] -> [1, D, H, W, 3]
        deformation = deformation.unsqueeze(0)

    # Create identity grid
    if ndim == 2:  # 2D
        grid = create_grid(image.shape[2:], image.device)
        grid = grid.unsqueeze(0).repeat(image.shape[0], 1, 1, 1)
    else:  # 3D
        grid = create_grid(image.shape[2:], image.device)
        grid = grid.unsqueeze(0).repeat(image.shape[0], 1, 1, 1, 1)

    # Add deformation to grid
    warped_grid = grid + deformation

    # Apply transformation
    warped = apply_transform(image, warped_grid, interp_mode)

    # Restore original shape if needed
    if len(original_image_shape) == 2:
        warped = warped.squeeze(0).squeeze(0)
    elif len(original_image_shape) == 3:
        warped = warped.squeeze(0).squeeze(0)

    return warped


def compute_gradient(image: torch.Tensor) -> torch.Tensor:
    """
    Compute image gradient using finite differences.

    Args:
        image: Input image tensor

    Returns:
        Gradient tensor
    """
    if len(image.shape) == 4:  # 2D: [B, C, H, W]
        # Compute gradients
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]

        # Pad to maintain size
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")

        # Stack gradients (replace channel dimension)
        gradient = torch.stack([grad_x, grad_y], dim=1)
        # Remove the original channel dimension that gets duplicated
        gradient = gradient.squeeze(2)

    elif len(image.shape) == 5:  # 3D: [B, C, D, H, W]
        # Compute gradients
        grad_z = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        grad_y = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        grad_x = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]

        # Pad to maintain size
        grad_z = F.pad(grad_z, (0, 0, 0, 0, 0, 1), mode="replicate")
        grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0), mode="replicate")
        grad_x = F.pad(grad_x, (0, 1, 0, 0, 0, 0), mode="replicate")

        # Stack gradients (replace channel dimension)
        gradient = torch.stack([grad_x, grad_y, grad_z], dim=1)
        # Remove the original channel dimension that gets duplicated
        gradient = gradient.squeeze(2)

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return gradient


def create_identity_transform(
    shape: Sequence[int], device: torch.device | None = None
) -> torch.Tensor:
    """
    Create identity transformation matrix.

    Args:
        shape: Spatial dimensions
        device: PyTorch device

    Returns:
        Identity transformation matrix
    """
    if device is None:
        device = torch.device("cpu")

    if len(shape) == 2:  # 2D
        matrix = torch.eye(2, 3, device=device)
    elif len(shape) == 3:  # 3D
        matrix = torch.eye(3, 4, device=device)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return matrix


def compose_transforms(
    transform1: torch.Tensor, transform2: torch.Tensor
) -> torch.Tensor:
    """
    Compose two affine transformation matrices.

    Args:
        transform1: First transformation matrix
        transform2: Second transformation matrix

    Returns:
        Composed transformation matrix
    """
    if transform1.shape[-1] == 3:  # 2D
        # Convert to homogeneous coordinates
        homo1 = torch.cat(
            [transform1, torch.tensor([[0, 0, 1]], device=transform1.device)], dim=0
        )
        homo2 = torch.cat(
            [transform2, torch.tensor([[0, 0, 1]], device=transform2.device)], dim=0
        )

        # Compose
        composed_homo = torch.matmul(homo2, homo1)

        # Extract 2x3 matrix
        composed = composed_homo[:2, :]

    elif transform1.shape[-1] == 4:  # 3D
        # Convert to homogeneous coordinates
        homo1 = torch.cat(
            [transform1, torch.tensor([[0, 0, 0, 1]], device=transform1.device)], dim=0
        )
        homo2 = torch.cat(
            [transform2, torch.tensor([[0, 0, 0, 1]], device=transform2.device)], dim=0
        )

        # Compose
        composed_homo = torch.matmul(homo2, homo1)

        # Extract 3x4 matrix
        composed = composed_homo[:3, :]

    else:
        raise ValueError(f"Unsupported transformation matrix shape: {transform1.shape}")

    return composed


def compute_target_registration_error(
    landmarks_fixed: torch.Tensor,
    landmarks_moving: torch.Tensor,
    transform: torch.Tensor | None = None,
    deformation: torch.Tensor | None = None,
) -> float:
    """
    Compute Target Registration Error (TRE) between corresponding landmarks.

    Args:
        landmarks_fixed: Fixed landmarks [N, ndim]
        landmarks_moving: Moving landmarks [N, ndim]
        transform: Affine transformation matrix (for affine registration)
        deformation: Deformation field (for deformable registration)

    Returns:
        Mean TRE in physical units
    """
    if transform is not None:
        # Apply affine transformation to moving landmarks
        if transform.shape[-1] == 3:  # 2D
            # Add homogeneous coordinate
            homo_landmarks = torch.cat(
                [
                    landmarks_moving,
                    torch.ones(
                        landmarks_moving.shape[0], 1, device=landmarks_moving.device
                    ),
                ],
                dim=1,
            )

            transformed_landmarks = torch.matmul(homo_landmarks, transform.T)

        elif transform.shape[-1] == 4:  # 3D
            # Add homogeneous coordinate
            homo_landmarks = torch.cat(
                [
                    landmarks_moving,
                    torch.ones(
                        landmarks_moving.shape[0], 1, device=landmarks_moving.device
                    ),
                ],
                dim=1,
            )

            transformed_landmarks = torch.matmul(homo_landmarks, transform.T)

    elif deformation is not None:
        # Apply deformation field to moving landmarks
        # Note: This is a simplified implementation
        # In practice, you'd need to interpolate the deformation field at landmark positions
        transformed_landmarks = landmarks_moving + deformation

    else:
        # No transformation applied
        transformed_landmarks = landmarks_moving

    # Compute Euclidean distances
    distances = torch.norm(landmarks_fixed - transformed_landmarks, dim=1)

    # Return mean TRE
    return float(distances.mean().item())
