"""
Utility functions for image I/O and transformations.

This module provides functions for converting between SimpleITK and PyTorch,
creating coordinate grids, and applying transformations.
"""

from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def sitk_to_torch(image: sitk.Image) -> torch.Tensor:
    """
    Convert SimpleITK image to PyTorch tensor.

    Args:
        image: SimpleITK image

    Returns:
        PyTorch tensor with shape [H, W] for 2D or [D, H, W] for 3D

    Note:
        - SimpleITK uses (z, y, x) o    # Determine dimension and field shape
    if len(deform_np.shape) == 3 and deform_np.shape[-1] == 2:
        # 2D deformation field [H, W, 2]
        dimension = 2
        field_shape = deform_np.shape[:2]  # (H, W)
    elif len(deform_np.shape) == 4 and deform_np.shape[-1] == 3:
        # 3D deformation field [D, H, W, 3]
        dimension = 3
        field_shape = deform_np.shape[:3]  # type: ignore[assignment]  # (D, H, W)we convert to PyTorch's (D, H, W)
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


def load_image(filepath: str) -> sitk.Image:
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
    filepath: str,
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
    image: torch.Tensor, transformed_grid: torch.Tensor
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
    expected_grid_coords = 2 if ndim == 2 else 3
    if transformed_grid.shape[-1] != expected_grid_coords:
        raise ValueError(
            f"Grid last dimension should be {expected_grid_coords} for {ndim}D images, got {transformed_grid.shape[-1]}"
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
            mode="bilinear",
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
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    # Restore original shape if needed
    if len(original_shape) == 2:
        warped = warped.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        warped = warped.squeeze(0).squeeze(0)

    return warped


def apply_deformation(image: torch.Tensor, deformation: torch.Tensor) -> torch.Tensor:
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
    warped = apply_transform(image, warped_grid)

    # Restore original shape if needed
    if len(original_image_shape) == 2:
        warped = warped.squeeze(0).squeeze(0)
    elif len(original_image_shape) == 3:
        warped = warped.squeeze(0).squeeze(0)

    return warped


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


def torch_affine_to_sitk_transform(
    affine_matrix: torch.Tensor, reference_image: sitk.Image | None = None
) -> sitk.AffineTransform:
    """
    Convert PyTorch affine transformation matrix to SimpleITK AffineTransform.

    Args:
        affine_matrix: Affine transformation matrix
            - 2D: [2, 3] matrix
            - 3D: [3, 4] matrix
        reference_image: Reference image for coordinate space (optional)

    Returns:
        SimpleITK AffineTransform

    Note:
        Both TorchRegister and SimpleITK use the same transform convention:
        - Transforms map from moving → fixed coordinates
        - The transform specifies where to sample from the moving image
          for each pixel position in the fixed image
        - A positive translation moves the image content in the negative direction

        For 2D: [[a, b, tx], [c, d, ty]]
        For 3D: [[a, b, c, tx], [d, e, f, ty], [g, h, i, tz]]
    """
    # Convert to numpy
    if affine_matrix.requires_grad:
        matrix = affine_matrix.detach().cpu().numpy()
    else:
        matrix = affine_matrix.cpu().numpy()

    # Determine dimensionality
    if matrix.shape == (2, 3):
        # 2D transform
        dimension = 2
        # Extract linear and translation parts
        linear = matrix[:, :2].flatten()  # [a, b, c, d]
        translation = matrix[:, 2]  # [tx, ty]
    elif matrix.shape == (3, 4):
        # 3D transform
        dimension = 3
        # Extract linear and translation parts
        linear = matrix[:, :3].flatten()  # [a, b, c, d, e, f, g, h, i]
        translation = matrix[:, 3]  # [tx, ty, tz]
    else:
        raise ValueError(f"Unsupported affine matrix shape: {matrix.shape}")

    # Create SimpleITK AffineTransform
    transform = sitk.AffineTransform(dimension)

    # Set parameters (linear part + translation)
    transform.SetMatrix(linear.astype(np.float64))
    transform.SetTranslation(translation.astype(np.float64))

    # If reference image provided, set center for rotation
    if reference_image is not None:
        size = reference_image.GetSize()
        spacing = reference_image.GetSpacing()
        origin = reference_image.GetOrigin()

        # Compute center of image in physical coordinates
        center = []
        for i in range(dimension):
            center.append(origin[i] + (size[i] - 1) * spacing[i] / 2.0)
        transform.SetCenter(center)

    return transform


def torch_deformation_to_sitk_transform(
    deformation_field: torch.Tensor, reference_image: sitk.Image
) -> sitk.DisplacementFieldTransform:
    """
    Convert PyTorch deformation field to SimpleITK DisplacementFieldTransform.

    Args:
        deformation_field: Deformation field tensor
            - 2D: [B, H, W, 2] or [H, W, 2] with [dx, dy] displacements
            - 3D: [B, D, H, W, 3] or [D, H, W, 3] with [dx, dy, dz] displacements
        reference_image: Reference image defining the coordinate space

    Returns:
        SimpleITK DisplacementFieldTransform

    Note:
        - Deformation vectors are in pixel coordinates and will be converted to physical coordinates
        - The reference image defines the spacing, origin, and direction for the field
    """
    # Remove batch dimension if present
    if len(deformation_field.shape) == 4 and deformation_field.shape[0] == 1:  # 2D
        deform = deformation_field.squeeze(0)  # [H, W, 2]
    elif len(deformation_field.shape) == 5 and deformation_field.shape[0] == 1:  # 3D
        deform = deformation_field.squeeze(0)  # [D, H, W, 3]
    else:
        deform = deformation_field

    # Convert to numpy
    if deform.requires_grad:
        deform_np = deform.detach().cpu().numpy()
    else:
        deform_np = deform.cpu().numpy()

    # Determine dimension
    if len(deform_np.shape) == 3 and deform_np.shape[-1] == 2:
        # 2D deformation field [H, W, 2]
        dimension = 2
    elif len(deform_np.shape) == 4 and deform_np.shape[-1] == 3:
        # 3D deformation field [D, H, W, 3]
        dimension = 3
    else:
        raise ValueError(f"Unsupported deformation field shape: {deform_np.shape}")

    # Get reference image properties
    spacing = reference_image.GetSpacing()
    origin = reference_image.GetOrigin()
    direction = reference_image.GetDirection()

    # Convert from pixel displacements to physical displacements
    # SimpleITK expects displacements in physical coordinates
    physical_deform = deform_np.copy()
    for i in range(dimension):
        physical_deform[..., i] *= spacing[i]

    # SimpleITK expects displacement field in (x, y, z) component order
    # but with spatial dimensions in (z, y, x) order (for 3D)
    if dimension == 3:
        # Reorder spatial dimensions from (D, H, W) to (W, H, D) for SimpleITK
        # and reorder vector components from (z, y, x) to (x, y, z)
        physical_deform = np.transpose(physical_deform, (2, 1, 0, 3))
        # Reorder vector components: [dx, dy, dz] -> [dx, dy, dz] (already correct)
    else:
        # For 2D: (H, W) -> (W, H) for SimpleITK
        physical_deform = np.transpose(physical_deform, (1, 0, 2))
        # Vector components already in correct order [dx, dy]

    # Create SimpleITK displacement field image
    # Convert to vector image (each pixel contains displacement vector)
    # SimpleITK requires float64 for displacement fields
    displacement_image = sitk.GetImageFromArray(
        physical_deform.astype(np.float64), isVector=True
    )

    # Set image properties to match reference
    displacement_image.SetSpacing(spacing)
    displacement_image.SetOrigin(origin)
    displacement_image.SetDirection(direction)

    # Create DisplacementFieldTransform
    transform = sitk.DisplacementFieldTransform(displacement_image)

    return transform


def torch_deformation_to_sitk_field(
    deformation_field: torch.Tensor, reference_image: sitk.Image
) -> sitk.Image:
    """
    Convert PyTorch deformation field to SimpleITK displacement field image.

    This is useful when you want the displacement field as an image rather than a transform.

    Args:
        deformation_field: Deformation field tensor (same format as torch_deformation_to_sitk_transform)
        reference_image: Reference image defining the coordinate space

    Returns:
        SimpleITK displacement field image (vector image)
    """
    transform = torch_deformation_to_sitk_transform(deformation_field, reference_image)
    return transform.GetDisplacementField()  # type: ignore[no-any-return]


def sitk_transform_to_torch_affine(
    transform: sitk.AffineTransform, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert SimpleITK AffineTransform to PyTorch affine matrix.

    Args:
        transform: SimpleITK AffineTransform
        dtype: Desired PyTorch dtype for output tensor (default: torch.float32)

    Returns:
        PyTorch affine matrix:
        - 2D: [2, 3] matrix
        - 3D: [3, 4] matrix
    """
    dimension = transform.GetDimension()

    # Get transformation parameters
    matrix = np.array(transform.GetMatrix())
    translation = np.array(transform.GetTranslation())

    if dimension == 2:
        # Reshape linear part to 2x2 matrix
        linear = matrix.reshape(2, 2)
        # Combine with translation
        affine_matrix = np.column_stack([linear, translation])  # [2, 3]
    elif dimension == 3:
        # Reshape linear part to 3x3 matrix
        linear = matrix.reshape(3, 3)
        # Combine with translation
        affine_matrix = np.column_stack([linear, translation])  # [3, 4]
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

    # Convert to requested dtype
    if dtype == torch.float32:
        return torch.from_numpy(affine_matrix.astype(np.float32))
    elif dtype == torch.float64:
        return torch.from_numpy(affine_matrix.astype(np.float64))
    else:
        return torch.from_numpy(affine_matrix).to(dtype)


def sitk_displacement_to_torch_deformation(
    displacement_field: sitk.Image, add_batch_dim: bool = True
) -> torch.Tensor:
    """
    Convert SimpleITK displacement field to PyTorch deformation field.

    Args:
        displacement_field: SimpleITK displacement field (vector image)
        add_batch_dim: Whether to add batch dimension

    Returns:
        PyTorch deformation field:
        - 2D: [H, W, 2] or [1, H, W, 2] if add_batch_dim=True
        - 3D: [D, H, W, 3] or [1, D, H, W, 3] if add_batch_dim=True
    """
    # Convert to numpy array
    displacement_np = sitk.GetArrayFromImage(displacement_field)

    # Get image properties
    spacing = displacement_field.GetSpacing()
    dimension = displacement_field.GetDimension()

    # Convert from physical displacements back to pixel displacements
    pixel_deform = displacement_np.copy()
    for i in range(dimension):
        pixel_deform[..., i] /= spacing[i]

    if dimension == 3:
        # SimpleITK gives us (W, H, D, 3), convert to (D, H, W, 3)
        pixel_deform = np.transpose(pixel_deform, (2, 1, 0, 3))
    else:
        # SimpleITK gives us (W, H, 2), convert to (H, W, 2)
        pixel_deform = np.transpose(pixel_deform, (1, 0, 2))

    # Convert to torch tensor
    deform_tensor = torch.from_numpy(pixel_deform.astype(np.float32))

    # Add batch dimension if requested
    if add_batch_dim:
        deform_tensor = deform_tensor.unsqueeze(0)

    return deform_tensor
