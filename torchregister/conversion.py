"""
Utility functions for converting between SimpleITK and PyTorch transformations.
"""

import numpy as np
import SimpleITK as sitk
import torch


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
