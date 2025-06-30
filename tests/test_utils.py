"""
Tests for utility functions module.
"""

import os
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from torchregister.io import (
    load_image,
    save_image,
    sitk_to_torch,
    torch_to_sitk,
)
from torchregister.transforms import (
    apply_deformation,
    apply_transform,
    compose_transforms,
    compute_gradient,
    compute_target_registration_error,
    create_grid,
    create_identity_transform,
)


class TestImageIO:
    """Test image I/O functions."""

    def test_sitk_to_torch_2d(self, create_sitk_image, device):
        """Test converting 2D SimpleITK image to PyTorch tensor."""
        # Create 2D array
        array = np.random.rand(32, 32).astype(np.float32)
        sitk_image = create_sitk_image(array)

        tensor = sitk_to_torch(sitk_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (32, 32)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.from_numpy(array))

    def test_sitk_to_torch_3d(self, create_sitk_image, device):
        """Test converting 3D SimpleITK image to PyTorch tensor."""
        # Create 3D array
        array = np.random.rand(16, 32, 32).astype(np.float32)
        sitk_image = create_sitk_image(array)

        tensor = sitk_to_torch(sitk_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (16, 32, 32)
        assert tensor.dtype == torch.float32

    def test_torch_to_sitk_2d(self, device):
        """Test converting 2D PyTorch tensor to SimpleITK image."""
        tensor = torch.rand(32, 32, device=device)

        sitk_image = torch_to_sitk(tensor)

        assert isinstance(sitk_image, sitk.Image)
        assert sitk_image.GetSize() == (32, 32)
        assert sitk_image.GetDimension() == 2

    def test_torch_to_sitk_3d(self, device):
        """Test converting 3D PyTorch tensor to SimpleITK image."""
        tensor = torch.rand(16, 32, 32, device=device)

        sitk_image = torch_to_sitk(tensor)

        assert isinstance(sitk_image, sitk.Image)
        assert sitk_image.GetSize() == (32, 32, 16)  # SimpleITK uses (x, y, z) ordering
        assert sitk_image.GetDimension() == 3

    def test_torch_to_sitk_with_reference(self, create_sitk_image, device):
        """Test converting tensor to SimpleITK with reference image metadata."""
        # Create reference image with specific metadata
        array = np.random.rand(16, 16).astype(np.float32)
        reference = create_sitk_image(array, spacing=(2.0, 2.0), origin=(10.0, 20.0))

        tensor = torch.rand(16, 16, device=device)
        sitk_image = torch_to_sitk(tensor, reference)

        assert sitk_image.GetSpacing() == reference.GetSpacing()
        assert sitk_image.GetOrigin() == reference.GetOrigin()
        assert sitk_image.GetDirection() == reference.GetDirection()

    def test_roundtrip_conversion(self, create_sitk_image, device, tolerance):
        """Test roundtrip conversion between SimpleITK and PyTorch."""
        # Create original array
        array = np.random.rand(16, 32).astype(np.float32)
        original_sitk = create_sitk_image(array)

        # Convert to tensor and back
        tensor = sitk_to_torch(original_sitk)
        reconstructed_sitk = torch_to_sitk(tensor)

        # Check that values are preserved
        original_array = sitk.GetArrayFromImage(original_sitk)
        reconstructed_array = sitk.GetArrayFromImage(reconstructed_sitk)

        assert np.allclose(original_array, reconstructed_array, **tolerance)

    def test_save_load_image_tensor(self, device):
        """Test saving and loading tensor as image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_image.nii.gz")

            # Create test tensor
            original_tensor = torch.rand(16, 32, device=device)

            # Save and load
            save_image(original_tensor, filepath)
            loaded_image = load_image(filepath)
            loaded_tensor = sitk_to_torch(loaded_image)

            assert torch.allclose(original_tensor.cpu(), loaded_tensor, atol=1e-6)

    def test_save_load_image_sitk(self, create_sitk_image):
        """Test saving and loading SimpleITK image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_image.nii.gz")

            # Create test image
            array = np.random.rand(16, 16).astype(np.float32)
            original_image = create_sitk_image(array)

            # Save and load
            save_image(original_image, filepath)
            loaded_image = load_image(filepath)

            # Check that arrays are the same
            original_array = sitk.GetArrayFromImage(original_image)
            loaded_array = sitk.GetArrayFromImage(loaded_image)

            assert np.allclose(original_array, loaded_array, atol=1e-6)


class TestGridOperations:
    """Test grid creation and transformation functions."""

    def test_create_grid_2d(self, device):
        """Test creating 2D coordinate grid."""
        shape = (16, 32)
        grid = create_grid(shape, device=device)

        assert grid.shape == (16, 32, 2)
        assert grid.device == device

        # Check coordinate ranges
        assert torch.allclose(grid[0, 0, :], torch.tensor([-1.0, -1.0], device=device))
        assert torch.allclose(grid[-1, -1, :], torch.tensor([1.0, 1.0], device=device))

    def test_create_grid_3d(self, device):
        """Test creating 3D coordinate grid."""
        shape = (8, 16, 32)
        grid = create_grid(shape, device=device)

        assert grid.shape == (8, 16, 32, 3)
        assert grid.device == device

        # Check coordinate ranges
        assert torch.allclose(
            grid[0, 0, 0, :], torch.tensor([-1.0, -1.0, -1.0], device=device)
        )
        assert torch.allclose(
            grid[-1, -1, -1, :], torch.tensor([1.0, 1.0, 1.0], device=device)
        )

    def test_apply_transform_2d(self, device):
        """Test applying transformation to 2D image."""
        # Create test image
        image = torch.rand(1, 1, 16, 16, device=device)

        # Create identity grid
        grid = create_grid((16, 16), device=device)

        # Apply identity transformation
        transformed = apply_transform(image, grid)

        assert transformed.shape == image.shape
        assert torch.allclose(transformed, image, atol=1e-5)

    def test_apply_transform_3d(self, device):
        """Test applying transformation to 3D image."""
        # Create test image
        image = torch.rand(1, 1, 8, 16, 16, device=device)

        # Create identity grid
        grid = create_grid((8, 16, 16), device=device)

        # Apply identity transformation
        transformed = apply_transform(image, grid)

        assert transformed.shape == image.shape
        assert torch.allclose(transformed, image, atol=1e-5)

    def test_apply_deformation_2d(self, device):
        """Test applying deformation field to 2D image."""
        # Create test image
        image = torch.rand(1, 1, 16, 16, device=device)

        # Create zero deformation (should preserve image)
        deformation = torch.zeros(1, 16, 16, 2, device=device)

        deformed = apply_deformation(image, deformation)

        assert deformed.shape == image.shape
        assert torch.allclose(deformed, image, atol=1e-5)

    def test_apply_deformation_with_translation(self, device):
        """Test applying translation deformation."""
        # Create test image with a pattern
        image = torch.zeros(1, 1, 16, 16, device=device)
        image[:, :, 4:8, 4:8] = 1.0  # Square pattern

        # Create constant translation deformation
        deformation = torch.zeros(1, 16, 16, 2, device=device)
        deformation[:, :, :, 0] = 0.25  # Translate right

        deformed = apply_deformation(image, deformation)

        assert deformed.shape == image.shape
        # Pattern should have moved (not exactly equal)
        assert not torch.allclose(deformed, image)


class TestTransformOperations:
    """Test transformation utility functions."""

    def test_create_identity_transform_2d(self, device):
        """Test creating 2D identity transform."""
        transform = create_identity_transform((16, 16), device=device)

        expected = torch.eye(2, 3, device=device)
        assert torch.allclose(transform, expected)

    def test_create_identity_transform_3d(self, device):
        """Test creating 3D identity transform."""
        transform = create_identity_transform((8, 16, 16), device=device)

        expected = torch.eye(3, 4, device=device)
        assert torch.allclose(transform, expected)

    def test_compose_transforms_2d(self, device, create_affine_transform_2d):
        """Test composing 2D affine transforms."""
        # Create two translation transforms
        t1 = create_affine_transform_2d(tx=0.1, ty=0.0)
        t2 = create_affine_transform_2d(tx=0.2, ty=0.0)

        # Compose transforms
        composed = compose_transforms(t1, t2)

        # Should result in combined translation
        expected = create_affine_transform_2d(tx=0.3, ty=0.0)
        assert torch.allclose(composed, expected, atol=1e-5)

    def test_compose_transforms_3d(self, device, create_affine_transform_3d):
        """Test composing 3D affine transforms."""
        # Create two translation transforms
        t1 = create_affine_transform_3d(tx=0.1, ty=0.0, tz=0.0)
        t2 = create_affine_transform_3d(tx=0.2, ty=0.0, tz=0.0)

        # Compose transforms
        composed = compose_transforms(t1, t2)

        # Translation components should be combined
        assert abs(composed[0, 3].item() - 0.3) < 1e-5

    def test_compose_transforms_invalid_shape(self, device):
        """Test composing transforms with invalid shapes."""
        t1 = torch.rand(2, 2, device=device)  # Invalid shape
        t2 = torch.rand(2, 3, device=device)

        with pytest.raises(ValueError, match="Unsupported transformation matrix shape"):
            compose_transforms(t1, t2)


class TestTargetRegistrationError:
    """Test Target Registration Error computation."""

    def test_tre_identical_landmarks(self, device, tolerance):
        """Test TRE with identical landmarks."""
        landmarks = torch.rand(10, 2, device=device)

        tre = compute_target_registration_error(landmarks, landmarks)

        assert abs(tre) < tolerance["atol"]

    def test_tre_with_affine_transform(self, device, create_affine_transform_2d):
        """Test TRE with affine transformation."""
        # Create landmarks
        landmarks_fixed = torch.rand(5, 2, device=device)

        # Apply known transformation to create moving landmarks
        transform = create_affine_transform_2d(tx=0.1, ty=0.05)

        # Create moving landmarks by applying the forward transform
        homo_landmarks = torch.cat(
            [landmarks_fixed, torch.ones(landmarks_fixed.shape[0], 1, device=device)],
            dim=1,
        )
        landmarks_moving = torch.matmul(homo_landmarks, transform.T)

        # The registration transform should be the inverse to map moving back to fixed
        # For the TRE computation, we use the inverse transform
        # Create inverse transform
        A = transform[:, :2]  # 2x2 rotation/scale part
        t = transform[:, 2]  # translation part
        A_inv = torch.inverse(A)
        t_inv = -torch.matmul(A_inv, t)
        inverse_transform = torch.cat([A_inv, t_inv.unsqueeze(1)], dim=1)

        # Compute TRE (should be close to zero when using the correct inverse transform)
        tre = compute_target_registration_error(
            landmarks_fixed, landmarks_moving, transform=inverse_transform
        )

        assert tre < 0.01  # Should be very small

    def test_tre_with_deformation(self, device):
        """Test TRE with deformation field."""
        landmarks_fixed = torch.rand(5, 2, device=device)

        # Create small deformation
        deformation = torch.rand(5, 2, device=device) * 0.05
        landmarks_moving = landmarks_fixed + deformation

        # Compute TRE
        tre = compute_target_registration_error(
            landmarks_fixed, landmarks_moving, deformation=deformation
        )

        assert tre < 0.1

    def test_tre_no_transform(self, device):
        """Test TRE without any transformation."""
        landmarks_fixed = torch.rand(5, 2, device=device)
        landmarks_moving = landmarks_fixed.clone()
        landmarks_moving[:, 0] += 0.1  # Offset only in x direction

        tre = compute_target_registration_error(landmarks_fixed, landmarks_moving)

        assert abs(tre - 0.1) < 0.01  # Should equal the offset


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_full_pipeline_2d(
        self, device, create_test_image_2d, create_affine_transform_2d
    ):
        """Test full 2D image processing pipeline."""
        # Create test image
        original = create_test_image_2d()

        # Convert to SimpleITK and back
        sitk_image = torch_to_sitk(original)
        tensor_image = sitk_to_torch(sitk_image)

        # Apply transformation
        transform = create_affine_transform_2d(tx=0.1, rotation=0.05)
        grid = create_grid(original.shape, device=device)

        # Apply transformation to grid (manually)
        homo_grid = torch.cat(
            [grid, torch.ones(*grid.shape[:-1], 1, device=device)], dim=-1
        )
        transformed_grid = torch.matmul(homo_grid, transform.T)

        # Transform image
        transformed_image = apply_transform(
            tensor_image.unsqueeze(0).unsqueeze(0), transformed_grid
        ).squeeze()

        # Simple min-max normalization for testing
        min_val = transformed_image.min()
        max_val = transformed_image.max()
        normalized = (transformed_image - min_val) / (max_val - min_val + 1e-8)

        assert normalized.shape == original.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_error_handling(self, device):
        """Test error handling in utility functions."""
        # Test invalid grid shape
        with pytest.raises(ValueError):
            create_grid((10,), device=device)  # 1D not supported

        # Test invalid transform shape
        with pytest.raises(ValueError):
            image = torch.rand(1, 1, 8, 8, device=device)
            invalid_grid = torch.rand(8, 8, 3, device=device)  # Wrong last dim for 2D
            apply_transform(image, invalid_grid)

    def test_numerical_stability(self, device):
        """Test numerical stability of utility functions."""
        # Test with very small values
        small_tensor = torch.ones(10, 10, device=device) * 1e-10

        # Simple min-max normalization for testing numerical stability
        min_val = small_tensor.min()
        max_val = small_tensor.max()
        normalized = (small_tensor - min_val) / (max_val - min_val + 1e-8)
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

        # Test gradient computation with constant image
        constant_image = torch.ones(1, 1, 8, 8, device=device)
        gradient = compute_gradient(constant_image)

        # Gradient of constant should be zero
        assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-6)
