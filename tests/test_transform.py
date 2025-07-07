"""
Tests for SimpleITK transform conversion utilities.
"""

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from torchregister import AffineRegistration
from torchregister.metrics import MSE
from torchregister.transforms import (
    torch_affine_to_sitk_transform,
)


class TestAffineTransformConversion:
    """Test conversion between PyTorch affine matrices and SimpleITK AffineTransforms."""

    def test_2d_identity_conversion(self, device, create_sitk_image):
        """Test 2D identity transform conversion."""
        # Create test images
        array = np.random.rand(64, 64).astype(np.float32)
        fixed_image = create_sitk_image(array, spacing=(1.0, 1.0), origin=(0.0, 0.0))
        moving_image = create_sitk_image(array, spacing=(1.0, 1.0), origin=(0.0, 0.0))

        # Create 2D identity matrix
        identity_2d = torch.eye(2, 3)

        # Convert to SimpleITK
        sitk_transform = torch_affine_to_sitk_transform(
            identity_2d, fixed_image, moving_image
        )

        linear = torch.Tensor(sitk_transform.GetMatrix()).reshape(2, 2)
        translation = torch.Tensor(sitk_transform.GetTranslation())

        # Verify conversion (should be close to identity)
        torch.testing.assert_close(linear, identity_2d[:, :2], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(translation, identity_2d[:, 2], atol=1e-5, rtol=1e-5)

    def test_3d_identity_conversion(self, device, create_sitk_image):
        """Test 3D identity transform conversion."""
        # Create test images
        array = np.random.rand(32, 32, 32).astype(np.float32)
        fixed_image = create_sitk_image(
            array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
        )
        moving_image = create_sitk_image(
            array, spacing=(1.2, 1.1, 1.0), origin=(10.0, 20.0, 30.0)
        )

        # Create 3D identity matrix
        identity_3d = torch.eye(3, 4, device=device)

        # Convert to SimpleITK
        sitk_transform = torch_affine_to_sitk_transform(
            identity_3d, fixed_image, moving_image
        )

        linear = torch.Tensor(sitk_transform.GetMatrix()).reshape(3, 3)
        translation = torch.Tensor(sitk_transform.GetTranslation())

        # Verify conversion (contains scaling+translation from fixed to moving)
        expected_linear = torch.diag(torch.Tensor([1.2, 1.1, 1.0]))
        expected_translation = torch.Tensor([10.0, 20.0, 30.0])
        torch.testing.assert_close(linear, expected_linear, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            translation, expected_translation, atol=1e-5, rtol=1e-5
        )

    def test_2d_translation_conversion(self, device, create_sitk_image):
        """Test 2D translation transform conversion."""
        # Create test images
        array = np.random.rand(64, 64).astype(np.float32)
        fixed_image = create_sitk_image(array, spacing=(1.0, 1.0), origin=(0.0, 0.0))
        moving_image = create_sitk_image(array, spacing=(1.0, 1.0), origin=(0.0, 0.0))

        # Create 2D translation matrix (small translation in normalized coordinates)
        translation_matrix = torch.tensor(
            [[1.0, 0.0, 0.1], [0.0, 1.0, -0.05]], device=device
        )

        # Convert to SimpleITK
        sitk_transform = torch_affine_to_sitk_transform(
            translation_matrix, fixed_image, moving_image
        )

        linear = torch.Tensor(sitk_transform.GetMatrix()).reshape(2, 2)
        translation = torch.Tensor(sitk_transform.GetTranslation())

        size = torch.Tensor(moving_image.GetSize())
        expected_translation = (size - 1) * translation_matrix[:, 2].cpu() / 2.0

        torch.testing.assert_close(linear, torch.eye(2, 2), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            translation, expected_translation, atol=1e-5, rtol=1e-5
        )

    def test_2d_rotation_conversion(self, device, create_sitk_image):
        """Test 2D rotation transform conversion."""
        # Create test images
        array = np.random.rand(64, 64)
        fixed_image = create_sitk_image(array, spacing=(1.0, 1.0), origin=(0.0, 0.0))
        moving_image = create_sitk_image(array, spacing=(1.0, 1.0), origin=(0.0, 0.0))

        # Create 2D rotation matrix (small rotation)
        angle = np.pi / 8  # 22.5 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            device=device,
            dtype=torch.float32,
        )

        # Convert to SimpleITK
        sitk_transform = torch_affine_to_sitk_transform(
            rotation_matrix, fixed_image, moving_image
        )

        linear = torch.Tensor(sitk_transform.GetMatrix()).reshape(2, 2)
        # TODO: test translation

        torch.testing.assert_close(
            linear, rotation_matrix[:, :2].cpu(), atol=1e-5, rtol=1e-5
        )

    def test_invalid_matrix_shape(self, device, create_sitk_image):
        """Test error handling for invalid matrix shapes."""
        # Create test images
        array = np.random.rand(64, 64).astype(np.float32)
        fixed_image = create_sitk_image(array, spacing=(1.0, 1.0), origin=(0.0, 0.0))
        moving_image = create_sitk_image(array, spacing=(1.0, 1.0), origin=(0.0, 0.0))

        # Test invalid 2D matrix
        invalid_matrix = torch.rand(3, 3, device=device)

        with pytest.raises(ValueError, match="Unsupported affine matrix shape"):
            torch_affine_to_sitk_transform(invalid_matrix, fixed_image, moving_image)


class TestTransformIntegration:
    """Test integration of transform conversion with registration results."""

    def test_affine_registration_to_sitk(self, device, create_test_image_2d):
        """Test converting affine registration result to SimpleITK."""
        # Create test images
        fixed = create_test_image_2d()
        moving = create_test_image_2d()

        # Create SimpleITK images for coordinate conversion
        fixed_sitk = sitk.Image([64, 64], sitk.sitkFloat32)
        fixed_sitk.SetSpacing([1.0, 1.0])
        fixed_sitk.SetOrigin([0.0, 0.0])

        moving_sitk = sitk.Image([64, 64], sitk.sitkFloat32)
        moving_sitk.SetSpacing([1.0, 1.0])
        moving_sitk.SetOrigin([0.0, 0.0])

        # Run registration
        mse = MSE()
        reg = AffineRegistration(
            similarity_metric=mse, num_iterations=[5]
        )  # Quick test
        transform_matrix, _ = reg.register(fixed, moving)

        # Convert to SimpleITK
        sitk_transform = torch_affine_to_sitk_transform(
            transform_matrix, fixed_sitk, moving_sitk
        )

        # Verify we get a valid transform
        assert isinstance(sitk_transform, sitk.AffineTransform)
        assert sitk_transform.GetDimension() == 2
