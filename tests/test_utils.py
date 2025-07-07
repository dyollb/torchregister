"""
Tests for utility functions module.
"""

import os
import tempfile

import numpy as np
import SimpleITK as sitk
import torch

from torchregister.io import (
    load_image,
    save_image,
    sitk_to_torch,
    torch_to_sitk,
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
