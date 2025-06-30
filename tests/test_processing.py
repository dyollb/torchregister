"""
Tests for image processing utilities.
"""

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from torchregister.processing import gaussian_blur, normalize_image, resample_image
from torchregister.transforms import compute_gradient


class TestGaussianBlur:
    """Test the Gaussian blur function."""

    def test_gaussian_blur_2d_shape_preservation(self):
        """Test that 2D Gaussian blur preserves tensor shape."""
        image = torch.randn(2, 3, 32, 32)  # [B, C, H, W]
        blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)

        assert blurred.shape == image.shape
        assert blurred.dtype == image.dtype
        assert blurred.device == image.device

    def test_gaussian_blur_3d_shape_preservation(self):
        """Test that 3D Gaussian blur preserves tensor shape."""
        image = torch.randn(1, 2, 16, 32, 32)  # [B, C, D, H, W]
        blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)

        assert blurred.shape == image.shape
        assert blurred.dtype == image.dtype
        assert blurred.device == image.device

    def test_gaussian_blur_smoothing_effect(self):
        """Test that Gaussian blur actually smooths the image."""
        # Create a noisy image
        torch.manual_seed(42)
        image = torch.randn(1, 1, 64, 64)

        # Apply blur
        blurred = gaussian_blur(image, kernel_size=7, sigma=2.0)

        # Blurred image should have lower variance (less noise)
        original_var = torch.var(image)
        blurred_var = torch.var(blurred)

        assert blurred_var < original_var, "Blurred image should have lower variance"

    def test_gaussian_blur_different_sigmas(self):
        """Test that larger sigma produces more blurring."""
        image = torch.randn(1, 1, 32, 32)

        blur_small = gaussian_blur(image, kernel_size=5, sigma=0.5)
        blur_large = gaussian_blur(image, kernel_size=9, sigma=2.0)

        # Larger sigma should produce more smoothing (lower variance)
        var_small = torch.var(blur_small)
        var_large = torch.var(blur_large)

        assert var_large < var_small, "Larger sigma should produce more smoothing"

    def test_gaussian_blur_kernel_size_odd(self):
        """Test that function works with odd kernel sizes."""
        image = torch.randn(1, 1, 16, 16)

        # These should work without error
        gaussian_blur(image, kernel_size=3, sigma=0.5)
        gaussian_blur(image, kernel_size=5, sigma=1.0)
        gaussian_blur(image, kernel_size=7, sigma=1.5)

    def test_gaussian_blur_single_channel(self):
        """Test Gaussian blur with single channel."""
        image = torch.randn(1, 1, 32, 32)
        blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)

        assert blurred.shape == image.shape

    def test_gaussian_blur_multi_channel(self):
        """Test Gaussian blur with multiple channels."""
        image = torch.randn(1, 5, 32, 32)  # 5 channels
        blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)

        assert blurred.shape == image.shape

    def test_gaussian_blur_batch_processing(self):
        """Test Gaussian blur with batch dimension."""
        image = torch.randn(4, 2, 32, 32)  # Batch of 4
        blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)

        assert blurred.shape == image.shape

    def test_gaussian_blur_3d_multi_batch_channel(self):
        """Test 3D Gaussian blur with multiple batches and channels."""
        image = torch.randn(2, 3, 8, 16, 16)  # [B=2, C=3, D=8, H=16, W=16]
        blurred = gaussian_blur(image, kernel_size=3, sigma=0.8)

        assert blurred.shape == image.shape

    def test_gaussian_blur_gradient_flow(self):
        """Test that gradients can flow through Gaussian blur."""
        image = torch.randn(1, 1, 16, 16, requires_grad=True)

        blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)
        loss = torch.sum(blurred)
        loss.backward()

        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_gaussian_blur_device_compatibility(self, device):
        """Test Gaussian blur works on different devices."""
        image = torch.randn(1, 1, 16, 16, device=device)
        blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)

        assert blurred.device == image.device
        assert blurred.shape == image.shape

    def test_gaussian_blur_zero_sigma(self):
        """Test behavior with zero sigma (should return similar image)."""
        image = torch.randn(1, 1, 16, 16)

        # Very small sigma should produce minimal change
        blurred = gaussian_blur(image, kernel_size=3, sigma=0.1)

        # Should be very similar to original
        diff = torch.mean(torch.abs(image - blurred))
        assert diff < 0.1, "Very small sigma should produce minimal change"


class TestImageResampling:
    """Test image resampling functions."""

    @pytest.fixture
    def create_sitk_image(self):
        """Create a SimpleITK image for testing."""

        def _create_image(array, spacing=(1.0, 1.0), origin=(0.0, 0.0)):
            image = sitk.GetImageFromArray(array)
            image.SetSpacing(spacing)
            image.SetOrigin(origin)
            return image

        return _create_image

    def test_resample_by_spacing(self, create_sitk_image):
        """Test resampling image by changing spacing."""
        # Create test image with known spacing
        array = np.random.rand(32, 32).astype(np.float32)
        original = create_sitk_image(array, spacing=(1.0, 1.0))

        # Resample to half spacing (double resolution)
        resampled = resample_image(original, new_spacing=(0.5, 0.5))

        assert resampled.GetSpacing() == (0.5, 0.5)
        # Size should approximately double
        assert resampled.GetSize()[0] >= 60  # Should be around 64
        assert resampled.GetSize()[1] >= 60

    def test_resample_by_size(self, create_sitk_image):
        """Test resampling image by changing size."""
        array = np.random.rand(32, 32).astype(np.float32)
        original = create_sitk_image(array, spacing=(1.0, 1.0))

        # Resample to specific size
        new_size = (64, 64)
        resampled = resample_image(original, new_size=new_size)

        assert resampled.GetSize() == new_size
        # Spacing should be adjusted accordingly
        expected_spacing = (32 / 64, 32 / 64)  # Original size / new size
        assert abs(resampled.GetSpacing()[0] - expected_spacing[0]) < 0.1
        assert abs(resampled.GetSpacing()[1] - expected_spacing[1]) < 0.1

    def test_resample_error_no_parameters(self, create_sitk_image):
        """Test that resampling without parameters raises error."""
        array = np.random.rand(16, 16).astype(np.float32)
        image = create_sitk_image(array)

        with pytest.raises(
            ValueError, match="Either new_spacing or new_size must be provided"
        ):
            resample_image(image)


class TestImageProcessing:
    """Test image processing utility functions."""

    @pytest.fixture
    def device(self):
        """Device fixture for testing."""
        return torch.device("cpu")

    def test_normalize_minmax(self, device):
        """Test min-max normalization."""
        # Create test tensor with known range
        tensor = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device=device)

        normalized = normalize_image(tensor, method="minmax")

        assert torch.allclose(normalized.min(), torch.tensor(0.0, device=device))
        assert torch.allclose(normalized.max(), torch.tensor(1.0, device=device))

    def test_normalize_zscore(self, device):
        """Test z-score normalization."""
        # Create test tensor
        tensor = torch.randn(100, device=device) * 5 + 10  # mean=10, std≈5

        normalized = normalize_image(tensor, method="zscore")

        # Should have approximately zero mean and unit std
        assert abs(normalized.mean().item()) < 0.1
        assert abs(normalized.std().item() - 1.0) < 0.1

    def test_normalize_invalid_method(self, device):
        """Test normalization with invalid method."""
        tensor = torch.rand(10, device=device)

        with pytest.raises(ValueError, match="Unsupported normalization method"):
            normalize_image(tensor, method="invalid")

    def test_compute_gradient_2d(self, device):
        """Test computing 2D image gradient."""
        # Create test image with linear gradient
        image = torch.zeros(1, 1, 8, 8, device=device)
        for i in range(8):
            image[:, :, :, i] = i  # Linear gradient in x direction

        gradient = compute_gradient(image)

        assert gradient.shape == (1, 2, 8, 8)  # [B, 2, H, W] for 2D gradients

        # x-gradient should be approximately constant (≈1)
        x_grad = gradient[:, 0, :, :]
        assert torch.allclose(
            x_grad[:, :, :-1], torch.ones_like(x_grad[:, :, :-1]), atol=0.1
        )

    def test_compute_gradient_3d(self, device):
        """Test computing 3D image gradient."""
        image = torch.rand(1, 1, 4, 8, 8, device=device)

        gradient = compute_gradient(image)

        assert gradient.shape == (1, 3, 4, 8, 8)  # [B, 3, D, H, W] for 3D gradients


# Note: normalize_image and resample_image tests moved here from test_utils.py
# to better reflect the submodule organization
