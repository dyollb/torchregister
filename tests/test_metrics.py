"""
Tests for metrics module (loss functions).
"""

import pytest
import torch

from torchregister.metrics import (
    LNCC,
    MSE,
    NCC,
    CombinedLoss,
    Dice,
    MattesMI,
    RegistrationLoss,
)


class TestBaseLoss:
    """Test the RegistrationLoss class."""

    def test_base_loss_is_abstract(self):
        """Test that RegistrationLoss is abstract and requires forward implementation."""
        base_loss = RegistrationLoss()
        with pytest.raises(
            NotImplementedError, match="Subclasses must implement the forward method"
        ):
            base_loss(torch.randn(1, 1, 10, 10), torch.randn(1, 1, 10, 10))


class TestNCC:
    """Test Normalized Cross-Correlation loss."""

    def test_ncc_identical_images(self, device, tolerance):
        """Test NCC with identical images."""
        ncc = NCC()

        # Create test images
        image = torch.randn(1, 1, 32, 32, device=device)

        # NCC of identical images should be -1 (perfect correlation)
        loss = ncc(image, image)
        assert torch.allclose(loss, torch.tensor(-1.0, device=device), **tolerance)

    def test_ncc_uncorrelated_images(self, device):
        """Test NCC with uncorrelated images."""
        ncc = NCC()

        # Create uncorrelated test images
        image1 = torch.randn(1, 1, 32, 32, device=device)
        image2 = torch.randn(1, 1, 32, 32, device=device)

        loss = ncc(image1, image2)
        # Should be close to 0 for uncorrelated images
        assert abs(loss.item()) < 1.0

    def test_ncc_batch_processing(self, device):
        """Test NCC with batch of images."""
        ncc = NCC()

        batch_size = 4
        images = torch.randn(batch_size, 1, 16, 16, device=device)

        loss = ncc(images, images)
        assert loss.shape == torch.Size([])  # Should return scalar

    def test_ncc_3d_images(self, device, tolerance):
        """Test NCC with 3D images."""
        ncc = NCC()

        image = torch.randn(1, 1, 8, 16, 16, device=device)

        loss = ncc(image, image)
        assert torch.allclose(loss, torch.tensor(-1.0, device=device), **tolerance)


class TestLNCC:
    """Test Local Normalized Cross-Correlation loss."""

    def test_lncc_identical_images(self, device, tolerance):
        """Test LNCC with identical images."""
        lncc = LNCC(window_size=5)

        image = torch.randn(1, 1, 32, 32, device=device)

        loss = lncc(image, image)
        assert torch.allclose(loss, torch.tensor(-1.0, device=device), **tolerance)

    def test_lncc_window_sizes(self, device):
        """Test LNCC with different window sizes."""
        image1 = torch.randn(1, 1, 32, 32, device=device)
        image2 = torch.randn(1, 1, 32, 32, device=device)

        for window_size in [3, 5, 9]:
            lncc = LNCC(window_size=window_size)
            loss = lncc(image1, image2)
            assert isinstance(loss.item(), float)

    def test_lncc_gradients(self, device):
        """Test that LNCC produces gradients."""
        lncc = LNCC()

        image1 = torch.randn(1, 1, 16, 16, device=device, requires_grad=True)
        image2 = torch.randn(1, 1, 16, 16, device=device)

        loss = lncc(image1, image2)
        loss.backward()

        assert image1.grad is not None
        assert not torch.allclose(image1.grad, torch.zeros_like(image1.grad))


class TestMSE:
    """Test Mean Squared Error loss."""

    def test_mse_identical_images(self, device, tolerance):
        """Test MSE with identical images."""
        mse = MSE()

        image = torch.randn(1, 1, 32, 32, device=device)

        loss = mse(image, image)
        assert torch.allclose(loss, torch.tensor(0.0, device=device), **tolerance)

    def test_mse_different_images(self, device):
        """Test MSE with different images."""
        mse = MSE()

        image1 = torch.ones(1, 1, 16, 16, device=device)
        image2 = torch.zeros(1, 1, 16, 16, device=device)

        loss = mse(image1, image2)
        assert torch.allclose(loss, torch.tensor(1.0, device=device))

    def test_mse_gradients(self, device):
        """Test that MSE produces gradients."""
        mse = MSE()

        image1 = torch.randn(1, 1, 16, 16, device=device, requires_grad=True)
        image2 = torch.randn(1, 1, 16, 16, device=device)

        loss = mse(image1, image2)
        loss.backward()

        assert image1.grad is not None


class TestMattesMI:
    """Test Mattes Mutual Information loss."""

    def test_mattes_mi_identical_images(self, device):
        """Test Mattes MI with identical images."""
        mi = MattesMI(bins=32)

        # Create image with sufficient intensity variation
        image = torch.randn(1, 1, 64, 64, device=device) * 2 + 5

        loss = mi(image, image)
        # Identical images should have high mutual information (negative loss)
        assert loss.item() < 0

    def test_mattes_mi_different_bins(self, device):
        """Test Mattes MI with different bin numbers."""
        image1 = torch.randn(1, 1, 32, 32, device=device)
        image2 = torch.randn(1, 1, 32, 32, device=device)

        for bins in [16, 32, 64]:
            mi = MattesMI(bins=bins)
            loss = mi(image1, image2)
            assert isinstance(loss.item(), float)

    def test_mattes_mi_gradients(self, device):
        """Test that Mattes MI produces gradients."""
        mi = MattesMI(bins=16)

        image1 = torch.randn(1, 1, 16, 16, device=device, requires_grad=True)
        image2 = torch.randn(1, 1, 16, 16, device=device)

        loss = mi(image1, image2)
        loss.backward()

        assert image1.grad is not None

    def test_mattes_mi_constant_images(self, device):
        """Test Mattes MI with constant value images (edge case)."""
        mi = MattesMI(bins=32)

        # Test with both images constant (same value)
        constant_image1 = torch.full((1, 1, 32, 32), 5.0, device=device)
        constant_image2 = torch.full((1, 1, 32, 32), 5.0, device=device)

        loss = mi(constant_image1, constant_image2)
        # Should not produce NaN or infinite values
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # For identical constant images, MI should be very close to 0
        # Allow for small numerical errors due to Parzen windowing and floating point precision
        assert abs(loss.item()) < 1e-6, (
            f"MI should be ~0 for identical images, got {loss.item()}"
        )

        # Test with one constant, one variable
        variable_image = torch.randn(1, 1, 32, 32, device=device) * 2 + 1
        constant_image = torch.full((1, 1, 32, 32), 3.0, device=device)

        loss = mi(constant_image, variable_image)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Test with different constant values
        constant_image1 = torch.full((1, 1, 32, 32), 2.0, device=device)
        constant_image2 = torch.full((1, 1, 32, 32), 8.0, device=device)

        loss = mi(constant_image1, constant_image2)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Test gradients with constant images
        constant_with_grad = torch.full(
            (1, 1, 16, 16), 4.0, device=device, requires_grad=True
        )
        variable_image = torch.randn(1, 1, 16, 16, device=device)

        loss = mi(constant_with_grad, variable_image)
        loss.backward()
        # Gradient should exist (even if it might be zero or small)
        assert constant_with_grad.grad is not None
        assert not torch.isnan(constant_with_grad.grad).any()
        assert not torch.isinf(constant_with_grad.grad).any()


class TestDice:
    """Test Dice coefficient loss."""

    def test_dice_identical_masks(self, device, tolerance):
        """Test Dice with identical binary masks."""
        dice = Dice()

        # Create binary mask
        mask = torch.randint(0, 2, (1, 1, 32, 32), device=device, dtype=torch.float32)

        loss = dice(mask, mask)
        assert torch.allclose(loss, torch.tensor(0.0, device=device), **tolerance)

    def test_dice_no_overlap(self, device, tolerance):
        """Test Dice with no overlap."""
        dice = Dice()

        mask1 = torch.zeros(1, 1, 16, 16, device=device)
        mask1[:, :, :8, :] = 1.0

        mask2 = torch.zeros(1, 1, 16, 16, device=device)
        mask2[:, :, 8:, :] = 1.0

        loss = dice(mask1, mask2)
        assert torch.allclose(loss, torch.tensor(1.0, device=device), **tolerance)

    def test_dice_partial_overlap(self, device):
        """Test Dice with partial overlap."""
        dice = Dice()

        mask1 = torch.zeros(1, 1, 16, 16, device=device)
        mask1[:, :, :12, :] = 1.0

        mask2 = torch.zeros(1, 1, 16, 16, device=device)
        mask2[:, :, 4:, :] = 1.0

        loss = dice(mask1, mask2)
        assert 0.0 < loss.item() < 1.0


class TestCombinedLoss:
    """Test combined loss function."""

    def test_combined_loss_creation(self, device):
        """Test creation of combined loss."""
        losses = {"ncc": NCC(), "mse": MSE()}
        weights = {"ncc": 1.0, "mse": 0.5}

        combined = CombinedLoss(losses, weights)
        assert isinstance(combined, CombinedLoss)
        assert isinstance(combined, RegistrationLoss)

    def test_combined_loss_computation(self, device):
        """Test combined loss computation."""
        losses = {"ncc": NCC(), "mse": MSE()}
        weights = {"ncc": 1.0, "mse": 0.5}

        combined = CombinedLoss(losses, weights)

        image1 = torch.randn(1, 1, 16, 16, device=device)
        image2 = torch.randn(1, 1, 16, 16, device=device)

        loss = combined(image1, image2)
        assert isinstance(loss.item(), float)

    def test_combined_loss_gradients(self, device):
        """Test that combined loss produces gradients."""
        losses = {"ncc": NCC(), "mse": MSE()}
        weights = {"ncc": 1.0, "mse": 0.5}

        combined = CombinedLoss(losses, weights)

        image1 = torch.randn(1, 1, 16, 16, device=device, requires_grad=True)
        image2 = torch.randn(1, 1, 16, 16, device=device)

        loss = combined(image1, image2)
        loss.backward()

        assert image1.grad is not None


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_all_metrics_with_same_input(self, device):
        """Test all metrics with the same input."""
        image1 = torch.randn(1, 1, 32, 32, device=device)
        image2 = torch.randn(1, 1, 32, 32, device=device)

        metrics = [
            NCC(),
            LNCC(window_size=5),
            MSE(),
            MattesMI(bins=32),
        ]

        for metric in metrics:
            loss = metric(image1, image2)
            assert isinstance(loss.item(), float)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_metrics_numerical_stability(self, device):
        """Test metrics numerical stability with extreme values."""
        # Very small values
        image1 = torch.ones(1, 1, 16, 16, device=device) * 1e-8
        image2 = torch.ones(1, 1, 16, 16, device=device) * 1e-8

        metrics = [NCC(), MSE()]

        for metric in metrics:
            loss = metric(image1, image2)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    @pytest.mark.parametrize("metric_class", [NCC, LNCC, MSE, MattesMI, Dice])
    def test_metric_reproducibility(self, metric_class, device, random_seed):
        """Test that metrics give reproducible results."""
        torch.manual_seed(random_seed)

        # Create metric
        if metric_class == LNCC:
            metric = metric_class(window_size=5)
        elif metric_class == MattesMI:
            metric = metric_class(bins=32)
        else:
            metric = metric_class()

        # Create test images
        image1 = torch.randn(1, 1, 16, 16, device=device)
        image2 = torch.randn(1, 1, 16, 16, device=device)

        # Compute loss twice
        loss1 = metric(image1, image2)
        loss2 = metric(image1, image2)

        assert torch.allclose(loss1, loss2)
