"""
Tests for affine registration module.
"""

import pytest
import torch

from torchregister.affine import AffineRegistration, AffineTransform
from torchregister.metrics import MSE, NCC
from torchregister.transforms import apply_transform, create_grid


class TestAffineTransform:
    """Test AffineTransform class."""

    def test_2d_identity_initialization(self, device):
        """Test 2D identity transform initialization."""
        transform = AffineTransform(ndim=2, init_identity=True).to(device)

        expected = torch.eye(2, 3, device=device)
        assert torch.allclose(transform.matrix, expected)

    def test_3d_identity_initialization(self, device):
        """Test 3D identity transform initialization."""
        transform = AffineTransform(ndim=3, init_identity=True).to(device)

        expected = torch.eye(3, 4, device=device)
        assert torch.allclose(transform.matrix, expected)

    def test_2d_random_initialization(self, device):
        """Test 2D random transform initialization."""
        transform = AffineTransform(ndim=2, init_identity=False).to(device)

        assert transform.matrix.shape == (2, 3)
        # Should not be identity matrix
        assert not torch.allclose(transform.matrix, torch.eye(2, 3, device=device))

    def test_2d_transform_application(self, device, tolerance):
        """Test applying 2D transformation to grid."""
        transform = AffineTransform(ndim=2, init_identity=True).to(device)

        # Create small grid
        grid = create_grid((4, 4), device=device)

        # Apply identity transform
        transformed = transform(grid)

        assert torch.allclose(grid, transformed, **tolerance)

    def test_3d_transform_application(self, device, tolerance):
        """Test applying 3D transformation to grid."""
        transform = AffineTransform(ndim=3, init_identity=True).to(device)

        # Create small grid
        grid = create_grid((4, 4, 4), device=device)

        # Apply identity transform
        transformed = transform(grid)

        assert torch.allclose(grid, transformed, **tolerance)

    def test_translation_transform_2d(self, device, create_affine_transform_2d):
        """Test 2D translation transformation."""
        # Create translation transform
        tx, ty = 0.2, -0.1
        matrix = create_affine_transform_2d(tx=tx, ty=ty)

        transform = AffineTransform(ndim=2, init_identity=False).to(device)
        transform.set_matrix(matrix)

        # Create point at origin
        origin = torch.tensor([[0.0, 0.0]], device=device)

        # Apply transformation
        transformed = transform(origin)

        expected = torch.tensor([[tx, ty]], device=device)
        assert torch.allclose(transformed, expected, atol=1e-4)

    def test_get_set_matrix(self, device, create_affine_transform_2d):
        """Test getting and setting transformation matrix."""
        transform = AffineTransform(ndim=2, init_identity=True).to(device)

        # Create new matrix
        new_matrix = create_affine_transform_2d(tx=0.5, ty=-0.3, rotation=0.1)

        # Set matrix
        transform.set_matrix(new_matrix)

        # Get matrix and compare
        retrieved_matrix = transform.get_matrix()
        assert torch.allclose(new_matrix, retrieved_matrix)


class TestAffineRegistration:
    """Test AffineRegistration class."""

    def test_registration_initialization(self):
        """Test registration object initialization."""
        ncc = NCC()
        reg = AffineRegistration(
            similarity_metric=ncc,
            num_scales=2,
            num_iterations=[50, 100],
            learning_rate=0.05,
        )

        assert reg.loss_fn is ncc
        assert reg.num_scales == 2
        assert reg.num_iterations == [50, 100]
        assert reg.learning_rate == 0.05

    def test_registration_initialization_with_metric_instance(self):
        """Test initialization with metric instance."""
        ncc = NCC()
        reg = AffineRegistration(
            similarity_metric=ncc,
            num_scales=2,
            num_iterations=[50, 100],
            learning_rate=0.05,
        )

        assert reg.loss_fn is ncc
        assert reg.num_scales == 2
        assert reg.num_iterations == [50, 100]
        assert reg.learning_rate == 0.05

    def test_invalid_similarity_metric_type(self):
        """Test initialization with invalid similarity metric type."""
        with pytest.raises(
            TypeError, match="similarity_metric must be an instance of RegistrationLoss"
        ):
            AffineRegistration(similarity_metric=123)

    def test_invalid_similarity_metric_string(self):
        """Test initialization with string similarity metric (no longer supported)."""
        with pytest.raises(
            TypeError, match="similarity_metric must be an instance of RegistrationLoss"
        ):
            AffineRegistration(similarity_metric="ncc")

    def test_pyramid_creation_2d(self, device, create_test_image_2d):
        """Test image pyramid creation for 2D images."""
        mse = MSE()
        reg = AffineRegistration(similarity_metric=mse, num_scales=3)

        # Create test image
        image = (
            create_test_image_2d().unsqueeze(0).unsqueeze(0)
        )  # Add batch and channel dims

        pyramid = reg._create_pyramid(image, num_scales=3)

        assert len(pyramid) == 3
        # Check that each level is smaller than the previous
        for i in range(1, len(pyramid)):
            prev_shape = pyramid[i - 1].shape[2:]
            curr_shape = pyramid[i].shape[2:]
            assert all(
                curr >= prev // 2
                for curr, prev in zip(curr_shape, prev_shape, strict=False)
            )

    def test_pyramid_creation_3d(self, device, create_test_image_3d):
        """Test image pyramid creation for 3D images."""
        mse = MSE()
        reg = AffineRegistration(similarity_metric=mse, num_scales=2)

        # Create test image
        image = (
            create_test_image_3d().unsqueeze(0).unsqueeze(0)
        )  # Add batch and channel dims

        pyramid = reg._create_pyramid(image, num_scales=2)

        assert len(pyramid) == 2
        # Check that second level is smaller
        assert all(
            s2 <= s1
            for s1, s2 in zip(pyramid[1].shape[2:], pyramid[0].shape[2:], strict=False)
        )

    def test_regularization_loss(self, device):
        """Test regularization loss computation."""
        mse = MSE()
        reg = AffineRegistration(similarity_metric=mse, regularization_weight=1.0)

        # Create transform with small deviation from identity
        transform = AffineTransform(ndim=2, init_identity=True).to(device)
        transform.matrix.data += 0.1

        reg_loss = reg._regularization_loss(transform)

        assert reg_loss.item() > 0
        assert isinstance(reg_loss.item(), float)

    def test_register_identical_images_2d(
        self, device, create_test_image_2d, tolerance
    ):
        """Test registration of identical 2D images."""
        mse = MSE()
        reg = AffineRegistration(
            similarity_metric=mse,
            num_scales=1,
            num_iterations=[10],  # Few iterations for speed
            learning_rate=0.1,
        )

        # Create identical images
        fixed = create_test_image_2d(noise_level=0.0)
        moving = fixed.clone()

        # Register
        transform_matrix, registered = reg.register(fixed, moving)

        # Should be close to identity
        # TODO: identity = torch.eye(2, 3, device=device)
        assert transform_matrix.shape == (2, 3)

        # Registered image should be similar to fixed
        mse_loss = torch.mean((fixed - registered) ** 2)
        assert mse_loss.item() < 0.1

    def test_register_translated_images_2d(
        self, device, create_test_image_2d, create_affine_transform_2d
    ):
        """Test registration of translated 2D images."""
        mse = MSE()
        reg = AffineRegistration(
            similarity_metric=mse,
            num_scales=1,
            num_iterations=[20],
            learning_rate=0.1,
        )

        # Create fixed image
        fixed = create_test_image_2d(noise_level=0.0)

        # Create moving image with known translation
        tx, ty = 0.1, -0.05
        transform_matrix = create_affine_transform_2d(tx=tx, ty=ty)

        # Apply transformation to create moving image
        grid = create_grid(fixed.shape, device=device)
        transform = AffineTransform(ndim=2, init_identity=False).to(device)
        transform.set_matrix(transform_matrix)
        transformed_grid = transform(grid)
        moving = apply_transform(
            fixed.unsqueeze(0).unsqueeze(0), transformed_grid
        ).squeeze()

        # Register
        estimated_transform, registered = reg.register(fixed, moving)

        # Check that translation is approximately recovered
        estimated_tx = estimated_transform[0, 2].item()
        estimated_ty = estimated_transform[1, 2].item()

        # Should recover approximate translation (allowing for some error)
        assert (
            abs(estimated_tx - (-tx)) < 0.1
        )  # Negative because we're registering moving to fixed
        assert abs(estimated_ty - (-ty)) < 0.1

    def test_register_with_initial_transform(
        self, device, create_test_image_2d, create_affine_transform_2d
    ):
        """Test registration with initial transformation."""
        mse = MSE()
        reg = AffineRegistration(
            similarity_metric=mse, num_scales=1, num_iterations=[5]
        )

        fixed = create_test_image_2d(noise_level=0.0)
        moving = create_test_image_2d(noise_level=0.0)

        # Provide initial transform
        initial_transform = create_affine_transform_2d(tx=0.1, ty=0.1)

        transform_matrix, registered = reg.register(fixed, moving, initial_transform)

        assert transform_matrix.shape == (2, 3)
        assert registered.shape == fixed.shape

    def test_evaluation_metrics(
        self, device, create_test_image_2d, create_affine_transform_2d
    ):
        """Test evaluation of registration quality."""
        mse = MSE()
        reg = AffineRegistration(similarity_metric=mse)

        fixed = create_test_image_2d(noise_level=0.0)
        moving = create_test_image_2d(noise_level=0.0)

        # Use identity transform
        transform_matrix = torch.eye(2, 3, device=device)

        metrics = reg.evaluate(fixed, moving, transform_matrix)

        assert "ncc" in metrics
        assert "mse" in metrics
        assert "transformation_matrix" in metrics
        assert isinstance(metrics["ncc"], float)
        assert isinstance(metrics["mse"], float)

    def test_register_tensor_inputs(self, device, create_test_image_2d):
        """Test registration with PyTorch tensor inputs."""
        mse = MSE()
        reg = AffineRegistration(
            similarity_metric=mse, num_scales=1, num_iterations=[5]
        )

        fixed = create_test_image_2d()
        moving = create_test_image_2d()

        transform_matrix, registered = reg.register(fixed, moving)

        assert isinstance(transform_matrix, torch.Tensor)
        assert isinstance(registered, torch.Tensor)
        assert transform_matrix.shape == (2, 3)

    def test_multi_scale_registration(self, device, create_test_image_2d):
        """Test multi-scale registration."""
        mse = MSE()
        reg = AffineRegistration(
            similarity_metric=mse, num_scales=3, num_iterations=[5, 5, 5]
        )

        fixed = create_test_image_2d()
        moving = create_test_image_2d()

        transform_matrix, registered = reg.register(fixed, moving)

        assert transform_matrix.shape == (2, 3)
        assert registered.shape == fixed.shape


class TestAffineRegistrationIntegration:
    """Integration tests for affine registration."""

    def test_registration_convergence(
        self, device, create_test_image_2d, create_affine_transform_2d
    ):
        """Test that registration converges for simple transformations."""
        mse = MSE()
        reg = AffineRegistration(
            similarity_metric=mse,
            num_scales=2,
            num_iterations=[50, 100],
            learning_rate=0.01,
        )

        # Create fixed image
        fixed = create_test_image_2d(noise_level=0.01)

        # Create moving image with known transformation
        true_transform = create_affine_transform_2d(tx=0.15, ty=-0.1, rotation=0.05)

        # Apply transformation
        grid = create_grid(fixed.shape, device=device)
        transform = AffineTransform(ndim=2, init_identity=False).to(device)
        transform.set_matrix(true_transform)
        transformed_grid = transform(grid)
        moving = apply_transform(
            fixed.unsqueeze(0).unsqueeze(0), transformed_grid
        ).squeeze()

        # Register
        estimated_transform, registered = reg.register(fixed, moving)

        # Check that MSE improved
        initial_mse = torch.mean((fixed - moving) ** 2)
        final_mse = torch.mean((fixed - registered) ** 2)

        assert final_mse < initial_mse

    def test_registration_different_metrics(self, device, create_test_image_2d):
        """Test registration with different similarity metrics."""
        metrics = [NCC(), MSE()]

        fixed = create_test_image_2d(noise_level=0.05)
        moving = create_test_image_2d(noise_level=0.05)

        for metric in metrics:
            reg = AffineRegistration(
                similarity_metric=metric, num_scales=1, num_iterations=[10]
            )

            transform_matrix, registered = reg.register(fixed, moving)

            assert transform_matrix.shape == (2, 3)
            assert registered.shape == fixed.shape

    def test_registration_with_metric_instances(self, device, create_test_image_2d):
        """Test registration with different similarity metric instances."""
        metrics = [NCC(), MSE()]

        fixed = create_test_image_2d(noise_level=0.05)
        moving = create_test_image_2d(noise_level=0.05)

        for metric in metrics:
            reg = AffineRegistration(
                similarity_metric=metric, num_scales=1, num_iterations=[10]
            )

            transform_matrix, registered = reg.register(fixed, moving)

            assert transform_matrix.shape == (2, 3)
            assert registered.shape == fixed.shape

    def test_registration_robustness_to_noise(
        self, device, create_test_image_2d, create_affine_transform_2d
    ):
        """Test registration robustness to noise."""
        ncc = NCC()  # NCC is more robust to noise
        reg = AffineRegistration(
            similarity_metric=ncc,
            num_scales=2,
            num_iterations=[30, 50],
            regularization_weight=0.1,
        )

        # Create images with noise
        fixed = create_test_image_2d(noise_level=0.1)

        # Create moving with known transform + noise
        true_transform = create_affine_transform_2d(tx=0.1, ty=0.05)
        grid = create_grid(fixed.shape, device=device)
        transform = AffineTransform(ndim=2, init_identity=False).to(device)
        transform.set_matrix(true_transform)
        transformed_grid = transform(grid)
        moving = apply_transform(
            fixed.unsqueeze(0).unsqueeze(0), transformed_grid
        ).squeeze()
        moving += torch.randn_like(moving) * 0.1  # Add noise

        # Register
        estimated_transform, registered = reg.register(fixed, moving)

        # Should still recover approximate transformation
        estimated_tx = estimated_transform[0, 2].item()
        estimated_ty = estimated_transform[1, 2].item()

        # Allow for larger error due to noise
        assert abs(estimated_tx - (-0.1)) < 0.2
        assert abs(estimated_ty - (-0.05)) < 0.2
