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
        transform = AffineTransform(ndim=2).to(device)

        expected = torch.eye(2, 3, device=device)
        assert torch.allclose(transform.matrix, expected)

    def test_3d_identity_initialization(self, device):
        """Test 3D identity transform initialization."""
        transform = AffineTransform(ndim=3).to(device)

        expected = torch.eye(3, 4, device=device)
        assert torch.allclose(transform.matrix, expected)

    def test_2d_transform_application(self, device, tolerance):
        """Test applying 2D transformation to grid."""
        transform = AffineTransform(ndim=2).to(device)

        # Create small image
        image = torch.rand(10, 20, device=device).unsqueeze(0).unsqueeze(0)

        # Apply identity transform
        transformed = transform(image)

        assert torch.allclose(image, transformed, **tolerance)

    def test_3d_transform_application(self, device, tolerance):
        """Test applying 3D transformation to grid."""
        transform = AffineTransform(ndim=3).to(device)

        # Create small grid
        image = torch.rand(4, 4, 4, device=device).unsqueeze(0).unsqueeze(0)

        # Apply identity transform
        transformed = transform(image)

        assert torch.allclose(image, transformed, **tolerance)

    def test_translation_transform_2d(self, device, create_affine_transform_2d):
        """Test 2D translation transformation."""
        # Create translation transform
        tx, ty = 0.2, -0.1
        matrix = create_affine_transform_2d(tx=tx, ty=ty)

        transform = AffineTransform(ndim=2, init_matrix=matrix).to(device)

        # Create point at origin
        origin = torch.tensor([[0.0, 0.0]], device=device)

        # Apply transformation
        transformed = transform(origin)

        expected = torch.tensor([[tx, ty]], device=device)
        assert torch.allclose(transformed, expected, atol=1e-4)

    def test_get_set_matrix(self, device, create_affine_transform_2d):
        """Test getting and setting transformation matrix."""
        transform = AffineTransform(ndim=2).to(device)

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
            shrink_factors=[4, 2],
            smoothing_sigmas=[2.0, 1.0],
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
            shrink_factors=[4, 2],
            smoothing_sigmas=[2.0, 1.0],
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
        reg = AffineRegistration(
            similarity_metric=mse,
            shrink_factors=[4, 2, 1],
            smoothing_sigmas=[2.0, 1.0, 0.0],
        )

        # Create test image
        image = (
            create_test_image_2d().unsqueeze(0).unsqueeze(0)
        )  # Add batch and channel dims

        pyramid = reg._create_pyramid(image)

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
        reg = AffineRegistration(
            similarity_metric=mse, shrink_factors=[4, 2], smoothing_sigmas=[2.0, 1.0]
        )

        # Create test image
        image = (
            create_test_image_3d().unsqueeze(0).unsqueeze(0)
        )  # Add batch and channel dims

        pyramid = reg._create_pyramid(image)

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
        transform = AffineTransform(ndim=2).to(device)
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
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
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
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
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
        transform = AffineTransform(ndim=2, init_matrix=transform_matrix).to(device)
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
            similarity_metric=mse,
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
            num_iterations=[5],
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
            similarity_metric=mse,
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
            num_iterations=[5],
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
            similarity_metric=mse,
            shrink_factors=[4, 2, 1],
            smoothing_sigmas=[2.0, 1.0, 0.0],
            num_iterations=[5, 5, 5],
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
            shrink_factors=[4, 2],
            smoothing_sigmas=[2.0, 1.0],
            num_iterations=[50, 100],
            learning_rate=0.01,
        )

        # Create fixed image
        fixed = create_test_image_2d(noise_level=0.01)

        # Create moving image with known transformation
        true_transform = create_affine_transform_2d(tx=0.15, ty=-0.1, rotation=0.05)

        # Apply transformation
        grid = create_grid(fixed.shape, device=device)
        transform = AffineTransform(ndim=2, init_matrix=true_transform).to(device)
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
                similarity_metric=metric,
                shrink_factors=[1],
                smoothing_sigmas=[0.0],
                num_iterations=[10],
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
                similarity_metric=metric,
                shrink_factors=[1],
                smoothing_sigmas=[0.0],
                num_iterations=[10],
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
            shrink_factors=[4, 2],
            smoothing_sigmas=[2.0, 1.0],
            num_iterations=[30, 50],
            regularization_weight=0.1,
        )

        # Create images with noise
        fixed = create_test_image_2d(noise_level=0.1)

        # Create moving with known transform + noise
        true_transform = create_affine_transform_2d(tx=0.1, ty=0.05)
        grid = create_grid(fixed.shape, device=device)
        transform = AffineTransform(ndim=2, init_matrix=true_transform).to(device)
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

    def test_registration_with_anisotropic_spacing(
        self, device, create_test_image_2d, create_affine_transform_2d
    ):
        """Test affine registration with anisotropic voxel spacing."""
        from torchregister.io import sitk_to_torch, torch_to_sitk

        mse = MSE()
        reg = AffineRegistration(
            similarity_metric=mse,
            shrink_factors=[2, 1],
            smoothing_sigmas=[1.0, 0.0],
            num_iterations=[50, 100],
            learning_rate=0.01,
        )

        # Create fixed image with anisotropic spacing
        fixed_tensor = create_test_image_2d(shape=(64, 128), noise_level=0.01)

        # Convert to SimpleITK with anisotropic spacing (2x difference)
        # Higher resolution in y-direction (height), lower in x-direction (width)
        fixed_sitk = torch_to_sitk(fixed_tensor)
        anisotropic_spacing = [2.0, 1.0]  # [x_spacing, y_spacing] - coarser in x
        fixed_sitk.SetSpacing(anisotropic_spacing)

        # Create moving image with known transformation in PHYSICAL coordinates
        # A translation of 4.0mm in x and 2.0mm in y should correspond to:
        # - 2 pixels in x direction (4.0mm / 2.0mm/pixel)
        # - 2 pixels in y direction (2.0mm / 1.0mm/pixel)
        true_transform = create_affine_transform_2d(
            tx=0.0625, ty=0.03125, rotation=0.0
        )  # Normalized coordinates

        # Apply transformation to create moving image
        grid = create_grid(fixed_tensor.shape, device=device)
        transform = AffineTransform(ndim=2, init_matrix=true_transform).to(device)
        transformed_grid = transform(grid)
        moving_tensor = apply_transform(
            fixed_tensor.unsqueeze(0).unsqueeze(0), transformed_grid
        ).squeeze()

        # Convert moving to SimpleITK with same spacing
        moving_sitk = torch_to_sitk(moving_tensor)
        moving_sitk.SetSpacing(anisotropic_spacing)

        # Register using SimpleITK images (spacing-aware)
        estimated_transform, registered_tensor = reg.register(fixed_sitk, moving_sitk)

        # Convert registered result back to SimpleITK format
        registered = torch_to_sitk(registered_tensor, reference_image=fixed_sitk)

        # Verify the estimated transformation accounts for anisotropic spacing
        # The registration should find a transformation that compensates for the spacing
        estimated_tx = estimated_transform[0, 2].item()
        estimated_ty = estimated_transform[1, 2].item()

        # Expected translations in normalized coordinates given the spacing difference
        expected_tx = -0.0625  # Inverse of applied transform
        expected_ty = -0.03125  # Inverse of applied transform

        # Allow for some registration error, but should be close
        assert abs(estimated_tx - expected_tx) < 1e-3, (
            f"X translation error too large: {estimated_tx} vs {expected_tx}"
        )
        assert abs(estimated_ty - expected_ty) < 1e-3, (
            f"Y translation error too large: {estimated_ty} vs {expected_ty}"
        )

        # Verify that registered image has better similarity than original moving
        final_mse = torch.mean((sitk_to_torch(fixed_sitk) - registered_tensor) ** 2)
        initial_mse = torch.mean(
            (sitk_to_torch(fixed_sitk) - sitk_to_torch(moving_sitk)) ** 2
        )

        assert final_mse < initial_mse, "Registration should improve image similarity"

        # Test that the spacing information is preserved in output
        assert registered.GetSpacing() == fixed_sitk.GetSpacing(), (
            "Output spacing should match fixed image"
        )
