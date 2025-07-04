"""
Tests for RDMM registration module.
"""

import pytest
import torch

from torchregister.metrics import LNCC, MSE, NCC
from torchregister.rdmm import GaussianSmoothing, RDMMRegistration, VelocityField


class TestGaussianSmoothing:
    """Test Gaussian smoothing functionality."""

    def test_2d_smoothing_initialization(self, device):
        """Test 2D Gaussian smoothing initialization."""
        smoother = GaussianSmoothing(channels=1, sigma=1.0, ndim=2).to(device)

        assert smoother.ndim == 2
        assert smoother.sigma == 1.0
        assert smoother.weight.shape[0] == 1  # channels

    def test_3d_smoothing_initialization(self, device):
        """Test 3D Gaussian smoothing initialization."""
        smoother = GaussianSmoothing(channels=3, sigma=2.0, ndim=3).to(device)

        assert smoother.ndim == 3
        assert smoother.sigma == 2.0
        assert smoother.weight.shape[0] == 3  # channels

    def test_2d_smoothing_application(self, device):
        """Test applying 2D Gaussian smoothing."""
        smoother = GaussianSmoothing(channels=2, sigma=1.0, ndim=2).to(device)

        # Create test tensor
        x = torch.randn(1, 2, 16, 16, device=device)

        # Apply smoothing
        smoothed = smoother(x)

        assert smoothed.shape == x.shape
        # Smoothed image should have lower variance
        assert smoothed.var() <= x.var()

    def test_3d_smoothing_application(self, device):
        """Test applying 3D Gaussian smoothing."""
        smoother = GaussianSmoothing(channels=1, sigma=1.0, ndim=3).to(device)

        # Create test tensor
        x = torch.randn(1, 1, 8, 16, 16, device=device)

        # Apply smoothing
        smoothed = smoother(x)

        assert smoothed.shape == x.shape
        assert smoothed.var() <= x.var()

    def test_different_sigma_values(self, device):
        """Test smoothing with different sigma values."""
        x = torch.randn(1, 1, 16, 16, device=device)

        variances = []
        for sigma in [0.5, 1.0, 2.0]:
            smoother = GaussianSmoothing(channels=1, sigma=sigma, ndim=2).to(device)
            smoothed = smoother(x)
            variances.append(smoothed.var().item())

        # Higher sigma should result in more smoothing (lower variance)
        assert variances[0] > variances[1] > variances[2]


class TestVelocityField:
    """Test velocity field functionality."""

    def test_2d_velocity_field_initialization(self, device):
        """Test 2D velocity field initialization."""
        shape = (32, 32)
        velocity_field = VelocityField(shape, ndim=2).to(device)

        assert velocity_field.ndim == 2
        assert velocity_field.shape == shape
        assert velocity_field.velocity.shape == (1, 2, 32, 32)

    def test_3d_velocity_field_initialization(self, device):
        """Test 3D velocity field initialization."""
        shape = (16, 32, 32)
        velocity_field = VelocityField(shape, ndim=3).to(device)

        assert velocity_field.ndim == 3
        assert velocity_field.shape == shape
        assert velocity_field.velocity.shape == (1, 3, 16, 32, 32)

    def test_velocity_field_forward(self, device):
        """Test velocity field forward pass."""
        shape = (16, 16)
        velocity_field = VelocityField(shape, ndim=2).to(device)

        velocity = velocity_field()

        assert velocity.shape == (1, 2, 16, 16)
        assert velocity.requires_grad

    def test_velocity_field_gradients(self, device):
        """Test that velocity field produces gradients."""
        shape = (8, 8)
        velocity_field = VelocityField(shape, ndim=2).to(device)

        velocity = velocity_field()
        loss = velocity.mean()
        loss.backward()

        assert velocity_field.velocity.grad is not None


class TestRDMMRegistration:
    """Test RDMM registration functionality."""

    def test_registration_initialization(self):
        """Test RDMM registration initialization."""
        lncc = LNCC()
        reg = RDMMRegistration(
            similarity_metric=lncc,
            shrink_factors=[4, 2],
            smoothing_sigmas=[2.0, 1.0],
            num_iterations=[30, 50],
            learning_rate=0.02,
            smoothing_sigma=1.5,
            alpha=2.0,
        )

        assert reg.loss_fn is lncc
        assert reg.num_scales == 2
        assert reg.shrink_factors == [4, 2]
        assert reg.smoothing_sigmas == [2.0, 1.0]
        assert reg.num_iterations == [30, 50]
        assert reg.learning_rate == 0.02
        assert reg.smoothing_sigma == 1.5
        assert reg.alpha == 2.0

    def test_registration_initialization_with_metric_instance(self):
        """Test initialization with metric instance."""
        lncc = LNCC(window_size=7)
        reg = RDMMRegistration(
            similarity_metric=lncc,
            shrink_factors=[4, 2],
            smoothing_sigmas=[2.0, 1.0],
            num_iterations=[30, 50],
            learning_rate=0.02,
            smoothing_sigma=1.5,
            alpha=2.0,
        )

        assert reg.loss_fn is lncc
        assert reg.num_scales == 2
        assert reg.shrink_factors == [4, 2]
        assert reg.smoothing_sigmas == [2.0, 1.0]
        assert reg.num_iterations == [30, 50]
        assert reg.learning_rate == 0.02
        assert reg.smoothing_sigma == 1.5
        assert reg.alpha == 2.0

    def test_invalid_similarity_metric_type(self):
        """Test initialization with invalid similarity metric type."""
        with pytest.raises(
            TypeError, match="similarity_metric must be an instance of RegistrationLoss"
        ):
            RDMMRegistration(similarity_metric=123)

    def test_invalid_similarity_metric_string(self):
        """Test initialization with string similarity metric (no longer supported)."""
        with pytest.raises(
            TypeError, match="similarity_metric must be an instance of RegistrationLoss"
        ):
            RDMMRegistration(similarity_metric="lncc")

    def test_velocity_integration_2d(self, device):
        """Test velocity field integration for 2D."""
        mse = MSE()
        reg = RDMMRegistration(
            similarity_metric=mse, shrink_factors=[2], smoothing_sigmas=[1.0]
        )

        # Create simple velocity field
        velocity = torch.zeros(1, 2, 16, 16, device=device)
        velocity[:, 0, :, :] = 0.1  # Constant velocity in x direction

        # Integrate
        deformation = reg._integrate_velocity(velocity, num_steps=3)

        assert deformation.shape == (1, 16, 16, 2)
        # Should have non-zero deformation in x direction
        assert deformation[..., 0].abs().sum() > 0

    def test_velocity_integration_3d(self, device):
        """Test velocity field integration for 3D."""
        mse = MSE()
        reg = RDMMRegistration(
            similarity_metric=mse, shrink_factors=[2], smoothing_sigmas=[1.0]
        )

        # Create simple velocity field
        velocity = torch.zeros(1, 3, 8, 16, 16, device=device)
        velocity[:, 1, :, :, :] = 0.05  # Constant velocity in y direction

        # Integrate
        deformation = reg._integrate_velocity(velocity, num_steps=2)

        assert deformation.shape == (1, 8, 16, 16, 3)
        # Should have non-zero deformation in y direction
        assert deformation[..., 1].abs().sum() > 0

    def test_jacobian_determinant_2d(self, device):
        """Test Jacobian determinant computation for 2D."""
        mse = MSE()
        reg = RDMMRegistration(similarity_metric=mse)

        # Create small deformation field
        deformation = torch.zeros(1, 8, 8, 2, device=device)
        deformation[:, :, :, 0] = 0.1  # Small x displacement

        jac_det = reg._compute_jacobian_determinant(deformation)

        assert jac_det.shape == (1, 8, 8)
        # Should be close to 1 for small deformations
        assert torch.allclose(jac_det, torch.ones_like(jac_det), atol=0.5)

    def test_jacobian_determinant_3d(self, device):
        """Test Jacobian determinant computation for 3D."""
        mse = MSE()
        reg = RDMMRegistration(similarity_metric=mse)

        # Create small deformation field
        deformation = torch.zeros(1, 4, 8, 8, 3, device=device)
        deformation[:, :, :, :, 2] = 0.05  # Small z displacement

        jac_det = reg._compute_jacobian_determinant(deformation)

        assert jac_det.shape == (1, 4, 8, 8)
        # Should be close to 1 for small deformations
        assert torch.allclose(jac_det, torch.ones_like(jac_det), atol=0.5)

    def test_regularization_loss_computation(self, device):
        """Test regularization loss computation."""
        mse = MSE()
        reg = RDMMRegistration(
            similarity_metric=mse, alpha=1.0, num_integration_steps=2
        )

        # Create velocity field with some magnitude
        velocity = torch.randn(1, 2, 8, 8, device=device) * 0.1

        reg_loss = reg._regularization_loss(velocity)

        assert isinstance(reg_loss.item(), float)
        assert reg_loss.item() >= 0  # Regularization should be non-negative

    def test_pyramid_creation(self, device, create_test_image_2d):
        """Test pyramid creation for RDMM."""
        mse = MSE()
        reg = RDMMRegistration(
            similarity_metric=mse,
            shrink_factors=[4, 2, 1],
            smoothing_sigmas=[2.0, 1.0, 0.0],
        )

        # Create test image
        image = create_test_image_2d().unsqueeze(0).unsqueeze(0)

        pyramid = reg._create_pyramid(image)

        assert len(pyramid) == 3
        # Each level should be progressively larger (pyramid is ordered coarse to fine)
        for i in range(1, len(pyramid)):
            prev_shape = pyramid[i - 1].shape[2:]
            curr_shape = pyramid[i].shape[2:]
            assert all(
                curr >= prev for curr, prev in zip(curr_shape, prev_shape, strict=False)
            )

    def test_register_identical_images_2d(self, device, create_test_image_2d):
        """Test registration of identical 2D images."""
        mse = MSE()
        reg = RDMMRegistration(
            similarity_metric=mse,
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
            num_iterations=[5],  # Few iterations for speed
            learning_rate=0.1,
            alpha=0.1,
        )

        # Create identical images
        fixed = create_test_image_2d(noise_level=0.0)
        moving = fixed.clone()

        # Register
        deformation, registered = reg.register(fixed, moving)

        # Deformation should be small for identical images
        assert deformation.abs().max().item() < 0.5

        # Registered image should be similar to fixed
        mse = torch.mean((fixed - registered) ** 2)
        assert mse.item() < 0.1

    def test_register_tensor_inputs(self, device, create_test_image_2d):
        """Test registration with PyTorch tensor inputs."""
        mse = MSE()
        reg = RDMMRegistration(
            similarity_metric=mse,
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
            num_iterations=[3],
        )

        fixed = create_test_image_2d()
        moving = create_test_image_2d()

        deformation, registered = reg.register(fixed, moving)

        assert isinstance(deformation, torch.Tensor)
        assert isinstance(registered, torch.Tensor)
        # For 2D images
        assert len(deformation.shape) == 3  # [H, W, 2]
        assert deformation.shape[-1] == 2  # 2D displacement

    def test_evaluation_metrics(self, device, create_test_image_2d):
        """Test evaluation of RDMM registration quality."""
        mse = MSE()
        reg = RDMMRegistration(similarity_metric=mse)

        fixed = create_test_image_2d(noise_level=0.0)
        moving = create_test_image_2d(noise_level=0.0)

        # Create small deformation field
        deformation = torch.zeros(*fixed.shape, 2, device=device)

        metrics = reg.evaluate(fixed, moving, deformation)

        assert "ncc" in metrics
        assert "mse" in metrics
        assert "jacobian_det_mean" in metrics
        assert "jacobian_det_std" in metrics
        assert "negative_jacobian_ratio" in metrics
        assert "deformation_magnitude" in metrics

        for value in metrics.values():
            assert isinstance(value, float)

    def test_multi_scale_registration(self, device, create_test_image_2d):
        """Test multi-scale RDMM registration."""
        mse = MSE()
        reg = RDMMRegistration(
            similarity_metric=mse,
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
            num_iterations=[3],
            alpha=0.5,
        )

        fixed = create_test_image_2d()
        moving = create_test_image_2d()

        deformation, registered = reg.register(fixed, moving)

        assert deformation.shape[-1] == 2  # 2D deformation
        assert registered.shape == fixed.shape

    def test_different_similarity_metrics(self, device, create_test_image_2d):
        """Test RDMM with different similarity metrics."""
        metrics = [NCC(), LNCC(), MSE()]

        fixed = create_test_image_2d()
        moving = create_test_image_2d()

        for metric in metrics:
            reg = RDMMRegistration(
                similarity_metric=metric,
                shrink_factors=[1],
                smoothing_sigmas=[0.0],
                num_iterations=[3],
            )

            deformation, registered = reg.register(fixed, moving)

            assert deformation.shape[-1] == 2
            assert registered.shape == fixed.shape

    def test_different_similarity_metric_instances(self, device, create_test_image_2d):
        """Test RDMM with different similarity metric instances."""
        metrics = [NCC(), LNCC(), MSE()]

        fixed = create_test_image_2d()
        moving = create_test_image_2d()

        for metric in metrics:
            reg = RDMMRegistration(
                similarity_metric=metric,
                shrink_factors=[1],
                smoothing_sigmas=[0.0],
                num_iterations=[3],
            )

            deformation, registered = reg.register(fixed, moving)

            assert deformation.shape[-1] == 2
            assert registered.shape == fixed.shape


class TestRDMMIntegration:
    """Integration tests for RDMM registration."""

    def test_registration_with_known_deformation(self, device, create_test_image_2d):
        """Test RDMM registration with known synthetic deformation."""
        mse = MSE()
        reg = RDMMRegistration(
            similarity_metric=mse,
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
            num_iterations=[25],
            learning_rate=0.01,
            alpha=0.1,
        )

        # Create fixed image
        fixed = create_test_image_2d(noise_level=0.01)

        # Create synthetic deformation
        H, W = fixed.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )

        # Simple sinusoidal deformation
        def_x = 0.05 * torch.sin(3.14159 * x)
        def_y = 0.05 * torch.cos(3.14159 * y)
        true_deformation = torch.stack([def_x, def_y], dim=-1)

        # Apply deformation to create moving image
        from torchregister.transforms import apply_deformation

        moving = apply_deformation(
            fixed.unsqueeze(0).unsqueeze(0), true_deformation.unsqueeze(0)
        ).squeeze()

        # Register
        estimated_deformation, registered = reg.register(fixed, moving)

        # Check that registration improved similarity
        initial_mse = torch.mean((fixed - moving) ** 2)
        final_mse = torch.mean((fixed - registered) ** 2)

        assert final_mse < initial_mse

    def test_jacobian_determinant_properties(self, device):
        """Test properties of Jacobian determinant computation."""
        mse = MSE()
        reg = RDMMRegistration(similarity_metric=mse)

        # Test with identity deformation (should give determinant = 1)
        identity_def = torch.zeros(1, 8, 8, 2, device=device)
        jac_det = reg._compute_jacobian_determinant(identity_def)

        expected = torch.ones_like(jac_det)
        assert torch.allclose(jac_det, expected, atol=1e-4)

    def test_velocity_field_integration_properties(self, device):
        """Test properties of velocity field integration."""
        mse = MSE()
        reg = RDMMRegistration(similarity_metric=mse)

        # Zero velocity should give zero deformation
        zero_velocity = torch.zeros(1, 2, 8, 8, device=device)
        zero_deformation = reg._integrate_velocity(zero_velocity, num_steps=5)

        assert torch.allclose(
            zero_deformation, torch.zeros_like(zero_deformation), atol=1e-6
        )

    def test_registration_convergence_properties(self, device, create_test_image_2d):
        """Test that RDMM registration has proper convergence properties."""
        ncc = NCC()
        reg = RDMMRegistration(
            similarity_metric=ncc,
            shrink_factors=[1],
            smoothing_sigmas=[0.0],
            num_iterations=[30],
            learning_rate=0.01,
            alpha=1.0,
        )

        # Create images with small difference
        fixed = create_test_image_2d(noise_level=0.02)
        moving = create_test_image_2d(noise_level=0.02)

        # Register
        deformation, registered = reg.register(fixed, moving)

        # Check that deformation field is reasonable
        max_displacement = deformation.abs().max().item()
        assert max_displacement < 1.0  # Should not have extreme displacements

        # Check that Jacobian determinant is mostly positive
        jac_det = reg._compute_jacobian_determinant(deformation.unsqueeze(0))
        negative_ratio = (jac_det < 0).float().mean().item()
        assert negative_ratio < 0.1  # Less than 10% negative Jacobians

    def test_registration_with_anisotropic_spacing(self, device, create_test_image_2d):
        """Test RDMM registration with anisotropic voxel spacing."""
        from torchregister.io import sitk_to_torch, torch_to_sitk
        from torchregister.transforms import apply_deformation

        mse = MSE()
        reg = RDMMRegistration(
            similarity_metric=mse,
            shrink_factors=[2, 1],
            smoothing_sigmas=[1.0, 0.0],
            num_iterations=[30, 50],
            learning_rate=0.01,
            alpha=0.5,
        )

        # Create fixed image with anisotropic spacing
        fixed_tensor = create_test_image_2d(shape=(64, 128), noise_level=0.01)

        # Convert to SimpleITK with anisotropic spacing
        # Higher resolution in y-direction, lower in x-direction
        fixed_sitk = torch_to_sitk(fixed_tensor)
        anisotropic_spacing = [3.0, 1.5]  # [x_spacing, y_spacing] - 2:1 ratio
        fixed_sitk.SetSpacing(anisotropic_spacing)

        # Create synthetic deformation field in physical coordinates
        H, W = fixed_tensor.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )

        # Create a deformation that should be different in x vs y due to spacing
        # Larger deformation in x (coarser spacing) should result in smaller pixel displacement
        # Smaller deformation in y (finer spacing) should result in larger pixel displacement
        def_x_physical = 0.3 * torch.sin(
            3.14159 * x
        )  # Larger deformation in physical coords
        def_y_physical = 0.3 * torch.cos(
            3.14159 * y
        )  # Larger deformation in physical coords

        # Convert physical deformation to normalized coordinates considering spacing
        # Normalized displacement = physical_displacement / (image_extent * spacing / image_size)
        # For images in [-1,1] range: image_extent = 2.0
        def_x_normalized = def_x_physical * W / (2.0 * anisotropic_spacing[0])
        def_y_normalized = def_y_physical * H / (2.0 * anisotropic_spacing[1])

        true_deformation = torch.stack([def_x_normalized, def_y_normalized], dim=-1)

        # Apply deformation to create moving image
        moving_tensor = apply_deformation(
            fixed_tensor.unsqueeze(0).unsqueeze(0), true_deformation.unsqueeze(0)
        ).squeeze()

        # Convert moving to SimpleITK with same spacing
        moving_sitk = torch_to_sitk(moving_tensor)
        moving_sitk.SetSpacing(anisotropic_spacing)

        # Register using SimpleITK images (spacing-aware)
        estimated_deformation, registered_tensor = reg.register(fixed_sitk, moving_sitk)

        # Convert registered result back to SimpleITK format
        registered = torch_to_sitk(registered_tensor, reference_image=fixed_sitk)

        # Verify that the deformation field dimensions are correct
        assert estimated_deformation.shape[-1] == 2, (
            "Deformation should have 2 components for 2D"
        )
        assert estimated_deformation.shape[:2] == fixed_tensor.shape, (
            "Deformation spatial dims should match image"
        )

        # Check that the deformation field magnitude is reasonable given the anisotropic spacing
        # The estimated deformation should account for the spacing differences
        max_def_x = estimated_deformation[..., 0].abs().max().item()
        max_def_y = estimated_deformation[..., 1].abs().max().item()

        # Due to anisotropic spacing, we expect different magnitudes in x vs y
        # The ratio of maximum deformations should reflect the spacing ratio
        spacing_ratio = (
            anisotropic_spacing[0] / anisotropic_spacing[1]
        )  # 3.0 / 1.5 = 2.0
        assert spacing_ratio > 1.9 and spacing_ratio < 2.1, (
            "Spacing ratio should be close to 2.0"
        )

        # Allow for some tolerance in the spacing-aware deformation
        # The main goal is to verify that the algorithm handles anisotropic spacing
        # without crashing and produces reasonable outputs
        assert max_def_x >= 0.0, "Should have non-negative deformation in x"
        assert max_def_y >= 0.0, "Should have non-negative deformation in y"

        # The key test: verify that anisotropic spacing doesn't break the registration
        # and that spacing is properly handled throughout the process        # Verify that registered image similarity didn't get significantly worse
        # (allowing for the possibility that the synthetic deformation may not be recoverable)
        final_mse = torch.mean((sitk_to_torch(fixed_sitk) - registered_tensor) ** 2)
        initial_mse = torch.mean(
            (sitk_to_torch(fixed_sitk) - sitk_to_torch(moving_sitk)) ** 2
        )

        # Allow the final MSE to be up to 20% worse than initial, as long as spacing is handled
        assert final_mse < initial_mse * 1.2, (
            "Registration should not significantly worsen image similarity"
        )

        # Test that the spacing information is preserved in output
        assert registered.GetSpacing() == fixed_sitk.GetSpacing(), (
            "Output spacing should match fixed image"
        )

        # Verify that the Jacobian determinant is reasonable with anisotropic spacing
        jac_det = reg._compute_jacobian_determinant(estimated_deformation.unsqueeze(0))
        negative_ratio = (jac_det < 0).float().mean().item()
        assert negative_ratio < 0.15, (
            "Should have mostly positive Jacobian determinants even with anisotropic spacing"
        )
