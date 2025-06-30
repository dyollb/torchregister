"""
Test configuration and fixtures for torchregister tests.
"""

import numpy as np
import pytest
import SimpleITK as sitk
import torch


@pytest.fixture
def device():
    """PyTorch device for testing."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def image_2d_shape():
    """Standard 2D image shape for testing."""
    return (64, 64)


@pytest.fixture
def image_3d_shape():
    """Standard 3D image shape for testing."""
    return (32, 64, 64)


@pytest.fixture
def create_test_image_2d(image_2d_shape, device):
    """Create a synthetic 2D test image."""

    def _create_image(shape=None, noise_level=0.1):
        if shape is None:
            shape = image_2d_shape

        H, W = shape

        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        )

        # Create synthetic image with geometric patterns
        image = torch.zeros(H, W)

        # Add circular pattern
        center_dist = torch.sqrt(x**2 + y**2)
        image += torch.exp(-5 * center_dist**2)

        # Add rectangular pattern
        rect_pattern = (torch.abs(x) < 0.5) & (torch.abs(y) < 0.3)
        image += rect_pattern.float() * 0.5

        # Add noise
        if noise_level > 0:
            image += torch.randn_like(image) * noise_level

        return image.to(device)

    return _create_image


@pytest.fixture
def create_test_image_3d(image_3d_shape, device):
    """Create a synthetic 3D test image."""

    def _create_image(shape=None, noise_level=0.1):
        if shape is None:
            shape = image_3d_shape

        D, H, W = shape

        # Create coordinate grids
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, D),
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij",
        )

        # Create synthetic image with geometric patterns
        image = torch.zeros(D, H, W)

        # Add spherical pattern
        center_dist = torch.sqrt(x**2 + y**2 + z**2)
        image += torch.exp(-3 * center_dist**2)

        # Add box pattern
        box_pattern = (torch.abs(x) < 0.4) & (torch.abs(y) < 0.4) & (torch.abs(z) < 0.4)
        image += box_pattern.float() * 0.3

        # Add noise
        if noise_level > 0:
            image += torch.randn_like(image) * noise_level

        return image.to(device)

    return _create_image


@pytest.fixture
def create_sitk_image():
    """Create SimpleITK test images."""

    def _create_sitk_image(array: np.ndarray, spacing=None, origin=None):
        image = sitk.GetImageFromArray(array)

        if spacing is not None:
            image.SetSpacing(spacing)
        if origin is not None:
            image.SetOrigin(origin)

        return image

    return _create_sitk_image


@pytest.fixture
def create_affine_transform_2d(device):
    """Create 2D affine transformation matrices."""

    def _create_transform(tx=0.0, ty=0.0, rotation=0.0, scale_x=1.0, scale_y=1.0):
        # Create transformation matrix
        cos_theta = torch.cos(torch.tensor(rotation))
        sin_theta = torch.sin(torch.tensor(rotation))

        matrix = torch.tensor(
            [
                [scale_x * cos_theta, -scale_x * sin_theta, tx],
                [scale_y * sin_theta, scale_y * cos_theta, ty],
            ],
            dtype=torch.float32,
            device=device,
        )

        return matrix

    return _create_transform


@pytest.fixture
def create_affine_transform_3d(device):
    """Create 3D affine transformation matrices."""

    def _create_transform(
        tx=0.0, ty=0.0, tz=0.0, rotation_x=0.0, rotation_y=0.0, rotation_z=0.0
    ):
        # Create rotation matrices
        cos_x, sin_x = (
            torch.cos(torch.tensor(rotation_x)),
            torch.sin(torch.tensor(rotation_x)),
        )
        cos_y, sin_y = (
            torch.cos(torch.tensor(rotation_y)),
            torch.sin(torch.tensor(rotation_y)),
        )
        cos_z, sin_z = (
            torch.cos(torch.tensor(rotation_z)),
            torch.sin(torch.tensor(rotation_z)),
        )

        # Rotation around X axis
        R_x = torch.tensor(
            [[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=torch.float32
        )

        # Rotation around Y axis
        R_y = torch.tensor(
            [[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=torch.float32
        )

        # Rotation around Z axis
        R_z = torch.tensor(
            [[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=torch.float32
        )

        # Combined rotation
        R = torch.matmul(torch.matmul(R_z, R_y), R_x)

        # Create transformation matrix
        matrix = torch.zeros(3, 4, dtype=torch.float32, device=device)
        matrix[:3, :3] = R
        matrix[:3, 3] = torch.tensor([tx, ty, tz])

        return matrix

    return _create_transform


@pytest.fixture
def tolerance():
    """Default tolerance for numerical comparisons."""
    return {"rtol": 1e-4, "atol": 1e-6}


class MockRegistration:
    """Mock registration class for testing."""

    def __init__(self):
        self.call_count = 0

    def register(self, fixed, moving):
        self.call_count += 1
        return torch.eye(2, 3), moving  # Return identity transform and moving image
