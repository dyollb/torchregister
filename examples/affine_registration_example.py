"""
Example script demonstrating affine registration with TorchRegister.

This script shows how to perform 2D affine registration between two synthetic images.
"""

import matplotlib.pyplot as plt
import torch

from torchregister import AffineRegistration
from torchregister.metrics import MSE, NCC
from torchregister.transforms import apply_transform, create_grid


def create_synthetic_image(shape=(128, 128), pattern_type="circles"):
    """Create synthetic test image with geometric patterns."""
    H, W = shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )

    image = torch.zeros(H, W)

    if pattern_type == "circles":
        # Add circular patterns
        for center, radius, intensity in [
            ((0.3, 0.3), 0.2, 0.8),
            ((-0.3, -0.3), 0.15, 0.6),
            ((0.0, 0.0), 0.1, 1.0),
        ]:
            cx, cy = center
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            image += intensity * torch.exp(-5 * (dist / radius) ** 2)

    elif pattern_type == "squares":
        # Add rectangular patterns
        for center, size, intensity in [
            ((0.2, 0.2), (0.3, 0.2), 0.7),
            ((-0.3, -0.1), (0.2, 0.4), 0.5),
        ]:
            cx, cy = center
            sx, sy = size
            mask = (torch.abs(x - cx) < sx) & (torch.abs(y - cy) < sy)
            image += intensity * mask.float()

    # Add some noise
    image += torch.randn_like(image) * 0.05

    return image


def create_transformed_image(image, tx=0.1, ty=0.05, rotation=0.1, scale=1.1):
    """Apply known transformation to create moving image."""
    device = image.device

    # Create transformation matrix
    cos_theta = torch.cos(torch.tensor(rotation))
    sin_theta = torch.sin(torch.tensor(rotation))

    transform_matrix = torch.tensor(
        [
            [scale * cos_theta, -scale * sin_theta, tx],
            [scale * sin_theta, scale * cos_theta, ty],
        ],
        dtype=torch.float32,
        device=device,
    )

    # Apply transformation
    grid = create_grid(image.shape, device=device)

    # Convert to homogeneous coordinates and apply transform
    homo_grid = torch.cat(
        [grid, torch.ones(*grid.shape[:-1], 1, device=device)], dim=-1
    )
    transformed_grid = torch.matmul(homo_grid, transform_matrix.T)

    # Sample image at transformed coordinates
    transformed_image = apply_transform(
        image.unsqueeze(0).unsqueeze(0), transformed_grid
    ).squeeze()

    return transformed_image, transform_matrix


def run_affine_registration_example():
    """Run complete affine registration example."""
    print("TorchRegister Affine Registration Example")
    print("=" * 50)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create synthetic images
    print("\n1. Creating synthetic images...")
    fixed_image = create_synthetic_image(shape=(128, 128), pattern_type="circles")
    fixed_image = fixed_image.to(device)

    # Create moving image with known transformation
    true_params = {"tx": 0.15, "ty": -0.1, "rotation": 0.2, "scale": 1.05}
    moving_image, true_transform = create_transformed_image(fixed_image, **true_params)

    print(f"Fixed image shape: {fixed_image.shape}")
    print(f"Moving image shape: {moving_image.shape}")
    print(f"True transformation parameters: {true_params}")

    # Compute initial similarity
    ncc_metric = NCC()
    mse_metric = MSE()

    initial_ncc = -ncc_metric(
        fixed_image.unsqueeze(0).unsqueeze(0), moving_image.unsqueeze(0).unsqueeze(0)
    ).item()
    initial_mse = mse_metric(
        fixed_image.unsqueeze(0).unsqueeze(0), moving_image.unsqueeze(0).unsqueeze(0)
    ).item()

    print("\nInitial similarity:")
    print(f"  NCC: {initial_ncc:.4f}")
    print(f"  MSE: {initial_mse:.4f}")

    # Initialize registration
    print("\n2. Setting up affine registration...")
    registration = AffineRegistration(
        similarity_metric="ncc",
        num_scales=3,
        num_iterations=[100, 150, 200],
        learning_rate=0.01,
        regularization_weight=0.001,
        device=device,
    )

    # Perform registration
    print("\n3. Performing registration...")
    estimated_transform, registered_image = registration.register(
        fixed_image, moving_image
    )

    # Compute final similarity
    final_ncc = -ncc_metric(
        fixed_image.unsqueeze(0).unsqueeze(0),
        registered_image.unsqueeze(0).unsqueeze(0),
    ).item()
    final_mse = mse_metric(
        fixed_image.unsqueeze(0).unsqueeze(0),
        registered_image.unsqueeze(0).unsqueeze(0),
    ).item()

    print("\nFinal similarity:")
    print(f"  NCC: {final_ncc:.4f} (improvement: {final_ncc - initial_ncc:.4f})")
    print(f"  MSE: {final_mse:.4f} (improvement: {initial_mse - final_mse:.4f})")

    # Analyze transformation parameters
    print("\n4. Transformation Analysis:")
    print("True transform matrix:")
    print(true_transform.detach().numpy())
    print("Estimated transform matrix:")
    print(estimated_transform.cpu().detach().numpy())

    # Extract transformation parameters (approximate)
    estimated_tx = estimated_transform[0, 2].item()
    estimated_ty = estimated_transform[1, 2].item()

    # Compute rotation and scale (simplified)
    estimated_scale_x = torch.norm(estimated_transform[:, 0]).item()
    # TODO: why is estimated_scale_y not used?
    # estimated_scale_y = torch.norm(estimated_transform[:, 1]).item()
    estimated_rotation = torch.atan2(
        estimated_transform[1, 0], estimated_transform[0, 0]
    ).item()

    print("\nParameter comparison:")
    print(
        f"Translation X: True={true_params['tx']:.3f}, Est={-estimated_tx:.3f}, Error={abs(true_params['tx'] + estimated_tx):.3f}"
    )
    print(
        f"Translation Y: True={true_params['ty']:.3f}, Est={-estimated_ty:.3f}, Error={abs(true_params['ty'] + estimated_ty):.3f}"
    )
    print(
        f"Rotation: True={true_params['rotation']:.3f}, Est={-estimated_rotation:.3f}, Error={abs(true_params['rotation'] + estimated_rotation):.3f}"
    )
    print(
        f"Scale: True={true_params['scale']:.3f}, Est={estimated_scale_x:.3f}, Error={abs(true_params['scale'] - estimated_scale_x):.3f}"
    )

    # Evaluate registration
    print("\n5. Evaluation metrics...")
    metrics = registration.evaluate(fixed_image, moving_image, estimated_transform)

    for metric_name, value in metrics.items():
        if metric_name != "transformation_matrix":
            print(f"  {metric_name}: {value:.4f}")

    # Visualization
    print("\n6. Creating visualization...")
    create_visualization(
        fixed_image.cpu(),
        moving_image.cpu(),
        registered_image.cpu(),
        initial_ncc,
        final_ncc,
    )

    print("\nRegistration completed successfully!")
    return {
        "fixed_image": fixed_image.cpu(),
        "moving_image": moving_image.cpu(),
        "registered_image": registered_image.cpu(),
        "true_transform": true_transform.cpu(),
        "estimated_transform": estimated_transform.cpu(),
        "metrics": metrics,
    }


def create_visualization(fixed, moving, registered, initial_ncc, final_ncc):
    """Create visualization of registration results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Images
    axes[0, 0].imshow(fixed.detach().numpy(), cmap="gray")
    axes[0, 0].set_title("Fixed Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(moving.detach().numpy(), cmap="gray")
    axes[0, 1].set_title("Moving Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(registered.detach().numpy(), cmap="gray")
    axes[0, 2].set_title("Registered Image")
    axes[0, 2].axis("off")

    # Bottom row: Difference images
    diff_initial = torch.abs(fixed - moving)
    diff_final = torch.abs(fixed - registered)

    im1 = axes[1, 0].imshow(diff_initial.detach().numpy(), cmap="hot")
    axes[1, 0].set_title(f"Initial Difference\n(NCC: {initial_ncc:.3f})")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0])

    im2 = axes[1, 1].imshow(diff_final.detach().numpy(), cmap="hot")
    axes[1, 1].set_title(f"Final Difference\n(NCC: {final_ncc:.3f})")
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1])

    # Improvement visualization
    improvement = diff_initial - diff_final
    im3 = axes[1, 2].imshow(improvement.detach().numpy(), cmap="RdBu_r")
    axes[1, 2].set_title("Improvement\n(Blue = Better)")
    axes[1, 2].axis("off")
    plt.colorbar(im3, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig("affine_registration_example.png", dpi=150, bbox_inches="tight")
    print("Visualization saved as 'affine_registration_example.png'")
    plt.show()


if __name__ == "__main__":
    # Run the example
    results = run_affine_registration_example()

    # Print summary
    print(f"\n{'=' * 50}")
    print("REGISTRATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Final NCC: {results['metrics']['ncc']:.4f}")
    print(f"Final MSE: {results['metrics']['mse']:.4f}")
    print("Example completed successfully!")
