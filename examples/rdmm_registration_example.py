"""
Example script demonstrating RDMM deformable registration with TorchRegister.

This script shows how to perform 2D deformable registration using the RDMM method.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchregister import RDMMRegistration
from torchregister.metrics import LNCC, MSE, NCC
from torchregister.transforms import apply_deformation


def create_synthetic_image_with_details(shape=(96, 96), complexity="medium"):
    """Create synthetic image with detailed structures for deformable registration."""
    H, W = shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )

    image = torch.zeros(H, W)

    if complexity == "simple":
        # Simple circular pattern
        center_dist = torch.sqrt(x**2 + y**2)
        image += torch.exp(-3 * center_dist**2)

    elif complexity == "medium":
        # Multiple structures
        # Large central structure
        center_dist = torch.sqrt(x**2 + y**2)
        image += 0.8 * torch.exp(-2 * center_dist**2)

        # Smaller peripheral structures
        for center in [(0.5, 0.3), (-0.4, 0.4), (0.2, -0.5), (-0.3, -0.3)]:
            cx, cy = center
            local_dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            image += 0.4 * torch.exp(-8 * local_dist**2)

        # Linear structures
        mask1 = torch.abs(y - 0.3 * x) < 0.05
        mask2 = torch.abs(y + 0.5 * x - 0.2) < 0.03
        image += 0.3 * mask1.float() + 0.25 * mask2.float()

    elif complexity == "high":
        # Very detailed structure
        # Multiple frequency components
        for freq in [2, 4, 6]:
            image += 0.1 * torch.sin(freq * np.pi * x) * torch.cos(freq * np.pi * y)

        # Radial patterns
        angle = torch.atan2(y, x)
        radius = torch.sqrt(x**2 + y**2)
        image += 0.3 * torch.sin(8 * angle) * torch.exp(-2 * radius**2)

        # Random blob-like structures
        for _i in range(8):
            cx = np.random.uniform(-0.7, 0.7)
            cy = np.random.uniform(-0.7, 0.7)
            sigma = np.random.uniform(0.1, 0.25)
            intensity = np.random.uniform(0.2, 0.6)

            local_dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            image += intensity * torch.exp(-(local_dist**2) / (2 * sigma**2))

    # Add some noise
    image += torch.randn_like(image) * 0.02

    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return image


def create_synthetic_deformation(shape, deformation_type="wave"):
    """Create synthetic deformation field."""
    H, W = shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )

    if deformation_type == "wave":
        # Sinusoidal deformation
        def_x = 0.1 * torch.sin(2 * np.pi * y) * torch.cos(np.pi * x)
        def_y = 0.08 * torch.cos(2 * np.pi * x) * torch.sin(np.pi * y)

    elif deformation_type == "radial":
        # Radial expansion/contraction
        radius = torch.sqrt(x**2 + y**2)
        angle = torch.atan2(y, x)

        # Radial component (expansion in center, contraction at edges)
        radial_factor = 0.1 * torch.exp(-2 * radius**2) * (1 - radius)

        def_x = radial_factor * torch.cos(angle)
        def_y = radial_factor * torch.sin(angle)

    elif deformation_type == "shear":
        # Shear deformation
        def_x = 0.05 * y + 0.03 * torch.sin(np.pi * x)
        def_y = 0.04 * x + 0.02 * torch.cos(np.pi * y)

    elif deformation_type == "localized":
        # Localized deformations (multiple centers)
        def_x = torch.zeros_like(x)
        def_y = torch.zeros_like(y)

        centers = [(0.3, 0.2), (-0.4, -0.3), (0.1, -0.5)]
        for cx, cy in centers:
            local_dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            weight = torch.exp(-10 * local_dist**2)

            def_x += 0.08 * weight * (x - cx)
            def_y += 0.08 * weight * (y - cy)

    # Stack to create deformation field
    deformation = torch.stack([def_x, def_y], dim=-1)

    return deformation


def run_rdmm_registration_example():
    """Run complete RDMM registration example."""
    print("TorchRegister RDMM Deformable Registration Example")
    print("=" * 55)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create synthetic images
    print("\n1. Creating synthetic images...")
    fixed_image = create_synthetic_image_with_details(
        shape=(96, 96), complexity="medium"
    ).to(device)

    # Create moving image with known deformation
    true_deformation = create_synthetic_deformation(
        fixed_image.shape, deformation_type="wave"
    ).to(device)

    moving_image = apply_deformation(
        fixed_image.unsqueeze(0).unsqueeze(0), true_deformation.unsqueeze(0)
    ).squeeze()

    print(f"Fixed image shape: {fixed_image.shape}")
    print(f"Moving image shape: {moving_image.shape}")
    print(f"True deformation magnitude: {torch.norm(true_deformation).item():.4f}")

    # Compute initial similarity
    ncc_metric = NCC()
    lncc_metric = LNCC(window_size=9)
    mse_metric = MSE()

    initial_ncc = -ncc_metric(
        fixed_image.unsqueeze(0).unsqueeze(0), moving_image.unsqueeze(0).unsqueeze(0)
    ).item()

    initial_lncc = -lncc_metric(
        fixed_image.unsqueeze(0).unsqueeze(0), moving_image.unsqueeze(0).unsqueeze(0)
    ).item()

    initial_mse = mse_metric(
        fixed_image.unsqueeze(0).unsqueeze(0), moving_image.unsqueeze(0).unsqueeze(0)
    ).item()

    print("\nInitial similarity:")
    print(f"  NCC: {initial_ncc:.4f}")
    print(f"  LNCC: {initial_lncc:.4f}")
    print(f"  MSE: {initial_mse:.4f}")

    # Initialize RDMM registration
    print("\n2. Setting up RDMM registration...")
    lncc = LNCC()
    registration = RDMMRegistration(
        similarity_metric=lncc,
        shrink_factors=[4, 2, 1], smoothing_sigmas=[2.0, 1.0, 0.0],
        num_iterations=[30, 50, 80],
        learning_rate=0.01,
        smoothing_sigma=1.5,
        alpha=1.0,  # Regularization weight
        num_integration_steps=7,
        device=device,
    )

    # Perform registration
    print("\n3. Performing deformable registration...")
    estimated_deformation, registered_image = registration.register(
        fixed_image, moving_image
    )

    # Compute final similarity
    final_ncc = -ncc_metric(
        fixed_image.unsqueeze(0).unsqueeze(0),
        registered_image.unsqueeze(0).unsqueeze(0),
    ).item()

    final_lncc = -lncc_metric(
        fixed_image.unsqueeze(0).unsqueeze(0),
        registered_image.unsqueeze(0).unsqueeze(0),
    ).item()

    final_mse = mse_metric(
        fixed_image.unsqueeze(0).unsqueeze(0),
        registered_image.unsqueeze(0).unsqueeze(0),
    ).item()

    print("\nFinal similarity:")
    print(f"  NCC: {final_ncc:.4f} (improvement: {final_ncc - initial_ncc:.4f})")
    print(f"  LNCC: {final_lncc:.4f} (improvement: {final_lncc - initial_lncc:.4f})")
    print(f"  MSE: {final_mse:.4f} (improvement: {initial_mse - final_mse:.4f})")

    # Analyze deformation field
    print("\n4. Deformation Field Analysis:")
    true_magnitude = torch.norm(true_deformation).item()
    estimated_magnitude = torch.norm(estimated_deformation).item()

    # Compute deformation error
    deformation_error = torch.norm(estimated_deformation - true_deformation.squeeze())
    deformation_mse = torch.mean(
        (estimated_deformation - true_deformation.squeeze()) ** 2
    )

    print(f"True deformation magnitude: {true_magnitude:.4f}")
    print(f"Estimated deformation magnitude: {estimated_magnitude:.4f}")
    print(f"Deformation error (L2 norm): {deformation_error:.4f}")
    print(f"Deformation MSE: {deformation_mse:.6f}")

    # Evaluate registration
    print("\n5. Detailed evaluation metrics...")
    metrics = registration.evaluate(fixed_image, moving_image, estimated_deformation)

    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Check for negative Jacobians (folding)
    jac_det = registration._compute_jacobian_determinant(
        estimated_deformation.unsqueeze(0)
    )
    negative_jac_ratio = (jac_det < 0).float().mean().item()
    min_jac = jac_det.min().item()

    print("\nDeformation quality:")
    print(
        f"  Negative Jacobian ratio: {negative_jac_ratio:.4f} ({negative_jac_ratio * 100:.1f}%)"
    )
    print(f"  Minimum Jacobian determinant: {min_jac:.4f}")
    print(f"  Mean Jacobian determinant: {jac_det.mean().item():.4f}")

    # Visualization
    print("\n6. Creating visualization...")
    create_rdmm_visualization(
        fixed_image.cpu(),
        moving_image.cpu(),
        registered_image.cpu(),
        true_deformation.cpu().squeeze(),
        estimated_deformation.cpu(),
        initial_lncc,
        final_lncc,
    )

    print("\nRDMM registration completed successfully!")
    return {
        "fixed_image": fixed_image.cpu(),
        "moving_image": moving_image.cpu(),
        "registered_image": registered_image.cpu(),
        "true_deformation": true_deformation.cpu(),
        "estimated_deformation": estimated_deformation.cpu(),
        "metrics": metrics,
    }


def create_rdmm_visualization(
    fixed, moving, registered, true_def, estimated_def, initial_lncc, final_lncc
):
    """Create comprehensive visualization of RDMM registration results."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Row 1: Images
    axes[0, 0].imshow(fixed.detach().numpy(), cmap="gray")
    axes[0, 0].set_title("Fixed Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(moving.detach().numpy(), cmap="gray")
    axes[0, 1].set_title("Moving Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(registered.detach().numpy(), cmap="gray")
    axes[0, 2].set_title("Registered Image")
    axes[0, 2].axis("off")

    # Overlay of fixed and registered
    overlay = torch.stack([fixed, registered, torch.zeros_like(fixed)], dim=-1)
    axes[0, 3].imshow(overlay.detach().numpy())
    axes[0, 3].set_title("Overlay (Red: Fixed, Green: Registered)")
    axes[0, 3].axis("off")

    # Row 2: Deformation fields
    # True deformation
    skip = 8  # Subsample for visualization
    H, W = true_def.shape[:2]
    y_coords, x_coords = np.mgrid[0:H:skip, 0:W:skip]

    true_def_x = true_def[::skip, ::skip, 0].detach().numpy()
    true_def_y = true_def[::skip, ::skip, 1].detach().numpy()

    axes[1, 0].quiver(
        x_coords,
        y_coords,
        true_def_x,
        -true_def_y,
        scale=1,
        scale_units="xy",
        angles="xy",
        alpha=0.7,
    )
    axes[1, 0].set_title("True Deformation Field")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].invert_yaxis()

    # Estimated deformation
    est_def_x = estimated_def[::skip, ::skip, 0].detach().numpy()
    est_def_y = estimated_def[::skip, ::skip, 1].detach().numpy()

    axes[1, 1].quiver(
        x_coords,
        y_coords,
        est_def_x,
        -est_def_y,
        scale=1,
        scale_units="xy",
        angles="xy",
        alpha=0.7,
    )
    axes[1, 1].set_title("Estimated Deformation Field")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].invert_yaxis()

    # Deformation magnitude
    true_mag = torch.norm(true_def, dim=-1).detach().numpy()
    est_mag = torch.norm(estimated_def, dim=-1).detach().numpy()

    im1 = axes[1, 2].imshow(true_mag, cmap="viridis")
    axes[1, 2].set_title("True Deformation Magnitude")
    axes[1, 2].axis("off")
    plt.colorbar(im1, ax=axes[1, 2], fraction=0.046)

    im2 = axes[1, 3].imshow(est_mag, cmap="viridis")
    axes[1, 3].set_title("Estimated Deformation Magnitude")
    axes[1, 3].axis("off")
    plt.colorbar(im2, ax=axes[1, 3], fraction=0.046)

    # Row 3: Error analysis
    diff_initial = torch.abs(fixed - moving)
    diff_final = torch.abs(fixed - registered)
    deformation_error = torch.norm(estimated_def - true_def, dim=-1)

    im3 = axes[2, 0].imshow(diff_initial.detach().numpy(), cmap="hot")
    axes[2, 0].set_title(f"Initial Difference\n(LNCC: {initial_lncc:.3f})")
    axes[2, 0].axis("off")
    plt.colorbar(im3, ax=axes[2, 0], fraction=0.046)

    im4 = axes[2, 1].imshow(diff_final.detach().numpy(), cmap="hot")
    axes[2, 1].set_title(f"Final Difference\n(LNCC: {final_lncc:.3f})")
    axes[2, 1].axis("off")
    plt.colorbar(im4, ax=axes[2, 1], fraction=0.046)

    im5 = axes[2, 2].imshow(deformation_error.detach().numpy(), cmap="plasma")
    axes[2, 2].set_title("Deformation Error")
    axes[2, 2].axis("off")
    plt.colorbar(im5, ax=axes[2, 2], fraction=0.046)

    # Registration improvement
    improvement = diff_initial - diff_final
    im6 = axes[2, 3].imshow(improvement.detach().numpy(), cmap="RdBu_r")
    axes[2, 3].set_title("Registration Improvement\n(Blue = Better)")
    axes[2, 3].axis("off")
    plt.colorbar(im6, ax=axes[2, 3], fraction=0.046)

    plt.tight_layout()
    plt.savefig("rdmm_registration_example.png", dpi=150, bbox_inches="tight")
    print("Visualization saved as 'rdmm_registration_example.png'")
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run the example
    results = run_rdmm_registration_example()

    # Print summary
    print(f"\n{'=' * 55}")
    print("RDMM REGISTRATION SUMMARY")
    print(f"{'=' * 55}")
    print(f"Final NCC: {results['metrics']['ncc']:.4f}")
    print(f"Final MSE: {results['metrics']['mse']:.4f}")
    print(
        f"Negative Jacobian ratio: {results['metrics']['negative_jacobian_ratio']:.4f}"
    )
    print(f"Mean Jacobian determinant: {results['metrics']['jacobian_det_mean']:.4f}")
    print("RDMM example completed successfully!")
