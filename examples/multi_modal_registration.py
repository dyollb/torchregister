#!/usr/bin/env python3
"""
Example: Multi-Modal 3D Image Registration

This example demonstrates how to register multi-modal 3D images (T1 and T2)
using TorchRegister. The key is to stack the modalities in the channel dimension.
"""

import numpy as np
import torch

from torchregister import RDMMRegistration
from torchregister.transforms import apply_deformation


def create_synthetic_multimodal_data():
    """Create synthetic T1 and T2-like images for demonstration."""
    # Image dimensions
    D, H, W = 32, 64, 64

    # Create synthetic T1 images (higher contrast between gray/white matter)
    t1_fixed = torch.randn(D, H, W) * 0.2 + 0.5
    t1_moving = torch.randn(D, H, W) * 0.2 + 0.5

    # Create synthetic T2 images (different contrast characteristics)
    t2_fixed = torch.randn(D, H, W) * 0.3 + 0.7
    t2_moving = torch.randn(D, H, W) * 0.3 + 0.7

    # Add some structure (simplified brain-like regions)
    center = (D // 2, H // 2, W // 2)
    for d in range(D):
        for h in range(H):
            for w in range(W):
                dist = np.sqrt(
                    (d - center[0]) ** 2 + (h - center[1]) ** 2 + (w - center[2]) ** 2
                )
                if dist < 15:  # "Brain tissue"
                    t1_fixed[d, h, w] += 0.3
                    t1_moving[d, h, w] += 0.3
                    t2_fixed[d, h, w] += 0.2
                    t2_moving[d, h, w] += 0.2
                elif dist < 20:  # "Gray matter"
                    t1_fixed[d, h, w] += 0.1
                    t1_moving[d, h, w] += 0.1
                    t2_fixed[d, h, w] += 0.4
                    t2_moving[d, h, w] += 0.4

    return t1_fixed, t2_fixed, t1_moving, t2_moving


def main():
    print("TorchRegister Multi-Modal Registration Example")
    print("=" * 50)

    # Create synthetic data
    print("Creating synthetic multi-modal data...")
    t1_fixed, t2_fixed, t1_moving, t2_moving = create_synthetic_multimodal_data()

    print(f"T1 Fixed shape: {t1_fixed.shape}")
    print(f"T2 Fixed shape: {t2_fixed.shape}")

    # Method 1: Joint multi-modal registration
    print("\n1. Joint Multi-Modal Registration")
    print("-" * 30)

    # Stack modalities in channel dimension
    fixed_multimodal = torch.stack([t1_fixed, t2_fixed], dim=0).unsqueeze(
        0
    )  # [1, 2, D, H, W]
    moving_multimodal = torch.stack([t1_moving, t2_moving], dim=0).unsqueeze(
        0
    )  # [1, 2, D, H, W]

    print(f"Multi-modal fixed shape: {fixed_multimodal.shape}")
    print(f"Channel 0 (T1) shape: {fixed_multimodal[0, 0].shape}")
    print(f"Channel 1 (T2) shape: {fixed_multimodal[0, 1].shape}")

    # Initialize registration with NCC (good for similar modalities)
    reg = RDMMRegistration(
        similarity_metric="ncc",
        num_scales=2,
        num_iterations=[50, 25],
        learning_rate=0.01,
    )

    # Perform registration
    print("Performing joint registration...")
    deformation, registered_multimodal = reg.register(
        fixed_multimodal, moving_multimodal
    )

    print(f"Deformation field shape: {deformation.shape}")
    print(f"Registered image shape: {registered_multimodal.shape}")

    # Extract individual registered modalities
    t1_registered = registered_multimodal[0, 0]  # [D, H, W]
    t2_registered = registered_multimodal[0, 1]  # [D, H, W]

    print(f"T1 registered shape: {t1_registered.shape}")
    print(f"T2 registered shape: {t2_registered.shape}")

    # Method 2: Single-modality registration with shared deformation
    print("\n2. Single-Modality Registration + Shared Deformation")
    print("-" * 50)

    # Register using primary modality (T1)
    t1_fixed_batch = t1_fixed.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    t1_moving_batch = t1_moving.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    print("Registering T1 images...")
    primary_deformation, t1_reg_primary = reg.register(t1_fixed_batch, t1_moving_batch)

    # Apply same deformation to T2
    print("Applying deformation to T2...")
    t2_moving_batch = t2_moving.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    t2_reg_shared = apply_deformation(t2_moving_batch, primary_deformation)

    print(f"T1 registered (primary) shape: {t1_reg_primary.shape}")
    print(f"T2 registered (shared) shape: {t2_reg_shared.shape}")

    # Compute similarity metrics for comparison
    print("\n3. Evaluation")
    print("-" * 15)

    # Evaluate joint registration
    metrics_joint = reg.evaluate(fixed_multimodal, moving_multimodal, deformation)
    print("Joint registration metrics:")
    for key, value in metrics_joint.items():
        print(f"  {key}: {value:.4f}")

    # Evaluate single-modality registration on T1
    metrics_single = reg.evaluate(t1_fixed_batch, t1_moving_batch, primary_deformation)
    print("\nSingle-modality (T1) registration metrics:")
    for key, value in metrics_single.items():
        print(f"  {key}: {value:.4f}")

    print("\nKey Points:")
    print("- Multi-modal images use channel dimension: [B, C, D, H, W]")
    print("- Same spatial transformation applied to all channels")
    print("- Similarity metrics computed per-channel then averaged")
    print(
        "- Choose appropriate metrics: NCC for similar modalities, MI for different ones"
    )

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
