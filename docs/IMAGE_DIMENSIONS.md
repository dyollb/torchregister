# Image Dimension Conventions in TorchRegister

This document outlines the core assumptions and conventions for image dimensions in the TorchRegister library, including how to handle multi-modal images.

## Table of Contents
- [Core Dimension Conventions](#core-dimension-conventions)
- [Supported Input Formats](#supported-input-formats)
- [Multi-Modal Images](#multi-modal-images)
- [Deformation Fields](#deformation-fields)
- [Coordinate Grids](#coordinate-grids)
- [Examples](#examples)

## Core Dimension Conventions

TorchRegister follows PyTorch's standard dimension ordering conventions:

### 2D Images
- **Spatial-only**: `[H, W]` - Height × Width
- **With batch**: `[B, C, H, W]` - Batch × Channels × Height × Width
- **Coordinate order**: `[x, y]` where x is width dimension, y is height dimension

### 3D Images
- **Spatial-only**: `[D, H, W]` - Depth × Height × Width
- **With batch**: `[B, C, D, H, W]` - Batch × Channels × Depth × Height × Width
- **Coordinate order**: `[x, y, z]` where x is width, y is height, z is depth

### Key Principles
1. **Batch dimension (B)**: Always first when present
2. **Channel dimension (C)**: Always second when present (after batch)
3. **Spatial dimensions**: Always last, in order `[H, W]` for 2D or `[D, H, W]` for 3D
4. **Coordinate convention**: `(x, y)` for 2D, `(x, y, z)` for 3D, following PyTorch's grid_sample convention

## Supported Input Formats

The library automatically handles inputs in various formats:

### Registration Functions Accept
```python
# Single images (automatically expanded to [1, 1, ...] internally)
fixed_2d = torch.rand(64, 64)              # [H, W]
fixed_3d = torch.rand(32, 64, 64)          # [D, H, W]

# Batched single-channel images
fixed_2d = torch.rand(1, 1, 64, 64)        # [B, C, H, W]
fixed_3d = torch.rand(1, 1, 32, 64, 64)    # [B, C, D, H, W]

# Multi-channel images (see Multi-Modal section)
fixed_2d = torch.rand(1, 2, 64, 64)        # [B, C, H, W]
fixed_3d = torch.rand(1, 2, 32, 64, 64)    # [B, C, D, H, W]
```

### SimpleITK Integration
```python
# SimpleITK images are automatically converted
sitk_image = sitk.ReadImage("image.nii.gz")
torch_tensor = sitk_to_torch(sitk_image)  # Converts to [D, H, W] or [H, W]
```

## Multi-Modal Images

### Multi-Modal 3D Images (T1, T2, FLAIR, etc.)

Multi-modal images should be represented using the **channel dimension**:

```python
# T1 and T2 images registered together
t1_image = torch.rand(32, 64, 64)  # [D, H, W]
t2_image = torch.rand(32, 64, 64)  # [D, H, W]

# Combine into multi-modal tensor
multi_modal = torch.stack([t1_image, t2_image], dim=0)  # [2, D, H, W]
multi_modal = multi_modal.unsqueeze(0)  # [1, 2, D, H, W] - [B, C, D, H, W]

# For registration
from torchregister import RDMMRegistration
from torchregister.metrics import NCC

ncc = NCC()
reg = RDMMRegistration(similarity_metric=ncc)
# Both fixed and moving should have same number of channels
fixed_multi = multi_modal  # [1, 2, D, H, W]
moving_multi = multi_modal  # [1, 2, D, H, W]

transform = reg.register(fixed_multi, moving_multi)
```

### Channel Processing

Metrics and registration algorithms process channels in the following ways:

1. **NCC/LNCC**: Computed per-channel, then averaged
2. **MSE**: Computed per-channel, then averaged
3. **Mattes MI**: Applied to combined multi-channel intensity space
4. **Deformation fields**: Shared across all channels (same spatial transformation)

### Multi-Modal Registration Strategies

```python
# Strategy 1: Joint registration (recommended)
# All channels contribute to similarity metric
multi_modal_fixed = torch.stack([t1_fixed, t2_fixed], dim=1)  # [1, 2, D, H, W]
multi_modal_moving = torch.stack([t1_moving, t2_moving], dim=1)  # [1, 2, D, H, W]

deformation = reg.register(multi_modal_fixed, multi_modal_moving)

# Strategy 2: Single modality registration, apply to all
# Register using primary modality, apply deformation to all
primary_deformation = reg.register(t1_fixed.unsqueeze(0).unsqueeze(0),
                                  t1_moving.unsqueeze(0).unsqueeze(0))

# Apply same deformation to all modalities
from torchregister.transforms import apply_deformation
t1_registered = apply_deformation(t1_moving.unsqueeze(0).unsqueeze(0), primary_deformation)
t2_registered = apply_deformation(t2_moving.unsqueeze(0).unsqueeze(0), primary_deformation)
```

## Deformation Fields

Deformation fields represent displacement vectors at each spatial location:

### 2D Deformation Fields
```python
# Shape: [B, H, W, 2] or [H, W, 2]
# Last dimension contains [dx, dy] displacement vectors
deformation_2d = torch.zeros(1, 64, 64, 2)  # [B, H, W, 2]
```

### 3D Deformation Fields
```python
# Shape: [B, D, H, W, 3] or [D, H, W, 3]
# Last dimension contains [dx, dy, dz] displacement vectors
deformation_3d = torch.zeros(1, 32, 64, 64, 3)  # [B, D, H, W, 3]
```

### Coordinate Convention
- Displacement vectors follow the same coordinate order as grids: `[x, y]` for 2D, `[x, y, z]` for 3D
- Positive displacement moves towards higher index values in each dimension

## Coordinate Grids

Coordinate grids define sampling positions for transformations:

### 2D Grids
```python
# Shape: [H, W, 2] or [B, H, W, 2]
# Values in range [-1, 1] following PyTorch grid_sample convention
grid_2d = create_grid((64, 64))  # [64, 64, 2]
```

### 3D Grids
```python
# Shape: [D, H, W, 3] or [B, D, H, W, 3]
# Values in range [-1, 1] following PyTorch grid_sample convention
grid_3d = create_grid((32, 64, 64))  # [32, 64, 64, 3]
```

## Examples

### Complete Multi-Modal 3D Registration Example

```python
import torch
from torchregister import RDMMRegistration
from torchregister.transforms import apply_deformation
import SimpleITK as sitk

# Load multi-modal data
t1_fixed = sitk.ReadImage("patient1_t1.nii.gz")
t2_fixed = sitk.ReadImage("patient1_t2.nii.gz")
t1_moving = sitk.ReadImage("patient2_t1.nii.gz")
t2_moving = sitk.ReadImage("patient2_t2.nii.gz")

# Convert to PyTorch tensors
t1_fixed_torch = sitk_to_torch(t1_fixed)    # [D, H, W]
t2_fixed_torch = sitk_to_torch(t2_fixed)    # [D, H, W]
t1_moving_torch = sitk_to_torch(t1_moving)  # [D, H, W]
t2_moving_torch = sitk_to_torch(t2_moving)  # [D, H, W]

# Create multi-modal tensors
fixed_multi = torch.stack([t1_fixed_torch, t2_fixed_torch], dim=0).unsqueeze(0)   # [1, 2, D, H, W]
moving_multi = torch.stack([t1_moving_torch, t2_moving_torch], dim=0).unsqueeze(0) # [1, 2, D, H, W]

# Perform registration
from torchregister.metrics import NCC

ncc = NCC()
reg = RDMMRegistration(
    similarity_metric=ncc,  # NCC computed per-channel, then averaged
    num_scales=3,
    num_iterations=[100, 50, 25]
)

deformation_field = reg.register(fixed_multi, moving_multi)
# deformation_field shape: [1, D, H, W, 3]

# Apply deformation to get registered images
registered_multi = apply_deformation(moving_multi, deformation_field)
# registered_multi shape: [1, 2, D, H, W]

# Extract individual registered modalities
t1_registered = registered_multi[0, 0]  # [D, H, W]
t2_registered = registered_multi[0, 1]  # [D, H, W]

# Convert back to SimpleITK for saving
t1_registered_sitk = torch_to_sitk(t1_registered, reference_image=t1_fixed)
t2_registered_sitk = torch_to_sitk(t2_registered, reference_image=t2_fixed)

sitk.WriteImage(t1_registered_sitk, "patient2_t1_registered.nii.gz")
sitk.WriteImage(t2_registered_sitk, "patient2_t2_registered.nii.gz")
```

### Single-Channel Registration with Multi-Channel Application

```python
# Register using primary modality (e.g., T1)
primary_fixed = t1_fixed_torch.unsqueeze(0).unsqueeze(0)   # [1, 1, D, H, W]
primary_moving = t1_moving_torch.unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]

deformation = reg.register(primary_fixed, primary_moving)

# Apply to all modalities
t1_reg = apply_deformation(primary_moving, deformation)
t2_reg = apply_deformation(t2_moving_torch.unsqueeze(0).unsqueeze(0), deformation)
```

## Important Notes

1. **Memory Efficiency**: Multi-modal registration requires more memory. Consider processing on GPU if available.

2. **Similarity Metrics**: Choose appropriate metrics for multi-modal data:
   - **NCC/LNCC**: Good for similar modalities (T1/T2)
   - **Mutual Information**: Better for different modalities (T1/CT)
   - **MSE**: Only for identical modalities

3. **Preprocessing**: Ensure all modalities are:
   - In the same physical space (same origin, spacing, direction)
   - Appropriately normalized
   - Of the same dimensions

4. **Channel Independence**: Each channel is processed independently for similarity metrics but shares the same spatial transformation.

5. **Batch Processing**: The batch dimension allows processing multiple subjects simultaneously, but each subject should have the same number of modalities.
