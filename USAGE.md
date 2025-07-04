# TorchRegister Usage Guide

## Installation

### Quick Installation
```bash
pip install torchregister
```

### Development Installation
```bash
git clone https://github.com/dyollb/torchregister.git
cd torchregister
python setup.py dev
```

## Quick Start

### Basic Affine Registration

```python
import torchregister
import SimpleITK as sitk
from torchregister.metrics import NCC

# Load images
fixed_image = sitk.ReadImage("fixed.nii.gz")
moving_image = sitk.ReadImage("moving.nii.gz")

# Initialize affine registration
ncc = NCC()
affine_reg = torchregister.AffineRegistration(
    similarity_metric=ncc,
    shrink_factors=[4, 2, 1], smoothing_sigmas=[2.0, 1.0, 0.0],
    num_iterations=[100, 150, 200],
    learning_rate=0.01
)

# Perform registration
transform_matrix, registered_image = affine_reg.register(fixed_image, moving_image)

# Save results
torchregister.save_image(registered_image, "registered.nii.gz", reference_image=fixed_image)
```

### Basic Deformable Registration (RDMM)

```python
from torchregister.metrics import LNCC

# Initialize RDMM registration
lncc = LNCC()
rdmm_reg = torchregister.RDMMRegistration(
    similarity_metric=lncc,
    shrink_factors=[4, 2, 1], smoothing_sigmas=[2.0, 1.0, 0.0],
    num_iterations=[50, 100, 150],
    learning_rate=0.01,
    smoothing_sigma=1.5,
    alpha=1.0
)

# Perform registration
deformation_field, registered_image = rdmm_reg.register(fixed_image, moving_image)

# Save results
torchregister.save_image(registered_image, "registered_rdmm.nii.gz", reference_image=fixed_image)
```

## Advanced Usage

### Custom Loss Functions

```python
from torchregister.metrics import NCC, LNCC, MSE, CombinedLoss

# Create combined loss
losses = {
    'ncc': NCC(),
    'lncc': LNCC(window_size=9)
}
weights = {
    'ncc': 0.7,
    'lncc': 0.3
}

combined_loss = CombinedLoss(losses, weights)

# Use in registration
ncc = NCC()
affine_reg = torchregister.AffineRegistration(
    similarity_metric=ncc,  # Use NCC as primary metric
    shrink_factors=[4, 2, 1], smoothing_sigmas=[2.0, 1.0, 0.0]
)
# Note: CombinedLoss can be set after initialization if needed
affine_reg.loss_fn = combined_loss
```

### Multi-Scale Registration

```python
# Configure multi-scale pyramid
mse = MSE()
affine_reg = torchregister.AffineRegistration(
    similarity_metric=mse,
    shrink_factors=[8, 4, 2, 1], smoothing_sigmas=[3.0, 2.0, 1.0, 0.0],  # 4-level pyramid
    num_iterations=[50, 100, 150, 200],  # Iterations per scale
    learning_rate=0.01,
    regularization_weight=0.001
)

# Registration will automatically create pyramid and register coarse-to-fine
transform_matrix, registered_image = affine_reg.register(fixed_image, moving_image)
```

### Evaluation and Metrics

```python
# Evaluate registration quality
metrics = affine_reg.evaluate(fixed_image, moving_image, transform_matrix)

print(f"Final NCC: {metrics['ncc']:.4f}")
print(f"Final MSE: {metrics['mse']:.4f}")

# For RDMM registration
rdmm_metrics = rdmm_reg.evaluate(fixed_image, moving_image, deformation_field)
print(f"Jacobian determinant mean: {rdmm_metrics['jacobian_det_mean']:.4f}")
print(f"Negative Jacobian ratio: {rdmm_metrics['negative_jacobian_ratio']:.4f}")
```

### Working with PyTorch Tensors

```python
import torch
from torchregister.io import sitk_to_torch, torch_to_sitk

# Convert SimpleITK to PyTorch
fixed_tensor = sitk_to_torch(fixed_image)
moving_tensor = sitk_to_torch(moving_image)

# Perform registration with tensors
transform_matrix, registered_tensor = affine_reg.register(fixed_tensor, moving_tensor)

# Convert back to SimpleITK
registered_sitk = torch_to_sitk(registered_tensor, reference_image=fixed_image)
```

## Configuration Options

### Affine Registration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `similarity_metric` | Similarity metric | "ncc" | "ncc", "mse" |
| `shrink_factors` | List of downsample factors per scale | [4, 2, 1] | e.g., [8, 4, 2, 1] |
| `smoothing_sigmas` | List of Gaussian smoothing sigmas in pixel units | [2.0, 1.0, 0.0] | e.g., [3.0, 2.0, 1.0, 0.0] |
| `num_iterations` | Iterations per scale | [100, 200, 300] | List of ints |
| `learning_rate` | Optimizer learning rate | 0.01 | 0.001-0.1 |
| `regularization_weight` | Regularization strength | 0.0 | 0.0-1.0 |

### RDMM Registration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `similarity_metric` | Similarity metric | "lncc" | "ncc", "lncc", "mse" |
| `shrink_factors` | List of downsample factors per scale | [4, 2, 1] | e.g., [8, 4, 2, 1] |
| `smoothing_sigmas` | List of Gaussian smoothing sigmas in pixel units | [2.0, 1.0, 0.0] | e.g., [3.0, 2.0, 1.0, 0.0] |
| `num_iterations` | Iterations per scale | [50, 100, 200] | List of ints |
| `learning_rate` | Optimizer learning rate | 0.01 | 0.001-0.1 |
| `smoothing_sigma` | Gaussian smoothing sigma | 1.0 | 0.5-3.0 |
| `alpha` | Regularization weight | 1.0 | 0.1-10.0 |
| `num_integration_steps` | Velocity integration steps | 7 | 3-10 |

## Similarity Metrics

### Normalized Cross-Correlation (NCC)
- Best for images with similar intensity distributions
- Robust to global intensity changes
- Range: [-1, 1], where 1 is perfect correlation

```python
from torchregister.metrics import NCC
ncc_loss = NCC()
```

### Local Normalized Cross-Correlation (LNCC)
- Better handling of local intensity variations
- Configurable window size
- More robust to noise than global NCC

```python
from torchregister.metrics import LNCC
lncc_loss = LNCC(window_size=9)  # 9x9 local windows
```

### Mean Squared Error (MSE)
- Simple intensity-based metric
- Fast computation
- Sensitive to intensity differences

```python
from torchregister.metrics import MSE
mse_loss = MSE()
```

### Mattes Mutual Information
- Information-theoretic metric
- Good for multi-modal registration
- Robust to intensity transformations

```python
from torchregister.metrics import MattesMI
mi_loss = MattesMI(bins=64)
```

### Dice Coefficient
- For segmentation overlap evaluation
- Used when segmentations are available

```python
from torchregister.metrics import Dice
dice_loss = Dice(smooth=1e-6)
```

## Best Practices

### Preprocessing
1. **Intensity normalization**: Normalize images to similar intensity ranges
2. **Resampling**: Ensure similar voxel spacing
3. **Cropping**: Remove unnecessary background

```python
from torchregister.processing import normalize_image, resample_image

# Normalize intensities
fixed_normalized = normalize_image(fixed_tensor, method="minmax")
moving_normalized = normalize_image(moving_tensor, method="minmax")

# Resample to isotropic spacing
fixed_resampled = resample_image(fixed_image, new_spacing=(1.0, 1.0, 1.0))
```

### Parameter Selection
1. **Start with fewer scales** for initial testing
2. **Use NCC or LNCC** for most applications
3. **Increase regularization** if deformations are too large
4. **Adjust learning rate** based on convergence behavior

### Performance Optimization
1. **Use GPU** when available
2. **Reduce image size** for faster prototyping
3. **Use fewer iterations** for initial experiments

```python
# Enable GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

affine_reg = torchregister.AffineRegistration(
    device=device
)
```

## Troubleshooting

### Common Issues

**Registration not converging:**
- Reduce learning rate
- Increase number of iterations
- Try different similarity metric
- Check image preprocessing

**Deformations too large (RDMM):**
- Increase regularization weight (`alpha`)
- Increase smoothing sigma
- Reduce learning rate

**Memory issues:**
- Reduce image size
- Use fewer pyramid scales
- Process on CPU instead of GPU

**Negative Jacobians (RDMM):**
- Increase regularization
- Reduce learning rate
- Use more integration steps

### Debug Mode

```python
# Enable verbose output
from torchregister.metrics import NCC

ncc = NCC()
affine_reg = torchregister.AffineRegistration(
    similarity_metric=ncc,
    num_iterations=[10]  # Fewer iterations for debugging
)

# Monitor loss during registration
transform_matrix, registered_image = affine_reg.register(fixed_image, moving_image)
```

## Examples

See the `examples/` directory for complete working examples:
- `affine_registration_example.py` - Complete affine registration workflow
- `rdmm_registration_example.py` - Deformable registration with RDMM

## API Reference

### Main Classes
- `torchregister.AffineRegistration` - Affine registration
- `torchregister.RDMMRegistration` - Deformable registration

### Metrics
- `torchregister.metrics.NCC` - Normalized Cross-Correlation
- `torchregister.metrics.LNCC` - Local Normalized Cross-Correlation
- `torchregister.metrics.MSE` - Mean Squared Error
- `torchregister.metrics.MattesMI` - Mattes Mutual Information
- `torchregister.metrics.Dice` - Dice Coefficient

### Utilities
- `torchregister.io.load_image()` - Load image from file
- `torchregister.io.save_image()` - Save image to file
- `torchregister.io.sitk_to_torch()` - Convert SimpleITK to PyTorch
- `torchregister.io.torch_to_sitk()` - Convert PyTorch to SimpleITK
- `torchregister.processing.normalize_image()` - Normalize image intensities
- `torchregister.processing.resample_image()` - Resample image resolution
- `torchregister.processing.gaussian_blur()` - Apply Gaussian blur filtering
