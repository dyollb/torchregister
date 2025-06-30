# TorchRegister

[![CI](https://github.com/dyollb/torchregister/actions/workflows/ci.yml/badge.svg)](https://github.com/dyollb/torchregister/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dyollb/torchregister/branch/main/graph/badge.svg)](https://codecov.io/gh/dyollb/torchregister)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based package for multi-scale affine and deformable image registration.

## Features

- **Multi-scale Registration**: Supports both affine and deformable registration
- **Differentiable Losses**: Implementation of various similarity metrics (NCC, LNCC, MSE, Mattes MI, Dice)
- **PyTorch Integration**: Fully differentiable and GPU-accelerated
- **SimpleITK IO**: Seamless integration with medical imaging formats
- **Multi-Modal Support**: Handle multi-channel images (T1, T2, FLAIR, etc.)
- **Comprehensive Testing**: Full test coverage with pytest

## Documentation

- **[Image Dimension Conventions](docs/IMAGE_DIMENSIONS.md)**: Complete guide to image dimensions and multi-modal support

## Installation

```bash
pip install torchregister
```

For development:

```bash
git clone https://github.com/yourusername/torchregister.git
cd torchregister
pip install -e ".[dev]"
```

## Quick Start

### Affine Registration

```python
import torchregister
import SimpleITK as sitk
from torchregister.metrics import NCC

# Load images
fixed_image = sitk.ReadImage("fixed.nii.gz")
moving_image = sitk.ReadImage("moving.nii.gz")

# Initialize affine registration
ncc = NCC()
affine_reg = torchregister.AffineRegistration(similarity_metric=ncc)

# Perform registration
transform, registered_image = affine_reg.register(fixed_image, moving_image)
```

### Deformable Registration (RDMM)

```python
from torchregister.metrics import LNCC

# Initialize RDMM registration
lncc = LNCC()
rdmm_reg = torchregister.RDMMRegistration(similarity_metric=lncc)

# Perform registration
deformation_field, registered_image = rdmm_reg.register(fixed_image, moving_image)
```

### Custom Loss Functions

```python
from torchregister.metrics import NCC, LNCC, MattesMI, Dice

# Use different similarity metrics
ncc_loss = NCC()
lncc_loss = LNCC(window_size=9)
mi_loss = MattesMI(bins=64)
dice_loss = Dice()
```

### SimpleITK Transform Conversion

Convert between TorchRegister transforms and SimpleITK transforms for integration with other libraries:

```python
import torchregister
import SimpleITK as sitk

# Convert PyTorch affine matrix to SimpleITK AffineTransform
affine_matrix = torch.eye(2, 3)  # 2D identity transform
sitk_transform = torchregister.torch_affine_to_sitk_transform(affine_matrix)

# Convert PyTorch deformation field to SimpleITK DisplacementFieldTransform
reference_image = sitk.Image([64, 64], sitk.sitkFloat32)
deformation_field = torch.zeros(64, 64, 2)  # 2D zero deformation
sitk_transform = torchregister.torch_deformation_to_sitk_transform(deformation_field, reference_image)

# Convert SimpleITK transforms back to PyTorch
torch_matrix = torchregister.sitk_transform_to_torch_affine(sitk_transform)
torch_deformation = torchregister.sitk_displacement_to_torch_deformation(displacement_field)
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- SimpleITK >= 2.3.0
- NumPy >= 1.24.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
