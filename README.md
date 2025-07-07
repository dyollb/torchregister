# TorchRegister

[![CI](https://github.com/dyollb/torchregister/actions/workflows/ci.yml/badge.svg)](https://github.com/dyollb/torchregister/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dyollb/torchregister/branch/main/graph/badge.svg)](https://codecov.io/gh/dyollb/torchregister)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based package for multi-scale affine image registration.

## Features

- **Multi-scale Registration**: Supports affine registration
- **Differentiable Losses**: Implementation of various similarity metrics (NCC, LNCC, MSE, Mattes MI, Dice)
- **PyTorch Integration**: Fully differentiable and GPU-accelerated
- **SimpleITK IO**: Seamless integration with medical imaging formats
- **Multi-Modal Support**: Handle multi-channel images (T1, T2, FLAIR, etc.)
- **Comprehensive Testing**: Full test coverage with pytest

## Documentation

- No documentation atm, but try out the [script](examples/register_affine_cli.py) in the examples folder

## Installation

```bash
pip install torchregister
```

For development:

```bash
git clone https://github.com/dyollb/torchregister.git
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

affine_reg = torchregister.AffineRegistration(similarity_metric=ncc)
affine, _ = affine_reg.register(fixed_image, moving_image)

sitk_transform = torchregister.torch_affine_to_sitk_transform(affine, fixed_image=fixed_image, moving_image=moving_image)
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
