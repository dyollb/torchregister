from pathlib import Path

import SimpleITK as sitk
import torch

import torchregister
from torchregister.conversion import torch_affine_to_sitk_transform
from torchregister.metrics import MattesMI


def register(
    moving_file: Path, fixed_file: Path, output_dir: Path, device: str | None = None
):
    moving_image = sitk.ReadImage(moving_file)
    fixed_image = sitk.ReadImage(fixed_file)

    loss = MattesMI()
    affine_reg = torchregister.AffineRegistration(
        similarity_metric=loss,
        shrink_factors=[4, 2],
        smoothing_sigmas=[2.0, 1.0],
        num_iterations=[40, 40],
        learning_rate=0.01,
        device=None if device is None else torch.device(device),
    )

    transform_matrix, registered_image = affine_reg.register(fixed_image, moving_image)

    output_dir.mkdir(exist_ok=True, parents=True)
    torchregister.io.save_image(
        registered_image, output_dir / "registered.nii.gz", reference_image=fixed_image
    )

    tx = torch_affine_to_sitk_transform(transform_matrix)
    sitk.WriteTransform(tx, output_dir / "tx.tfm")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetSize(fixed_image.GetSize())
    resampler.SetOutputPixelType(moving_image.GetPixelID())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    resampler.SetTransform(tx)
    sitk.WriteImage(
        resampler.Execute(moving_image),
        output_dir / "registered (sitk transformed).nii.gz",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("moving_file", type=Path, help="Path to the moving image file")
    parser.add_argument("fixed_file", type=Path, help="Path to the fixed image file")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save the output files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for running tensor computations",
    )
    args = parser.parse_args()

    register(args.moving_file, args.fixed_file, args.output_dir, args.device)
