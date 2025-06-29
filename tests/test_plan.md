# Test Plan Overview

This document provides a detailed overview of all tests implemented in the TorchRegister project. Each test is designed to ensure the robustness and correctness of the functionalities provided by the library.

## Detailed Test Overview Table

| Test File                          | Test Name                                        | Purpose                                                                 |
|------------------------------------|--------------------------------------------------|-------------------------------------------------------------------------|
| **test_metrics.py**                |                                                  |                                                                         |
|                                    | `test_ncc_identical_images`                     | Verify NCC returns 1.0 for identical images                           |
|                                    | `test_ncc_uncorrelated_images`                  | Check NCC behavior with uncorrelated random images                     |
|                                    | `test_ncc_batch_processing`                     | Test NCC with batch dimensions                                         |
|                                    | `test_ncc_3d_images`                            | Validate NCC computation for 3D images                                 |
|                                    | `test_lncc_identical_images`                    | Verify LNCC returns 1.0 for identical images                          |
|                                    | `test_lncc_window_sizes`                        | Test LNCC with different window sizes                                  |
|                                    | `test_lncc_gradients`                           | Check LNCC gradient computation                                         |
|                                    | `test_mse_identical_images`                     | Verify MSE returns 0.0 for identical images                           |
|                                    | `test_mse_different_images`                     | Check MSE behavior with different images                               |
|                                    | `test_mse_gradients`                            | Test MSE gradient computation                                           |
|                                    | `test_mattes_mi_identical_images`               | Verify Mattes mutual information for identical images                  |
|                                    | `test_mattes_mi_different_bins`                 | Test Mattes MI with different histogram bin counts                     |
|                                    | `test_mattes_mi_gradients`                      | Check Mattes MI gradient computation                                    |
|                                    | `test_mattes_mi_constant_images`                | Check Mattes MI with constant images                                  |
|                                    | `test_dice_identical_masks`                     | Verify Dice coefficient returns 1.0 for identical masks               |
|                                    | `test_dice_no_overlap`                          | Check Dice behavior with non-overlapping masks                         |
|                                    | `test_dice_partial_overlap`                     | Test Dice with partially overlapping masks                             |
|                                    | `test_combined_loss_creation`                   | Verify combined loss function creation and configuration                |
|                                    | `test_combined_loss_computation`                | Test combined loss computation with multiple metrics                    |
|                                    | `test_combined_loss_gradients`                  | Check combined loss gradient computation                                |
|                                    | `test_all_metrics_with_same_input`              | Compare all metrics with identical inputs                               |
|                                    | `test_metrics_numerical_stability`              | Test metric stability with extreme values                              |
|                                    | `test_metric_reproducibility`                   | Verify metrics produce consistent results across runs                   |
| **test_affine.py**                 |                                                  |                                                                         |
|                                    | `test_2d_identity_initialization`               | Check 2D affine transform initializes to identity                      |
|                                    | `test_3d_identity_initialization`               | Check 3D affine transform initializes to identity                      |
|                                    | `test_2d_random_initialization`                 | Test 2D affine transform with random initialization                    |
|                                    | `test_2d_transform_application`                 | Verify 2D affine transform correctly applies to coordinates            |
|                                    | `test_3d_transform_application`                 | Verify 3D affine transform correctly applies to coordinates            |
|                                    | `test_translation_transform_2d`                 | Test pure translation transform in 2D                                  |
|                                    | `test_get_set_matrix`                           | Check getting and setting transformation matrix                         |
|                                    | `test_registration_initialization`              | Verify affine registration initializes correctly                       |
|                                    | `test_invalid_similarity_metric`                | Check error handling for invalid similarity metrics                    |
|                                    | `test_pyramid_creation_2d`                      | Test image pyramid creation for 2D images                              |
|                                    | `test_pyramid_creation_3d`                      | Test image pyramid creation for 3D images                              |
|                                    | `test_regularization_loss`                      | Verify regularization loss computation                                  |
|                                    | `test_register_identical_images_2d`             | Test registration of identical 2D images                               |
|                                    | `test_register_translated_images_2d`            | Test registration with known translation                               |
|                                    | `test_register_with_initial_transform`          | Test registration with initial transform guess                         |
|                                    | `test_evaluation_metrics`                       | Verify registration evaluation metrics                                  |
|                                    | `test_register_tensor_inputs`                   | Test registration with tensor inputs instead of images                 |
|                                    | `test_multi_scale_registration`                 | Test multi-scale registration approach                                 |
|                                    | `test_registration_convergence`                 | Verify registration converges to correct solution                      |
|                                    | `test_registration_different_metrics`           | Test registration with different similarity metrics                    |
|                                    | `test_registration_robustness_to_noise`         | Test registration robustness with noisy images                        |
| **test_rdmm.py**                   |                                                  |                                                                         |
|                                    | `test_2d_smoothing_initialization`              | Check 2D Gaussian smoothing kernel initialization                      |
|                                    | `test_3d_smoothing_initialization`              | Check 3D Gaussian smoothing kernel initialization                      |
|                                    | `test_2d_smoothing_application`                 | Test 2D Gaussian smoothing application                                 |
|                                    | `test_3d_smoothing_application`                 | Test 3D Gaussian smoothing application                                 |
|                                    | `test_different_sigma_values`                   | Test smoothing with different sigma parameters                         |
|                                    | `test_2d_velocity_field_initialization`         | Check 2D velocity field initialization                                 |
|                                    | `test_3d_velocity_field_initialization`         | Check 3D velocity field initialization                                 |
|                                    | `test_velocity_field_forward`                   | Test velocity field forward pass                                       |
|                                    | `test_velocity_field_gradients`                 | Check velocity field gradient computation                               |
|                                    | `test_registration_initialization`              | Verify RDMM registration initializes correctly                         |
|                                    | `test_invalid_similarity_metric`                | Check error handling for invalid similarity metrics                    |
|                                    | `test_velocity_integration_2d`                  | Test 2D velocity field integration to deformation                      |
|                                    | `test_velocity_integration_3d`                  | Test 3D velocity field integration to deformation                      |
|                                    | `test_jacobian_determinant_2d`                  | Verify 2D Jacobian determinant computation                             |
|                                    | `test_jacobian_determinant_3d`                  | Verify 3D Jacobian determinant computation                             |
|                                    | `test_regularization_loss_computation`          | Test regularization loss for velocity fields                           |
|                                    | `test_pyramid_creation`                         | Test image pyramid creation for RDMM                                   |
|                                    | `test_register_identical_images_2d`             | Test RDMM registration of identical images                             |
|                                    | `test_register_tensor_inputs`                   | Test RDMM registration with tensor inputs                              |
|                                    | `test_evaluation_metrics`                       | Verify RDMM registration evaluation metrics                            |
|                                    | `test_multi_scale_registration`                 | Test multi-scale RDMM registration                                     |
|                                    | `test_different_similarity_metrics`             | Test RDMM with different similarity metrics                            |
|                                    | `test_registration_with_known_deformation`      | Test RDMM registration with known deformation                          |
|                                    | `test_jacobian_determinant_properties`          | Verify mathematical properties of Jacobian determinant                 |
|                                    | `test_velocity_field_integration_properties`    | Check properties of velocity field integration                         |
|                                    | `test_registration_convergence_properties`      | Verify RDMM registration convergence properties                        |
| **test_utils.py**                  |                                                  |                                                                         |
|                                    | `test_sitk_to_torch_2d`                         | Test conversion from SimpleITK to PyTorch tensor (2D)                  |
|                                    | `test_sitk_to_torch_3d`                         | Test conversion from SimpleITK to PyTorch tensor (3D)                  |
|                                    | `test_torch_to_sitk_2d`                         | Test conversion from PyTorch tensor to SimpleITK (2D)                  |
|                                    | `test_torch_to_sitk_3d`                         | Test conversion from PyTorch tensor to SimpleITK (3D)                  |
|                                    | `test_torch_to_sitk_with_reference`             | Test tensor to SimpleITK conversion with reference image               |
|                                    | `test_roundtrip_conversion`                     | Verify round-trip image conversion preserves data                      |
|                                    | `test_save_load_image_tensor`                   | Test saving and loading PyTorch tensors as images                      |
|                                    | `test_save_load_image_sitk`                     | Test saving and loading SimpleITK images                               |
|                                    | `test_resample_by_spacing`                      | Test image resampling by target spacing                                |
|                                    | `test_resample_by_size`                         | Test image resampling by target size                                   |
|                                    | `test_resample_error_no_parameters`             | Check error handling when no resampling parameters provided            |
|                                    | `test_create_grid_2d`                           | Test coordinate grid creation for 2D                                   |
|                                    | `test_create_grid_3d`                           | Test coordinate grid creation for 3D                                   |
|                                    | `test_apply_transform_2d`                       | Test affine transform application in 2D                                |
|                                    | `test_apply_transform_3d`                       | Test affine transform application in 3D                                |
|                                    | `test_apply_deformation_2d`                     | Test deformation field application in 2D                               |
|                                    | `test_apply_deformation_with_translation`       | Test deformation field with translation component                      |
|                                    | `test_normalize_minmax`                         | Test min-max normalization                                             |
|                                    | `test_normalize_zscore`                         | Test z-score normalization                                             |
|                                    | `test_normalize_invalid_method`                 | Check error handling for invalid normalization methods                 |
|                                    | `test_compute_gradient_2d`                      | Test gradient computation for 2D images                                |
|                                    | `test_compute_gradient_3d`                      | Test gradient computation for 3D images                                |
|                                    | `test_create_identity_transform_2d`             | Test identity transform creation in 2D                                 |
|                                    | `test_create_identity_transform_3d`             | Test identity transform creation in 3D                                 |
|                                    | `test_compose_transforms_2d`                    | Test composition of multiple transforms in 2D                          |
|                                    | `test_compose_transforms_3d`                    | Test composition of multiple transforms in 3D                          |
|                                    | `test_compose_transforms_invalid_shape`         | Check error handling for invalid transform shapes                      |
|                                    | `test_tre_identical_landmarks`                  | Test target registration error with identical landmarks                |
|                                    | `test_tre_with_affine_transform`                | Test TRE calculation with affine transform                             |
|                                    | `test_tre_with_deformation`                     | Test TRE calculation with deformation field                            |
|                                    | `test_tre_no_transform`                         | Test TRE when no transform is applied                                  |
|                                    | `test_full_pipeline_2d`                         | Test complete registration pipeline in 2D                              |
|                                    | `test_error_handling`                           | Test various error conditions and edge cases                           |
|                                    | `test_numerical_stability`                      | Test numerical stability with extreme values                           |
| **test_transform_conversion.py**   |                                                  |                                                                         |
|                                    | `test_2d_identity_conversion`                   | Test conversion of 2D identity affine transform                        |
|                                    | `test_3d_identity_conversion`                   | Test conversion of 3D identity affine transform                        |
|                                    | `test_2d_translation_conversion`                | Test conversion of 2D translation transform                            |
|                                    | `test_2d_rotation_conversion`                   | Test conversion of 2D rotation transform                               |
|                                    | `test_3d_translation_conversion`                | Test conversion of 3D translation transform                            |
|                                    | `test_invalid_matrix_shape`                     | Check error handling for invalid matrix shapes                         |
|                                    | `test_with_reference_image`                     | Test transform conversion with reference image                         |
|                                    | `test_2d_zero_deformation_conversion`           | Test conversion of zero deformation field in 2D                        |
|                                    | `test_3d_zero_deformation_conversion`           | Test conversion of zero deformation field in 3D                        |
|                                    | `test_2d_nonzero_deformation_conversion`        | Test conversion of non-zero deformation field in 2D                    |
|                                    | `test_3d_nonzero_deformation_conversion`        | Test conversion of non-zero deformation field in 3D                    |
|                                    | `test_deformation_without_batch_dim`            | Test deformation conversion without batch dimension                     |
|                                    | `test_deformation_field_as_image`               | Test deformation field to SimpleITK image conversion                   |
|                                    | `test_batch_dimension_handling`                 | Test handling of batch dimensions in conversions                       |
|                                    | `test_invalid_deformation_shape`                | Check error handling for invalid deformation shapes                    |
|                                    | `test_affine_registration_to_sitk`              | Test conversion of affine registration results to SimpleITK            |
|                                    | `test_rdmm_registration_to_sitk`                | Test conversion of RDMM registration results to SimpleITK              |
|                                    | `test_transform_application_consistency`        | Verify transform application consistency between TorchRegister and SimpleITK |
|                                    | `test_transform_direction_consistency`          | Confirm transform direction convention matches between libraries        |
|                                    | `test_affine_round_trip_2d`                     | Test round-trip conversion of 2D affine transforms                     |
|                                    | `test_affine_round_trip_3d`                     | Test round-trip conversion of 3D affine transforms                     |
|                                    | `test_deformation_round_trip_2d`                | Test round-trip conversion of 2D deformation fields                    |
|                                    | `test_deformation_round_trip_3d`                | Test round-trip conversion of 3D deformation fields                    |

## Notes
- All tests are run using `pytest`.
- Comprehensive coverage is ensured for transform conversions, applications, and registration outputs.
- Tests include both 2D and 3D cases where applicable.
- Dtype handling (float32/float64) is explicitly tested in relevant sections.
- Transform direction convention (moving → fixed) is verified across all tests.

Refer to individual test files for detailed implementation and assertions.

Refer to individual test files for detailed implementation and assertions.
