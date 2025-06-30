# GitHub Copilot Instructions for torchregister

This document provides guidance for GitHub Copilot when working with this Python project.

## Project Overview
torchregister is a Python library for image registration using PyTorch. It provides implementations for various registration methods including affine and RDMM (Registration with Deformable Motion Models).

## Python Best Practices

### Code Style
- Follow PEP 8 style guidelines
- Use 4 spaces for indentation (not tabs)
- Maximum line length of 88 characters (Black formatter standard)
- Use descriptive variable and function names
- Add docstrings to all functions, classes, and modules, unless the signature is very clear

### Function Design
- Functions should do one thing and do it well
- Use type hints for function parameters and return values
- Provide clear docstrings with descriptions, parameters, and return values
- Use default parameter values where appropriate

```python
def example_function(param1: torch.Tensor, param2: float = 0.5) -> torch.Tensor:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of the return value
    """
    # Function implementation
```

### Error Handling
- Use try/except blocks to handle exceptions appropriately
- Raise specific exceptions with informative messages
- Validate input parameters

### Testing
- Write unit tests for all functions
- Use pytest for testing
- Include edge cases in tests

### Imports
- Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports (e.g., numpy, torch)
  3. Local application/library specific imports
- Sort imports alphabetically within each group

### PyTorch Specific
- Use torch.nn.Module for model components
- Ensure tensors are on the correct device (CPU/GPU)
- Use vectorized operations where possible for performance
- Properly handle tensor dimensions and shapes

### Documentation
- Document complex algorithms and implementation details
- Include examples in docstrings for non-trivial functions
- Keep documentation updated with code changes

## Project Structure
- Keep related functionality in appropriate modules
- Use relative imports within the package
- Maintain backward compatibility when making changes

## Dependencies
- Only add necessary dependencies
- Only add loose version requirements pyproject.toml

## Copilot Behavior
- Before refactoring code, make a brief plan and ask user for feedback or confirmation