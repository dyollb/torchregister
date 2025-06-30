# Contributing to TorchRegister

Thank you for your interest in contributing to TorchRegister! We welcome contributions that help improve the library.

## Pull Request Guidelines

When submitting a pull request, please ensure the following:

### 1. Purpose Statement
- [ ] Clearly state the purpose and scope of your changes in the PR description
- [ ] Reference any related issues using `#issue-number`
- [ ] Explain what problem your PR solves or what feature it adds

### 2. Code Quality
- [ ] Pre-commit hooks have been installed and run successfully
  ```bash
  pip install pre-commit
  pre-commit install
  pre-commit run --all-files
  ```

### 3. Testing
- [ ] All tests pass locally
  ```bash
  python -m pytest tests/ -v
  ```
- [ ] New functionality includes appropriate tests
- [ ] Test coverage is maintained or improved

### 4. Documentation
- [ ] Update documentation if your changes affect the public API
- [ ] Update examples if relevant
- [ ] Follow existing code style and conventions

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`
5. Make your changes
6. Run tests: `python -m pytest tests/`
7. Run pre-commit: `pre-commit run --all-files`
8. Commit and push your changes
9. Submit a pull request

## Questions?

If you have questions about contributing, please open an issue for discussion.
