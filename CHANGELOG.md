# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows PEP 440 versioning.

## [Unreleased]
### Added
### Changed
### Fixed

## [0.1.1] - 2025-12-29 
### Added
- New metrics module for evaluating medical image synthesis (FID, MMD, MS-SSIM).
- Example script for running metrics (`examples/compute_metrics.py`).
- Additional tests for metrics and utilities (MONAI-dependent tests are skipped if MONAI is not installed).
- Updated and added a comprehensive README.md
- Improvements & Bux Fixes (Tests, Examples, Feature Extractor)

### Changed
- Improved MedicalNet feature extractor usability and checkpoint loading robustness.

## [0.1.0] - 2025-12-28 **YANKED**
### Added
- Initial release with MedicalNet-based feature extraction.

### Deprecated
- **YANKED on PyPI: do not use this release.** Please upgrade to `>= 0.1.1`.


[Unreleased]: https://github.com/yasinzaii/medmetric/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/yasinzaii/medmetric/compare/v0.1.0...v0.1.1


