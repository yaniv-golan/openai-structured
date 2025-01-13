# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-01-12

### Added

- Automated PyPI publishing workflow using GitHub Actions
  - Secure publishing using OpenID Connect (OIDC) authentication
  - Triggered automatically on new GitHub releases
  - Built and published using Poetry

### Changed

- Improved Read the Docs build configuration
  - Simplified dependency management using Poetry
  - Enhanced documentation build process reliability

### Fixed

- Documentation build failures due to missing sphinx-rtd-theme
  - Added explicit theme installation in pre-install phase
  - Ensured correct environment setup for documentation builds

## [0.4.0] - 2025-01-15

### Added

- Documentation improvements for token limits
- Enhanced JSON parsing error messages with context (±50 chars) and position information

### Changed

- Unified model references across documentation (gpt-4o, gpt-4o-mini, O1)
- Clarified that model token limits are based on current OpenAI specifications
- Improved model version documentation with date format requirements
- Removed duplicate example files for better maintainability

### Fixed

- Corrected model references in documentation
- Fixed incorrect version references
- Code formatting improvements

## [0.3.0] - 2025-01-15

### Added

- Command-line interface (ostruct)
- JSON schema validation support
- Token counting and limits
- Multiple file input support
- stdin support
- Comprehensive CLI documentation
- CLI test suite with subprocess testing

### Changed

- Made jsonschema an optional dependency
- Updated Python requirement to >=3.9
- Enhanced error handling for CLI operations
- Improved documentation structure

### Fixed

- Token limit validation
- Template placeholder validation
- Output directory handling
- API error handling in CLI

## [0.2.0] - 2025-01-12

### Added

- Initial release of openai-structured
- Support for structured output using Pydantic models
- Async streaming capabilities
- Comprehensive test suite
- Documentation with examples
- GitHub Actions CI/CD pipeline

### Changed

- Updated development dependencies
- Improved error handling
- Enhanced type hints

### Fixed

- Proper cleanup of async resources
- Event loop handling in tests

## [0.1.0] - 2024-01-09

### Added

- Initial project structure
- Basic OpenAI API integration
- Pydantic model support
- Async streaming support
- Error handling
- Documentation framework

[Unreleased]: https://github.com/yaniv-golan/openai-structured/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yaniv-golan/openai-structured/releases/tag/v0.1.0
