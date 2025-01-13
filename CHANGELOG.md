# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2025-01-20

### Added

- New API functions for structured output
  - Added `openai_structured_call` for single responses
  - Added `openai_structured_stream` for streaming responses
  - Added `supports_structured_output` for checking model compatibility and version requirements
- Enhanced streaming support with efficient buffer management
  - Added `StreamBuffer` class for optimized memory usage
  - Implemented buffer cleanup mechanism for large responses
  - Added configurable buffer size limits and thresholds
- Improved error handling and reporting
  - Added detailed JSON parsing error messages with context
  - Enhanced validation error reporting with response snippets
  - Added response ID tracking in error messages
- Comprehensive logging system
  - Added detailed debug logging for API requests and responses
  - Added structured logging with payload information
  - Added streaming-specific debug logs
- Enhanced model version management
  - Added `ModelVersion` class for semantic version comparison
  - Added strict version validation for supported models
  - Updated model specifications and token limits

### Changed

- Rename CLI command from oai-structured-cli to ostruct
- Update documentation with comprehensive CLI guide
- Add supported models documentation
- Expand error handling documentation
- Improved type safety with strict type hints
- Enhanced error hierarchy and exception handling
- Optimized streaming performance with chunked processing

### Fixed

- Buffer overflow handling in streaming responses
- JSON parsing error handling with context information
- Model version validation and comparison
- Type hints and validation in API responses

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

[Unreleased]: https://github.com/yaniv-golan/openai-structured/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yaniv-golan/openai-structured/releases/tag/v0.1.0
