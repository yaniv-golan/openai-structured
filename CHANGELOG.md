# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-03-10

### Breaking Changes

- Renamed `model_name` field to `openai_model_name` in `ModelCapabilities` class
  - Any code referencing the `model_name` property will need to be updated
  - All constructor calls using the `model_name` parameter must be changed to use `openai_model_name`

### Fixed

- Resolved an issue where using field names starting with "model_" in Pydantic models would trigger protected namespace warnings

## [2.1.0] - 2025-03-05

### Added

- Enhanced registry update mechanism with HTTP caching
  - Added `RegistryUpdateStatus` enum and `RegistryUpdateResult` dataclass for detailed status reporting
  - Added `check_for_updates` method to check for available updates without downloading
  - Added `--check` option to `openai-structured-refresh` command
  - Added cache metadata tracking via `.yml.meta` files for ETag and Last-Modified headers

### Changed

- Improved HTTP request efficiency with conditional requests
  - Added support for ETag and If-Modified-Since headers
  - Reduced bandwidth usage by avoiding unnecessary downloads
- Enhanced verbose output in CLI
  - Added local registry file path to output
  - Improved token size display with human-readable K format
  - Added detailed status information in verbose mode
- Updated error handling with more specific status codes and messages

### Fixed

- Improved streaming response handling to prevent content loss
  - Modified `StreamBuffer.close()` to extract final responses before closing
  - Updated stream functions to properly handle and yield final content
  - Made streaming tests more resilient to API response variations

## [2.0.1] - 2025-02-28

### Added

- Added support for GPT-4.5 Preview model:
  - New dated model `gpt-4.5-preview-2025-02-27` with 128K context window and 16K output tokens
  - New alias `gpt-4.5-preview` pointing to the dated model
  - Full parameter validation and constraint support

### Fixed

- Improved `update_fallbacks.py` script to properly handle ModelVersion as dictionaries
- Updated fallback models with the latest configuration

## [2.0.0] - 2025-02-20

### Breaking Changes

- Model Registry Architecture:
  - Introduced singleton ModelRegistry for centralized model management
  - Strict version validation for all model names
  - Required model capabilities validation through registry
  - Removed direct model support checks in favor of registry-based validation

- Parameter Validation:
  - Added mandatory parameter validation for all model parameters
  - Parameters must now match model-specific constraints
  - `max_output_tokens` and `max_completion_tokens` are now mutually exclusive
  - Removed support for unvalidated parameters

- Error Handling:
  - Restructured error hierarchy with new specific error types
  - Changed error message format to include more context
  - Added new `TokenParameterError` for token parameter conflicts
  - Some error types now include additional context (e.g., available models, parameter constraints)

### Added

- Enhanced model registry with comprehensive validation
- Support for new model variants and aliases
- Comprehensive parameter validation with constraints
- Live testing capabilities
- Improved type hints and documentation
- New error types for better error handling
- Enhanced streaming support with better buffer management

### Changed

- Complete refactor of model registry and validation system
- Improved error handling and reporting
  - Enhanced error messages with more descriptive information
  - Added parameter type details and allowed values
  - Included model version format examples
  - Added categorized lists of available models
  - Added suggestions for using aliases
- Enhanced type safety across the codebase
- Updated model specifications and requirements
- Improved documentation
  - Added comprehensive error handling examples
  - Enhanced exception documentation
  - Added testing examples section
  - Updated model registry documentation
- Improved code organization
  - Fixed linting issues and removed unused imports
  - Enhanced import organization in test files
  - Improved code readability and maintainability

### Removed

- Deprecated model versions and outdated validation methods
- Legacy error handling patterns

## [1.3.0] - 2025-02-15

### Added

- Support for token limit validation
  - Added `get_context_window_limit` function
  - Added `get_default_token_limit` function
  - Added `TokenLimitError` exception class
- Enhanced model token limit documentation
  - Updated model specifications
  - Added token limit examples
  - Improved error handling documentation

### Changed

- Improved error handling for token limits
  - Added token limit validation in request preparation
  - Enhanced error messages with detailed token information
  - Updated examples with token limit handling

## [1.2.0] - 2025-02-08

### Added

- New `ClosedBufferError` class for better error handling of closed buffer states
- Enhanced mypy configuration for improved type checking and package structure

### Changed

- Improved error handling in StreamBuffer
  - Fixed incorrect error type when writing to closed buffer
  - Better error variable naming for clarity
  - Enhanced error documentation
- Fixed import paths to use installed package structure
- Updated mypy configuration for better src layout support
  - Added proper path configuration
  - Improved cache management
  - Enhanced import handling
  - Added Pydantic-specific overrides

### Fixed

- Bug fix: Using correct error type (`ClosedBufferError` instead of `BufferOverflowError`) when writing to closed buffer
- Import path in test files to use installed package path

## [1.1.0] - 2025-02-01

### Added

- Support for o3-mini model
  - 200K context window
  - 100K output tokens
  - Minimum version: 2025-01-31
- Enhanced model validation
  - Added test cases for o3-mini model
  - Added version validation for o3-mini
  - Added support for o3-mini alias

### Changed

- Updated documentation with o3-mini model specifications
- Improved code organization
  - Removed unused imports
  - Enhanced test structure

## [1.0.0] - 2025-01-26

### Added

- First stable release with complete feature set
- Finalized API for structured output processing
- Comprehensive documentation and examples
- Full test coverage and type safety

### Changed

- Moved CLI functionality to separate `ostruct-cli` package
- Stabilized public API interfaces
- Enhanced error handling and reporting
- Improved streaming performance and reliability

### Removed

- Deprecated CLI functionality (now available in `ostruct-cli` package)

## [0.9.2] - 2025-01-25

### Added

- Added deprecation warning for CLI functionality, which will be moved to `ostruct-cli` package in version 1.0.0
- Added deprecation notices in CLI documentation (README.md, cli.rst, cli_quickstart.rst)

### Changed

- Updated documentation to point users to the future `ostruct` repository for CLI features

## [0.9.1] - 2025-01-25

### Changed

- Improved file cache reliability
  - Switched to nanosecond precision for file modification times
  - Added file size validation for cache invalidation
  - Enhanced logging for better debugging
  - Increased retry logic in CI environments
- Enhanced test robustness in CI environments
  - Added verbose output and debug logging to pytest
  - Improved file system synchronization with `os.fsync()`
  - Increased retry attempts for file stat changes

## [0.9.0] - 2025-01-24

### Added

- New Code Review Use Case Example
  - Added complete code review automation example
  - Includes security, style, and performance analysis
  - CI/CD integration examples for GitHub Actions and GitLab
  - Structured output schema for review results
- Enhanced Template Engine
  - Added new `CommentExtension` for ignoring variables in comment blocks
  - Introduced dedicated template environment configuration
  - Added validation mode for template processing
  - Enhanced template filters and utilities
- Security Improvements
  - Added `--allowed-dir` option for explicit directory access control
  - Support for loading allowed directories from file with `@file` syntax
  - Multiple allowed directories support
  - Enhanced path traversal prevention
- New Dependencies
  - Added `ijson` ^3.2.3 for efficient JSON streaming
  - Added `chardet` ^5.0.0 for character encoding detection
  - Added `typing-extensions` ^4.9.0 for enhanced type support
  - Added example dependencies: `tenacity` and `asyncio-throttle`

### Changed

- Template Processing Architecture
  - Split template functionality into specialized modules:
    - `template_env.py` for environment configuration
    - `template_extensions.py` for custom extensions
    - `template_validation.py` for validation logic
  - Improved template error handling with dedicated exceptions
  - Enhanced template validation with SafeUndefined support
- Code Organization
  - Moved test models to dedicated support package
  - Improved modularity of CLI components
  - Enhanced test organization and coverage
- Documentation
  - Updated CLI documentation with new security features
  - Added comprehensive use case examples
  - Improved integration guides

### Fixed

- Template Validation
  - Fixed handling of undefined variables in templates
  - Improved error messages for validation failures
  - Enhanced handling of edge cases in template rendering
- Security
  - Fixed potential path traversal vulnerabilities
  - Improved file access validation
  - Enhanced security manager checks
- Type Safety
  - Added missing type annotations
  - Fixed type checking issues with pytest decorators
  - Improved overall type safety across the codebase

## [0.8.0] - 2025-01-22

### Added

- Template system prompts with YAML frontmatter support
  - Added support for template metadata and configuration
  - Enhanced template validation and processing
  - Added template filters for data transformation
- Comprehensive security management for file operations
  - Added SecurityManager for path validation
  - Implemented secure file access controls
  - Added path traversal protection
- New template rendering capabilities
  - Added lazy content loading for improved performance
  - Enhanced template context validation
  - Added support for complex data structures in templates

### Changed

- Complete CLI architecture overhaul
  - Split monolithic cli.py into specialized modules
  - Improved code organization and maintainability
  - Enhanced error handling with custom exceptions
- Enhanced FileInfo handling
  - Improved encapsulation and security
  - Added better file metadata management
  - Implemented efficient content caching
- Progress control moved to command-line flags
  - Simplified progress reporting configuration
  - Added granular control over progress updates

### Fixed

- Template validation and processing
  - Fixed template placeholder validation
  - Improved error reporting for template issues
  - Enhanced type safety in template rendering
- File operation security
  - Fixed path traversal vulnerabilities
  - Enhanced file access validation
  - Improved error handling for file operations

## [0.7.0] - 2024-01-20

### Added

- New `--dry-run` option for CLI operations
  - Preview template output without writing files
  - Test template processing without side effects
- Enhanced stream buffer implementation
  - Added new `StreamBuffer` class for efficient streaming
  - Improved memory management for large responses
  - Added buffer cleanup mechanisms
- Comprehensive Jinja2 template support
  - Added advanced template filters
  - Enhanced template processing capabilities
  - Improved template error handling

### Changed

- Improved template engine architecture
  - Refactored template processing for better maintainability
  - Enhanced test coverage for template operations
  - Optimized template rendering performance
- Enhanced documentation
  - Updated API documentation with new features
  - Improved CLI documentation with examples
  - Added comprehensive template usage guide
- Standardized OpenAI Structured Outputs references
  - Unified terminology across documentation
  - Improved consistency in API references
  - Enhanced example clarity

### Fixed

- Buffer management and error handling
  - Improved stream buffer cleanup
  - Enhanced error reporting for template issues
  - Fixed memory management in streaming operations
- Documentation and examples
  - Fixed incorrect schema references
  - Updated outdated examples
  - Corrected API usage documentation

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
- Added missing aiohttp type hints for mypy

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

[0.8.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yaniv-golan/openai-structured/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yaniv-golan/openai-structured/releases/tag/v0.1.0
