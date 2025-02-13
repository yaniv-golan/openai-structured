"""Tests for improved error messages."""

import pytest

from openai_structured.errors import (
    InvalidDateError,
    ModelNotSupportedError,
    OpenAIClientError,
    TokenParameterError,
    VersionTooOldError,
)
from openai_structured.testing import create_test_registry


def test_unsupported_model_error():
    """Test error message for unsupported model."""
    registry = create_test_registry()

    with pytest.raises(ModelNotSupportedError) as exc_info:
        registry.get_capabilities("unsupported-model")

    error_msg = str(exc_info.value)
    assert "Model 'unsupported-model' is not supported" in error_msg
    assert "Available models:" in error_msg
    assert "Dated models:" in error_msg
    assert "Aliases:" in error_msg
    assert "Note: For dated models, use format: base-YYYY-MM-DD" in error_msg


def test_unsupported_base_model_error():
    """Test error message for unsupported base model."""
    registry = create_test_registry()

    with pytest.raises(ModelNotSupportedError) as exc_info:
        registry.get_capabilities("invalid-2024-08-06")

    error_msg = str(exc_info.value)
    assert "Base model 'invalid' is not supported" in error_msg
    assert "Available base models:" in error_msg
    assert "Note: Base model names are case-sensitive" in error_msg


def test_version_too_old_error():
    """Test error message for version too old."""
    registry = create_test_registry()

    with pytest.raises(VersionTooOldError) as exc_info:
        # gpt-4o has min version 2024-08-06
        registry.get_capabilities("gpt-4o-2024-07-01")

    error_msg = str(exc_info.value)
    assert (
        "Model 'gpt-4o-2024-07-01' version 2024-07-01 is too old" in error_msg
    )
    assert "Minimum supported version:" in error_msg
    assert (
        "Note: Use the alias 'gpt-4o' to always get the latest version"
        in error_msg
    )


def test_invalid_date_error():
    """Test error message for invalid date format."""
    registry = create_test_registry()

    with pytest.raises(InvalidDateError) as exc_info:
        registry.get_capabilities("gpt-4o-2024-13-01")  # Invalid month

    error_msg = str(exc_info.value)
    assert "Invalid date format in model version:" in error_msg
    assert "Use format: YYYY-MM-DD (e.g. 2024-08-06)" in error_msg


def test_parameter_not_supported_error():
    """Test error message for unsupported parameter."""
    registry = create_test_registry()
    capabilities = registry.get_capabilities("test-model")

    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter("invalid_param", 1.0)

    error_msg = str(exc_info.value)
    assert (
        "Parameter 'invalid_param' is not supported by model 'test-model'"
        in error_msg
    )
    assert "Supported parameters:" in error_msg


def test_numeric_parameter_type_error():
    """Test error message for invalid numeric parameter type."""
    registry = create_test_registry()
    capabilities = registry.get_capabilities("test-model")

    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter(
            "temperature", "0.7"
        )  # String instead of number

    error_msg = str(exc_info.value)
    assert "Parameter 'temperature' must be a number, got str" in error_msg
    assert "Allowed types:" in error_msg


def test_numeric_parameter_range_error():
    """Test error message for numeric parameter out of range."""
    registry = create_test_registry()
    capabilities = registry.get_capabilities("test-model")

    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter("temperature", 2.5)  # Above max value

    error_msg = str(exc_info.value)
    assert "Parameter 'temperature' must be between" in error_msg
    assert "Description:" in error_msg
    assert "Current value: 2.5" in error_msg


def test_enum_parameter_error():
    """Test error message for invalid enum parameter value."""
    registry = create_test_registry()
    capabilities = registry.get_capabilities("test-o1")

    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter("reasoning_effort", "invalid")

    error_msg = str(exc_info.value)
    assert (
        "Invalid value 'invalid' for parameter 'reasoning_effort'" in error_msg
    )
    assert "Description:" in error_msg
    assert "Allowed values:" in error_msg


def test_token_parameter_error():
    """Test error message for using both token parameters."""
    registry = create_test_registry()
    capabilities = registry.get_capabilities("test-model")
    used_params = {"max_output_tokens"}

    with pytest.raises(TokenParameterError) as exc_info:
        capabilities.validate_parameter(
            "max_completion_tokens", 1000, used_params=used_params
        )

    error_msg = str(exc_info.value)
    assert (
        "Cannot specify both 'max_output_tokens' and 'max_completion_tokens' parameters"
        in error_msg
    )
    assert "Choose one:" in error_msg
    assert "max_output_tokens (recommended)" in error_msg
    assert "max_completion_tokens (legacy)" in error_msg
