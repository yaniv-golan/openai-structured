"""Tests for the testing utilities."""

import pytest

from openai_structured.errors import (
    ModelNotSupportedError,
    OpenAIClientError,
    VersionTooOldError,
)
from openai_structured.testing import (
    create_enum_constraint,
    create_numeric_constraint,
    create_test_registry,
    get_test_capabilities,
)


def test_create_test_registry() -> None:
    """Test creating a test registry."""
    # Test with default configuration
    registry = create_test_registry()
    assert registry is not None

    # Test basic model capabilities
    capabilities = registry.get_capabilities("test-model")
    assert capabilities.context_window == 4096
    assert capabilities.max_output_tokens == 2048
    assert capabilities.supports_structured

    # Test GPT-4 model capabilities
    gpt4o = registry.get_capabilities("gpt-4o")
    assert gpt4o.context_window == 128000
    assert gpt4o.max_output_tokens == 16384
    assert any(
        param.ref == "numeric_constraints.temperature"
        for param in gpt4o.supported_parameters
    )

    # Test O1 model capabilities
    o1 = registry.get_capabilities("test-o1")
    assert o1.context_window == 200000
    assert o1.max_output_tokens == 100000
    assert any(
        param.ref == "enum_constraints.reasoning_effort"
        for param in o1.supported_parameters
    )


def test_create_test_registry_with_custom_config() -> None:
    """Test creating a test registry with custom configuration."""
    # Create custom model config
    model_config = {
        "dated_models": {
            "custom-model-2024-01-01": {
                "context_window": 8192,
                "max_output_tokens": 4096,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": [
                    {"ref": "numeric_constraints.temperature"}
                ],
                "min_version": {
                    "year": 2024,
                    "month": 1,
                    "day": 1,
                },
            }
        },
        "aliases": {"custom-model": "custom-model-2024-01-01"},
    }

    # Create custom constraints config
    constraints_config = {
        "numeric_constraints": {
            "custom_param": {
                "type": "numeric",
                "min_value": 0.0,
                "max_value": 10.0,
                "description": "Custom parameter",
                "allow_float": True,
                "allow_int": True,
            }
        }
    }

    # Create registry with custom config
    registry = create_test_registry(
        model_config=model_config,
        constraints_config=constraints_config,
    )

    # Test custom model
    capabilities = registry.get_capabilities("custom-model")
    assert capabilities.context_window == 8192
    assert capabilities.max_output_tokens == 4096
    assert capabilities.supports_structured


def test_get_test_capabilities() -> None:
    """Test getting test capabilities."""
    # Test with default values
    capabilities = get_test_capabilities()
    assert capabilities.model_name == "test-model"
    assert capabilities.context_window == 4096
    assert capabilities.max_output_tokens == 2048
    assert capabilities.supports_structured
    assert capabilities.supports_streaming

    # Test with custom values
    custom_caps = get_test_capabilities(
        model_name="custom-model",
        context_window=8192,
        max_output_tokens=4096,
        supported_parameters=[
            {"ref": "numeric_constraints.temperature"},
            {"ref": "numeric_constraints.top_p"},
        ],
    )
    assert custom_caps.model_name == "custom-model"
    assert custom_caps.context_window == 8192
    assert custom_caps.max_output_tokens == 4096
    assert len(custom_caps.supported_parameters) == 2


def test_create_numeric_constraint() -> None:
    """Test creating numeric constraints."""
    # Test basic constraint
    constraint = create_numeric_constraint(0.0, 2.0)
    assert constraint.type == "numeric"
    assert constraint.min_value == 0.0
    assert constraint.max_value == 2.0
    assert constraint.allow_int
    assert constraint.allow_float

    # Test with custom options
    custom = create_numeric_constraint(
        min_value=1.0,
        max_value=10.0,
        description="Custom constraint",
        allow_int=False,
        allow_float=True,
    )
    assert custom.min_value == 1.0
    assert custom.max_value == 10.0
    assert custom.description == "Custom constraint"
    assert not custom.allow_int
    assert custom.allow_float


def test_create_enum_constraint() -> None:
    """Test creating enum constraints."""
    # Test basic constraint
    values = ["low", "medium", "high"]
    constraint = create_enum_constraint(values)
    assert constraint.type == "enum"
    assert constraint.allowed_values == values

    # Test with description
    custom = create_enum_constraint(
        values,
        description="Custom enum constraint",
    )
    assert custom.allowed_values == values
    assert custom.description == "Custom enum constraint"


def test_parameter_validation() -> None:
    """Test parameter validation with test utilities."""
    registry = create_test_registry()
    capabilities = registry.get_capabilities("test-gpt4o")

    # Test valid parameters
    capabilities.validate_parameter("temperature", 0.7)
    capabilities.validate_parameter("top_p", 0.9)

    # Test invalid parameters
    with pytest.raises(OpenAIClientError, match="must be between"):
        capabilities.validate_parameter("temperature", 2.5)

    with pytest.raises(OpenAIClientError, match="not supported"):
        capabilities.validate_parameter("invalid_param", 1.0)


def test_version_validation() -> None:
    """Test version validation with test utilities."""
    registry = create_test_registry()

    # Test valid version
    registry.get_capabilities("gpt-4o-2024-08-06")

    # Test too old version
    with pytest.raises(VersionTooOldError):
        registry.get_capabilities("test-gpt4o-2024-07-01")

    # Test unsupported model
    with pytest.raises(ModelNotSupportedError):
        registry.get_capabilities("unsupported-model")
