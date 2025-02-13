"""Example: Testing with custom model configurations."""

import pytest

from openai_structured import OpenAIClientError, VersionTooOldError
from openai_structured.model_registry import ModelRegistry
from openai_structured.testing import (
    create_enum_constraint,
    create_numeric_constraint,
    create_test_registry,
)


@pytest.fixture
def custom_test_registry():
    """Custom test registry with specific model configuration."""
    # Create custom model config
    model_config = {
        "dated_models": {
            "test-custom-model-2024-01-01": {
                "context_window": 8192,
                "max_output_tokens": 4096,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": [
                    {"ref": "numeric_constraints.temperature"},
                    {"ref": "numeric_constraints.top_p"},
                ],
                "min_version": {
                    "year": 2024,
                    "month": 1,
                    "day": 1,
                },
            }
        },
        "aliases": {"test-custom-model": "test-custom-model-2024-01-01"},
    }

    # Create custom constraints config
    constraints_config = {
        "numeric_constraints": {
            "temperature": {
                "type": "numeric",
                "min_value": 0.0,
                "max_value": 1.0,
                "description": "Custom temperature range",
                "allow_float": True,
                "allow_int": False,
            },
            "top_p": {
                "type": "numeric",
                "min_value": 0.0,
                "max_value": 1.0,
                "description": "Custom top_p range",
                "allow_float": True,
                "allow_int": False,
            },
        }
    }

    # Create and return registry with custom configuration
    registry = create_test_registry(
        model_config=model_config,
        constraints_config=constraints_config,
    )
    yield registry
    ModelRegistry.cleanup()


def test_custom_model_capabilities(custom_test_registry):
    """Test custom model capabilities."""
    # Test basic capabilities
    capabilities = custom_test_registry.get_capabilities("test-custom-model")
    assert capabilities.context_window == 8192
    assert capabilities.max_output_tokens == 4096
    assert capabilities.supports_structured
    assert capabilities.supports_streaming

    # Test parameter validation
    capabilities.validate_parameter("temperature", 0.5)
    capabilities.validate_parameter("top_p", 0.9)

    # Test invalid parameters
    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter("temperature", 1.5)
    assert "must be between" in str(exc_info.value)

    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter("invalid_param", 1.0)
    assert "not supported by model" in str(exc_info.value)

    # Test version validation
    custom_test_registry.get_capabilities("test-custom-model-2024-01-01")
    with pytest.raises(VersionTooOldError) as exc_info:
        custom_test_registry.get_capabilities("test-custom-model-2023-12-31")
    assert "is too old" in str(exc_info.value)


def test_custom_parameter_validation(test_registry):
    """Test parameter validation with custom constraints."""
    # Create custom constraints
    temp_constraint = create_numeric_constraint(
        min_value=0.0,
        max_value=1.0,
        description="Custom temperature range",
        allow_float=True,
        allow_int=False,
    )

    effort_constraint = create_enum_constraint(
        allowed_values=["low", "medium", "high"],
        description="Custom reasoning effort levels",
    )

    # Create registry with custom constraints
    constraints_config = {
        "numeric_constraints": {
            "temperature": temp_constraint.model_dump(),
        },
        "enum_constraints": {
            "reasoning_effort": effort_constraint.model_dump(),
        },
    }

    registry = create_test_registry(constraints_config=constraints_config)
    capabilities = registry.get_capabilities("test-model")

    # Test numeric parameters
    capabilities.validate_parameter("temperature", 0.7)  # Float OK
    with pytest.raises(OpenAIClientError, match="must be a float"):
        capabilities.validate_parameter(
            "temperature", 1
        )  # Integer not allowed

    # Test enum parameters
    o1_caps = registry.get_capabilities("o1")
    o1_caps.validate_parameter("reasoning_effort", "medium")  # Valid value
    with pytest.raises(OpenAIClientError, match="Invalid value"):
        o1_caps.validate_parameter("reasoning_effort", "invalid")
