"""Example: Testing token limit validation."""

import pytest

from openai_structured import OpenAIClientError, TokenParameterError
from openai_structured.model_registry import ModelRegistry
from openai_structured.testing import (
    create_test_registry,
    get_test_capabilities,
)


@pytest.fixture
def token_test_registry():
    """Create a test registry with token-specific configuration."""
    # Create model config with token limits
    model_config = {
        "dated_models": {
            "test-token-model-2024-01-01": {
                "context_window": 4096,
                "max_output_tokens": 2048,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": [
                    {"ref": "numeric_constraints.max_completion_tokens"},
                ],
                "min_version": {
                    "year": 2024,
                    "month": 1,
                    "day": 1,
                },
            }
        },
        "aliases": {"test-token-model": "test-token-model-2024-01-01"},
    }

    # Create constraints config with token parameters
    constraints_config = {
        "numeric_constraints": {
            "max_completion_tokens": {
                "type": "numeric",
                "min_value": 1,
                "max_value": 2048,
                "description": "Maximum completion tokens",
                "allow_float": False,
                "allow_int": True,
            }
        }
    }

    # Create and return registry
    registry = create_test_registry(
        model_config=model_config,
        constraints_config=constraints_config,
    )
    yield registry
    ModelRegistry.cleanup()


def test_token_limit_validation(token_test_registry):
    """Test token limit validation."""
    capabilities = token_test_registry.get_capabilities("test-token-model")

    # Test valid token limits
    capabilities.validate_parameter("max_completion_tokens", 1000)

    # Test invalid token limits
    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter("max_completion_tokens", 3000)
    assert "must not exceed" in str(exc_info.value)

    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter(
            "max_completion_tokens", "not a number"
        )
    assert "must be an integer" in str(exc_info.value)

    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter("max_completion_tokens", 1.5)
    assert "must be an integer" in str(exc_info.value)


def test_token_parameter_conflicts(test_registry):
    """Test token parameter conflict handling."""
    # Get test capabilities with both token parameters
    capabilities = get_test_capabilities(
        openai_model_name="test-model",
        max_output_tokens=4096,
        supported_parameters=[
            {"ref": "numeric_constraints.max_output_tokens"},
            {"ref": "numeric_constraints.max_completion_tokens"},
        ],
    )
    used_params = set()

    # Test max_output_tokens
    capabilities.validate_parameter(
        "max_output_tokens", 1000, used_params=used_params
    )

    # Test conflict with max_completion_tokens
    with pytest.raises(TokenParameterError) as exc_info:
        capabilities.validate_parameter(
            "max_completion_tokens", 1000, used_params=used_params
        )
    assert (
        "Cannot specify both 'max_output_tokens' and 'max_completion_tokens' parameters"
        in str(exc_info.value)
    )


def test_token_parameter_types(test_registry):
    """Test token parameter type validation."""
    # Get test capabilities with token parameter
    capabilities = get_test_capabilities(
        openai_model_name="test-model",
        max_output_tokens=4096,
        supported_parameters=[
            {"ref": "numeric_constraints.max_output_tokens"},
        ],
    )

    # Test valid integer
    capabilities.validate_parameter("max_output_tokens", 1000)

    # Test invalid float
    with pytest.raises(OpenAIClientError, match="must be an integer"):
        capabilities.validate_parameter("max_output_tokens", 1000.5)

    # Test invalid string
    with pytest.raises(OpenAIClientError, match="must be an integer"):
        capabilities.validate_parameter("max_output_tokens", "1000")
