"""Example: Basic parameter validation testing."""

import pytest

from openai_structured import OpenAIClientError


def test_basic_parameter_validation(test_registry):
    """Test basic parameter validation."""
    # Get capabilities for a test model
    capabilities = test_registry.get_capabilities("test-model")

    # Test valid parameters
    capabilities.validate_parameter("temperature", 0.7)
    capabilities.validate_parameter("top_p", 0.9)

    # Test invalid parameters
    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter("temperature", 2.5)
    assert "must be between" in str(exc_info.value)

    with pytest.raises(OpenAIClientError) as exc_info:
        capabilities.validate_parameter("invalid_param", 1.0)
    assert "not supported by model" in str(exc_info.value)
