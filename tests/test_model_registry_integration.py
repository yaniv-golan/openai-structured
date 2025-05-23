"""Tests for integration with the openai-model-registry package."""

from typing import Any, Optional, Set
from unittest.mock import MagicMock, patch

import pytest
from openai_model_registry import ModelCapabilities, ModelRegistry
from openai_model_registry.errors import (
    ModelNotSupportedError,
    VersionTooOldError,
)

from openai_structured.client import supports_structured_output


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create a mock registry with test models."""
    # Create a mock registry
    registry = MagicMock()

    # Create capabilities for different models
    gpt4o_capabilities = MagicMock(spec=ModelCapabilities)
    gpt4o_capabilities.context_window = 128000
    gpt4o_capabilities.max_output_tokens = 16384
    gpt4o_capabilities.supports_streaming = True
    gpt4o_capabilities.supports_structured = True

    o1_capabilities = MagicMock(spec=ModelCapabilities)
    o1_capabilities.context_window = 200000
    o1_capabilities.max_output_tokens = 100000
    o1_capabilities.supports_streaming = True
    o1_capabilities.supports_structured = True

    # Set up get_capabilities to return appropriate mock capabilities
    def mock_get_capabilities(model: str) -> MagicMock:
        if model in [
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-09-01",
        ]:
            return gpt4o_capabilities
        elif model in [
            "o1",
            "o1-2024-12-17",
            "o3-mini",
            "o3-mini-2025-01-31",
            "o3-mini-2025-02-01",
        ]:
            return o1_capabilities
        elif model in [
            "gpt-4o-2024-07-01",
            "gpt-4o-mini-2024-07-17",
            "o1-2024-12-16",
            "o3-mini-2025-01-30",
        ]:
            raise VersionTooOldError(
                "Version too old", model, "minimum_version"
            )
        else:
            raise ModelNotSupportedError(f"Model {model} not supported")

    registry.get_capabilities = mock_get_capabilities

    return registry


@pytest.mark.parametrize(
    "model_name,expected",
    [
        # Alias cases
        ("gpt-4o", True),  # Basic alias
        ("gpt-4o-mini", True),  # Mini variant
        ("o1", True),  # O1 model
        ("o3-mini", True),  # O3 mini model
        ("gpt-3.5-turbo", False),  # Unsupported model
        # Dated versions - valid
        ("gpt-4o-2024-08-06", True),  # Base version
        ("gpt-4o-mini-2024-07-18", True),  # Mini base version
        ("o1-2024-12-17", True),  # O1 base version
        ("o3-mini-2025-01-31", True),  # O3 mini base version
        ("gpt-4o-2024-09-01", True),  # Newer version
        ("o3-mini-2025-02-01", True),  # Newer version
        # Edge cases
        ("", False),  # Empty string
        ("not-a-model", False),  # Random string
    ],
)
def test_supports_structured_output(
    mock_registry: MagicMock, model_name: str, expected: bool
) -> None:
    """Test model support validation with various model names."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        assert supports_structured_output(model_name) == expected


def test_get_capabilities(mock_registry: MagicMock) -> None:
    """Test getting model capabilities."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        # Test getting a valid model
        gpt4o_capabilities = mock_registry.get_capabilities("gpt-4o")
        assert gpt4o_capabilities.context_window == 128000
        assert gpt4o_capabilities.max_output_tokens == 16384
        assert gpt4o_capabilities.supports_streaming is True
        assert gpt4o_capabilities.supports_structured is True


def test_version_errors(mock_registry: MagicMock) -> None:
    """Test error cases with model versions."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        # Test unsupported model
        with pytest.raises(ModelNotSupportedError):
            mock_registry.get_capabilities("nonexistent-model")

        # For known models with old versions, test version too old error
        with pytest.raises(VersionTooOldError):
            mock_registry.get_capabilities("gpt-4o-2024-07-01")


def test_parameter_validation_with_mock(mock_registry: MagicMock) -> None:
    """Test parameter validation functionality using mock registry."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        # Get capabilities from our mock
        capabilities = mock_registry.get_capabilities("gpt-4o")

        # Set up mock behavior for validate_parameter
        def validate_side_effect(
            param_name: str, value: Any, used_params: Optional[Set[str]] = None
        ) -> None:
            if param_name == "temperature":
                if value < 0 or value > 2:
                    raise Exception(
                        f"Temperature must be between 0 and 2, got {value}"
                    )
            elif param_name == "nonexistent_param":
                raise Exception(f"Parameter {param_name} not supported")

        capabilities.validate_parameter.side_effect = validate_side_effect

        # Test valid parameters
        capabilities.validate_parameter("temperature", 0.7)

        # Test invalid parameters
        with pytest.raises(
            Exception, match="Temperature must be between 0 and 2"
        ):
            capabilities.validate_parameter("temperature", 2.5)

        with pytest.raises(Exception, match="not supported"):
            capabilities.validate_parameter("nonexistent_param", 123)
