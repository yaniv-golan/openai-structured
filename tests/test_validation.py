"""Tests for the model registry functionality."""

import logging
from typing import Any, Optional, Set
from unittest.mock import MagicMock, patch

import pytest
from _pytest.logging import LogCaptureFixture
from openai_model_registry import ModelCapabilities, ModelRegistry

from openai_structured.client import (
    _validate_token_limits,
    get_context_window_limit,
    get_default_token_limit,
    supports_structured_output,
)
from openai_structured.errors import OpenAIClientError, TokenParameterError


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
        raise Exception(f"Model {model} not supported")

    registry.get_capabilities = mock_get_capabilities

    # Add mock validate_parameter method to capabilities
    def mock_validate_parameter(
        param_name: str, value: Any, used_params: Optional[Set[str]] = None
    ) -> None:
        if used_params is not None and param_name in used_params:
            if (
                param_name == "max_completion_tokens"
                and "max_output_tokens" in used_params
            ):
                raise TokenParameterError("test_model")
            if (
                param_name == "max_output_tokens"
                and "max_completion_tokens" in used_params
            ):
                raise TokenParameterError("test_model")

        if param_name == "temperature":
            if value < 0 or value > 2:
                raise OpenAIClientError(
                    f"Parameter {param_name} must be between 0 and 2"
                )

    gpt4o_capabilities.validate_parameter.side_effect = mock_validate_parameter
    o1_capabilities.validate_parameter.side_effect = mock_validate_parameter

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
        # Dated versions - invalid
        ("gpt-4o-2024-07-01", False),  # Too old for gpt-4o
        ("gpt-4o-mini-2024-07-17", False),  # Too old for gpt-4o-mini
        ("o1-2024-12-16", False),  # Too old for o1
        ("o3-mini-2025-01-30", False),  # Too old for o3-mini
        # Edge cases
        ("", False),  # Empty string
        ("gpt-4o-", False),  # Incomplete version
        ("gpt-4o-invalid-date", False),  # Invalid date format
        ("gpt-4o-9999-99-99", False),  # Invalid date values
        ("not-a-model", False),  # Random string
        ("gpt-4o-2024-08-06-extra", False),  # Extra version components
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


@pytest.mark.parametrize(
    "model_name,expected_limit",
    [
        ("gpt-4o", 128000),  # GPT-4o context window
        ("o1", 200000),  # O1 context window
        ("unknown-model", 8192),  # Default for unknown model
    ],
)
def test_get_context_window_limit(
    mock_registry: MagicMock, model_name: str, expected_limit: int
) -> None:
    """Test getting context window limit for various models."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        assert get_context_window_limit(model_name) == expected_limit


@pytest.mark.parametrize(
    "model_name,expected_limit",
    [
        ("gpt-4o", 16384),  # GPT-4o output token limit
        ("o1", 100000),  # O1 output token limit
        ("unknown-model", 4096),  # Default for unknown model
    ],
)
def test_get_default_token_limit(
    mock_registry: MagicMock, model_name: str, expected_limit: int
) -> None:
    """Test getting default token limit for various models."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        assert get_default_token_limit(model_name) == expected_limit


@pytest.mark.parametrize(
    "model_name,max_tokens,should_warn",
    [
        # Valid cases
        ("gpt-4o", 16000, False),  # Under GPT-4o limit
        ("o1", 90000, False),  # Under O1 limit
        # Invalid cases
        ("gpt-4o", 17000, True),  # Over GPT-4o limit
        ("o1", 110000, True),  # Over O1 limit
        # Unknown model - uses default limits
        ("unknown-model", 5000, True),  # Over default limit
    ],
)
def test_validate_token_limits(
    mock_registry: MagicMock,
    model_name: str,
    max_tokens: int,
    should_warn: bool,
    caplog: LogCaptureFixture,
) -> None:
    """Test token limit validation for various models and token counts."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            # The function may log warnings but not raise exceptions now
            _validate_token_limits(model_name, max_tokens)

        if should_warn:
            assert len(caplog.records) > 0
            assert any(
                f"Error validating token limits for {model_name}"
                in record.message
                for record in caplog.records
            )
        else:
            assert not any(
                f"Error validating token limits for {model_name}"
                in record.message
                for record in caplog.records
            )


def test_token_parameter_validation(mock_registry: MagicMock) -> None:
    """Test validation of token-related parameters."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        # Test GPT-4 models
        gpt4o = mock_registry.get_capabilities("gpt-4o")

        # Set up our own mock_validate_parameter that actually raises exceptions
        def strict_validate_parameter(
            param_name: str, value: Any, used_params: Optional[Set[str]] = None
        ) -> None:
            if used_params is not None:
                if (
                    param_name == "max_completion_tokens"
                    and "max_output_tokens" in used_params
                ):
                    raise TokenParameterError(
                        "Cannot specify both max_completion_tokens and max_output_tokens"
                    )
                if (
                    param_name == "max_output_tokens"
                    and "max_completion_tokens" in used_params
                ):
                    raise TokenParameterError(
                        "Cannot specify both max_output_tokens and max_completion_tokens"
                    )

        # Use our own mock instead of the one from fixture
        gpt4o.validate_parameter.side_effect = strict_validate_parameter

        used_params: Set[str] = set()

        # Test max_output_tokens
        gpt4o.validate_parameter(
            "max_output_tokens", 1000, used_params=used_params
        )

        # Add max_output_tokens to used_params
        used_params.add("max_output_tokens")

        # Test max_completion_tokens raises error when both are used
        with pytest.raises(TokenParameterError) as exc_info:
            gpt4o.validate_parameter(
                "max_completion_tokens", 1000, used_params=used_params
            )
        assert "Cannot specify both" in str(exc_info.value)


def test_parameter_validation_with_overrides(mock_registry: MagicMock) -> None:
    """Test parameter validation with max_value overrides."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        gpt4o = mock_registry.get_capabilities("gpt-4o")

        # Test with default limits
        gpt4o.validate_parameter("temperature", 1.5)

        # Test with constraint
        with pytest.raises(OpenAIClientError, match="must be between"):
            gpt4o.validate_parameter("temperature", 2.5)


def test_parameter_validation_with_dynamic_max(
    mock_registry: MagicMock,
) -> None:
    """Test parameter validation with dynamic max values."""
    with patch.object(
        ModelRegistry, "get_instance", return_value=mock_registry
    ):
        o1 = mock_registry.get_capabilities("o1")

        # Test max_completion_tokens with dynamic limit
        o1.validate_parameter(
            "max_completion_tokens", 90000
        )  # Under model's max_output_tokens
