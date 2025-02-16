"""Tests for the model registry functionality."""

from typing import Optional, Set

import pytest

from openai_structured.client import (
    _validate_token_limits,
    get_context_window_limit,
    get_default_token_limit,
    supports_structured_output,
)
from openai_structured.errors import (
    OpenAIClientError,
    TokenLimitError,
    TokenParameterError,
)
from openai_structured.model_registry import ModelRegistry


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
def test_supports_structured_output(model_name: str, expected: bool) -> None:
    """Test model support validation with various model names.

    Args:
        model_name: Name of the model to test
        expected: Expected result of the validation

    This test covers:
    1. Basic alias validation
    2. Dated version validation (minimum and newer versions)
    3. Invalid version handling
    4. Edge cases and error conditions
    """
    assert supports_structured_output(model_name) == expected


@pytest.mark.parametrize(
    "model_name,expected_limit",
    [
        # Alias cases
        ("gpt-4o", 128_000),  # Basic GPT-4 model
        ("gpt-4o-mini", 128_000),  # GPT-4 mini model
        ("o1", 200_000),  # o1 model
        ("o3-mini", 200_000),  # o3-mini model
        # Dated versions
        ("gpt-4o-2024-08-06", 128_000),  # GPT-4 base version
        ("gpt-4o-mini-2024-07-18", 128_000),  # GPT-4 mini base version
        ("o1-2024-12-17", 200_000),  # o1 base version
        ("o3-mini-2025-01-31", 200_000),  # o3-mini base version
        # Edge cases
        ("unknown-model", 8_192),  # Unknown model gets default
        ("", 8_192),  # Empty string gets default
    ],
)
def test_context_window_limit(model_name: str, expected_limit: int) -> None:
    """Test context window limit calculation for various models."""
    assert get_context_window_limit(model_name) == expected_limit


@pytest.mark.parametrize(
    "model_name,expected_limit",
    [
        # Alias cases
        ("gpt-4o", 16_384),  # Basic GPT-4 model
        ("gpt-4o-mini", 16_384),  # GPT-4 mini model
        ("o1", 100_000),  # o1 model
        ("o3-mini", 100_000),  # o3-mini model
        # Dated versions
        ("gpt-4o-2024-08-06", 16_384),  # GPT-4 base version
        ("gpt-4o-mini-2024-07-18", 16_384),  # GPT-4 mini base version
        ("o1-2024-12-17", 100_000),  # o1 base version
        ("o3-mini-2025-01-31", 100_000),  # o3-mini base version
        # Edge cases
        ("unknown-model", 4_096),  # Unknown model gets default
        ("", 4_096),  # Empty string gets default
    ],
)
def test_default_token_limit(model_name: str, expected_limit: int) -> None:
    """Test default token limit calculation for various models."""
    assert get_default_token_limit(model_name) == expected_limit


@pytest.mark.parametrize(
    "model_name,max_tokens,should_raise",
    [
        # Valid cases - GPT-4 models
        ("gpt-4o", 16_384, False),  # Exactly at limit
        ("gpt-4o", 16_000, False),  # Under limit
        ("gpt-4o-mini", 16_384, False),  # Exactly at limit
        ("gpt-4o-mini", 16_000, False),  # Under limit
        ("gpt-4o-2024-08-06", 16_384, False),  # Base version at limit
        (
            "gpt-4o-mini-2024-07-18",
            16_384,
            False,
        ),  # Mini base version at limit
        # Valid cases - o1/o3 models
        ("o1", 100_000, False),  # Exactly at limit
        ("o1", 90_000, False),  # Under limit
        ("o3-mini", 100_000, False),  # Exactly at limit
        ("o3-mini", 90_000, False),  # Under limit
        ("o1-2024-12-17", 100_000, False),  # Base version at limit
        ("o3-mini-2025-01-31", 100_000, False),  # Base version at limit
        # Edge cases
        (None, None, False),  # No token limit specified
        # Invalid cases - GPT-4 models
        ("gpt-4o", 16_385, True),  # Just over limit
        ("gpt-4o-mini", 16_385, True),  # Just over limit
        ("gpt-4o-2024-08-06", 16_385, True),  # Base version over limit
        # Invalid cases - o1/o3 models
        ("o1", 100_001, True),  # Just over limit
        ("o3-mini", 150_000, True),  # Well over limit
        ("o1-2024-12-17", 100_001, True),  # Base version over limit
        ("o3-mini-2025-01-31", 150_000, True),  # Base version well over limit
        # Invalid cases - unknown models
        ("unknown-model", 5_000, True),  # Over default limit
    ],
)
def test_validate_token_limits(
    model_name: Optional[str],
    max_tokens: Optional[int],
    should_raise: bool,
) -> None:
    """Test token limit validation for various models and token counts."""
    if should_raise:
        with pytest.raises(TokenLimitError) as exc_info:
            _validate_token_limits(model_name or "", max_tokens)
        assert exc_info.value.requested_tokens == max_tokens
        assert exc_info.value.model_limit == get_default_token_limit(
            model_name or ""
        )
    else:
        try:
            _validate_token_limits(model_name or "", max_tokens)
        except TokenLimitError as e:
            pytest.fail(f"Unexpected TokenLimitError: {e}")
        except OpenAIClientError as e:
            if "not supported" not in str(e):
                raise  # Re-raise if it's not a model support error


def test_token_parameter_validation(registry: ModelRegistry) -> None:
    """Test validation of token-related parameters."""
    # Test GPT-4 models
    gpt4o = registry.get_capabilities("gpt-4o")
    used_params: Set[str] = set()

    # Test max_output_tokens
    gpt4o.validate_parameter(
        "max_output_tokens", 1000, used_params=used_params
    )

    # Test max_completion_tokens raises error when both are used
    with pytest.raises(TokenParameterError) as exc_info:
        gpt4o.validate_parameter(
            "max_completion_tokens", 1000, used_params=used_params
        )
    assert "Cannot specify both" in str(exc_info.value)

    # Test o1/o3 models
    o1 = registry.get_capabilities("o1")
    used_params = set()

    # Test max_completion_tokens
    o1.validate_parameter(
        "max_completion_tokens", 1000, used_params=used_params
    )

    # Test max_output_tokens raises error when both are used
    with pytest.raises(TokenParameterError) as exc_info:
        o1.validate_parameter(
            "max_output_tokens", 1000, used_params=used_params
        )
    assert "Cannot specify both" in str(exc_info.value)


def test_parameter_validation_with_overrides(registry: ModelRegistry) -> None:
    """Test parameter validation with max_value overrides."""
    gpt4o = registry.get_capabilities("gpt-4o")

    # Find temperature parameter reference
    temp_ref = None
    for ref in gpt4o.supported_parameters:
        if ref.ref == "numeric_constraints.temperature":
            temp_ref = ref
            break

    assert temp_ref is not None

    # Test with default max value
    gpt4o.validate_parameter("temperature", 1.5)

    # Test with override
    temp_ref.max_value = 1.0
    with pytest.raises(OpenAIClientError, match="must be between"):
        gpt4o.validate_parameter("temperature", 1.5)

    # Reset override
    temp_ref.max_value = None


def test_parameter_validation_with_dynamic_max(
    registry: ModelRegistry,
) -> None:
    """Test parameter validation with dynamic max values."""
    o1 = registry.get_capabilities("o1")

    # Test max_completion_tokens with dynamic limit
    o1.validate_parameter(
        "max_completion_tokens", 90_000
    )  # Under model's max_output_tokens

    with pytest.raises(OpenAIClientError, match="must not exceed"):
        o1.validate_parameter(
            "max_completion_tokens", 150_000
        )  # Over model's max_output_tokens
