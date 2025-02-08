import pytest
from openai import OpenAI
from pydantic import BaseModel

from openai_structured.client import (
    _validate_token_limits,
    get_context_window_limit,
    get_default_token_limit,
    openai_structured_call,
    openai_structured_stream,
    supports_structured_output,
)
from openai_structured.errors import OpenAIClientError, TokenLimitError


@pytest.mark.parametrize(
    "model_name,expected",
    [
        # Alias cases
        ("gpt-4o", True),  # Basic alias
        ("gpt-3.5-turbo", False),  # Unsupported model
        ("o3-mini", True),  # New o3-mini alias
        # Dated versions - valid
        ("gpt-4o-2024-08-06", True),  # Minimum version
        ("gpt-4o-2024-09-01", True),  # Newer version
        ("o3-mini-2025-01-31", True),  # Minimum o3-mini version
        ("o3-mini-2025-02-01", True),  # Newer o3-mini version
        # Dated versions - invalid
        ("gpt-4o-2024-07-01", False),  # Too old
        ("gpt-4o-2023-12-01", False),  # Much too old
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
        ("o1", 200_000),  # o1 model
        ("o3-mini", 200_000),  # o3-mini model (corrected from 128K to 200K)
        # Dated versions
        ("gpt-4o-2024-08-06", 128_000),  # GPT-4 dated version
        ("o1-2024-12-17", 200_000),  # o1 dated version
        ("o3-mini-2025-01-31", 200_000),  # o3-mini dated version
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
        ("o1", 100_000),  # o1 model
        ("o3-mini", 100_000),  # o3-mini model (corrected from 16K to 100K)
        # Dated versions
        ("gpt-4o-2024-08-06", 16_384),  # GPT-4 dated version
        ("o1-2024-12-17", 100_000),  # o1 dated version
        ("o3-mini-2025-01-31", 100_000),  # o3-mini dated version
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
        # Valid cases
        ("gpt-4o", 16_384, False),  # Exactly at limit
        ("gpt-4o", 16_000, False),  # Under limit
        ("o1", 100_000, False),  # Exactly at limit
        ("o3-mini", 100_000, False),  # Exactly at limit
        ("o3-mini", 90_000, False),  # Under limit
        (None, None, False),  # No token limit specified
        # Invalid cases
        ("gpt-4o", 16_385, True),  # Just over limit
        ("o1", 100_001, True),  # Just over limit
        ("o3-mini", 150_000, True),  # Well over limit
        ("unknown-model", 5_000, True),  # Over default limit
    ],
)
def test_validate_token_limits(
    model_name: str, max_tokens: int, should_raise: bool
) -> None:
    """Test token limit validation for various models and token counts."""
    if should_raise:
        with pytest.raises(TokenLimitError) as exc_info:
            _validate_token_limits(model_name, max_tokens)
        assert exc_info.value.requested_tokens == max_tokens
        assert exc_info.value.model_limit == get_default_token_limit(
            model_name
        )
    else:
        try:
            _validate_token_limits(model_name, max_tokens)
        except TokenLimitError as e:
            pytest.fail(f"Unexpected TokenLimitError: {e}")


def test_o1_fixed_parameters() -> None:
    """Test that o1 models enforce fixed parameters."""
    client = OpenAI()

    # Test streaming rejection for o1-2024-12-17
    with pytest.raises(
        OpenAIClientError,
        match=(
            "o1-2024-12-17 does not support streaming.*"
            "Supported values are: false.*"
            "Use o1-preview, o1-mini, or a different model"
        ),
    ):
        openai_structured_stream(
            client=client,
            model="o1-2024-12-17",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
        )

    # Test that streaming is allowed for o1-preview
    try:
        openai_structured_stream(
            client=client,
            model="o1-preview",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
        )
    except OpenAIClientError as e:
        if "streaming" in str(e).lower():
            pytest.fail("o1-preview should support streaming")

    # Test temperature enforcement
    with pytest.raises(
        OpenAIClientError,
        match="o1 models have fixed parameters that cannot be modified",
    ):
        openai_structured_call(
            client=client,
            model="o1",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
            temperature=0.5,
        )

    # Test top_p enforcement
    with pytest.raises(
        OpenAIClientError,
        match="o1 models have fixed parameters that cannot be modified",
    ):
        openai_structured_call(
            client=client,
            model="o1",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
            top_p=0.5,
        )

    # Test frequency_penalty enforcement
    with pytest.raises(
        OpenAIClientError,
        match="o1 models have fixed parameters that cannot be modified",
    ):
        openai_structured_call(
            client=client,
            model="o1",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
            frequency_penalty=0.5,
        )

    # Test presence_penalty enforcement
    with pytest.raises(
        OpenAIClientError,
        match="o1 models have fixed parameters that cannot be modified",
    ):
        openai_structured_call(
            client=client,
            model="o1",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
            presence_penalty=0.5,
        )


def test_o3_fixed_parameters() -> None:
    """Test that o3 models enforce fixed parameters and streaming restrictions."""
    client = OpenAI()

    # Test streaming rejection for main o3 model
    with pytest.raises(
        OpenAIClientError,
        match="The main o3 model does not support streaming.*Use o3-mini or o3-mini-high",
    ):
        openai_structured_stream(
            client=client,
            model="o3",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
        )

    # Test that streaming is allowed for o3-mini
    try:
        openai_structured_stream(
            client=client,
            model="o3-mini",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
        )
    except OpenAIClientError as e:
        if "streaming" in str(e).lower():
            pytest.fail("o3-mini should support streaming")

    # Test that streaming is allowed for o3-mini-high
    try:
        openai_structured_stream(
            client=client,
            model="o3-mini-high",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
        )
    except OpenAIClientError as e:
        if "streaming" in str(e).lower():
            pytest.fail("o3-mini-high should support streaming")

    # Test temperature enforcement
    with pytest.raises(
        OpenAIClientError,
        match="o3 models have fixed parameters that cannot be modified",
    ):
        openai_structured_call(
            client=client,
            model="o3",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
            temperature=0.5,
        )

    # Test top_p enforcement
    with pytest.raises(
        OpenAIClientError,
        match="o3 models have fixed parameters that cannot be modified",
    ):
        openai_structured_call(
            client=client,
            model="o3",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
            top_p=0.5,
        )

    # Test frequency_penalty enforcement
    with pytest.raises(
        OpenAIClientError,
        match="o3 models have fixed parameters that cannot be modified",
    ):
        openai_structured_call(
            client=client,
            model="o3",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
            frequency_penalty=0.5,
        )

    # Test presence_penalty enforcement
    with pytest.raises(
        OpenAIClientError,
        match="o3 models have fixed parameters that cannot be modified",
    ):
        openai_structured_call(
            client=client,
            model="o3",
            output_schema=BaseModel,
            system_prompt="test",
            user_prompt="test",
            presence_penalty=0.5,
        )
