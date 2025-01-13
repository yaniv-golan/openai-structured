"""Unit tests for the openai_structured client module."""

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio

from openai_structured import (
    ModelNotSupportedError,
    ModelVersionError,
    openai_structured_call,
    supports_structured_output,
)
from openai_structured.model_version import ModelVersion


class DummySchema(BaseModel):
    """Dummy schema for testing."""

    value: str


def test_supports_structured_output() -> None:
    """Test model support validation."""
    # Test aliases
    assert supports_structured_output("gpt-4o") is True
    assert supports_structured_output("gpt-4o-mini") is True
    assert supports_structured_output("o1") is True
    assert supports_structured_output("gpt-3.5-turbo") is False
    assert supports_structured_output("gpt-4") is False

    # Test dated versions - valid
    assert supports_structured_output("gpt-4o-2024-08-06") is True  # Minimum version
    assert supports_structured_output("gpt-4o-2024-09-01") is True  # Newer version
    assert supports_structured_output("gpt-4o-mini-2024-07-18") is True  # Minimum version
    assert supports_structured_output("gpt-4o-mini-2024-08-01") is True  # Newer version
    assert supports_structured_output("o1-2024-12-17") is True  # Minimum version
    assert supports_structured_output("o1-2025-01-01") is True  # Newer version

    # Test dated versions - invalid
    assert supports_structured_output("gpt-4o-2024-08-05") is False  # Too old
    assert supports_structured_output("gpt-4o-mini-2024-07-17") is False  # Too old
    assert supports_structured_output("o1-2024-12-16") is False  # Too old

    # Test invalid formats
    assert supports_structured_output("invalid-model") is False
    assert supports_structured_output("gpt-4o-invalid-date") is False
    assert supports_structured_output("gpt-4o-2024-13-01") is False  # Invalid month
    assert supports_structured_output("gpt-4o-2024-04-31") is False  # Invalid day
    assert supports_structured_output("gpt-4o-2024-02-30") is False  # Invalid day


def test_invalid_model() -> None:
    """Test that using an unsupported model raises an error."""
    with pytest.raises(ModelNotSupportedError) as exc_info:
        asyncio.run(
            openai_structured_call(
                client=AsyncOpenAI(),
                model="gpt-3.5-turbo",
                output_schema=DummySchema,
                user_prompt="test",
                system_prompt="test",
            )
        )
    assert "does not support structured output" in str(exc_info.value)
