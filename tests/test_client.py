"""Unit tests for the openai_structured client module."""

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from openai_structured import openai_structured_call
from openai_structured.errors import ModelNotSupportedError


class DummySchema(BaseModel):
    """Dummy schema for testing."""

    value: str


@pytest.mark.asyncio
async def test_invalid_model() -> None:
    """Test that using an unsupported model raises an error."""
    async with AsyncOpenAI(api_key="test-key") as client:
        with pytest.raises(ModelNotSupportedError) as exc_info:
            await openai_structured_call(
                client=client,
                model="gpt-4",
                output_schema=DummySchema,
                user_prompt="test",
                system_prompt="test",
            )
        assert "does not support structured output" in str(exc_info.value)
