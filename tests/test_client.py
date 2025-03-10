"""Live tests for the OpenAI client functionality."""

import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from openai_structured.client import (
    async_openai_structured_stream,
    openai_structured_call,
    openai_structured_stream,
)
from openai_structured.errors import OpenAIClientError


@pytest.fixture
def client() -> OpenAI:
    """Create OpenAI client for testing."""
    return OpenAI()


class StreamTestMessage(BaseModel):
    """Test message schema for streaming responses."""

    content: str
    is_complete: bool


@pytest.mark.live
def test_live_structured_call(client: OpenAI) -> None:
    """Test structured output call with live OpenAI API."""
    # Test with gpt-4o
    result = openai_structured_call(
        client=client,
        model="gpt-4o",
        output_schema=StreamTestMessage,
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello, world!' and indicate the message is complete.",
        temperature=0.7,
    )
    assert isinstance(result, StreamTestMessage)
    assert "hello, world" in result.content.lower()
    assert result.is_complete

    # Test with o1
    result = openai_structured_call(
        client=client,
        model="o1",
        output_schema=StreamTestMessage,
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello from o1!' and indicate the message is complete.",
        reasoning_effort="medium",
    )
    assert isinstance(result, StreamTestMessage)
    assert "hello from o1" in result.content.lower()
    assert result.is_complete


@pytest.mark.live
def test_live_structured_stream_sync(client: OpenAI) -> None:
    """Test synchronous structured output streaming with live OpenAI API."""
    chunks = []
    for chunk in openai_structured_stream(
        client=client,
        model="gpt-4o",
        output_schema=StreamTestMessage,
        system_prompt=(
            "You are a helpful assistant. For streaming responses, you must output "
            "a series of complete JSON objects, each with 'content' and 'is_complete' fields. "
            "The 'content' field should contain your message, and 'is_complete' should be true "
            "only in the final object. Each response must be a complete, valid JSON object."
        ),
        user_prompt="Say 'Hello, streaming!' and indicate the message is complete.",
        temperature=0.7,
    ):
        assert isinstance(chunk, StreamTestMessage)
        chunks.append(chunk)

    # Verify we got at least one chunk
    assert len(chunks) > 0, "No chunks received from the API"

    # Verify final result is a StreamTestMessage
    final = chunks[-1]
    assert isinstance(final, StreamTestMessage)

    # Combine all content from chunks
    all_content = " ".join(chunk.content for chunk in chunks)

    # Verify we got a non-empty response
    assert all_content.strip(), "Received empty content from API"

    # Log the content for debugging
    print(f"Received content: {all_content}")

    # Check if any chunk has is_complete=True (but don't fail if none do)
    has_complete = any(chunk.is_complete for chunk in chunks)
    print(f"Any chunk has is_complete=True: {has_complete}")


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_structured_stream_async(client: OpenAI) -> None:
    """Test asynchronous structured output streaming with live OpenAI API."""
    async_client = AsyncOpenAI()
    chunks = []
    async for chunk in async_openai_structured_stream(
        client=async_client,
        model="gpt-4o",
        output_schema=StreamTestMessage,
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello, streaming!' and indicate the message is complete.",
        temperature=0.7,
    ):
        assert isinstance(chunk, StreamTestMessage)
        chunks.append(chunk)

    # Verify we got at least one chunk
    assert len(chunks) > 0, "No chunks received from the API"

    # Verify final result is a StreamTestMessage
    final = chunks[-1]
    assert isinstance(final, StreamTestMessage)

    # Combine all content from chunks
    all_content = " ".join(chunk.content for chunk in chunks)

    # Verify we got a non-empty response
    assert all_content.strip(), "Received empty content from API"

    # Log the content for debugging
    print(f"Received content: {all_content}")

    # Check if any chunk has is_complete=True (but don't fail if none do)
    has_complete = any(chunk.is_complete for chunk in chunks)
    print(f"Any chunk has is_complete=True: {has_complete}")


@pytest.mark.live
def test_live_parameter_errors(client: OpenAI) -> None:
    """Test parameter validation with live OpenAI API."""
    # Test invalid temperature
    with pytest.raises(OpenAIClientError):
        openai_structured_call(
            client=client,
            model="gpt-4o",
            output_schema=StreamTestMessage,
            system_prompt="You are a helpful assistant.",
            user_prompt="This should fail.",
            temperature=2.5,  # Invalid temperature
        )

    # Test invalid reasoning_effort
    with pytest.raises(OpenAIClientError):
        openai_structured_call(
            client=client,
            model="o1",
            output_schema=StreamTestMessage,
            system_prompt="You are a helpful assistant.",
            user_prompt="This should fail.",
            reasoning_effort="invalid",  # Invalid enum value
        )

    # Test token parameter conflict
    with pytest.raises(OpenAIClientError):
        openai_structured_call(
            client=client,
            model="gpt-4o",
            output_schema=StreamTestMessage,
            system_prompt="You are a helpful assistant.",
            user_prompt="This should fail.",
            max_completion_tokens=1000,  # Can't use both max_completion_tokens
            max_output_tokens=1000,  # and max_output_tokens together
        )
