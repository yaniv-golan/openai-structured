"""Integration tests for the openai_structured client module."""

import os
from typing import Any

import pytest
from openai import AsyncOpenAI, OpenAI

from openai_structured import (
    StreamConfig,
    async_openai_structured_call,
    async_openai_structured_stream,
    openai_structured_call,
    openai_structured_stream,
)
from openai_structured.examples.schemas import SentimentMessage
from openai_structured.testing import (
    create_structured_response,
    create_structured_stream_response,
    create_error_response,
    create_rate_limit_response,
)


class LogCapture:
    """Helper class to capture log messages."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def __call__(
        self,
        level: int,
        message: str,
        data: dict[str, Any],
    ) -> None:
        self.messages.append(
            {"level": level, "message": message, "data": data}
        )


@pytest.mark.skipif(  # type: ignore[misc]
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live  # type: ignore[misc]
def test_live_sync_api() -> None:
    """Test synchronous API calls with real OpenAI API."""
    log_capture = LogCapture()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    result = openai_structured_call(
        client=client,
        model="gpt-4o",
        output_schema=SentimentMessage,
        user_prompt="What is the sentiment of 'I love pizza'?",
        system_prompt="You analyze sentiment of text.",
        on_log=log_capture,
    )

    assert isinstance(result, SentimentMessage)
    assert result.sentiment in ["positive", "negative", "neutral"]
    assert len(log_capture.messages) > 0


@pytest.mark.skipif(  # type: ignore[misc]
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live  # type: ignore[misc]
def test_live_sync_streaming() -> None:
    """Test synchronous streaming with real OpenAI API."""
    log_capture = LogCapture()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = list(openai_structured_stream(
        client=client,
        model="gpt-4o",
        output_schema=SentimentMessage,
        user_prompt="What is the sentiment of 'I hate mondays'?",
        system_prompt="You analyze sentiment of text.",
        on_log=log_capture,
    ))

    assert len(results) > 0
    assert all(isinstance(r, SentimentMessage) for r in results)
    assert all(r.sentiment in ["positive", "negative", "neutral"] for r in results)
    assert len(log_capture.messages) > 0


@pytest.mark.skipif(  # type: ignore[misc]
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live  # type: ignore[misc]
@pytest.mark.asyncio  # type: ignore[misc]
async def test_live_async_api() -> None:
    """Test asynchronous API calls with real OpenAI API."""
    log_capture = LogCapture()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    result = await async_openai_structured_call(
        client=client,
        model="gpt-4o",
        output_schema=SentimentMessage,
        user_prompt="What is the sentiment of 'I am neutral about this'?",
        system_prompt="You analyze sentiment of text.",
        on_log=log_capture,
    )

    assert isinstance(result, SentimentMessage)
    assert result.sentiment in ["positive", "negative", "neutral"]
    assert len(log_capture.messages) > 0


@pytest.mark.skipif(  # type: ignore[misc]
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live  # type: ignore[misc]
@pytest.mark.asyncio  # type: ignore[misc]
async def test_live_async_streaming() -> None:
    """Test asynchronous streaming with real OpenAI API."""
    log_capture = LogCapture()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = []
    async for result in async_openai_structured_stream(
        client=client,
        model="gpt-4o",
        output_schema=SentimentMessage,
        user_prompt="What is the sentiment of 'This is great!'?",
        system_prompt="You analyze sentiment of text.",
        on_log=log_capture,
    ):
        results.append(result)

    assert len(results) > 0
    assert all(isinstance(r, SentimentMessage) for r in results)
    assert all(r.sentiment in ["positive", "negative", "neutral"] for r in results)
    assert len(log_capture.messages) > 0


def test_live_parameter_validation() -> None:
    """Test parameter validation."""
    client = OpenAI(api_key="dummy")

    # Test invalid model
    with pytest.raises(ValueError):
        openai_structured_call(
            client=client,
            model="invalid-model",
            output_schema=SentimentMessage,
            user_prompt="test",
        )

    # Test missing system prompt
    result = openai_structured_call(
        client=client,
        model="gpt-4o",
        output_schema=SentimentMessage,
        user_prompt="test",
    )
    assert isinstance(result, SentimentMessage)

    # Test stream config validation
    with pytest.raises(ValueError):
        list(openai_structured_stream(
            client=client,
            model="gpt-4o",
            output_schema=SentimentMessage,
            user_prompt="test",
            stream_config=StreamConfig(max_buffer_size=-1),
        ))


def test_mock_rate_limit() -> None:
    """Test rate limit handling with mocks."""
    client = MagicMock()
    client.chat.completions.create = create_rate_limit_response(
        max_requests=2,
        reset_after=60
    )

    # First two calls should succeed
    result1 = openai_structured_call(
        client=client,
        model="gpt-4o",
        output_schema=SentimentMessage,
        user_prompt="test1",
    )
    assert isinstance(result1, SentimentMessage)

    result2 = openai_structured_call(
        client=client,
        model="gpt-4o",
        output_schema=SentimentMessage,
        user_prompt="test2",
    )
    assert isinstance(result2, SentimentMessage)

    # Third call should raise rate limit error
    with pytest.raises(Exception) as exc:  # Replace with specific rate limit error
        openai_structured_call(
            client=client,
            model="gpt-4o",
            output_schema=SentimentMessage,
            user_prompt="test3",
        )
    assert "rate limit" in str(exc.value).lower()
