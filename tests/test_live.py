"""Integration tests for the openai_structured client module."""

# Standard library imports
import os
from typing import Any

# Third-party imports
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

# Local imports
from openai_structured import (
    StreamConfig,
    async_openai_structured_call,
    async_openai_structured_stream,
    openai_structured_call,
    openai_structured_stream,
)


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""

    message: str = Field(..., description="The analyzed message")
    sentiment: str = Field(
        ...,
        pattern="(?i)^(positive|negative|neutral|mixed)$",
        description="Sentiment of the message",
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


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live
def test_live_sync_api() -> None:
    """Test synchronous API calls with real OpenAI API."""
    log_capture = LogCapture()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    result = openai_structured_call(
        client=client,
        model="gpt-4o-2024-08-06",
        output_schema=SentimentResponse,
        user_prompt="What is the sentiment of 'I love pizza'?",
        system_prompt="You analyze sentiment of text.",
        on_log=log_capture,
    )

    assert isinstance(result, SentimentResponse)
    assert result.sentiment.lower() in [
        "positive",
        "negative",
        "neutral",
        "mixed",
    ]
    assert len(log_capture.messages) > 0


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live
def test_live_sync_streaming() -> None:
    """Test synchronous streaming with real OpenAI API."""
    log_capture = LogCapture()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    config = StreamConfig(
        max_buffer_size=1024 * 1024,  # 1MB
        cleanup_threshold=512 * 1024,  # 512KB
        chunk_size=8192,  # 8KB
    )

    results = []
    for result in openai_structured_stream(
        client=client,
        model="gpt-4o-2024-08-06",
        output_schema=SentimentResponse,
        user_prompt="What is the sentiment of 'I hate mondays'?",
        system_prompt="You analyze sentiment of text.",
        stream_config=config,
        on_log=log_capture,
    ):
        assert isinstance(result, SentimentResponse)
        assert result.sentiment.lower() in [
            "positive",
            "negative",
            "neutral",
            "mixed",
        ]
        results.append(result)
        if len(results) >= 2:  # Test first two results
            break

    assert len(results) > 0
    assert len(log_capture.messages) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live
async def test_live_async_api() -> None:
    """Test async API calls with real OpenAI API."""
    log_capture = LogCapture()
    async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as client:
        result = await async_openai_structured_call(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=SentimentResponse,
            user_prompt="What is the sentiment of 'I love pizza'?",
            system_prompt="You analyze sentiment of text.",
            on_log=log_capture,
            timeout=30.0,  # Test timeout parameter
        )

        assert isinstance(result, SentimentResponse)
        assert result.sentiment.lower() in [
            "positive",
            "negative",
            "neutral",
            "mixed",
        ]
        assert len(log_capture.messages) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live
async def test_live_async_streaming() -> None:
    """Test async streaming with real OpenAI API."""
    log_capture = LogCapture()
    config = StreamConfig(
        max_buffer_size=1024 * 1024,  # 1MB
        cleanup_threshold=512 * 1024,  # 512KB
        chunk_size=8192,  # 8KB
    )

    async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as client:
        results = []
        async for result in async_openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=SentimentResponse,
            user_prompt="What is the sentiment of 'I hate mondays'?",
            system_prompt="You analyze sentiment of text.",
            stream_config=config,
            on_log=log_capture,
        ):
            assert isinstance(result, SentimentResponse)
            assert result.sentiment.lower() in [
                "positive",
                "negative",
                "neutral",
                "mixed",
            ]
            results.append(result)
            if len(results) >= 2:  # Test first two results
                break

        assert len(results) > 0
        assert len(log_capture.messages) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live
async def test_live_parameter_validation() -> None:
    """Test parameter validation with real API calls."""
    async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as client:
        # Test temperature limits
        with pytest.raises(Exception):
            await async_openai_structured_call(
                client=client,
                model="gpt-4o-2024-08-06",
                output_schema=SentimentResponse,
                user_prompt="test",
                system_prompt="test",
                temperature=2.5,  # Invalid temperature
            )

        # Test timeout behavior
        with pytest.raises(Exception):
            await async_openai_structured_call(
                client=client,
                model="gpt-4o-2024-08-06",
                output_schema=SentimentResponse,
                user_prompt="test",
                system_prompt="test",
                timeout=0.001,  # Very short timeout
            )
