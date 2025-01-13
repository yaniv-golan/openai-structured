"""Unit tests for the openai_structured client module."""

# Standard library imports
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock

import pytest

# Third-party imports
from openai import APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from pydantic import BaseModel

# Local imports
from openai_structured.client import (
    StreamConfig,
    async_openai_structured_call,
    async_openai_structured_stream,
    openai_structured_call,
    openai_structured_stream,
)
from openai_structured.errors import (
    OpenAIClientError,
    StreamBufferError,
    StreamInterruptedError,
)


class DummySchema(BaseModel):
    """Dummy schema for testing."""

    sentiment: str


class LogCapture:
    """Helper class to capture log messages."""

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    def __call__(
        self,
        level: Union[int, str],
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.messages.append(
            {"level": level, "message": message, "data": data or {}}
        )


def test_temperature_validation() -> None:
    """Test temperature parameter validation."""
    # Create mock client with proper spec
    mock_client = MagicMock(spec=OpenAI)
    mock_chat = MagicMock()
    mock_completions = MagicMock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    # Test valid temperatures
    for temp in [0.0, 1.0, 2.0]:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"sentiment": "positive"}'))
        ]
        mock_completions.create.return_value = mock_response

        openai_structured_call(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
            temperature=temp,
        )

    # Test invalid temperatures
    for temp in [-0.1, 2.1]:
        with pytest.raises(
            OpenAIClientError, match="Temperature must be between 0 and 2"
        ):
            openai_structured_call(
                client=mock_client,
                model="gpt-4o",
                output_schema=DummySchema,
                system_prompt="test",
                user_prompt="test",
                temperature=temp,
            )


def test_top_p_validation() -> None:
    """Test top_p parameter validation."""
    # Create mock client with proper spec
    mock_client = MagicMock(spec=OpenAI)
    mock_chat = MagicMock()
    mock_completions = MagicMock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    # Test valid top_p values
    for top_p in [0.0, 0.5, 1.0]:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"sentiment": "positive"}'))
        ]
        mock_completions.create.return_value = mock_response

        openai_structured_call(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
            top_p=top_p,
        )

    # Test invalid top_p values
    for top_p in [-0.1, 1.1]:
        with pytest.raises(
            OpenAIClientError, match="Top-p must be between 0 and 1"
        ):
            openai_structured_call(
                client=mock_client,
                model="gpt-4o",
                output_schema=DummySchema,
                system_prompt="test",
                user_prompt="test",
                top_p=top_p,
            )


def test_frequency_penalty_validation() -> None:
    """Test frequency_penalty parameter validation."""
    # Create mock client with proper spec
    mock_client = MagicMock(spec=OpenAI)
    mock_chat = MagicMock()
    mock_completions = MagicMock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    # Test valid frequency penalty values
    for penalty in [-2.0, 0.0, 2.0]:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"sentiment": "positive"}'))
        ]
        mock_completions.create.return_value = mock_response

        openai_structured_call(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
            frequency_penalty=penalty,
        )

    # Test invalid frequency penalty values
    for penalty in [-2.1, 2.1]:
        with pytest.raises(
            OpenAIClientError,
            match="frequency_penalty must be between -2 and 2",
        ):
            openai_structured_call(
                client=mock_client,
                model="gpt-4o",
                output_schema=DummySchema,
                system_prompt="test",
                user_prompt="test",
                frequency_penalty=penalty,
            )


def test_presence_penalty_validation() -> None:
    """Test presence_penalty parameter validation."""
    # Create mock client with proper spec
    mock_client = MagicMock(spec=OpenAI)
    mock_chat = MagicMock()
    mock_completions = MagicMock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    # Test valid presence penalty values
    for penalty in [-2.0, 0.0, 2.0]:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"sentiment": "positive"}'))
        ]
        mock_completions.create.return_value = mock_response

        openai_structured_call(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
            presence_penalty=penalty,
        )

    # Test invalid presence penalty values
    for penalty in [-2.1, 2.1]:
        with pytest.raises(
            OpenAIClientError,
            match="presence_penalty must be between -2 and 2",
        ):
            openai_structured_call(
                client=mock_client,
                model="gpt-4o",
                output_schema=DummySchema,
                system_prompt="test",
                user_prompt="test",
                presence_penalty=penalty,
            )


def test_sync_streaming() -> None:
    """Test synchronous streaming interface."""
    # Create mock responses
    mock_responses = [
        MagicMock(
            choices=[
                MagicMock(delta=MagicMock(content='{"sentiment": "positive"}'))
            ]
        )
    ]

    # Create mock completions
    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_responses

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = mock_chat

    stream_config = StreamConfig(max_buffer_size=100)
    for result in openai_structured_stream(
        client=mock_client,
        model="gpt-4o",
        output_schema=DummySchema,
        system_prompt="test",
        user_prompt="test",
        stream_config=stream_config,
    ):
        assert isinstance(result, DummySchema)


def test_empty_response_handling() -> None:
    """Test handling of empty responses."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.choices = []

    # Create mock completions
    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_response

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = mock_chat

    with pytest.raises(OpenAIClientError) as exc_info:
        openai_structured_call(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
        )
    assert "empty response" in str(exc_info.value).lower()


def test_invalid_json_handling() -> None:
    """Test handling of invalid JSON responses."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="invalid json {"))
    ]

    # Create mock completions
    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_response

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = mock_chat

    with pytest.raises(OpenAIClientError) as exc_info:
        openai_structured_call(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
        )
    assert "invalid json" in str(exc_info.value).lower()


def test_invalid_schema_handling() -> None:
    """Test handling of responses that don't match the schema."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"invalid": "field"}'))
    ]

    # Create mock completions
    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_response

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = mock_chat

    with pytest.raises(OpenAIClientError) as exc_info:
        openai_structured_call(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
        )
    assert "validation failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_api_error_handling() -> None:
    """Test handling of OpenAI API errors."""
    # Create mock request
    mock_request = MagicMock()
    mock_request.method = "POST"
    mock_request.url = "https://api.openai.com/v1/chat/completions"
    mock_request.headers = {}
    mock_request.body = b"{}"

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create.side_effect = APITimeoutError(request=mock_request)

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    with pytest.raises(APITimeoutError):
        await async_openai_structured_call(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
        )


@pytest.mark.asyncio
async def test_rate_limit_handling() -> None:
    """Test handling of rate limit errors."""
    # Create mock response for rate limit error
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"retry-after": "30"}
    mock_response.text = "Rate limit exceeded"
    mock_response.json.return_value = {
        "error": {"message": "Rate limit exceeded"}
    }

    # Create mock completions with rate limit error
    mock_completions = AsyncMock()
    mock_completions.create.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    with pytest.raises(RateLimitError):
        await async_openai_structured_call(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
        )


def test_stream_interruption() -> None:
    """Test handling of stream interruption."""

    # Create mock stream that raises connection error
    def mock_stream() -> Generator[MagicMock, None, None]:
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="{"))])
        raise ConnectionError("Connection lost")

    # Create mock completions
    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_stream()

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = mock_chat

    with pytest.raises(StreamInterruptedError):
        for _ in openai_structured_stream(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
        ):
            pass


@pytest.mark.asyncio
async def test_async_stream_success() -> None:
    """Test successful async streaming."""
    # Create mock responses
    mock_responses = [
        MagicMock(
            choices=[MagicMock(delta=MagicMock(content='{"sentiment": '))]
        ),
        MagicMock(choices=[MagicMock(delta=MagicMock(content='"positive"'))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="}"))]),
    ]

    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        for response in mock_responses:
            yield response

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream())

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    results = []
    async for result in async_openai_structured_stream(
        client=mock_client,
        model="gpt-4o",
        output_schema=DummySchema,
        system_prompt="test",
        user_prompt="test",
    ):
        results.append(result)

    assert len(results) == 1
    assert isinstance(results[0], DummySchema)
    assert results[0].sentiment == "positive"


@pytest.mark.asyncio
async def test_async_stream_buffer_overflow() -> None:
    """Test handling of buffer overflow in async streaming."""
    # Create mock response with large content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(delta=MagicMock(content="x" * 100))]

    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        yield mock_response

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream())

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    stream_config = StreamConfig(max_buffer_size=10)  # Very small buffer
    with pytest.raises(StreamBufferError):
        async for _ in async_openai_structured_stream(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
            stream_config=stream_config,
        ):
            pass


@pytest.mark.asyncio
async def test_async_stream_interruption() -> None:
    """Test handling of stream interruption in async streaming."""

    # Create mock stream that raises connection error
    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="{"))])
        raise ConnectionError("Connection lost")

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream())

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    with pytest.raises(StreamInterruptedError):
        async for _ in async_openai_structured_stream(
            client=mock_client,
            model="gpt-4o",
            output_schema=DummySchema,
            system_prompt="test",
            user_prompt="test",
        ):
            pass


@pytest.mark.asyncio
async def test_async_call_success() -> None:
    """Test successful async API call."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"sentiment": "positive"}'))
    ]

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create.return_value = mock_response

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    result = await async_openai_structured_call(
        client=mock_client,
        model="gpt-4o",
        output_schema=DummySchema,
        system_prompt="test",
        user_prompt="test",
    )
    assert isinstance(result, DummySchema)
    assert result.sentiment == "positive"


@pytest.mark.asyncio
async def test_async_call_with_logging() -> None:
    """Test async API call with logging callback."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"sentiment": "positive"}'))
    ]

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create.return_value = mock_response

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    log_capture = LogCapture()

    result = await async_openai_structured_call(
        client=mock_client,
        model="gpt-4o",
        output_schema=DummySchema,
        system_prompt="test",
        user_prompt="test",
        on_log=log_capture,
    )

    assert isinstance(result, DummySchema)
    assert result.sentiment == "positive"
    assert len(log_capture.messages) > 0


@pytest.mark.asyncio
async def test_async_stream_with_logging() -> None:
    """Test async streaming with logging callback."""
    # Create mock responses
    mock_responses = [
        MagicMock(
            choices=[MagicMock(delta=MagicMock(content='{"sentiment": '))]
        ),
        MagicMock(choices=[MagicMock(delta=MagicMock(content='"positive"'))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="}"))]),
    ]

    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        for response in mock_responses:
            yield response

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream())

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client with proper spec
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    log_capture = LogCapture()

    results = []
    async for result in async_openai_structured_stream(
        client=mock_client,
        model="gpt-4o",
        output_schema=DummySchema,
        system_prompt="test",
        user_prompt="test",
        on_log=log_capture,
    ):
        results.append(result)

    assert len(results) == 1
    assert isinstance(results[0], DummySchema)
    assert results[0].sentiment == "positive"
    assert len(log_capture.messages) > 0


@pytest.mark.asyncio
async def test_async_api_call() -> None:
    """Test async API call with mock response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"sentiment": "positive"}'))
    ]

    mock_completions = AsyncMock()
    mock_completions.create.return_value = mock_response
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    result = await async_openai_structured_call(
        client=mock_client,
        model="gpt-4o",
        output_schema=DummySchema,
        system_prompt="test",
        user_prompt="test",
    )

    assert isinstance(result, DummySchema)
    assert result.sentiment == "positive"
