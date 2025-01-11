"""Tests for the openai_structured client module."""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI, OpenAI

from openai_structured import (
    EmptyResponseError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
    OpenAIClientError,
    openai_structured_call,
    openai_structured_stream,
)
from tests.models import MockResponseModel

from .test_live import SentimentResponse


class AsyncOpenAIMock(AsyncOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(api_key="test-key")
        self._chat = AsyncChatMock()

    @property
    def chat(self):
        return self._chat

    def __instancecheck__(self, instance):
        return isinstance(instance, AsyncOpenAI)


class AsyncChatMock:
    def __init__(self):
        self.completions = AsyncCompletionsMock()


class AsyncCompletionsMock:
    async def create(self, *args, **kwargs):
        return AsyncStreamMock(
            ['{"message": "Hello",', '"sentiment": "positive"}']
        )


class AsyncStreamMock:
    def __init__(self, chunks: list[str]):
        self.chunks = [AsyncCompletionChunk(chunk) for chunk in chunks]
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


class AsyncCompletionChunk:
    def __init__(self, content: str):
        self.choices = [
            type(
                "Choice",
                (),
                {"delta": type("Delta", (), {"content": content})()},
            )
        ]


@pytest.fixture
def mock_async_client():
    """Create a mock async OpenAI client."""
    return AsyncOpenAIMock()


@pytest.fixture
def client() -> OpenAI:
    """Create a test OpenAI client."""
    return OpenAI()


@pytest.fixture
def mock_response() -> Dict[str, Any]:
    """Create a mock response."""
    return {"value": "test response"}


@pytest.fixture
def mock_client(mock_response: Dict[str, Any]) -> MagicMock:
    """Create a mock OpenAI client."""
    mock = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=json.dumps(mock_response)))
    ]
    mock_completion.id = "test-id"
    mock.chat.completions.create.return_value = mock_completion
    return mock


@pytest.fixture
def mock_stream_client():
    """Create a mock async OpenAI client for streaming."""
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content='{"message": "Test message", "sentiment": "positive"}'
            )
        )
    ]

    async def mock_create(*args, **kwargs):
        yield mock_chunk

    # Set up the mock hierarchy with proper spec
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)

    return mock_client


def test_unsupported_model(client: OpenAI) -> None:
    """Test that unsupported models raise appropriate error."""
    with pytest.raises(ModelNotSupportedError):
        openai_structured_call(
            client=client,
            model="unsupported-model",
            output_schema=MockResponseModel,
            user_prompt="test",
            system_prompt="test",
        )


def test_successful_call(mock_client: MagicMock) -> None:
    """Test successful API call."""
    result = openai_structured_call(
        client=mock_client,
        model="gpt-4o-2024-08-06",
        output_schema=MockResponseModel,
        user_prompt="test",
        system_prompt="test",
    )
    assert isinstance(result, MockResponseModel)
    assert result.value == "test response"


def test_empty_response(mock_client: MagicMock) -> None:
    """Test empty response handling."""
    mock_completion = MagicMock()
    mock_completion.choices = []  # Empty choices list
    mock_completion.id = "test-id"
    mock_client.chat.completions.create.return_value = mock_completion
    with pytest.raises(EmptyResponseError) as exc_info:
        openai_structured_call(
            client=mock_client,
            model="gpt-4o-2024-08-06",
            output_schema=MockResponseModel,
            user_prompt="test",
            system_prompt="test",
        )
    assert "OpenAI API returned an empty response" in str(exc_info.value)


def test_invalid_json_response(mock_client: MagicMock) -> None:
    """Test invalid JSON response handling."""
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="invalid json"))
    ]
    mock_completion.id = "test-id"
    mock_client.chat.completions.create.return_value = mock_completion
    with pytest.raises(InvalidResponseFormatError) as exc_info:
        openai_structured_call(
            client=mock_client,
            model="gpt-4o-2024-08-06",
            output_schema=MockResponseModel,
            user_prompt="test",
            system_prompt="test",
        )
    assert "Response validation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_successful_stream(mock_stream_client):
    """Test successful streaming of structured output."""
    responses = []
    async for response in openai_structured_stream(
        client=mock_stream_client,
        model="gpt-4o-2024-08-06",
        output_schema=SentimentResponse,
        system_prompt="Analyze sentiment",
        user_prompt="How's the weather?",
    ):
        responses.append(response)

    assert len(responses) == 1
    assert isinstance(responses[0], SentimentResponse)
    assert responses[0].message == "Test message"
    assert responses[0].sentiment == "positive"


@pytest.mark.asyncio
async def test_stream_error_handling(mock_stream_client):
    """Test error handling during streaming."""
    mock_stream_client.chat.completions.create = AsyncMock(
        side_effect=Exception("Stream error")
    )

    with pytest.raises(OpenAIClientError) as exc_info:
        async for _ in openai_structured_stream(
            client=mock_stream_client,
            model="gpt-4o-2024-08-06",
            output_schema=SentimentResponse,
            system_prompt="Analyze sentiment",
            user_prompt="How's the weather?",
        ):
            pass
    assert "Stream error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invalid_client_type():
    """Test validation of AsyncOpenAI client type."""
    invalid_client = MagicMock()

    with pytest.raises(TypeError) as exc_info:
        async for _ in openai_structured_stream(
            client=invalid_client,
            model="gpt-4o",
            output_schema=SentimentResponse,
            system_prompt="Analyze sentiment",
            user_prompt="How's the weather?",
        ):
            pass

    assert "Streaming operations require " "AsyncOpenAI client" in str(
        exc_info.value
    )
