"""Tests for the openai_structured client module."""

import json
from typing import Any, AsyncIterator, Dict, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import OpenAI

from openai_structured import (
    EmptyResponseError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
    openai_structured_call,
    openai_structured_stream,
)
from tests.models import MockResponseModel


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
    mock.chat.completions.create.return_value = mock_completion
    return mock


@pytest.fixture
def mock_stream_client(mock_response: Dict[str, Any]) -> MagicMock:
    """Create a mock OpenAI client for streaming."""
    mock = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(delta=MagicMock(content=json.dumps(mock_response)))
    ]

    async def async_iter():
        yield mock_chunk

    mock.chat.completions.create = AsyncMock(return_value=async_iter())
    return mock


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
        model="gpt-4",
        output_schema=MockResponseModel,
        user_prompt="test",
        system_prompt="test",
    )
    assert isinstance(result, MockResponseModel)
    assert result.value == "test response"


def test_empty_response(mock_client: MagicMock) -> None:
    """Test empty response handling."""
    mock_client.chat.completions.create.return_value.choices = []
    with pytest.raises(EmptyResponseError):
        openai_structured_call(
            client=mock_client,
            model="gpt-4",
            output_schema=MockResponseModel,
            user_prompt="test",
            system_prompt="test",
        )


def test_invalid_json_response(mock_client: MagicMock) -> None:
    """Test invalid JSON response handling."""
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="invalid json"))
    ]
    with pytest.raises(InvalidResponseFormatError):
        openai_structured_call(
            client=mock_client,
            model="gpt-4",
            output_schema=MockResponseModel,
            user_prompt="test",
            system_prompt="test",
        )


@pytest.mark.asyncio
async def test_stream_unsupported_model(client: OpenAI) -> None:
    """Test that streaming with unsupported models raises appropriate error."""
    with pytest.raises(ModelNotSupportedError):
        async for _ in cast(
            AsyncIterator[MockResponseModel],
            openai_structured_stream(
                client=client,
                model="unsupported-model",
                output_schema=MockResponseModel,
                user_prompt="test",
                system_prompt="test",
            ),
        ):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_successful_stream(
    mock_stream_client: MagicMock,
) -> None:
    """Test successful streaming."""
    count = 0
    async for result in cast(
        AsyncIterator[MockResponseModel],
        openai_structured_stream(
            client=mock_stream_client,
            model="gpt-4",
            output_schema=MockResponseModel,
            user_prompt="test",
            system_prompt="test",
        ),
    ):
        assert isinstance(result, MockResponseModel)
        assert result.value == "test response"
        count += 1
    assert count == 1
