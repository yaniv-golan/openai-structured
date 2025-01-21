"""Tests for the OpenAI structured client module using live API calls."""

import logging
from typing import AsyncGenerator, Any
import pytest
import pytest_asyncio
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI

from openai_structured.client import (
    openai_structured_call,
    async_openai_structured_call,
    openai_structured_stream,
    async_openai_structured_stream,
    JSONParseError,
    ValidationError,
    InvalidResponseFormatError,
)

class SimpleResponse(BaseModel):
    """Simple schema for testing basic responses."""
    message: str

@pytest_asyncio.fixture
async def openai_client() -> AsyncGenerator[AsyncOpenAI, None]:
    """Create OpenAI client for testing."""
    client = AsyncOpenAI()
    yield client
    await client.close()

@pytest.mark.live
class TestBasicFunctionality:
    """Test basic functionality using live API."""

    @pytest.mark.asyncio
    async def test_create_chat_completion(self, openai_client: AsyncOpenAI) -> None:
        """Test basic chat completion with live API."""
        result = await async_openai_structured_call(
            client=openai_client,
            user_prompt="Return a simple greeting as JSON with message field",
            output_schema=SimpleResponse,
            model="gpt-4o-2024-08-06",
            system_prompt="You are a helpful assistant that returns JSON responses."
        )
        
        assert isinstance(result, SimpleResponse)
        assert isinstance(result.message, str)
        assert len(result.message) > 0

    @pytest.mark.asyncio
    async def test_create_chat_completion_stream(self, openai_client: AsyncOpenAI) -> None:
        """Test streaming chat completion with live API."""
        responses = []
        async for response in async_openai_structured_stream(
            client=openai_client,
            user_prompt="Return a simple greeting as JSON with message field",
            model="gpt-4o-2024-08-06",
            output_schema=SimpleResponse,
            system_prompt="You are a helpful assistant that returns JSON responses."
        ):
            responses.append(response)
        
        # Verify we got at least one response
        assert len(responses) > 0
        
        # Verify each response is valid
        for response in responses:
            assert isinstance(response, SimpleResponse)
            assert isinstance(response.message, str)
            assert len(response.message) > 0

@pytest.mark.live
class TestErrorHandling:
    """Test error handling with live API."""

    @pytest.mark.asyncio
    async def test_validation_error(self, openai_client: AsyncOpenAI) -> None:
        """Test handling of validation errors."""
        with pytest.raises(InvalidResponseFormatError):
            await async_openai_structured_call(
                client=openai_client,
                user_prompt="Return JSON without a message field",
                output_schema=SimpleResponse,
                model="gpt-4o-2024-08-06",
                system_prompt="You are a helpful assistant. Return JSON with an 'invalid' field instead of 'message'."
            )

@pytest.mark.live
class TestLogging:
    """Test logging with live API."""

    @pytest.mark.asyncio
    async def test_basic_logging(self, openai_client: AsyncOpenAI, caplog: pytest.LogCaptureFixture) -> None:
        """Test basic logging of requests."""
        caplog.set_level(logging.INFO)
        
        await async_openai_structured_call(
            client=openai_client,
            user_prompt="Return a simple greeting as JSON with message field",
            output_schema=SimpleResponse,
            model="gpt-4o-2024-08-06",
            system_prompt="You are a helpful assistant that returns JSON responses."
        )
        
        # Check for httpx log messages
        assert "HTTP Request: POST https://api.openai.com/v1/chat/completions" in caplog.text
        assert "HTTP/1.1 200 OK" in caplog.text

    @pytest.mark.asyncio
    async def test_stream_logging(self, openai_client: AsyncOpenAI, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging for streaming requests."""
        caplog.set_level(logging.INFO)
        
        async for _ in async_openai_structured_stream(
            client=openai_client,
            user_prompt="Return a simple greeting as JSON with message field",
            output_schema=SimpleResponse,
            model="gpt-4o-2024-08-06",
            system_prompt="You are a helpful assistant that returns JSON responses."
        ):
            pass
        
        # Check for httpx log messages
        assert "HTTP Request: POST https://api.openai.com/v1/chat/completions" in caplog.text
        assert "HTTP/1.1 200 OK" in caplog.text
