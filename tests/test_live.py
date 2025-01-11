"""Live API tests for the openai_structured client module."""

import logging
import os

import pytest
import pytest_asyncio
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from openai_structured import openai_structured_call, openai_structured_stream


class SentimentResponse(BaseModel):
    """Test response model for sentiment analysis."""

    message: str
    sentiment: str


@pytest.fixture
def sync_openai_client():
    """Create a synchronous OpenAI client for testing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


@pytest_asyncio.fixture
async def async_openai_client():
    """Provide an async OpenAI client for testing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key)
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def test_prompts():
    """Test prompts for consistency across tests."""
    return {
        "system": (
            "You are a friendly assistant that provides messages "
            "with sentiment analysis."
        ),
        "user": "Say hello and analyze the sentiment of your message.",
    }


@pytest.fixture
def debug_logger() -> logging.Logger:
    """Create a debug logger for tests."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
@pytest.mark.live
def test_sync_api(sync_openai_client, test_prompts):
    """Test synchronous API call."""
    result = openai_structured_call(
        client=sync_openai_client,
        model="gpt-4o-2024-08-06",
        output_schema=SentimentResponse,
        system_prompt=test_prompts["system"],
        user_prompt=test_prompts["user"],
    )

    assert isinstance(result, SentimentResponse)
    assert isinstance(result.message, str)
    assert isinstance(result.sentiment, str)
    assert len(result.message) > 0
    assert len(result.sentiment) > 0


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
@pytest.mark.live
@pytest.mark.asyncio
async def test_async_stream(async_openai_client):
    """Test streaming structured output from the OpenAI API."""
    responses = []
    async for response in openai_structured_stream(
        client=async_openai_client,
        model="gpt-4o-2024-08-06",
        output_schema=SentimentResponse,
        system_prompt="You are a helpful assistant.",
        user_prompt="What do you think about AI?",
    ):
        responses.append(response)
        assert isinstance(response, SentimentResponse)
        assert response.message
        assert response.sentiment in ["positive", "negative", "neutral"]


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
@pytest.mark.live
def test_invalid_model(async_openai_client, test_prompts):
    """Test error handling with invalid model."""
    with pytest.raises(Exception) as exc_info:
        openai_structured_call(
            client=async_openai_client,
            model="invalid-model",
            output_schema=SentimentResponse,
            system_prompt=test_prompts["system"],
            user_prompt=test_prompts["user"],
        )
    assert "does not support structured output" in str(exc_info.value)
