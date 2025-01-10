import os
import pytest
from openai import OpenAI
from pydantic import BaseModel

from openai_structured.client import openai_structured_call, openai_structured_stream

class SentimentResponse(BaseModel):
    message: str
    sentiment: str

@pytest.fixture
def openai_client():
    """Create OpenAI client with proper API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "test-key":
        raise ValueError("Invalid API key. Please set a valid OPENAI_API_KEY environment variable.")
    
    return OpenAI(
        api_key=api_key,
        organization=None  # Ensure no organization is set that might override settings
    )

@pytest.fixture
def test_prompts():
    """Test prompts for consistency across tests."""
    return {
        "system": "You are a friendly assistant that provides messages with sentiment analysis.",
        "user": "Say hello and analyze the sentiment of your message."
    }

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set"
)
@pytest.mark.live
def test_sync_api(openai_client, test_prompts):
    """Test synchronous API call."""
    result = openai_structured_call(
        client=openai_client,
        model="gpt-4o-2024-08-06",
        output_schema=SentimentResponse,
        system_prompt=test_prompts["system"],
        user_prompt=test_prompts["user"],
    )
    
    # Validate response structure
    assert isinstance(result, SentimentResponse)
    assert isinstance(result.message, str)
    assert isinstance(result.sentiment, str)
    assert len(result.message) > 0
    assert len(result.sentiment) > 0

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set"
)
@pytest.mark.live
@pytest.mark.asyncio
async def test_async_stream(openai_client, test_prompts):
    """Test asynchronous streaming API call."""
    chunks_received = 0
    last_chunk = None
    
    async for chunk in openai_structured_stream(
        client=openai_client,
        model="gpt-4o-2024-08-06",
        output_schema=SentimentResponse,
        system_prompt=test_prompts["system"],
        user_prompt=test_prompts["user"],
    ):
        # Validate each chunk
        assert isinstance(chunk, SentimentResponse)
        assert isinstance(chunk.message, str)
        assert isinstance(chunk.sentiment, str)
        
        chunks_received += 1
        last_chunk = chunk
    
    # Validate we received chunks and final result
    assert chunks_received > 0, "No chunks received from stream"
    assert last_chunk is not None, "No final chunk received"
    assert len(last_chunk.message) > 0, "Final message is empty"
    assert len(last_chunk.sentiment) > 0, "Final sentiment is empty"

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set"
)
@pytest.mark.live
def test_invalid_model(openai_client, test_prompts):
    """Test error handling with invalid model."""
    with pytest.raises(Exception) as exc_info:
        openai_structured_call(
            client=openai_client,
            model="invalid-model",
            output_schema=SentimentResponse,
            system_prompt=test_prompts["system"],
            user_prompt=test_prompts["user"],
        )
    assert "does not support structured output" in str(exc_info.value) 