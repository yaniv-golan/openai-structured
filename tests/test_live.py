"""Integration tests for the openai_structured client module."""

import os

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from openai_structured import openai_structured_call, openai_structured_stream


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""

    message: str = Field(..., description="The analyzed message")
    sentiment: str = Field(
        ...,
        pattern="(?i)^(positive|negative|neutral|mixed)$",
        description="Sentiment of the message",
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live
async def test_live_api() -> None:
    """Test live API calls with real OpenAI API."""
    async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as client:
        result = await openai_structured_call(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=SentimentResponse,
            user_prompt="What is the sentiment of 'I love pizza'?",
            system_prompt="You analyze sentiment of text.",
        )

        assert isinstance(result, SentimentResponse)
        assert result.sentiment.lower() in [
            "positive",
            "negative",
            "neutral",
            "mixed",
        ]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.live
async def test_live_streaming() -> None:
    """Test streaming with real OpenAI API."""
    async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as client:
        async for result in openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=SentimentResponse,
            user_prompt="What is the sentiment of 'I hate mondays'?",
            system_prompt="You analyze sentiment of text.",
        ):
            assert isinstance(result, SentimentResponse)
            assert result.sentiment.lower() in [
                "positive",
                "negative",
                "neutral",
                "mixed",
            ]
            break  # Test just the first result
