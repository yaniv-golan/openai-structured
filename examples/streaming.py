"""Async streaming example for openai-structured."""

import asyncio
import os
from typing import NoReturn

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from openai_structured import (
    BufferOverflowError,
    ModelNotSupportedError,
    OpenAIClientError,
    openai_structured_stream,
)


class AnalysisResult(BaseModel):
    """Progressive analysis result."""

    topic: str = Field(..., description="Current topic being analyzed")
    insight: str = Field(..., min_length=10)
    confidence: float = Field(..., ge=0.0, le=1.0)


async def main() -> NoReturn:
    """Run the streaming example."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize async client with API key
    client = AsyncOpenAI(api_key=api_key)

    try:
        # Define prompts
        system_prompt = """
        You are an AI analyst providing progressive insights.
        For each response:
        1. Focus on a specific topic
        2. Provide a detailed insight
        3. Include a confidence score
        """
        user_prompt = (
            "Analyze the impact of AI on different aspects of society."
        )

        # Stream structured responses
        print("\nStreaming analysis results:\n")
        async for result in openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=AnalysisResult,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            print(f"Topic: {result.topic}")
            print(f"Insight: {result.insight}")
            print(f"Confidence: {result.confidence:.2f}\n")

    except ModelNotSupportedError as e:
        print(f"\nError: Model not supported - {e}")
    except BufferOverflowError as e:
        print(f"\nError: Buffer overflow - {e}")
    except OpenAIClientError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        await client.close()  # Cleanup resources


if __name__ == "__main__":
    asyncio.run(main())
