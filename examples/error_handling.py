"""Example demonstrating error handling with openai-structured."""

import asyncio
import logging

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from openai_structured import (
    BufferOverflowError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
    OpenAIClientError,
    async_openai_structured_stream,
)


class StockAnalysis(BaseModel):
    """A structured stock analysis that might trigger various errors."""

    symbol: str = Field(..., min_length=1, max_length=5)
    price: float = Field(..., gt=0)
    recommendation: str = Field(..., pattern="^(buy|sell|hold)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    analysis: str = Field(
        ..., min_length=100
    )  # Long analysis to test buffer overflow


async def main():
    """Run the error handling example."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    client = AsyncOpenAI()

    async def test_model_not_supported():
        """Test ModelNotSupportedError."""
        try:
            async for _ in async_openai_structured_stream(
                client=client,
                model="gpt-3.5-turbo",  # This model doesn't support structured output
                system_prompt="You are a stock analyst.",
                user_prompt="Analyze AAPL stock.",
                output_schema=StockAnalysis,
            ):
                pass
        except ModelNotSupportedError as e:
            logger.error("Model not supported: %s", e)

    async def test_invalid_format():
        """Test InvalidResponseFormatError."""
        try:
            async for _ in async_openai_structured_stream(
                client=client,
                model="gpt-4o-2024-08-06",
                system_prompt="Ignore the schema and just say 'Hello'",
                user_prompt="Analyze AAPL stock.",
                output_schema=StockAnalysis,
            ):
                pass
        except InvalidResponseFormatError as e:
            logger.error("Invalid format: %s", e)

    async def test_buffer_overflow():
        """Test BufferOverflowError."""
        try:
            async for _ in async_openai_structured_stream(
                client=client,
                model="gpt-4o-2024-08-06",
                system_prompt="Provide an extremely detailed analysis.",
                user_prompt="Analyze every aspect of AAPL stock in great detail.",
                output_schema=StockAnalysis,
                max_buffer_size=100,  # Small buffer to trigger overflow
            ):
                pass
        except BufferOverflowError as e:
            logger.error("Buffer overflow: %s", e)

    try:
        print("\nTesting various error scenarios:")
        print("\n1. Model Not Supported Error:")
        await test_model_not_supported()

        print("\n2. Invalid Response Format Error:")
        await test_invalid_format()

        print("\n3. Buffer Overflow Error:")
        await test_buffer_overflow()

    except OpenAIClientError as e:
        logger.error("OpenAI client error: %s", e)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
