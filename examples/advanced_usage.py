"""Advanced usage example for openai-structured."""

import asyncio
import logging
import os
from typing import List, NoReturn, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from openai_structured import (
    BufferOverflowError,
    EmptyResponseError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
    OpenAIClientError,
    openai_structured_stream,
)


class NewsArticle(BaseModel):
    """A structured news article."""

    title: str = Field(..., min_length=10, max_length=100)
    summary: str = Field(..., min_length=50)
    topics: List[str] = Field(..., min_items=1)
    sentiment: str = Field(
        ...,
        pattern="^(positive|negative|neutral)$",
        description="Sentiment of the article (positive/negative/neutral)",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: Optional[str] = None


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("openai_structured_example")
    logger.setLevel(logging.DEBUG)

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


async def process_articles(
    client: AsyncOpenAI, logger: logging.Logger
) -> None:
    """Process news articles with error handling."""
    system_prompt = """
    You are a news analyst that processes articles and provides structured analysis.
    For each article:
    1. Create a clear, concise title
    2. Provide a detailed summary
    3. List relevant topics
    4. Analyze the sentiment
    5. Include a confidence score
    6. Optionally include the source
    """
    user_prompt = """
    Analyze these recent tech industry developments:
    1. AI advancements in healthcare
    2. Cybersecurity challenges
    3. Environmental impact of cloud computing
    """

    try:
        async for article in openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=NewsArticle,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,  # Lower temperature for more focused output
            logger=logger,  # Enable detailed logging
        ):
            logger.info("Received article: %s", article.title)
            print("\nArticle Analysis:")
            print(f"Title: {article.title}")
            print(f"Summary: {article.summary}")
            print(f"Topics: {', '.join(article.topics)}")
            print(f"Sentiment: {article.sentiment}")
            print(f"Confidence: {article.confidence:.2f}")
            if article.source:
                print(f"Source: {article.source}")
            print("-" * 80)

    except ModelNotSupportedError as e:
        logger.error("Model not supported: %s", e)
        print(f"\nError: {e}")
    except InvalidResponseFormatError as e:
        logger.error("Invalid response format: %s", e)
        print(f"\nError: Response validation failed - {e}")
    except EmptyResponseError as e:
        logger.error("Empty response received: %s", e)
        print(f"\nError: {e}")
    except BufferOverflowError as e:
        logger.error("Buffer overflow: %s", e)
        print(f"\nError: Response too large - {e}")
    except OpenAIClientError as e:
        logger.error("OpenAI client error: %s", e)
        print(f"\nError: API error - {e}")
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\nUnexpected error: {e}")


async def main() -> NoReturn:
    """Run the advanced example."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize logger and client
    logger = setup_logging()
    client = AsyncOpenAI(api_key=api_key)

    try:
        await process_articles(client, logger)
    except Exception as e:
        logger.exception("Fatal error in main")
        print(f"\nFatal error: {e}")
    finally:
        # Cleanup
        await client.close()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    asyncio.run(main())
