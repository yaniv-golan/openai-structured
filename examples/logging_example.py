"""Example demonstrating logging with openai-structured."""

import asyncio
import logging
from typing import List, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from openai_structured import async_openai_structured_stream


class MovieReview(BaseModel):
    """A structured movie review."""

    title: str = Field(..., min_length=3)
    year: int = Field(..., ge=1900, le=2024)
    genre: List[Literal["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]]
    rating: float = Field(..., ge=0, le=10)
    strengths: List[str] = Field(..., min_items=1)
    weaknesses: List[str] = Field(..., min_items=1)
    verdict: str = Field(..., min_length=20)


def setup_logging():
    """Configure logging with custom format and levels."""
    # Create logger
    logger = logging.getLogger("openai_structured")
    logger.setLevel(logging.DEBUG)

    # Create console handler with custom format
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Create file handler for errors
    error_file = logging.FileHandler("errors.log")
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)
    logger.addHandler(error_file)

    return logger


async def process_movie_review(client: AsyncOpenAI, logger: logging.Logger):
    """Process a movie review with detailed logging."""
    logger.info("Starting movie review processing")

    try:
        system_prompt = "You are a professional film critic. Provide detailed movie reviews."
        user_prompt = (
            "Review the movie 'Inception' (2010) by Christopher Nolan."
        )

        logger.debug(
            "Making API call with prompts",
            extra={"system_prompt": system_prompt, "user_prompt": user_prompt},
        )

        async for review in async_openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            temperature=0.3,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=MovieReview,
            on_log=lambda level, msg, data=None: logger.log(
                level, msg, extra={"data": data} if data else None
            ),
        ):
            logger.info(
                "Received movie review", extra={"review": review.model_dump()}
            )
            print("\nMovie Review:")
            print(f"Title: {review.title} ({review.year})")
            print(f"Genre: {', '.join(review.genre)}")
            print(f"Rating: {review.rating}/10")
            print("\nStrengths:")
            for strength in review.strengths:
                print(f"- {strength}")
            print("\nWeaknesses:")
            for weakness in review.weaknesses:
                print(f"- {weakness}")
            print(f"\nVerdict: {review.verdict}")

    except Exception as e:
        logger.error(f"Error processing movie review: {str(e)}", exc_info=True)
        raise


async def main():
    """Run the example with logging configuration."""
    logger = setup_logging()
    logger.info("Starting application")

    client = AsyncOpenAI()
    try:
        await process_movie_review(client, logger)
    finally:
        await client.close()
        logger.info("Application finished")


if __name__ == "__main__":
    asyncio.run(main())
