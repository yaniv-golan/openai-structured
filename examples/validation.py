"""Example demonstrating Pydantic validation features with openai-structured."""

import asyncio
from typing import List, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, conint, constr

from openai_structured import (
    InvalidResponseFormatError,
    async_openai_structured_call,
)


class BookReview(BaseModel):
    """A structured book review with extensive validation."""

    title: constr(min_length=3, max_length=100) = Field(
        ..., description="Book title"
    )
    author: constr(min_length=2, max_length=50) = Field(
        ..., description="Book author"
    )
    rating: conint(ge=1, le=5) = Field(
        ..., description="Rating from 1 to 5 stars"
    )
    genre: List[
        Literal["fiction", "non-fiction", "mystery", "sci-fi", "romance"]
    ] = Field(
        ..., min_items=1, max_items=3, description="Book genres (1-3 genres)"
    )
    page_count: conint(ge=1) = Field(..., description="Number of pages")
    recommended_age: Literal["children", "young adult", "adult"] = Field(
        ..., description="Recommended age group"
    )
    summary: constr(min_length=50, max_length=500) = Field(
        ..., description="Book summary"
    )


async def main():
    """Run the validation example."""
    client = AsyncOpenAI()

    try:
        # Make a structured call to get a book review
        review = await async_openai_structured_call(
            client=client,
            model="gpt-4o-2024-08-06",
            system_prompt="You are a book reviewer. Provide detailed book reviews with accurate metadata.",
            user_prompt="Review 'Dune' by Frank Herbert.",
            output_schema=BookReview,
        )

        print("\nBook Review:")
        print(f"Title: {review.title}")
        print(f"Author: {review.author}")
        print(f"Rating: {'⭐' * review.rating}")
        print(f"Genres: {', '.join(review.genre)}")
        print(f"Pages: {review.page_count}")
        print(f"Recommended for: {review.recommended_age}")
        print(f"\nSummary:\n{review.summary}")

    except InvalidResponseFormatError as e:
        print(f"\nValidation Error: {e}")
        # The error will include details about which fields failed validation

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
