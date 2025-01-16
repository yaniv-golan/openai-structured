"""Basic async example for openai-structured."""

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from openai_structured import async_openai_structured_call


class Person(BaseModel):
    """A simple person schema."""

    name: str = Field(..., description="Full name of the person")
    age: int = Field(..., ge=0, le=120, description="Age in years")
    occupation: str = Field(..., description="Current occupation or job title")
    hobbies: list[str] = Field(..., min_items=1, max_items=5, description="List of hobbies (1-5 items)")


async def main():
    """Run the basic async example."""
    client = AsyncOpenAI()

    # Make a structured call to get person info
    person = await async_openai_structured_call(
        client=client,
        model="gpt-4o-2024-08-06",
        system_prompt="You are a helpful assistant that provides example person profiles.",
        user_prompt="Generate a profile for a software engineer.",
        output_schema=Person,
    )

    print("\nPerson Profile:")
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Occupation: {person.occupation}")
    print("Hobbies:")
    for hobby in person.hobbies:
        print(f"- {hobby}")


if __name__ == "__main__":
    asyncio.run(main())
