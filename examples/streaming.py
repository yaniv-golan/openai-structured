# examples/streaming_example.py
import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel

from openai_structured import async_openai_structured_stream


class TodoItem(BaseModel):
    task: str
    priority: str


async def main():
    client = AsyncOpenAI()
    async for item in async_openai_structured_stream(
        client=client,
        model="gpt-4o-2024-08-06",
        output_schema=TodoItem,
        user_prompt=(
            "Create a list of 3 tasks for a software developer "
            "with different priorities"
        ),
        system_prompt="Generate tasks with priorities (high/medium/low)",
    ):
        print(f"Task: {item.task}, Priority: {item.priority}")


if __name__ == "__main__":
    asyncio.run(main())
