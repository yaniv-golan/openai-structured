# examples/streaming_example.py
import asyncio

from openai import OpenAI
from openai_structured import openai_structured_stream
from pydantic import BaseModel


class TodoItem(BaseModel):
    task: str
    priority: str


async def main():
    client = OpenAI()
    async for item in openai_structured_stream(
        client=client,
        model="gpt-4",
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
