# examples/streaming_example.py
import asyncio
import os
from typing import List

from openai import OpenAI
from pydantic import BaseModel

from openai_structured.client import openai_structured_stream


class TodoItem(BaseModel):
    task: str
    priority: str


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

user_prompt = "Create a list of three todo items."
system_prompt = "Generate a list of tasks with their priorities in JSON format."


async def main():
    try:
        async for item in openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=TodoItem,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        ):
            print(f"Received item: {item}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
