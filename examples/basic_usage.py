"""Basic usage example for openai-structured."""

import os
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field

from openai_structured import (
    ModelNotSupportedError,
    OpenAIClientError,
    openai_structured_call,
)


class TodoItem(BaseModel):
    """A single todo item with priority."""

    task: str = Field(..., description="The task description")
    priority: str = Field(
        ...,
        pattern="^(high|medium|low|High|Medium|Low)$",
        description="Priority level (high/medium/low)",
    )


class TodoList(BaseModel):
    """A list of todo items."""

    items: List[TodoItem]
    total_count: int = Field(..., ge=0)


def main() -> None:
    """Run the example."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize client with API key
    client = OpenAI(api_key=api_key)

    try:
        # Define prompts
        system_prompt = """
        You are a task manager that creates todo lists.
        Each task should have a clear description and priority level.
        """
        user_prompt = (
            "Create a todo list with 3 tasks for a software developer."
        )

        # Get structured response
        result = openai_structured_call(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=TodoList,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Process the response
        print(f"\nTotal tasks: {result.total_count}")
        print("\nTasks:")
        for item in result.items:
            print(f"- {item.task} (Priority: {item.priority})")

    except ModelNotSupportedError as e:
        print(f"\nError: Model not supported - {e}")
    except OpenAIClientError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        client.close()  # Cleanup resources


if __name__ == "__main__":
    main()
