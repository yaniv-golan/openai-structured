Examples
========

This section provides examples of common use cases for openai-structured. Each example
demonstrates how to use the library effectively with real-world scenarios.

Basic Usage
-----------

The basic example demonstrates:

* Defining Pydantic models with field validation
* Making API calls with proper error handling
* Processing structured responses
* Using environment variables for configuration

.. code-block:: python

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
        client = OpenAI()

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

Expected output:

.. code-block:: text

    Total tasks: 3

    Tasks:
    - Implement user authentication system (Priority: high)
    - Write unit tests for core functionality (Priority: medium)
    - Update API documentation (Priority: low)

Streaming Example
-------------------

For handling large responses or when you want to process data as it arrives,
you can use the streaming API. This example demonstrates:

* Async streaming functionality with proper error handling
* Processing structured items as they arrive
* Using type hints and proper Python typing
* Handling various error cases

.. code-block:: python

    import asyncio
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
        client = AsyncOpenAI()

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

Expected output:

.. code-block:: text

    Streaming analysis results:

    Topic: Employment and Workforce
    Insight: AI automation is reshaping job markets, creating new roles while displacing traditional ones
    Confidence: 0.95

    Topic: Healthcare
    Insight: AI-powered diagnostics are improving early disease detection and treatment planning
    Confidence: 0.88

    Topic: Education
    Insight: Personalized learning platforms are adapting to individual student needs
    Confidence: 0.92

Advanced Usage
--------------

This example shows more advanced features:

* Custom system prompts for better control
* Error handling with specific exceptions
* Complex data structures with multiple fields
* Proper Python packaging structure 