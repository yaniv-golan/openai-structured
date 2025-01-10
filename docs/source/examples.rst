Examples
========

This section provides examples of common use cases for openai-structured. Each example
demonstrates how to use the library effectively with real-world scenarios.

Basic Usage
-----------

The simplest use case is extracting structured data from text. This example shows how to:

* Define a Pydantic model for the expected structure
* Make a simple API call
* Handle the structured response

.. code-block:: python

    from openai import OpenAI
    from openai_structured import openai_structured_call
    from pydantic import BaseModel

    class UserInfo(BaseModel):
        name: str
        age: int

    def main():
        client = OpenAI()
        result = openai_structured_call(
            client=client,
            model="gpt-4",
            output_schema=UserInfo,
            user_prompt="Tell me about John who is 30 years old",
            system_prompt="Extract user information"
        )
        print(f"Name: {result.name}, Age: {result.age}")

    if __name__ == "__main__":
        main()

Expected output:

.. code-block:: text

    Name: John, Age: 30

Streaming Example
-------------------

For handling large responses or when you want to process data as it arrives,
you can use the streaming API. This example demonstrates:

* Async streaming functionality
* Processing structured items as they arrive
* Using line continuation for long strings

.. code-block:: python

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
            system_prompt="Generate tasks with priorities (high/medium/low)"
        ):
            print(f"Task: {item.task}, Priority: {item.priority}")

    if __name__ == "__main__":
        asyncio.run(main())

Expected output:

.. code-block:: text

    Task: Implement user authentication, Priority: high
    Task: Write unit tests for API endpoints, Priority: medium
    Task: Update API documentation, Priority: low

Advanced Usage
--------------

This example shows more advanced features:

* Custom system prompts for better control
* Error handling with specific exceptions
* Complex data structures with multiple fields
* Proper Python packaging structure

.. code-block:: python

    from openai import OpenAI
    from openai_structured import (
        openai_structured_call,
        OpenAIClientError,
    )
    from pydantic import BaseModel

    class ProductInfo(BaseModel):
        name: str
        price: float
        description: str

    def main():
        client = OpenAI()
        try:
            result = openai_structured_call(
                client=client,
                model="gpt-4",
                output_schema=ProductInfo,
                user_prompt=(
                    "Tell me about a high-end laptop with detailed specifications"
                ),
                system_prompt="Extract product details with exact pricing"
            )
            print(f"{result.name}: ${result.price}")
            print(f"Description: {result.description}")
        except OpenAIClientError as error:
            print(f"Error occurred: {error}")

    if __name__ == "__main__":
        main()

Expected output:

.. code-block:: text

    MacBook Pro 16": $2499.00
    Description: High-performance laptop with M2 Pro chip, 16GB RAM, 512GB SSD...

Common Patterns
-------------

Error Handling
~~~~~~~~~~~~

Always wrap API calls in try-except blocks to handle specific exceptions:

.. code-block:: python

    from openai_structured import (
        openai_structured_call,
        OpenAIClientError,
        APIResponseError,
        ModelNotSupportedError,
        EmptyResponseError,
        InvalidResponseFormatError,
    )

    try:
        result = openai_structured_call(...)
    except ModelNotSupportedError:
        print("Please use one of: gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview")
    except APIResponseError as e:
        print(f"API error: {e}")
    except EmptyResponseError:
        print("Received empty response from API")
    except InvalidResponseFormatError as e:
        print(f"Could not parse response: {e}")
    except OpenAIClientError as e:
        print(f"Other error: {e}")

Environment Variables
~~~~~~~~~~~~~~~~~~

Set up your environment properly:

.. code-block:: bash

    # In your shell or .env file
    export OPENAI_API_KEY=your-api-key-here

    # Optional: set default model
    export OPENAI_MODEL=gpt-4

Custom System Prompts
~~~~~~~~~~~~~~~~~~

Customize the system prompt for better results:

.. code-block:: python

    system_prompt = """
    Extract precise information from the input.
    Ensure all numeric values are accurate.
    Format strings consistently.
    Return only valid JSON matching the schema.
    """

Additional Tips
-------------

* Always use type hints and Pydantic models for better code safety
* Handle exceptions appropriately in production code
* Consider using environment variables for API keys
* Use async streaming for large responses or real-time processing
* Test your models with various inputs to ensure robustness
* Set appropriate timeouts for your use case
* Consider rate limiting in production environments 