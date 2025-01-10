Examples
========

Basic Usage
-----------

Extract structured data from text:

.. code-block:: python

    from openai import OpenAI
    from openai_structured import openai_structured_call
    from pydantic import BaseModel

    class UserInfo(BaseModel):
        name: str
        age: int

    client = OpenAI()
    result = openai_structured_call(
        client=client,
        model="gpt-4",
        output_schema=UserInfo,
        user_prompt="Tell me about John who is 30 years old",
        system_prompt="Extract user information"
    )
    print(f"Name: {result.name}, Age: {result.age}")

Streaming Example
-------------------

Process structured data as it arrives:

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
            user_prompt="Create a list of tasks",
            system_prompt="Generate tasks with priorities"
        ):
            print(f"Task: {item.task}, Priority: {item.priority}")

    asyncio.run(main())

Advanced Usage
--------------

Use custom system prompts and error handling:

.. code-block:: python

    from openai import OpenAI
    from openai_structured import openai_structured_call, OpenAIClientError
    from pydantic import BaseModel

    class ProductInfo(BaseModel):
        name: str
        price: float
        description: str

    try:
        client = OpenAI()
        result = openai_structured_call(
            client=client,
            model="gpt-4",
            output_schema=ProductInfo,
            user_prompt="Tell me about a laptop",
            system_prompt="Extract product details with exact pricing"
        )
        print(f"{result.name}: ${result.price}")
        print(f"Description: {result.description}")
    except OpenAIClientError as e:
        print(f"Error: {e}") 