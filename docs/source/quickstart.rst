.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

Quickstart
=========

Installation
-----------

Using pip::

    pip install openai-structured

Using Poetry::

    poetry add openai-structured

Basic Usage
----------

The library provides a simple way to extract structured data from OpenAI models:

.. code-block:: python

    import asyncio
    from pydantic import BaseModel, Field
    from openai_structured import async_openai_structured_stream, StreamConfig
    from openai_structured.errors import StreamBufferError, ValidationError

    # Define your output schema
    class MovieReview(BaseModel):
        title: str
        rating: float
        summary: str
        pros: list[str]
        cons: list[str]

    async def analyze_movie():
        # Convert Pydantic model to JSON Schema
        schema = MovieReview.model_json_schema()

        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a movie critic."},
            {"role": "user", "content": "Review the movie 'Inception'"}
        ]

        try:
            # Stream the response
            async for chunk in async_openai_structured_stream(
                messages=messages,
                schema=schema,
                model="gpt-4o-2024-08-06",
                temperature=0.7,
                stream_config=StreamConfig(
                    max_buffer_size=1024 * 1024  # 1MB
                )
            ):
                # Process each chunk as it arrives
                review = MovieReview(**chunk)
                print(f"Processing: {review.title}")
                print(f"Rating: {review.rating}/10")
                print(f"Summary: {review.summary}")
                print("\nPros:")
                for pro in review.pros:
                    print(f"- {pro}")
                print("\nCons:")
                for con in review.cons:
                    print(f"- {con}")

        except ValidationError as e:
            print(f"Invalid response: {e}")
        except StreamBufferError as e:
            print(f"Buffer overflow: {e}")
        except StreamInterruptedError as e:
            print(f"Stream interrupted: {e}")
        except TokenLimitError as e:
            print(f"Token limit exceeded: {e}")

    # Run the async function
    asyncio.run(analyze_movie())

Stream Configuration
-----------------

Configure streaming behavior:

.. code-block:: python

    from openai_structured import StreamConfig

    # Default configuration
    config = StreamConfig()  # 1MB buffer, 512KB cleanup

    # Custom configuration
    config = StreamConfig(
        max_buffer_size=2 * 1024 * 1024,  # 2MB
        cleanup_threshold=1024 * 1024,     # 1MB
        chunk_size=16 * 1024              # 16KB
    )

    async for chunk in async_openai_structured_stream(
        messages=messages,
        schema=schema,
        stream_config=config
    ):
        process_chunk(chunk)

Error Handling
------------

The library provides robust error handling to help you build resilient applications. Here's a realistic example analyzing sentiment from customer reviews:

.. code-block:: python

    try:
        async for chunk in async_openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=SentimentAnalysis,  # Pydantic model defining the structure
            system_prompt="You are a sentiment analysis expert. You analyze customer reviews to extract sentiment, key phrases, and emotional tone.",
            user_prompt="""Analyze the sentiment in the following customer review. Map the results as follows:
                - Extract the overall sentiment (positive/negative/neutral) into the 'sentiment' field
                - Place the confidence score (0-1) into the 'confidence' field
                - List the key positive phrases in 'positive_phrases'
                - List the key negative phrases in 'negative_phrases'
                - Summarize the emotional tone in 'tone'
                
                Review: {{ review.content }}""",
            file_vars={"review": "customer_review.txt"}
        ):
            process_sentiment_results(chunk)
    except ValueError as e:
        if "token limit" in str(e).lower():
            print(f"Token limit exceeded: {e}")
            print("Consider reducing input size or using a model with larger context")
        else:
            raise
    except StreamBufferError as e:
        print(f"Buffer overflow: {e}")
    except StreamInterruptedError as e:
        print(f"Stream interrupted: {e}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    except APIError as e:
        print(f"API error: {e}")
    finally:
        await client.close()

**Key Error Types:**

*   **ValueError:**  Can indicate a token limit issue. If you encounter this, check the total token count using the `--verbose` flag in the CLI or by logging the prompt length in your code. Reduce the input size or use a model with a larger context window.
*   **StreamBufferError:** Occurs if the internal buffer for streaming is exceeded. Adjust `StreamConfig` parameters if necessary.
*   **StreamInterruptedError:** Indicates that the stream was interrupted before completion. Implement retries if needed.
*   **ValidationError:**  Signals that the generated output doesn't conform to the provided schema. Review your schema and prompt.
*   **APIError:** Represents an error from the OpenAI API. Check the error message for details and consult the OpenAI documentation.

File Processing
~~~~~~~~~~~~~

Process files efficiently:

.. code-block:: python

    import aiofiles

    async def analyze_file(filepath: str):
        async with aiofiles.open(filepath, 'r') as f:
            content = await f.read()

        messages = [
            {"role": "system", "content": "Analyze this document."},
            {"role": "user", "content": content}
        ]

        async for chunk in async_openai_structured_stream(
            messages=messages,
            schema=schema,
            model="o1",  # Use optimized model for large files
            timeout=120.0  # Longer timeout for large files
        ):
            await process_chunk(chunk)

Supported Models
--------------

Production Models
~~~~~~~~~~~~~~~

* ``gpt-4o-2024-08-06``
    - GPT-4 with structured output
    - 128K context window
    - 16K output tokens
    - Full JSON schema support

* ``gpt-4o-mini-2024-07-18``
    - Smaller GPT-4 variant
    - 128K context window
    - 16K output tokens
    - Optimized for faster responses

* ``o1-2024-12-17``
    - Optimized for structured data
    - 200K context window
    - 100K output tokens
    - Best for large structured outputs

* ``o3-mini-2025-01-31``
    - Mini variant optimized for structured data
    - 200K context window
    - 100K output tokens
    - Efficient for large outputs

Development Aliases
~~~~~~~~~~~~~~~~

* ``gpt-4o``: Latest GPT-4 structured model
* ``gpt-4o-mini``: Latest mini variant
* ``o1``: Latest optimized model
* ``o3-mini``: Latest mini optimized model

.. note::
    Use dated versions in production for stability.
    Aliases automatically use the latest compatible version.

Environment Variables
------------------

The library uses these environment variables:

* ``OPENAI_API_KEY`` (required)
    OpenAI API key for authentication

* ``OPENAI_API_BASE`` (optional)
    Custom API endpoint URL

* ``OPENAI_API_VERSION`` (optional)
    Specific API version to use 

Advanced Usage
------------

Complex Schema
~~~~~~~~~~~~

Use Pydantic for complex schemas:

.. code-block:: python

    from typing import Literal
    from pydantic import BaseModel, Field

    class Character(BaseModel):
        name: str
        age: int = Field(minimum=0, maximum=150)
        occupation: str
        skills: list[str]

    class MovieAnalysis(BaseModel):
        title: str
        rating: float = Field(minimum=0, maximum=10)
        summary: str
        themes: list[str]

    # Convert to JSON Schema
    schema = MovieAnalysis.model_json_schema()

    async def analyze_movie():
        try:
            async for chunk in async_openai_structured_stream(
                messages=[
                    {"role": "system", "content": "Analyze this movie."},
                    {"role": "user", "content": "Analyze 'The Matrix'"}
                ],
                schema=schema
            ):
                analysis = MovieAnalysis(**chunk)
                print(f"Analyzing {analysis.title}...")
                for character in analysis.characters:
                    print(f"- {character.name}: {character.role}")
        except ValidationError as e:
            print(f"Validation error: {e}")

Error Handling
------------

The library provides comprehensive error handling:

.. code-block:: python

    from openai_structured.errors import (
        StreamBufferError,
        StreamInterruptedError,
        ValidationError,
        TokenLimitError
    )

    try:
        async for chunk in async_openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=OutputSchema,
            system_prompt="Analyze this text",
            user_prompt="Sample text to analyze",
        ):
            process_chunk(chunk)
    except ValueError as e:
        if "token limit" in str(e).lower():
            print(f"Token limit exceeded: {e}")
            print("Consider reducing input size or using a model with larger context")
        else:
            raise
    except StreamBufferError as e:
        print(f"Buffer overflow: {e}")
    except StreamInterruptedError as e:
        print(f"Stream interrupted: {e}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    except APIError as e:
        print(f"API error: {e}")
    finally:
        await client.close() 