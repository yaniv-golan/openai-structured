.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

Examples
=======

This section provides examples of common use cases for working with `OpenAI Structured Outputs <https://platform.openai.com/docs/guides/function-calling>`_ using the ``openai-structured`` library.

Basic Examples
------------

Movie Review Analysis
~~~~~~~~~~~~~~~~~~

Extract structured movie reviews using OpenAI Structured Outputs with streaming:

.. code-block:: python

    import logging
    from pydantic import BaseModel, Field
    from openai import AsyncOpenAI, APIError, APITimeoutError
    from openai_structured import async_openai_structured_stream, StreamConfig
    from openai_structured.errors import (
        StreamBufferError,
        StreamInterruptedError,
        StreamParseError,
        ValidationError,
        ModelNotSupportedError
    )
    from typing import Optional

    # Configure application logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Library logging callback - the library does not use Python's logging infrastructure directly
    # Instead, it calls this callback for all internal logs, giving you full control over log handling
    def log_callback(level: int, message: str, data: Optional[dict] = None):
        """Custom logging callback to handle library logs.
        
        The library will call this function for all internal logs, allowing you to:
        - Filter logs by level
        - Format messages and data as needed
        - Route logs to your logging system
        - Add additional context or processing
        
        Note: The library itself does not use Python's logging infrastructure.
        This callback is the only way the library outputs logs.
        """
        # Example: Route library logs through application logger
        if level == logging.DEBUG:
            logger.debug("Library: " + message, data or {})
        elif level == logging.INFO:
            logger.info("Library: " + message, data or {})
        elif level == logging.WARNING:
            logger.warning("Library: " + message, data or {})
        elif level == logging.ERROR:
            logger.error("Library: " + message, data or {})

    class MovieReview(BaseModel):
        title: str
        rating: float = Field(minimum=0, maximum=10)
        summary: str
        pros: list[str]
        cons: list[str]

    async def analyze_movie(title: str):
        client = AsyncOpenAI()  # Initialize client

        try:
            # Application log
            logger.info("Starting analysis of movie: %s", title)

            # Use OpenAI Structured Outputs with streaming
            async for chunk in async_openai_structured_stream(
                client=client,
                model="gpt-4o-2024-08-06",
                output_schema=MovieReview,
                system_prompt="You are a movie critic.",
                user_prompt=f"Review the movie '{title}'",
                stream_config=StreamConfig(
                    max_buffer_size=1024 * 1024,  # 1MB
                    cleanup_threshold=512 * 1024   # 512KB
                ),
                timeout=30.0,
                on_log=log_callback  # Library will use this for all logging
            ):
                # Application logs
                logger.info("Received review for: %s", chunk.title)
                print(f"Title: {chunk.title}")
                print(f"Rating: {chunk.rating}/10")
                print(f"Summary: {chunk.summary}")
                print("\nPros:")
                for pro in chunk.pros:
                    print(f"- {pro}")
                print("\nCons:")
                for con in chunk.cons:
                    print(f"- {con}")

        except StreamBufferError as e:
            # Application error logging
            logger.error("Failed to process stream: %s", e)
            logger.info("Hint: Try increasing buffer size or adjusting cleanup threshold")

        except StreamInterruptedError as e:
            logger.error("Stream interrupted: %s", e)
            logger.info("Check network connection and API status")

        except StreamParseError as e:
            logger.error(
                "Parse error after %d attempts: %s",
                e.attempts, e.last_error
            )
            logger.debug("Buffer cleanup completed")

        except ValidationError as e:
            logger.error("Invalid analysis format: %s", e)
            logger.debug("Error context: %s", e.errors())

        except APITimeoutError as e:
            logger.error("API timeout: %s", e)
            logger.info("Consider increasing timeout for large files")

        except APIError as e:
            logger.error("API error: %s", e)
            if e.status_code == 429:
                logger.info("Rate limit exceeded, implement backoff")
            elif e.status_code >= 500:
                logger.info("Server error, retry with exponential backoff")

        except ModelNotSupportedError as e:
            logger.error("Model not supported: %s", e)
            logger.info("Supported versions: %s", e.supported_versions)
        finally:
            await client.close()  # Cleanup resources

Code Analysis
~~~~~~~~~~~

Analyze code using OpenAI Structured Outputs with custom rules and streaming:

.. code-block:: python

    import logging
    import aiofiles
    from typing import Literal
    from pydantic import BaseModel, Field
    from openai import AsyncOpenAI, APIError, APITimeoutError
    from openai_structured import async_openai_structured_stream, StreamConfig
    from openai_structured.errors import (
        StreamBufferError,
        StreamInterruptedError,
        StreamParseError,
        ValidationError,
        ModelNotSupportedError
    )

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class CodeIssue(BaseModel):
        severity: Literal["high", "medium", "low"]
        line_number: int = Field(ge=1)
        description: str
        suggestion: str

    class CodeAnalysis(BaseModel):
        file_name: str
        language: str
        issues: list[CodeIssue]
        best_practices: list[str]
        improvement_summary: str

    async def analyze_code(file_path: str):
        client = AsyncOpenAI()

        try:
            # Read file with proper error handling
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    code = await f.read()
            except IOError as e:
                logger.error("Failed to read file: %s", e)
                return

            # Configure stream with larger buffer for code analysis
            config = StreamConfig(
                max_buffer_size=2 * 1024 * 1024,  # 2MB for large files
                cleanup_threshold=1024 * 1024,     # 1MB (50% of max)
                chunk_size=16 * 1024              # 16KB chunks
            )

            # Use OpenAI Structured Outputs with streaming
            async for chunk in async_openai_structured_stream(
                client=client,
                model="gpt-4o-2024-08-06",  # Model with OpenAI Structured Outputs support
                output_schema=CodeAnalysis,
                system_prompt="You are a code review expert.",
                user_prompt=f"Analyze this code:\n\n{code}",
                temperature=0.2,  # Lower temperature for analysis
                stream_config=config,
                timeout=60.0  # Longer timeout for large files
            ):
                # Log buffer size changes
                if config.should_log_size():
                    logger.info(
                        "Buffer size: %d bytes",
                        config.total_bytes
                    )

                logger.info("Analyzing %s", chunk.file_name)
                print(f"\nAnalysis for {chunk.file_name}:")
                print(f"Language: {chunk.language}")
                
                print("\nIssues:")
                for issue in chunk.issues:
                    print(f"[{issue.severity.upper()}] Line {issue.line_number}")
                    print(f"  Problem: {issue.description}")
                    print(f"  Suggestion: {issue.suggestion}")
                
                print("\nBest Practices:")
                for practice in chunk.best_practices:
                    print(f"- {practice}")
                
                print(f"\nSummary: {chunk.improvement_summary}")

        except StreamBufferError as e:
            logger.error("Buffer overflow: %s", e)
            logger.info("Consider increasing buffer size or processing chunks faster")
        except StreamInterruptedError as e:
            logger.error("Stream interrupted: %s", e)
            logger.info("Check network connection and API status")
        except StreamParseError as e:
            logger.error(
                "Parse error after %d attempts: %s (max attempts: %d)",
                e.attempts, e.last_error, StreamBuffer.MAX_PARSE_ERRORS
            )
            logger.debug("Buffer cleanup completed")
        except ValidationError as e:
            logger.error("Invalid analysis format: %s", e)
            logger.debug("Error context: %s", e.errors())
        except APITimeoutError as e:
            logger.error("API timeout: %s", e)
            logger.info("Consider increasing timeout for large files")
        except APIError as e:
            logger.error("API error: %s", e)
            if e.status_code == 429:
                logger.info("Rate limit exceeded, implement backoff")
            elif e.status_code >= 500:
                logger.info("Server error, retry with exponential backoff")
        except ModelNotSupportedError as e:
            logger.error("Model not supported: %s", e)
            logger.info("Supported versions: %s", e.supported_versions)
        finally:
            await client.close()  # Cleanup resources

Buffer Management
~~~~~~~~~~~~~~

Configure buffer settings for different OpenAI Structured Outputs use cases:

.. code-block:: python

    import logging
    from openai import AsyncOpenAI
    from openai_structured import StreamConfig, async_openai_structured_stream
    from openai_structured.errors import StreamBufferError, StreamParseError

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Default configuration (1MB buffer)
    config = StreamConfig()  # Uses ijson for efficient parsing

    # Large responses (2MB buffer)
    large_config = StreamConfig(
        max_buffer_size=2 * 1024 * 1024,    # 2MB
        cleanup_threshold=1024 * 1024,       # 1MB (50% of max)
        chunk_size=16 * 1024                # 16KB
    )

    # Memory-constrained (256KB buffer)
    small_config = StreamConfig(
        max_buffer_size=256 * 1024,    # 256KB
        cleanup_threshold=128 * 1024,   # 128KB (50% of max)
        chunk_size=4 * 1024            # 4KB
    )

    async def process_with_config(config: StreamConfig):
        client = AsyncOpenAI()

        try:
            async for chunk in async_openai_structured_stream(
                client=client,
                model="gpt-4o-2024-08-06",
                output_schema=OutputSchema,
                system_prompt="Process this data.",
                user_prompt="Sample input",
                stream_config=config
            ):
                # Monitor buffer size changes
                if config.should_log_size():
                    logger.info(
                        "Buffer size: %d bytes (max: %d, cleanup at: %d)",
                        config.total_bytes,
                        config.max_buffer_size,
                        config.cleanup_threshold
                    )
                process_chunk(chunk)

        except StreamBufferError as e:
            # Buffer exceeded max size after MAX_CLEANUP_ATTEMPTS
            logger.error(
                "Buffer overflow with %d bytes limit after %d cleanup attempts: %s",
                config.max_buffer_size,
                StreamBuffer.MAX_CLEANUP_ATTEMPTS,
                e
            )
            if hasattr(e, '_cleanup_stats'):
                logger.debug("Cleanup stats: %s", e._cleanup_stats)

        except StreamParseError as e:
            # Failed to parse after MAX_PARSE_ERRORS attempts
            logger.error(
                "Parse error after %d attempts (max: %d): %s",
                e.attempts,
                StreamBuffer.MAX_PARSE_ERRORS,
                e.last_error
            )
            logger.debug("Buffer cleanup completed")

        finally:
            await client.close()

Model Support
~~~~~~~~~~~

Use different models with version validation:

.. code-block:: python

    from openai import AsyncOpenAI
    from openai_structured import async_openai_structured_stream
    from openai_structured.errors import ModelNotSupportedError

    async def use_models():
        client = AsyncOpenAI()

        try:
            # Production model with specific version
            async for chunk in async_openai_structured_stream(
                client=client,
                model="gpt-4o-2024-08-06",  # Specific version
                output_schema=OutputSchema,
                system_prompt="Process this.",
                user_prompt="Sample input",
                max_tokens=8000  # Model-specific limit
            ):
                process_chunk(chunk)

            # Development alias (latest compatible version)
            async for chunk in async_openai_structured_stream(
                client=client,
                model="gpt-4o",  # Latest version
                output_schema=OutputSchema,
                system_prompt="Process this.",
                user_prompt="Sample input"
            ):
                process_chunk(chunk)

            # Optimized model for large responses
            async for chunk in async_openai_structured_stream(
                client=client,
                model="o1-2024-12-17",  # Large context window
                output_schema=OutputSchema,
                system_prompt="Process this.",
                user_prompt="Sample input",
                max_tokens=50000  # Up to 100K tokens
            ):
                process_chunk(chunk)

        except ModelNotSupportedError as e:
            print(f"Model version error: {e}")
            print("Supported versions:")
            for model, version in e.supported_versions.items():
                print(f"- {model}: {version}")

        finally:
            await client.close()

Example Schemas
==============

The library provides example schemas and patterns to help you get started.

Basic Usage
----------

The simplest way to use the library is with the ``SimpleMessage`` schema:

.. code-block:: python

    from openai import OpenAI
    from openai_structured import openai_structured
    from openai_structured.examples.schemas import SimpleMessage

    client = OpenAI()
    result = openai_structured(
        client=client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        user_prompt="What is the capital of France?"
    )
    print(result.message)  # "The capital of France is Paris."

Available Schemas
--------------

1. SimpleMessage
~~~~~~~~~~~~~~

A basic schema for text responses:

.. code-block:: python

    from openai_structured.examples.schemas import SimpleMessage

    class SimpleMessage(BaseModel):
        """Simple schema with a single message field."""
        message: str

Use this when you just need the model's response as text.

2. SentimentMessage
~~~~~~~~~~~~~~~~

A more complex schema that includes sentiment analysis:

.. code-block:: python

    from openai_structured.examples.schemas import SentimentMessage

    class SentimentMessage(BaseModel):
        """Schema for sentiment analysis responses."""
        message: str = Field(..., description="The analyzed message")
        sentiment: str = Field(
            ...,
            pattern="(?i)^(positive|negative|neutral|mixed)$",
            description="Sentiment of the message"
        )

Use this when you need both content and sentiment analysis:

.. code-block:: python

    result = openai_structured(
        client=client,
        model="gpt-4o",
        output_schema=SentimentMessage,
        user_prompt="How do you feel about AI?"
    )
    print(f"Message: {result.message}")
    print(f"Sentiment: {result.sentiment}")

Creating Your Own Schemas
----------------------

You can use these examples as templates for your own schemas:

1. Basic Pattern
~~~~~~~~~~~~~

.. code-block:: python

    from pydantic import BaseModel

    class YourSchema(BaseModel):
        field1: str
        field2: int

2. With Validation
~~~~~~~~~~~~~~~

.. code-block:: python

    from pydantic import BaseModel, Field

    class YourValidatedSchema(BaseModel):
        field1: str = Field(..., description="Field description")
        field2: int = Field(..., gt=0, description="Must be positive")

3. With Complex Types
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from typing import List, Optional
    from pydantic import BaseModel

    class YourComplexSchema(BaseModel):
        items: List[str]
        details: Optional[dict]

Best Practices
------------

1. **Clear Field Names**
   - Use descriptive names
   - Follow Python naming conventions
   - Add field descriptions

2. **Appropriate Validation**
   - Add type hints
   - Use Field() for constraints
   - Include pattern validation where needed

3. **Documentation**
   - Add class docstrings
   - Document field meanings
   - Include usage examples

4. **Type Safety**
   - Use appropriate types
   - Consider Optional fields
   - Add proper type hints
```