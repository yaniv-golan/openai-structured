.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

Examples
=======

This section provides examples of common use cases for the ``openai-structured`` library.

Basic Examples
------------

Movie Review Analysis
~~~~~~~~~~~~~~~~~~

Extract structured movie reviews with streaming:

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

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class MovieReview(BaseModel):
        title: str
        rating: float = Field(minimum=0, maximum=10)
        summary: str
        pros: list[str]
        cons: list[str]

    async def analyze_movie(title: str):
        client = AsyncOpenAI()  # Initialize client

        try:
            async for chunk in async_openai_structured_stream(
                client=client,
                model="gpt-4o-2024-08-06",  # Use specific version
                output_schema=MovieReview,
                system_prompt="You are a movie critic.",
                user_prompt=f"Review the movie '{title}'",
                stream_config=StreamConfig(
                    max_buffer_size=1024 * 1024,  # 1MB
                    cleanup_threshold=512 * 1024   # 512KB
                ),
                timeout=30.0  # 30 second timeout
            ):
                logger.info("Processing review for %s", chunk.title)
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
            logger.error("Buffer overflow: %s", e)
            logger.info("Consider increasing buffer size or processing chunks faster")
        except StreamInterruptedError as e:
            logger.error("Stream interrupted: %s", e)
            logger.info("Check network connection and API status")
        except StreamParseError as e:
            logger.error(
                "Parse error after %d attempts: %s",
                e.attempts, e.last_error
            )
            logger.debug("Cleanup stats: %s", e.__dict__.get('_cleanup_stats', {}))
        except ValueError as e:
            if "token limit" in str(e).lower():
                logger.error("Token limit exceeded: %s", e)
                logger.info("Consider reducing input size or using a model with larger context")
            else:
                raise
        except ValidationError as e:
            logger.error("Invalid review format: %s", e)
            logger.debug("Error context: %s", e.errors())
        except APITimeoutError as e:
            logger.error("API timeout: %s", e)
            logger.info("Consider increasing timeout or optimizing request")
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

Analyze code with custom rules and streaming:

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

            async for chunk in async_openai_structured_stream(
                client=client,
                model="gpt-4o-2024-08-06",  # Use specific version
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
            if hasattr(e, '_cleanup_stats'):
                logger.debug("Cleanup stats: %s", e._cleanup_stats)
        except StreamInterruptedError as e:
            logger.error("Stream interrupted: %s", e)
            logger.info("Check network connection and API status")
        except StreamParseError as e:
            logger.error(
                "Parse error after %d attempts: %s (max attempts: %d)",
                e.attempts, e.last_error, StreamBuffer.MAX_PARSE_ERRORS
            )
            logger.debug("Cleanup stats: %s", e.__dict__.get('_cleanup_stats', {}))
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

Configure buffer settings for different use cases:

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
            logger.debug("Cleanup stats: %s", e.__dict__.get('_cleanup_stats', {}))

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
```