.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

API Reference
============

This document details the API for the ``openai-structured`` library, which provides a Python interface for working with `OpenAI Structured Outputs <https://platform.openai.com/docs/guides/function-calling>`_.

Version Compatibility
------------------

Python Support
~~~~~~~~~~~~

* Python 3.9+: Full support
* Python 3.8: Limited support (no TypedDict)
* Python 3.7 and below: Not supported

API Versions
~~~~~~~~~~

* OpenAI API: v2024-02-15 or later
* JSON Schema: Draft 7
* Pydantic: v2.0+

Client
------

.. module:: openai_structured.client

The client module provides functions for working with OpenAI Structured Outputs, featuring streaming support and efficient buffer management.

Functions
~~~~~~~~~

.. function:: async_openai_structured_stream(*, messages: List[Dict[str, str]], schema: Dict[str, Any], model: str = "gpt-4o", temperature: float = 0.0, max_tokens: Optional[int] = None, top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, timeout: float = 60.0, stream_config: Optional[StreamConfig] = None, validate_schema: bool = True, on_log: Optional[Callable[[str, Any], Awaitable[None]]] = None) -> AsyncGenerator[Dict[str, Any], None]

    Make a streaming OpenAI API call using OpenAI Structured Outputs.

    :param messages: List of chat messages in OpenAI format
    :param schema: JSON Schema defining the expected response structure
    :param model: Model to use (default: "gpt-4o")
    :param temperature: Sampling temperature (default: 0.0)
    :param max_tokens: Maximum tokens to generate (default: model-specific)
    :param top_p: Top-p sampling parameter (default: 1.0)
    :param frequency_penalty: Frequency penalty (default: 0.0)
    :param presence_penalty: Presence penalty (default: 0.0)
    :param timeout: API timeout in seconds (default: 60.0)
    :param stream_config: Stream configuration (default: None)
    :param validate_schema: Whether to validate response against schema (default: True)
    :param on_log: Optional callback for structured logging events (default: None)
        - Receives LogEvent objects with event type and data
        - Used for custom logging, monitoring, and debugging
        - Sensitive data is automatically redacted
    :return: AsyncGenerator yielding structured data chunks
    :raises: Various exceptions (see Error Handling section)

    Example::

        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Analyze this text: " + text}
        ]

        async for chunk in async_openai_structured_stream(
            messages=messages,
            schema=schema,
            model="gpt-4o",
            temperature=0.7,
            stream_config=StreamConfig(
                max_buffer_size=1024 * 1024,  # 1MB
                cleanup_threshold=512 * 1024   # 512KB
            )
        ):
            print(chunk)

.. function:: supports_structured_output(model_name: str) -> bool

    Check if a model supports OpenAI Structured Outputs.

    This function validates whether a given model name supports OpenAI Structured Outputs,
    handling both aliases and dated versions. For dated versions, it ensures they meet
    minimum version requirements.

    :param model_name: The model name to validate. Can be either:
        - an alias (e.g., "gpt-4o")
        - dated version (e.g., "gpt-4o-2024-08-06")
        - newer version (e.g., "gpt-4o-2024-09-01")
    :return: True if the model supports OpenAI Structured Outputs, False otherwise

    Example::

        # Check alias
        if supports_structured_output("gpt-4o"):
            print("Model supports OpenAI Structured Outputs")

        # Check dated version
        if supports_structured_output("gpt-4o-2024-08-06"):
            print("Version is supported")

        # Check unsupported model
        if not supports_structured_output("gpt-3.5-turbo"):
            print("Model does not support OpenAI Structured Outputs")

    Notes:
        - Aliases (e.g., "gpt-4o") are automatically resolved to the latest compatible version
        - Dated versions must meet minimum version requirements
        - For dated versions, both the base model and date are validated
        - Newer versions are accepted if the base model is supported

.. function:: get_context_window_limit(model_name: str) -> int

    Get the context window limit (maximum total tokens) for a given model.

    :param model_name: The model name (e.g., "gpt-4o", "o1", "o3-mini")
    :return: Maximum context window size in tokens

    Example::

        limit = get_context_window_limit("gpt-4o")  # Returns 128,000
        limit = get_context_window_limit("o1")      # Returns 200,000
        limit = get_context_window_limit("o3-mini") # Returns 200,000

.. function:: get_default_token_limit(model_name: str) -> int

    Get the default output token limit for a given model.

    :param model_name: The model name (e.g., "gpt-4o", "o1", "o3-mini")
    :return: Maximum output tokens allowed

    Example::

        limit = get_default_token_limit("gpt-4o")  # Returns 16,384
        limit = get_default_token_limit("o1")      # Returns 100,000
        limit = get_default_token_limit("o3-mini") # Returns 100,000

    Note: The actual usable output tokens may be slightly less due to invisible reasoning tokens.

Classes
~~~~~~~

.. class:: StreamConfig

    Configuration for streaming behavior with OpenAI Structured Outputs.

    :param max_buffer_size: Maximum buffer size in bytes (default: 1MB)
    :param cleanup_threshold: Buffer cleanup threshold in bytes (default: 512KB)
    :param chunk_size: Stream chunk size in bytes (default: 8KB)
    :param max_cleanup_attempts: Maximum number of cleanup attempts (default: 3)
    :param max_parse_errors: Maximum number of parse errors before failing (default: 5)
    :param log_size_threshold: Size change that triggers logging (default: 100KB)

    Example::

        config = StreamConfig(
            max_buffer_size=1024 * 1024,  # 1MB
            cleanup_threshold=512 * 1024,  # 512KB
            chunk_size=8192               # 8KB
        )

.. class:: StreamBuffer

    Internal buffer management for streaming OpenAI Structured Outputs responses.

    :param config: StreamConfig instance controlling buffer behavior
    :param schema: Optional Pydantic model class for validation

    Attributes:
        total_bytes: Current buffer size in bytes
        parse_errors: Number of parse errors encountered
        cleanup_attempts: Number of cleanup attempts performed
        _cleanup_stats: Dictionary tracking cleanup operations:
            - strategy: Cleanup strategy used (ijson_parsing or pattern_matching)
            - cleaned_bytes: Number of bytes cleaned
            - error_context: Context around errors when they occur
            - validation_error: Details of validation errors
            - json_error: Details of JSON parsing errors

    Methods:
        write(content: str) -> None
            Write content to the buffer. Raises BufferOverflowError if size exceeds limit.

        process_stream_chunk(content: str, on_log: Optional[Callable]) -> Optional[Any]
            Process a stream chunk and return parsed content if complete.

        cleanup() -> None
            Attempt to clean the buffer by finding and preserving valid JSON.

        reset() -> None
            Reset the buffer state while preserving configuration.

        close() -> None
            Close the buffer and clean up resources.

    Example::

        buffer = StreamBuffer(
            config=StreamConfig(),
            schema=MyPydanticModel
        )

        try:
            result = buffer.process_stream_chunk(chunk)
            if result:
                print(f"Valid data: {result}")
        except BufferError as e:
            print(f"Buffer error: {e}")

Errors
------

.. module:: openai_structured.errors

The errors module defines custom exceptions used by the library.

Exceptions
~~~~~~~~~

.. exception:: APIResponseError

    Base exception for API response errors. Contains detailed information about the failed response.

    Attributes:
        - response_id (Optional[str]): The OpenAI response ID for tracking and debugging
        - content (Optional[str]): The raw response content that caused the error

    Example::

        try:
            result = await async_openai_structured_call(...)
        except APIResponseError as e:
            print(f"Error ID: {e.response_id}")
            print(f"Error content: {e.content}")
            print(f"Error message: {str(e)}")

.. exception:: InvalidResponseFormatError

    Raised when the API response doesn't match the expected format.
    Inherits from APIResponseError, providing response_id and content.

    Example::

        try:
            result = await async_openai_structured_call(...)
        except InvalidResponseFormatError as e:
            print(f"Invalid format in response {e.response_id}")
            print(f"Raw content: {e.content}")

.. exception:: EmptyResponseError

    Raised when the API returns an empty response.
    Inherits from APIResponseError, providing response_id and content.

    Example::

        try:
            result = await async_openai_structured_call(...)
        except EmptyResponseError as e:
            print(f"Empty response with ID: {e.response_id}")

.. exception:: StreamBufferError

    Raised when stream buffer limits are exceeded.

    Causes:
        - Buffer size exceeds limit
        - Cleanup fails
        - Memory allocation fails

    Example::

        try:
            async for chunk in async_openai_structured_stream(...):
                process_chunk(chunk)
        except StreamBufferError as e:
            print(f"Buffer overflow: {e}")

.. exception:: StreamInterruptedError

    Raised when the stream is interrupted unexpectedly.

    Causes:
        - Network issues
        - API errors
        - Client disconnection
        - Timeouts

    Example::

        try:
            async for chunk in async_openai_structured_stream(...):
                process_chunk(chunk)
        except StreamInterruptedError as e:
            print(f"Stream interrupted: {e}")

.. exception:: StreamParseError

    Raised when stream content cannot be parsed.

    Causes:
        - Invalid JSON
        - Schema mismatch
        - Encoding issues
        - Partial response

    Example::

        try:
            async for chunk in async_openai_structured_stream(...):
                process_chunk(chunk)
        except StreamParseError as e:
            print(f"Parse error: {e}")

.. exception:: ValidationError

    Raised when schema validation fails.

    Causes:
        - Schema violations
        - Type mismatches
        - Missing fields
        - Format errors

    Example::

        try:
            async for chunk in async_openai_structured_stream(...):
                process_chunk(chunk)
        except ValidationError as e:
            print(f"Validation error: {e}")

.. note::
    Token limit validation is performed using the `validate_token_limits` function, which raises a `ValueError` if limits are exceeded.

.. exception:: TokenLimitError

    Raised when token limits are exceeded for a model.

    Attributes:
        - requested_tokens (Optional[int]): The number of tokens requested
        - model_limit (Optional[int]): The maximum token limit for the model

    Example::

        try:
            result = await async_openai_structured_call(
                client=client,
                model="gpt-4o",
                max_tokens=20_000,  # Exceeds limit
                output_schema=MySchema,
                user_prompt="..."
            )
        except TokenLimitError as e:
            print(f"Token limit exceeded: requested {e.requested_tokens} tokens")
            print(f"Model limit is {e.model_limit} tokens")

    Note:
        Token limits vary by model:
        - gpt-4o: 16,384 output tokens
        - o1: 100,000 output tokens
        - o3-mini: 100,000 output tokens

.. exception:: TokenParameterError

    Raised when both max_output_tokens and max_completion_tokens are used.
    These parameters are mutually exclusive as they control the same functionality.
    Only one should be used in a request.

    Example::

        try:
            client.complete(
                "gpt-4o",
                max_output_tokens=100,
                max_completion_tokens=100
            )
        except TokenParameterError as e:
            print(f"Token error: {e}")
            # Output:
            # "Cannot specify both 'max_output_tokens' and 'max_completion_tokens' parameters.
            # These parameters are mutually exclusive as they control the same functionality.
            # Choose one:
            # - max_output_tokens (recommended)
            # - max_completion_tokens (legacy)"

.. exception:: ModelNotSupportedError

    Raised when a model is not supported by the client.

    Example::

        try:
            registry.get_capabilities("unsupported-model")
        except ModelNotSupportedError as e:
            print(f"Model error: {e}")
            # Output:
            # "Model 'unsupported-model' is not supported.
            # Available models:
            # - Dated models: gpt-4o-2024-08-06, o1-2024-12-17
            # - Aliases: gpt-4o, o1
            # Note: For dated models, use format: base-YYYY-MM-DD (e.g. gpt-4o-2024-08-06)"

.. exception:: VersionTooOldError

    Raised when a model version is older than the minimum supported version.

    Example::

        try:
            registry.get_capabilities("gpt-4o-2024-07-01")
        except VersionTooOldError as e:
            print(f"Version error: {e}")
            # Output:
            # "Model 'gpt-4o-2024-07-01' version 2024-07-01 is too old.
            # Minimum supported version: 2024-08-06
            # Note: Use the alias 'gpt-4o' to always get the latest version"

.. exception:: InvalidDateError

    Raised when a model version has invalid date components.

    Example::

        try:
            registry.get_capabilities("gpt-4o-2024-13-01")
        except InvalidDateError as e:
            print(f"Date error: {e}")
            # Output:
            # "Invalid date format in model version: Month must be between 1 and 12
            # Use format: YYYY-MM-DD (e.g. 2024-08-06)"

.. exception:: OpenAIClientError

    Base exception for client-side errors. Used for various validation and parameter errors.

    Example::

        try:
            capabilities.validate_parameter("reasoning_effort", "invalid")
        except OpenAIClientError as e:
            print(f"Parameter error: {e}")
            # Output:
            # "Invalid value 'invalid' for parameter 'reasoning_effort'.
            # Description: Controls the model's reasoning depth.
            # Allowed values: low, medium, high"

Error Handling Examples
~~~~~~~~~~~~~~~~~~~~

Here are comprehensive examples of handling different error scenarios:

Basic Error Recovery
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openai_structured import (
        APIResponseError, StreamBufferError, StreamInterruptedError,
        StreamParseError, ValidationError, ModelNotSupportedError,
        StreamBuffer
    )
    from openai_structured.errors import TokenLimitError
    from openai import APIError, RateLimitError, APITimeoutError

    async def process_with_basic_recovery():
        stream_config = StreamConfig(
            max_buffer_size=1024 * 1024,  # 1MB
            cleanup_threshold=512 * 1024   # 512KB
        )
        buffer = StreamBuffer(config=stream_config)

        try:
            async for chunk in async_openai_structured_stream(
                model="gpt-4o",
                output_schema=OutputSchema,
                system_prompt="Analyze this",
                user_prompt="Sample text",
                stream_config=stream_config
            ):
                process_chunk(chunk)

        except ModelNotSupportedError as e:
            # Handle model compatibility issues
            print(f"Model not supported: {e}")
            print("Available models: gpt-4o, gpt-4o-mini, o1")

        except ValidationError as e:
            # Handle schema validation failures
            print(f"Schema validation failed: {e}")
            print("Fields with errors:", e.errors())

        except StreamBufferError as e:
            # Handle buffer-related issues
            print(f"Buffer error: {e}")
            if hasattr(e, '_cleanup_stats'):
                print("Cleanup attempts:", e._cleanup_stats['attempts'])
                print("Last buffer size:", e._cleanup_stats['bytes_before'])

        except StreamParseError as e:
            # Handle JSON parsing issues
            print(f"Parse error after {e.attempts} attempts")
            print(f"Last error: {e.last_error}")

        except APIResponseError as e:
            # Handle API response issues with detailed info
            print(f"API Response Error (ID: {e.response_id})")
            print(f"Response content: {e.content}")

Advanced Error Recovery
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from typing import Optional, Dict, Any
    import asyncio
    from tenacity import retry, stop_after_attempt, wait_exponential

    class ErrorHandler:
        def __init__(self, max_retries: int = 3):
            self.max_retries = max_retries
            self.current_attempt = 0
            self.last_error: Optional[Exception] = None
            self.cleanup_stats: Dict[str, Any] = {}

        async def process_with_retry(self):
            while self.current_attempt < self.max_retries:
                try:
                    async for chunk in async_openai_structured_stream(
                        client=client,
                        model="gpt-4o",
                        output_schema=OutputSchema,
                        system_prompt="Analyze this",
                        user_prompt="Sample text",
                        timeout=30.0
                    ):
                        await self.process_chunk(chunk)
                    break  # Success, exit loop

                except (StreamBufferError, ValidationError) as e:
                    # Don't retry these errors
                    self.log_error("Permanent error, not retrying", e)
                    raise

                except StreamInterruptedError as e:
                    # Retry with exponential backoff
                    await self.handle_interrupted_stream(e)

                except APITimeoutError:
                    # Retry with increased timeout
                    await self.handle_timeout()

                except RateLimitError:
                    # Retry with increased wait time
                    await self.handle_rate_limit()

                except APIResponseError as e:
                    # Log detailed response info and retry
                    await self.handle_api_response_error(e)

                except Exception as e:
                    # Unexpected error
                    self.log_error("Unexpected error", e)
                    raise

                self.current_attempt += 1

            if self.last_error:
                raise self.last_error

        async def handle_interrupted_stream(self, error: StreamInterruptedError):
            self.last_error = error
            wait_time = min(2 ** self.current_attempt, 30)  # Max 30 seconds
            self.log_error(f"Stream interrupted, retrying in {wait_time}s", error)
            await asyncio.sleep(wait_time)

        async def handle_timeout(self):
            new_timeout = 30 * (self.current_attempt + 1)  # Increase timeout
            self.log_error(f"Timeout, retrying with {new_timeout}s timeout")
            # Update client timeout for next attempt

        async def handle_rate_limit(self):
            wait_time = 30 * (self.current_attempt + 1)  # Increase wait time
            self.log_error(f"Rate limited, waiting {wait_time}s")
            await asyncio.sleep(wait_time)

        async def handle_api_response_error(self, error: APIResponseError):
            self.last_error = error
            self.log_error(
                f"API error (ID: {error.response_id})",
                f"Content: {error.content}"
            )
            await asyncio.sleep(5)  # Brief wait before retry

        def log_error(self, message: str, error: Optional[Exception] = None):
            print(f"Attempt {self.current_attempt + 1}/{self.max_retries}: {message}")
            if error:
                print(f"Error details: {error}")

    # Usage
    handler = ErrorHandler(max_retries=3)
    await handler.process_with_retry()

These examples demonstrate:

1. Different error handling strategies:
   - Simple error catching and reporting
   - Sophisticated retry logic with exponential backoff
   - Error-specific handling and recovery

2. Proper resource cleanup using ``finally``

3. Detailed error information extraction:
   - Response IDs from APIResponseError
   - Cleanup statistics from StreamBufferError
   - Parse attempt counts from StreamParseError

4. Advanced retry mechanisms:
   - Rate limit handling with increasing delays
   - Timeout handling with increasing timeouts
   - Stream interruption recovery

5. Structured error logging and monitoring

Example Usage
------------

Basic Streaming
~~~~~~~~~~~~~

.. code-block:: python

    from openai_structured import async_openai_structured_stream, StreamConfig
    from openai_structured.errors import StreamBufferError, ValidationError

    async def process_stream():
        try:
            async for chunk in async_openai_structured_stream(
                client=client,
                model="gpt-4o-2024-08-06",
                output_schema=OutputSchema,
                system_prompt="Analyze this text",
                user_prompt="Sample text to analyze",
                stream_config=StreamConfig(
                    max_buffer_size=1024 * 1024
                )
            ):
                print(chunk)
        except ValueError as e:
            if "token limit" in str(e).lower():
                print(f"Token limit exceeded: {e}")
            else:
                raise
        except StreamBufferError as e:
            print(f"Buffer error: {e}")
        except ValidationError as e:
            print(f"Validation error: {e}")

Error Recovery
~~~~~~~~~~~~

.. code-block:: python

    from openai_structured.errors import StreamInterruptedError
    import asyncio

    async def process_with_retry(max_retries=3):
        last_error = None
        for attempt in range(max_retries):
            try:
                async for chunk in async_openai_structured_stream(...):
                    process_chunk(chunk)
                break
            except StreamInterruptedError as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)
            except (StreamBufferError, ValidationError) as e:
                # Don't retry these errors
                raise

        if last_error:
            raise last_error

Resource Management
~~~~~~~~~~~~~~~~

.. code-block:: python

    async def process_with_timeout():
        try:
            async for chunk in async_openai_structured_stream(
                messages=[...],
                schema={...},
                timeout=30.0
            ):
                process_chunk(chunk)
        except asyncio.TimeoutError:
            print("Operation timed out")
        finally:
            cleanup_resources()

Schema Validation
~~~~~~~~~~~~~~~

.. code-block:: python

    from openai_structured.errors import ValidationError

    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[A-Za-z]+$"
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150
            }
        },
        "required": ["name", "age"]
    }

    try:
        async for chunk in async_openai_structured_stream(
            messages=[...],
            schema=schema,
            validate_schema=True
        ):
            process_chunk(chunk)
    except ValidationError as e:
        print(f"Validation failed: {e}")

Schema Validation
~~~~~~~~~~~~~~~

.. code-block:: python

    from openai_structured.errors import ValidationError

    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[A-Za-z]+$"
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150
            }
        },
        "required": ["name", "age"]
    }

    try:
        async for chunk in async_openai_structured_stream(
            messages=[...],
            schema=schema,
            validate_schema=True
        ):
            process_chunk(chunk)
    except ValidationError as e:
        print(f"Validation failed: {e}")

Error Handling
~~~~~~~~~~~~~

The library raises the following exceptions:

* ``StreamBufferError``: Raised when the buffer size exceeds the configured maximum.
* ``StreamInterruptedError``: Raised when the stream is interrupted before completion.
* ``StreamParseError``: Raised when the stream content cannot be parsed as valid JSON.
* ``ValidationError``: Raised when the response does not match the provided schema.
* ``APIError``: Raised when the OpenAI API returns an error.
* ``ValueError``: Raised in several cases:
    * When token limits are exceeded (input too long or output limit exceeded)
    * When an invalid model version is provided
    * When schema validation fails

Example error handling:

.. code-block:: python

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

Logging Events
~~~~~~~~~~~~

The library provides structured logging through the ``on_log`` callback:

.. class:: LogEvent

    Structured logging event.

    :param type: Event type (e.g., "buffer.size", "stream.start", "error")
    :param data: Event data (sensitive information automatically redacted)

    Security:
        The library automatically redacts sensitive information in logs:
        - API keys and tokens
        - Authentication headers
        - Other security-sensitive fields
        This protection applies to all logging events, including errors and API responses.

Common event types:

* ``buffer.size``: Buffer size changes
* ``stream.start``: Stream creation
* ``stream.end``: Stream completion
* ``stream.chunk``: Chunk received
* ``cleanup.stats``: Buffer cleanup statistics
* ``error``: Error details (sensitive data redacted)
* ``parse.attempt``: Parse attempt details
* ``validation``: Schema validation results

Example logging implementation::

    import logging
    logger = logging.getLogger(__name__)

    async def log_callback(event: LogEvent, level: str):
        # All events are automatically redacted for security
        if event.type == "error":
            logger.error("Error: %s", event.data, exc_info=True)  # API keys and auth data redacted
        elif event.type == "buffer.size":
            logger.info("Buffer size: %d bytes", event.data["size"])
        elif event.type == "cleanup.stats":
            logger.debug("Cleanup stats: %s", event.data)
        else:
            logger.debug("Event %s: %s", event.type, event.data)

    async for chunk in async_openai_structured_stream(
        model="gpt-4o-2024-08-06",
        output_schema=OutputSchema,
        system_prompt="Analyze this text",
        user_prompt="Sample text to analyze",
        on_log=log_callback
    ):
        process_chunk(chunk)

Data Processing Features
=====================

The template engine includes powerful data processing capabilities for analyzing and transforming structured data.

Data Transformation
-----------------

.. code-block:: python

    # Sort items by a key
    {{ items|sort_by('timestamp') }}

    # Group items by category
    {% set grouped = items|group_by('category') %}
    {% for category, items in grouped.items() %}
        {{ category }}: {{ items|length }} items
    {% endfor %}

    # Filter items
    {{ items|filter_by('status', 'active') }}

    # Extract values
    {{ items|pluck('name') }}

    # Get unique values
    {{ items|unique }}

    # Count frequencies
    {{ items|frequency }}

Aggregation Functions
-------------------

.. code-block:: python

    # Basic aggregation
    {% set stats = data|aggregate('value') %}
    Count: {{ stats.count }}
    Sum: {{ stats.sum }}
    Average: {{ stats.avg }}
    Min: {{ stats.min }}
    Max: {{ stats.max }}

    # Aggregate nested data
    {% set user_stats = users|aggregate('age') %}
    Average age: {{ user_stats.avg }}

Data Analysis
------------

.. code-block:: python

    # Generate data summary
    {% set summary = summarize(data) %}
    Total records: {{ summary.total_records }}
    {% for field, stats in summary.fields.items() %}
        {{ field }}:
        - Type: {{ stats.type }}
        - Unique values: {{ stats.unique_values }}
        - Null count: {{ stats.null_count }}
    {% endfor %}

    # Create pivot tables
    {% set pivot = pivot_table(data, index='category', values='amount', aggfunc='sum') %}
    {{ pivot|dict_to_table }}

Table Formatting
--------------

.. code-block:: python

    # Basic table
    {{ table(['Name', 'Age'], [['Alice', 25], ['Bob', 30]]) }}

    # Aligned table
    {{ align_table(['Name', 'Age'], [['Alice', 25], ['Bob', 30]], ['left', 'right']) }}

    # Convert dict to table
    {{ stats|dict_to_table }}

    # Convert list to table
    {{ users|list_to_table(headers=['Name', 'Age']) }}

    # Auto-format any data structure
    {{ data|auto_table }}

Examples
--------

Here are some practical examples combining multiple features:

.. code-block:: python

    # Analyze user activity by category
    {% set user_activity = data|group_by('category') %}
    {% for category, items in user_activity.items() %}
        Category: {{ category }}
        {{ items|aggregate('duration')|dict_to_table }}
    {% endfor %}

    # Generate summary report
    {% set stats = data|aggregate('value') %}
    {% set distribution = data|pluck('category')|frequency %}

    Summary Statistics:
    {{ stats|dict_to_table }}

    Category Distribution:
    {{ distribution|dict_to_table }}

    # Create detailed pivot analysis
    {% set pivot_data = pivot_table(data,
                                  index='category',
                                  values='amount',
                                  aggfunc='mean') %}
    Average Amount by Category:
    {{ pivot_data|dict_to_table }}

Testing
-------

The library provides utilities for testing applications that use openai-structured.

Stream Testing
~~~~~~~~~~~~

When testing streaming functionality, you should test both the iteration mechanism and content processing:

.. code-block:: python

    # Test streaming functionality
    results = list(openai_structured_stream(
        client=client,  # Use a real client with test credentials
        model="gpt-4o",
        output_schema=SimpleMessage,
        user_prompt="test"
    ))

    # Verify results
    assert len(results) > 0
    for result in results:
        assert isinstance(result, SimpleMessage)

Error Handling
~~~~~~~~~~~~

Test error scenarios by configuring your client with invalid credentials or using network conditions that would trigger errors:

.. code-block:: python

    from openai_structured.errors import StreamInterruptedError

    # Test with invalid API key
    client = OpenAI(api_key="invalid-key")

    with pytest.raises(StreamInterruptedError):
        list(openai_structured_stream(
            client=client,
            model="gpt-4o",
            output_schema=SimpleMessage,
            user_prompt="test"
        ))

Async Testing
~~~~~~~~~~~

For async code, use pytest-asyncio and test both successful and error cases:

.. code-block:: python

    @pytest.mark.asyncio
    async def test_async_stream():
        client = AsyncOpenAI()  # Configure with test credentials

        results = []
        async for result in async_openai_structured_stream(
            client=client,
            model="gpt-4o",
            output_schema=SimpleMessage,
            user_prompt="test"
        ):
            results.append(result)

        assert len(results) > 0
        for result in results:
            assert isinstance(result, SimpleMessage)

.. note::
    o1 and o3 models only support a limited set of parameters:

    - max_completion_tokens
    - reasoning_effort

    Attempting to use other parameters (temperature, top_p, etc.) will raise an OpenAIClientError.

Model Support
------------

The library supports the following OpenAI models with structured output:

Production Models
~~~~~~~~~~~~~~~

- **gpt-4o-2024-08-06**
    - Full JSON schema support
    - 128K context window
    - 16K output tokens
    - Supports streaming

- **gpt-4o-mini-2024-07-18**
    - 128K context window
    - 16K output tokens
    - Supports streaming

- **o1-2024-12-17**
    - 200K context window
    - 100K output tokens
    - Limited parameter support
    - Does not support streaming

- **o3-mini-2025-01-31**
    - 200K context window
    - 100K output tokens
    - Limited parameter support
    - Supports streaming

Limited Parameter Support
~~~~~~~~~~~~~~~~~~~~

o1 and o3 models only support the following parameters:

- max_completion_tokens
- reasoning_effort

Attempting to use other parameters (temperature, top_p, etc.) will raise an OpenAIClientError.

Streaming Support
~~~~~~~~~~~~~~

Different models have varying streaming support:

.. code-block:: python

    # o1-2024-12-17 does not support streaming
    try:
        async for response in async_openai_structured_stream(
            model="o1-2024-12-17",  # Will raise 400 error
            stream=True
        ):
            process_response(response)
    except OpenAIClientError as e:
        # Error: "Unsupported value: 'stream' does not support true with this model"
        handle_error(e)

    # o3 main model does not support streaming
    try:
        async for response in async_openai_structured_stream(
            model="o3",  # Will raise 400 error
            stream=True
        ):
            process_response(response)
    except OpenAIClientError as e:
        # Error: "The main o3 model does not support streaming"
        handle_error(e)

    # o3-mini and o3-mini-high support streaming
    async for response in async_openai_structured_stream(
        model="o3-mini",  # Works correctly
        stream=True
    ):
        process_response(response)

Exceptions
---------

.. py:exception:: OpenAIClientError

   Base exception for client-side errors. Raised in several cases:

   - When attempting to use unsupported parameters with o1/o3 models
   - When model version is not supported
   - When validation fails
