"""Client for making structured OpenAI API calls.

This module provides functions for making structured calls to OpenAI's API,
ensuring responses conform to specified Pydantic models.

Constants:
    DEFAULT_TEMPERATURE (float): Default temperature for sampling (0.0)
    DEFAULT_TIMEOUT (float): Default timeout in seconds for API calls (60.0)

For production use, it's recommended to implement retry and rate limiting logic:

Example with retries:
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    def my_resilient_call():
        result = openai_structured_call(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=MySchema,
            user_prompt="...",
            system_prompt="..."
        )
        return result

Example with rate limiting:
    from asyncio_throttle import Throttler

    async with Throttler(rate_limit=100, period=60):
        result = await async_openai_structured_call(...)

See documentation for more examples and best practices.
"""

# Standard library imports
import asyncio
import json
import logging
import re
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    List,
    NoReturn,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
)

# Third-party imports
import aiohttp
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types import ResponseFormatJSONSchema
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)

# Import from openai-model-registry for model validation
from openai_model_registry import ModelRegistry, ModelVersion
from openai_model_registry.constraints import NumericConstraint
from openai_model_registry.errors import (
    ModelNotSupportedError,
    ModelRegistryError,
)
from pydantic import BaseModel, ValidationError
from typing_extensions import ParamSpec

# Local imports
from .buffer import StreamBuffer, StreamConfig
from .errors import (
    BufferOverflowError,
    EmptyResponseError,
    InvalidResponseFormatError,
    JSONParseError,
    OpenAIClientError,
    StreamBufferError,
    StreamInterruptedError,
    StreamParseError,
    TokenLimitError,
)
from .logging import LogCallback, LogLevel

# Define public API
__all__ = [
    # Main functions
    "openai_structured_call",
    "async_openai_structured_call",
    "openai_structured_stream",
    "async_openai_structured_stream",
    # Configuration
    "StreamConfig",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    # Type hints
    "LogCallback",
    # Exceptions
    "StreamInterruptedError",
    "StreamParseError",
    "StreamBufferError",
    "BufferOverflowError",
    "ModelNotSupportedError",
    "InvalidResponseFormatError",
    "JSONParseError",
    "ValidationError",
    "TokenLimitError",
]

# Create logger
logger = logging.getLogger(__name__)

# Type variables and aliases
ClientT = TypeVar("ClientT", bound=Union[OpenAI, AsyncOpenAI])
P = ParamSpec("P")
R = TypeVar("R")

# Constants
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 60.0

# Format: "{base_model}-{YYYY}-{MM}-{DD}"
# Examples:
#   - "gpt-4o-2024-08-06"      -> ("gpt-4o", "2024-08-06")
#   - "gpt-4-turbo-2024-08-06" -> ("gpt-4-turbo", "2024-08-06")
#
# Pattern components:
# ^           Start of string
# ([\w-]+?)   Group 1: Base model name
#   [\w-]     Allow word chars (a-z, A-Z, 0-9, _) and hyphens
#   +?        One or more chars (non-greedy)
# -           Literal hyphen separator
# (...)       Group 2: Date in YYYY-MM-DD format
#   \d{4}     Exactly 4 digits for year
#   -         Literal hyphen
#   \d{2}     Exactly 2 digits for month
#   -         Literal hyphen
#   \d{2}     Exactly 2 digits for day
# $           End of string
MODEL_VERSION_PATTERN = re.compile(r"^([\w-]+?)-(\d{4}-\d{2}-\d{2})$")

# Model token limits (based on OpenAI specifications as of 2024):
# Models can be specified using either aliases or dated versions:
# Aliases:
# - gpt-4o: 128K context window, 16K max output tokens (minimum version: 2024-08-06)
# - gpt-4o-mini: 128K context window, 16K max output tokens (minimum version: 2024-07-18)
# - o1: 200K context window, 100K max output tokens (minimum version: 2024-12-17)
# - o3-mini: 200K context window, 100K max output tokens (minimum version: 2025-01-31)
#
# When using aliases (e.g., "gpt-4o"), OpenAI will automatically use the latest
# compatible version. We validate that the model meets our minimum version
# requirements, but the actual version resolution is handled by OpenAI.
#
# Dated versions (recommended for production):
# - gpt-4o-2024-08-06: 128K context window, 16K max output tokens
# - gpt-4o-mini-2024-07-18: 128K context window, 16K max output tokens
# - o1-2024-12-17: 200K context window, 100K max output tokens
# - o3-mini-2025-01-31: 200K context window, 100K max output tokens
#
# Note: These limits may change as OpenAI updates their models
# Note: Actual output may be lower than these theoretical limits due to invisible
# reasoning tokens that count against the same budget

# Model token limits defined as constants
MODEL_CONTEXT_WINDOWS = {
    "o1": 200_000,  # o1 models
    "gpt-4o": 128_000,  # gpt-4o models
    "o3-mini": 200_000,  # o3-mini models
}

MODEL_OUTPUT_LIMITS = {
    "o1": 100_000,  # o1 models
    "gpt-4o": 16_384,  # gpt-4o models
    "o3-mini": 100_000,  # o3-mini models
}

# Model version mapping - maps aliases to minimum supported versions
OPENAI_API_SUPPORTED_MODELS = {
    "gpt-4o": ModelVersion(2024, 8, 6),  # Minimum supported version
    "gpt-4o-mini": ModelVersion(2024, 7, 18),  # Minimum supported version
    "o1": ModelVersion(2024, 12, 17),  # Minimum supported version
    "o3-mini": ModelVersion(2025, 1, 31),  # Minimum supported version
}


def validate_parameters(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to validate common parameters."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Get model name from kwargs or first positional argument
        model = kwargs.get("model")
        if model is None and len(args) > 0:
            model = str(args[0])  # Convert to str to ensure type safety
        if not model:
            raise OpenAIClientError("Model name is required")

        # Get model capabilities
        registry = ModelRegistry.get_instance()
        capabilities = registry.get_capabilities(str(model))

        # Validate all parameters using the model's capabilities
        for param_name, value in kwargs.items():
            if param_name in {
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_completion_tokens",
                "reasoning_effort",
            }:
                # Get the parameter constraint to determine type
                param_ref = None
                for ref in capabilities.supported_parameters:
                    if param_name in ref.ref:
                        param_ref = ref
                        break

                if param_ref is None:
                    continue  # Skip validation for unsupported parameters

                constraint = registry.get_parameter_constraint(param_ref.ref)

                # Convert string values based on constraint type
                if isinstance(value, str):
                    if isinstance(constraint, NumericConstraint):
                        try:
                            value = float(value)
                            kwargs[param_name] = value
                        except ValueError:
                            raise OpenAIClientError(
                                f"Invalid {param_name} value: {value}"
                            )

                try:
                    capabilities.validate_parameter(param_name, value)
                except ModelRegistryError as e:
                    raise OpenAIClientError(
                        f"Model registry error: {e}"
                    ) from e

        return func(*args, **kwargs)

    return wrapper


def supports_structured_output(model_name: str) -> bool:
    """Check if the model supports structured output.

    Args:
        model_name: Name of the model to check

    Returns:
        bool: True if the model supports structured output
    """
    try:
        registry = ModelRegistry.get_instance()
        capabilities = registry.get_capabilities(model_name)
        return capabilities.supports_structured
    except Exception:
        return False


def _validate_client_type(client: Any, expected_type: str = "sync") -> None:
    """Validates that the client is of the expected type (sync/async).

    Args:
        client: The client to validate
        expected_type: Either "sync" or "async"

    Raises:
        TypeError: If client type doesn't match expected_type
    """
    # For official clients, trust the type
    if isinstance(client, (OpenAI, AsyncOpenAI)):
        is_async = isinstance(client, AsyncOpenAI)
    else:
        # For mocks and custom implementations, check required attributes
        if not hasattr(client, "chat"):
            raise TypeError("Client missing required attribute: 'chat'")
        if not hasattr(client.chat, "completions"):
            raise TypeError("Client missing required attribute: 'completions'")
        if not hasattr(client.chat.completions, "create"):
            raise TypeError("Client missing required attribute: 'create'")

        # Check for async methods:
        # 1. Presence of acreate method
        # 2. Coroutine functions for custom implementations
        create_method = client.chat.completions.create
        acreate_method = getattr(client.chat.completions, "acreate", None)
        is_async = bool(acreate_method) or asyncio.iscoroutinefunction(
            create_method
        )

    if expected_type == "async" and not is_async:
        raise TypeError(
            "Async client required but got sync client. "
            "Use AsyncOpenAI or a compatible mock client"
        )
    elif expected_type == "sync" and is_async:
        raise TypeError(
            "Sync client required but got async client. "
            "Use OpenAI or a compatible mock client"
        )


def _validate_request(
    model: str,
    client: Union[OpenAI, AsyncOpenAI],
    expected_type: Type[ClientT],
) -> None:
    """Validate the request parameters.

    This function performs two key validations:
    1. Client interface and sync/async type validation using duck typing
       (see _validate_client_type for details on this design choice)
    2. Model support validation to ensure the model can handle structured output

    Args:
        model: The model name to validate
        client: The client to validate (can be official client or compatible mock)
        expected_type: Used to determine if we need sync or async validation

    Raises:
        TypeError: If client validation fails
        ModelNotSupportedError: If model doesn't support structured output
    """
    _validate_client_type(
        client, "sync" if expected_type == OpenAI else "async"
    )
    if not supports_structured_output(model):
        raise ModelNotSupportedError(
            f"Model {model} does not support structured output"
        )


def _create_chat_messages(
    system_prompt: str,
    user_prompt: str,
) -> List[ChatCompletionMessageParam]:
    """Create chat messages for the OpenAI API."""
    return [
        cast(
            ChatCompletionMessageParam,
            {"role": "system", "content": system_prompt},
        ),
        cast(
            ChatCompletionMessageParam,
            {"role": "user", "content": user_prompt},
        ),
    ]


def _create_request_params(
    model: str,
    messages: List[ChatCompletionMessageParam],
    schema: dict[str, Any],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    reasoning_effort: Optional[str],
    top_p: Optional[float],
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
    stream: bool,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Create request parameters for the API call."""
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": cast(
            ResponseFormatJSONSchema,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                },
            },
        ),
        "stream": stream,
    }

    # Only add optional parameters if they are specified and supported by the model
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if frequency_penalty is not None:
        params["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        params["presence_penalty"] = presence_penalty
    if max_output_tokens is not None:
        params["max_output_tokens"] = max_output_tokens
    if max_completion_tokens is not None:
        params["max_completion_tokens"] = max_completion_tokens
    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort
    if timeout is not None:
        params["timeout"] = timeout

    return params


def _get_schema(model_class: Type[BaseModel]) -> dict[str, Any]:
    """Get JSON schema for a model class."""
    schema = model_class.model_json_schema()
    assert isinstance(schema, dict)
    return schema


def _parse_json_response(
    content: Optional[str],
    output_schema: Type[BaseModel],
    response_id: Optional[str] = None,
) -> BaseModel:
    """Parse and validate JSON response."""
    if not content:
        raise EmptyResponseError("OpenAI API returned empty response")

    try:
        return output_schema.model_validate_json(content)
    except ValidationError as e:
        raise InvalidResponseFormatError(
            f"Response validation failed: {e}\n"
            f"Received content (first 200 chars): {content[:200]}",
            response_id=response_id,
        ) from e
    except json.JSONDecodeError as e:
        error_pos = e.pos
        context = content[
            max(0, error_pos - 50) : min(len(content), error_pos + 50)
        ]
        if error_pos > 50:
            context = "..." + context
        if error_pos + 50 < len(content):
            context = context + "..."

        raise JSONParseError(
            f"Invalid JSON at position {error_pos}: {e.msg}\n"
            f"Context: {context}\n"
            f"Full response (first 200 chars): {content[:200]}",
            response_id=response_id,
        ) from e


def _redact_string(text: str) -> str:
    """Redact sensitive patterns from a string."""
    if not text:
        return text
    # Redact OpenAI API keys (sk-...)
    return re.sub(r"sk-[a-zA-Z0-9]{32,}", "[REDACTED-API-KEY]", text)


def _redact_sensitive_data(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive data from logging output."""
    if not data:
        return data

    redacted = data.copy()
    sensitive_keys = {
        "api_key",
        "authorization",
        "key",
        "secret",
        "password",
        "token",
        "access_token",
        "refresh_token",
    }

    for key in redacted:
        value = redacted[key]
        # Redact any key containing sensitive terms
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            redacted[key] = "[REDACTED]"
        # Redact Authorization headers
        elif key.lower() == "headers" and isinstance(value, dict):
            headers = value.copy()
            for header in headers:
                if header.lower() in {"authorization", "x-api-key"}:
                    headers[header] = "[REDACTED]"
            redacted[key] = headers
        # Redact sensitive patterns in strings
        elif isinstance(value, str):
            redacted[key] = _redact_string(value)
        # Handle error messages and nested structures
        elif key in {"error", "error_message"} and isinstance(value, str):
            redacted[key] = _redact_string(value)

    return redacted


def _log_event(
    on_log: Optional[LogCallback],
    level: LogLevel,
    message: str,
    data: Optional[dict[str, Any]] = None,
) -> None:
    """Utility function for consistent logging with sensitive data redaction."""
    if on_log:
        safe_data = _redact_sensitive_data(data or {})
        on_log(level.value, message, safe_data)


def _handle_api_error(
    error: Exception, on_log: Optional[LogCallback] = None
) -> NoReturn:
    """Handle OpenAI API errors with enhanced logging."""
    if on_log:
        # First log basic error info
        _log_event(
            on_log,
            LogLevel.ERROR,
            "OpenAI API error",
            {
                "error": _redact_string(str(error)),
                "error_type": type(error).__name__,
            },
        )

        # Then log detailed error data if it's an OpenAI error
        if isinstance(error, OpenAIError):
            error_data: dict[str, Any] = {
                "error_type": error.__class__.__name__,
                "error_message": _redact_string(str(error)),
                "status_code": getattr(error, "status_code", None),
                "request_id": getattr(error, "request_id", None),
                "should_retry": isinstance(
                    error,
                    (APIConnectionError, APITimeoutError, RateLimitError),
                ),
                "retry_after": getattr(error, "retry_after", None),
            }
            _log_event(
                on_log, LogLevel.ERROR, "OpenAI API error details", error_data
            )

    # Re-raise OpenAI errors and our own errors as is, wrap others in OpenAIClientError
    if isinstance(
        error,
        (
            OpenAIError,
            InvalidResponseFormatError,
            JSONParseError,
            EmptyResponseError,
        ),
    ):
        raise error
    raise OpenAIClientError(
        f"Unexpected error during API call: {error}"
    ) from error


def _log_request_start(
    on_log: Optional[LogCallback],
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    streaming: bool = False,
    output_schema: Optional[Type[BaseModel]] = None,
) -> None:
    """Log the start of an API request."""
    if output_schema is None:
        return
    _log_event(
        on_log,
        LogLevel.DEBUG,
        "Starting OpenAI request",
        {
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "streaming": streaming,
            "output_schema": output_schema.__name__,
        },
    )


def _prepare_request(
    model: str,
    client: Union[OpenAI, AsyncOpenAI],
    expected_type: Type[ClientT],
    output_schema: Type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    reasoning_effort: Optional[str],
    top_p: Optional[float],
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
    stream: bool,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Prepare common request parameters and validate inputs."""
    _validate_request(model, client, expected_type)
    _validate_token_limits(model, max_output_tokens)
    messages = _create_chat_messages(system_prompt, user_prompt)
    schema = _get_schema(output_schema)

    # Get model capabilities to check supported parameters
    registry = ModelRegistry.get_instance()
    capabilities = registry.get_capabilities(str(model))

    # Validate parameters against model capabilities
    if temperature is not None:
        capabilities.validate_parameter("temperature", temperature)
    if top_p is not None:
        capabilities.validate_parameter("top_p", top_p)
    if frequency_penalty is not None:
        capabilities.validate_parameter("frequency_penalty", frequency_penalty)
    if presence_penalty is not None:
        capabilities.validate_parameter("presence_penalty", presence_penalty)
    if max_output_tokens is not None:
        capabilities.validate_parameter("max_output_tokens", max_output_tokens)
    if max_completion_tokens is not None:
        capabilities.validate_parameter(
            "max_completion_tokens", max_completion_tokens
        )
    if reasoning_effort is not None:
        capabilities.validate_parameter("reasoning_effort", reasoning_effort)

    return _create_request_params(
        model=model,
        messages=messages,
        schema=schema,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stream=stream,
        timeout=timeout,
    )


def _handle_stream_error(
    e: Exception, on_log: Optional[LogCallback]
) -> NoReturn:
    """Handle stream-specific errors consistently."""
    # Handle known error types first
    if isinstance(e, StreamBufferError):
        raise

    if isinstance(e, (StreamParseError, StreamInterruptedError)):
        _log_event(
            on_log,
            LogLevel.ERROR,
            "Stream parsing failed",
            {"error": str(e)},
        )
        raise

    if isinstance(
        e,
        (
            aiohttp.ClientError,
            aiohttp.ServerTimeoutError,
            asyncio.TimeoutError,
            ConnectionError,
        ),
    ):
        _log_event(
            on_log,
            LogLevel.ERROR,
            "Stream interrupted",
            {"error": str(e), "error_type": type(e).__name__},
        )
        raise StreamInterruptedError(
            f"Stream was interrupted: {e}. "
            "Check your network connection and API status."
        ) from e

    # Handle OpenAI errors directly
    if isinstance(e, OpenAIError):
        raise

    # Convert unknown exceptions to StreamInterruptedError
    raise StreamInterruptedError(f"Stream was interrupted: {e}") from e


def _process_stream_chunk(
    chunk: Any,
    buffer: StreamBuffer,
    output_schema: Type[BaseModel],
    on_log: Optional[LogCallback] = None,
) -> Optional[BaseModel]:
    """Process a single stream chunk with error handling."""
    try:
        _log_event(
            on_log,
            LogLevel.DEBUG,
            "Processing chunk",
            {
                "id": getattr(chunk, "id", None),
                "model": getattr(chunk, "model", None),
                "system_fingerprint": getattr(
                    chunk, "system_fingerprint", None
                ),
            },
        )

        if not hasattr(chunk, "choices"):
            _log_event(
                on_log,
                LogLevel.DEBUG,
                "Chunk missing 'choices' attribute",
                {"error": "No choices attribute"},
            )
            return None

        if not chunk.choices:
            _log_event(
                on_log,
                LogLevel.DEBUG,
                "Chunk has no choices, skipping",
                {"error": "Empty choices"},
            )
            return None

        choice = chunk.choices[0]
        _log_event(
            on_log,
            LogLevel.DEBUG,
            "First choice details",
            {
                "index": getattr(choice, "index", None),
                "finish_reason": getattr(choice, "finish_reason", None),
            },
        )

        try:
            delta = choice.delta
            _log_event(
                on_log,
                LogLevel.DEBUG,
                "Delta details",
                {
                    "role": getattr(delta, "role", None),
                    "content": getattr(delta, "content", None),
                    "function_call": getattr(delta, "function_call", None),
                    "tool_calls": getattr(delta, "tool_calls", None),
                },
            )
        except (KeyError, AttributeError) as e:
            _log_event(
                on_log,
                LogLevel.ERROR,
                "Error accessing delta",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "chunk_id": getattr(chunk, "id", None),
                    "choice_index": (
                        getattr(choice, "index", None)
                        if chunk.choices
                        else None
                    ),
                    "buffer_size": buffer.size,
                    "buffer_content": buffer.getvalue(),
                },
            )
            return None

        if not hasattr(delta, "content") or delta.content is None:
            _log_event(
                on_log,
                LogLevel.DEBUG,
                "Skipping chunk without content",
                {"error": "No content in delta"},
            )
            return None

        _log_event(
            on_log,
            LogLevel.DEBUG,
            "Processing content",
            {"content": delta.content},
        )
        try:
            # Write the content to the buffer
            buffer.write(delta.content)
            _log_event(
                on_log,
                LogLevel.DEBUG,
                "Buffer state after write",
                {
                    "buffer_size": buffer.size,
                    "buffer_content": buffer.getvalue(),
                    "starts_with_brace": buffer.getvalue()
                    .strip()
                    .startswith("{"),
                    "ends_with_brace": buffer.getvalue().strip().endswith("}"),
                },
            )

            # Only try to parse if we have a complete JSON object
            current_content = buffer.getvalue()
            if current_content.strip().startswith(
                "{"
            ) and current_content.strip().endswith("}"):
                try:
                    _log_event(
                        on_log,
                        LogLevel.DEBUG,
                        "Attempting to parse complete JSON object",
                        {"content": current_content},
                    )
                    model_instance = output_schema.model_validate_json(
                        current_content
                    )
                    _log_event(
                        on_log,
                        LogLevel.DEBUG,
                        "Successfully parsed model",
                        {"model": str(model_instance)},
                    )
                    buffer.reset()
                    return model_instance
                except ValidationError as e:
                    _log_event(
                        on_log,
                        LogLevel.ERROR,
                        "Validation error during chunk processing",
                        {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "content": current_content,
                            "buffer_size": buffer.size,
                            "schema": output_schema.__name__,
                            "validation_errors": e.errors(),
                        },
                    )
                    # Attempt cleanup if buffer is getting large
                    if buffer.size > buffer.config.cleanup_threshold:
                        _log_event(
                            on_log,
                            LogLevel.DEBUG,
                            "Buffer exceeded cleanup threshold after validation error, cleaning up",
                            {"size_bytes": buffer.size},
                        )
                        buffer.cleanup()
                except json.JSONDecodeError as e:
                    _log_event(
                        on_log,
                        LogLevel.ERROR,
                        "JSON decode error during chunk processing",
                        {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "content": current_content,
                            "buffer_size": buffer.size,
                            "error_position": e.pos,
                            "error_line": e.lineno,
                            "error_column": e.colno,
                        },
                    )
                    # Attempt cleanup if buffer is getting large
                    if buffer.size > buffer.config.cleanup_threshold:
                        _log_event(
                            on_log,
                            LogLevel.DEBUG,
                            "Buffer exceeded cleanup threshold after decode error, cleaning up",
                            {"size_bytes": buffer.size},
                        )
                        buffer.cleanup()

            # Only attempt cleanup if buffer is getting large
            if buffer.size > buffer.config.cleanup_threshold:
                _log_event(
                    on_log,
                    LogLevel.DEBUG,
                    "Buffer exceeded cleanup threshold, cleaning up",
                    {"size_bytes": buffer.size},
                )
                buffer.cleanup()

        except (BufferOverflowError, StreamParseError) as e:
            _log_event(
                on_log,
                LogLevel.ERROR,
                "Critical stream buffer error",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "buffer_size": buffer.size,
                    "buffer_content": buffer.getvalue(),
                    "buffer_config": {
                        "max_size": buffer.config.max_buffer_size,
                        "cleanup_threshold": buffer.config.cleanup_threshold,
                        "chunk_size": buffer.config.chunk_size,
                    },
                },
            )
            raise
    except (KeyError, AttributeError) as e:
        _log_event(
            on_log,
            LogLevel.ERROR,
            "Error processing chunk",
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "chunk": str(chunk),
                "chunk_type": str(type(chunk)),
                "chunk_attributes": (
                    dir(chunk) if hasattr(chunk, "__dict__") else None
                ),
                "buffer_size": buffer.size,
                "buffer_content": buffer.getvalue(),
                "processing_stage": "chunk_processing",
            },
        )
        return None

    return None


@validate_parameters
def openai_structured_call(
    client: OpenAI,
    model: str,
    output_schema: Type[BaseModel],
    user_prompt: str,
    system_prompt: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    on_log: Optional[LogCallback] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
) -> BaseModel:
    """Make a structured call to the OpenAI API.

    Args:
        client: OpenAI client instance
        model: Model name (e.g., "gpt-4o-2024-08-06")
        output_schema: Pydantic model class for response validation
        user_prompt: User's request to process
        system_prompt: System instructions for the model
        temperature: Sampling temperature (0-2)
        max_output_tokens: Maximum tokens in response
        max_completion_tokens: Maximum tokens to generate
        reasoning_effort: Optional reasoning effort
        top_p: Top-p sampling parameter (0-1)
        frequency_penalty: Frequency penalty (-2 to 2)
        presence_penalty: Presence penalty (-2 to 2)
        on_log: Optional logging callback
        timeout: Request timeout in seconds (default: 60s)

    Returns:
        Validated instance of output_schema

    Raises:
        TimeoutError: If the request exceeds the timeout
        OpenAIClientError: For client-side errors
        APIConnectionError: For network-related errors
        InvalidResponseFormatError: If response fails validation
    """
    try:
        _log_request_start(
            on_log,
            model,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            max_output_tokens,
            max_completion_tokens,
            reasoning_effort,
            False,
            output_schema,
        )

        params = _prepare_request(
            model,
            client,
            OpenAI,
            output_schema,
            system_prompt,
            user_prompt,
            temperature,
            max_output_tokens,
            max_completion_tokens,
            reasoning_effort,
            top_p,
            frequency_penalty,
            presence_penalty,
            False,
            timeout,
        )

        response = client.chat.completions.create(**params)

        if not response.choices or not response.choices[0].message.content:
            raise EmptyResponseError("OpenAI API returned an empty response.")

        _log_event(
            on_log,
            LogLevel.DEBUG,
            "Request completed",
            {"response_id": response.id},
        )

        return _parse_json_response(
            response.choices[0].message.content,
            output_schema,
            response_id=response.id,
        )

    except Exception as e:
        _handle_api_error(e, on_log)


@validate_parameters
async def async_openai_structured_call(
    client: AsyncOpenAI,
    model: str,
    output_schema: Type[BaseModel],
    user_prompt: str,
    system_prompt: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    on_log: Optional[LogCallback] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
) -> BaseModel:
    """Make an async structured call to the OpenAI API.

    This is the async version of openai_structured_call. It requires an AsyncOpenAI client
    and should be used in async contexts.

    Args:
        client: AsyncOpenAI client instance
        model: Model name (e.g., "gpt-4o-2024-08-06")
        output_schema: Pydantic model class for response validation
        user_prompt: User's request to process
        system_prompt: System instructions for the model
        temperature: Sampling temperature (0-2)
        max_output_tokens: Maximum tokens in response
        max_completion_tokens: Maximum tokens to generate
        reasoning_effort: Optional reasoning effort
        top_p: Top-p sampling parameter (0-1)
        frequency_penalty: Frequency penalty (-2 to 2)
        presence_penalty: Presence penalty (-2 to 2)
        on_log: Optional logging callback
        timeout: Request timeout in seconds (default: 60s)

    Returns:
        Validated instance of output_schema

    Raises:
        asyncio.TimeoutError: If the request exceeds the timeout
        OpenAIClientError: For client-side errors
        APIConnectionError: For network-related errors
        InvalidResponseFormatError: If response fails validation
    """
    try:
        _log_request_start(
            on_log,
            model,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            max_output_tokens,
            max_completion_tokens,
            reasoning_effort,
            False,
            output_schema,
        )

        params = _prepare_request(
            model,
            client,
            AsyncOpenAI,
            output_schema,
            system_prompt,
            user_prompt,
            temperature,
            max_output_tokens,
            max_completion_tokens,
            reasoning_effort,
            top_p,
            frequency_penalty,
            presence_penalty,
            False,
            timeout,
        )

        response = await client.chat.completions.create(**params)

        if not response.choices or not response.choices[0].message.content:
            raise EmptyResponseError("OpenAI API returned an empty response.")

        _log_event(
            on_log,
            LogLevel.DEBUG,
            "Request completed",
            {"response_id": response.id},
        )

        return _parse_json_response(
            response.choices[0].message.content,
            output_schema,
            response_id=response.id,
        )

    except Exception as e:
        _handle_api_error(e, on_log)


@validate_parameters
async def async_openai_structured_stream(
    client: AsyncOpenAI,
    model: str,
    output_schema: Type[BaseModel],
    user_prompt: str,
    system_prompt: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    on_log: Optional[LogCallback] = None,
    stream_config: Optional[StreamConfig] = None,
    timeout: Optional[float] = None,
) -> AsyncGenerator[BaseModel, None]:
    """Stream structured output from OpenAI's API asynchronously.

    This is the asynchronous version of openai_structured_stream. It provides
    the same functionality but can be used in async contexts for better
    performance and resource utilization.

    Args:
        client: AsyncOpenAI client instance
        model: Model name (e.g., "gpt-4o-2024-08-06")
        output_schema: Pydantic model class defining the expected output structure
        user_prompt: User message containing the actual request
        system_prompt: System message to guide the model's behavior
        temperature: Controls randomness (0.0-2.0, default: 0.7)
        max_output_tokens: Maximum tokens to generate (optional)
        max_completion_tokens: Maximum tokens to generate (optional)
        reasoning_effort: Optional reasoning effort
        top_p: Nucleus sampling parameter (optional)
        frequency_penalty: Frequency penalty parameter (optional)
        presence_penalty: Presence penalty parameter (optional)
        on_log: Optional callback for logging events
        stream_config: Optional configuration for stream buffer management
        timeout: Optional timeout in seconds for the API request

    Returns:
        AsyncGenerator yielding structured objects matching output_schema

    Raises:
        ModelNotSupportedError: If model doesn't support structured output
        StreamInterruptedError: If stream is interrupted (e.g., network issues)
        StreamParseError: If response can't be parsed into schema
        StreamBufferError: If buffer management fails
        BufferOverflowError: If buffer size exceeds limit
        APIError: For OpenAI API errors
        APITimeoutError: If request exceeds timeout
        ValidationError: If response doesn't match schema

    Examples:
        >>> from pydantic import BaseModel
        >>> from openai import AsyncOpenAI
        >>> import asyncio
        >>>
        >>> class StoryChapter(BaseModel):
        ...     title: str
        ...     content: str
        ...
        >>> async def main():
        ...     client = AsyncOpenAI()
        ...     prompt = "Write a story about a magical forest"
        ...
        ...     # Basic streaming
        ...     async for chapter in async_openai_structured_stream(
        ...         client=client,
        ...         model="gpt-4o-2024-08-06",
        ...         output_schema=StoryChapter,
        ...         system_prompt="You are a creative writer",
        ...         user_prompt=prompt
        ...     ):
        ...         print(f"Chapter: {chapter.title}")
        ...         print(chapter.content)
        ...
        ...     # With custom buffer config and timeout
        ...     config = StreamConfig(
        ...         max_buffer_size=2 * 1024 * 1024,  # 2MB
        ...         cleanup_threshold=1024 * 1024,     # 1MB
        ...         chunk_size=4096                    # 4KB
        ...     )
        ...     async for chapter in async_openai_structured_stream(
        ...         client=client,
        ...         model="gpt-4o-2024-08-06",
        ...         output_schema=StoryChapter,
        ...         system_prompt="You are a creative writer",
        ...         user_prompt=prompt,
        ...         stream_config=config,
        ...         timeout=30.0
        ...     ):
        ...         print(f"Chapter: {chapter.title}")
        ...         print(chapter.content)
        ...
        >>> asyncio.run(main())
    """
    buffer = None
    stream = None

    try:
        _log_request_start(
            on_log,
            model,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            max_output_tokens,
            max_completion_tokens,
            reasoning_effort,
            True,
            output_schema,
        )

        params = _prepare_request(
            model,
            client,
            AsyncOpenAI,
            output_schema,
            system_prompt,
            user_prompt,
            temperature,
            max_output_tokens,
            max_completion_tokens,
            reasoning_effort,
            top_p,
            frequency_penalty,
            presence_penalty,
            True,
            timeout,
        )

        buffer = StreamBuffer(
            config=stream_config or StreamConfig(), schema=output_schema
        )

        _log_event(
            on_log,
            LogLevel.DEBUG,
            "Creating streaming completion",
            {"status": "starting"},
        )
        stream = await client.chat.completions.create(**params)
        _log_event(
            on_log, LogLevel.DEBUG, "Stream created", {"status": "ready"}
        )

        async for chunk in stream:
            _log_event(
                on_log,
                LogLevel.DEBUG,
                "Received chunk from API",
                {"chunk": str(chunk)},
            )

            if not hasattr(chunk, "choices") or not chunk.choices:
                _log_event(
                    on_log,
                    LogLevel.DEBUG,
                    "Skipping chunk without choices",
                    {"chunk": str(chunk)},
                )
                continue

            choice = chunk.choices[0]
            if not hasattr(choice, "delta"):
                _log_event(
                    on_log,
                    LogLevel.DEBUG,
                    "Skipping chunk without delta",
                    {"choice": str(choice)},
                )
                continue

            delta = choice.delta
            if not hasattr(delta, "content") or delta.content is None:
                _log_event(
                    on_log,
                    LogLevel.DEBUG,
                    "Skipping chunk with no content",
                    {"error": "No content in delta"},
                )
                continue

            _log_event(
                on_log,
                LogLevel.DEBUG,
                "Processing content",
                {"content": delta.content},
            )

            try:
                result = _process_stream_chunk(
                    chunk=chunk,
                    buffer=buffer,
                    output_schema=output_schema,
                    on_log=on_log,
                )

                if result is not None:
                    _log_event(
                        on_log,
                        LogLevel.DEBUG,
                        "Yielding parsed result",
                        {"result": str(result)},
                    )
                    yield result

            except (KeyError, AttributeError) as e:
                _log_event(
                    on_log,
                    LogLevel.ERROR,
                    "Error processing chunk in async_openai_structured_stream function",
                    {
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                continue

        if buffer is not None:
            final_chunk = buffer.close()
            if final_chunk is not None:
                _log_event(
                    on_log,
                    LogLevel.DEBUG,
                    "Got final result from buffer close",
                    {"result": str(final_chunk)},
                )
                yield final_chunk

    except Exception as e:
        _handle_stream_error(e, on_log)

    finally:
        if buffer:
            # Already closed in normal flow, only close here if exception occurred
            if not buffer._closed:
                buffer.close()
        if stream and hasattr(stream, "close"):
            try:
                await stream.close()
            except Exception as e:
                _log_event(
                    on_log,
                    LogLevel.WARNING,
                    "Error closing stream",
                    {"error": str(e)},
                )


def validate_template(template: str, available_vars: Set[str]) -> None:
    """Validate that a template string only uses available variables.

    Args:
        template: Template string with {var} placeholders
        available_vars: Set of variable names that can be used

    Raises:
        ValueError: If template uses undefined variables or has invalid syntax
    """
    # Find all {var} placeholders
    placeholders = set(re.findall(r"{([^}]+)}", template))

    # Check for undefined variables
    undefined = placeholders - available_vars
    if undefined:
        raise ValueError(
            f"Template uses undefined variables: {', '.join(sorted(undefined))}"
        )

    # Validate template syntax by trying to format with dummy values
    dummy_values = {var: "" for var in available_vars}
    try:
        template.format(**dummy_values)
    except ValueError as e:
        raise ValueError(f"Invalid template syntax: {e}")


@validate_parameters
def openai_structured_stream(
    client: OpenAI,
    model: str,
    output_schema: Type[BaseModel],
    user_prompt: str,
    system_prompt: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    on_log: Optional[LogCallback] = None,
    stream_config: Optional[StreamConfig] = None,
    timeout: Optional[float] = None,
) -> Generator[BaseModel, None, None]:
    """Stream structured output from OpenAI's API.

    This function streams responses from OpenAI's API, parsing each chunk into
    a structured output that matches the provided schema. It's useful for
    handling long responses or when you want to process results incrementally.

    Args:
        client: OpenAI client instance
        model: Model name (e.g., "gpt-4o-2024-08-06")
        output_schema: Pydantic model class defining the expected output structure
        user_prompt: User message containing the actual request
        system_prompt: System message to guide the model's behavior
        temperature: Controls randomness (0.0-2.0, default: 0.7)
        max_output_tokens: Maximum tokens to generate (optional)
        max_completion_tokens: Maximum tokens to generate (optional)
        reasoning_effort: Optional reasoning effort
        top_p: Nucleus sampling parameter (optional)
        frequency_penalty: Frequency penalty parameter (optional)
        presence_penalty: Presence penalty parameter (optional)
        on_log: Optional callback for logging events
        stream_config: Optional configuration for stream buffer management
        timeout: Optional timeout in seconds for the API request

    Returns:
        Generator yielding structured objects matching output_schema

    Raises:
        ModelNotSupportedError: If model doesn't support structured output
        StreamInterruptedError: If stream is interrupted (e.g., network issues)
        StreamParseError: If response can't be parsed into schema
        StreamBufferError: If buffer management fails
        BufferOverflowError: If buffer size exceeds limit
        APIError: For OpenAI API errors
        APITimeoutError: If request exceeds timeout
        ValidationError: If response doesn't match schema

    Examples:
        >>> from pydantic import BaseModel
        >>> from openai import OpenAI
        >>>
        >>> class StoryChapter(BaseModel):
        ...     title: str
        ...     content: str
        ...
        >>> client = OpenAI()
        >>> prompt = "Write a story about a magical forest"
        >>>
        >>> # Basic streaming
        >>> for chapter in openai_structured_stream(
        ...     client=client,
        ...     model="gpt-4o-2024-08-06",
        ...     output_schema=StoryChapter,
        ...     system_prompt="You are a creative writer",
        ...     user_prompt=prompt
        ... ):
        ...     print(f"Chapter: {chapter.title}")
        ...     print(chapter.content)
        ...
        >>> # With custom buffer config and timeout
        >>> config = StreamConfig(
        ...     max_buffer_size=2 * 1024 * 1024,  # 2MB
        ...     cleanup_threshold=1024 * 1024,     # 1MB
        ...     chunk_size=4096                    # 4KB
        ... )
        >>> for chapter in openai_structured_stream(
        ...     client=client,
        ...     model="gpt-4o-2024-08-06",
        ...     output_schema=StoryChapter,
        ...     system_prompt="You are a creative writer",
        ...     user_prompt=prompt,
        ...     stream_config=config,
        ...     timeout=30.0
        ... ):
        ...     print(f"Chapter: {chapter.title}")
        ...     print(chapter.content)
    """
    buffer = None
    stream = None

    try:
        _log_request_start(
            on_log,
            model,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            max_output_tokens,
            max_completion_tokens,
            reasoning_effort,
            True,
            output_schema,
        )

        params = _prepare_request(
            model,
            client,
            OpenAI,
            output_schema,
            system_prompt,
            user_prompt,
            temperature,
            max_output_tokens,
            max_completion_tokens,
            reasoning_effort,
            top_p,
            frequency_penalty,
            presence_penalty,
            True,
            timeout,
        )

        buffer = StreamBuffer(
            config=stream_config or StreamConfig(), schema=output_schema
        )

        _log_event(
            on_log,
            LogLevel.DEBUG,
            "Creating streaming completion",
            {"status": "starting"},
        )

        stream = client.chat.completions.create(**params)

        _log_event(
            on_log, LogLevel.DEBUG, "Stream created", {"status": "ready"}
        )

        for chunk in stream:
            _log_event(
                on_log,
                LogLevel.DEBUG,
                "Received stream chunk",
                {
                    "id": getattr(chunk, "id", None),
                    "model": getattr(chunk, "model", None),
                    "object": getattr(chunk, "object", None),
                },
            )

            if not hasattr(chunk, "choices"):
                _log_event(
                    on_log,
                    LogLevel.DEBUG,
                    "Chunk missing choices attribute",
                    {"chunk": str(chunk)},
                )
                continue

            if not chunk.choices:
                _log_event(
                    on_log,
                    LogLevel.DEBUG,
                    "Chunk has empty choices",
                    {"chunk": str(chunk)},
                )
                continue

            choice = chunk.choices[0]
            _log_event(
                on_log,
                LogLevel.DEBUG,
                "Processing choice",
                {
                    "index": getattr(choice, "index", None),
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "has_delta": hasattr(choice, "delta"),
                },
            )

            try:
                delta = choice.delta
                if not hasattr(delta, "content") or delta.content is None:
                    _log_event(
                        on_log,
                        LogLevel.DEBUG,
                        "Skipping chunk without content",
                        {"error": "No content in delta"},
                    )
                    continue

                _log_event(
                    on_log,
                    LogLevel.DEBUG,
                    "Processing content",
                    {"content": delta.content},
                )
                result = buffer.process_stream_chunk(delta.content)
                if result is not None:
                    _log_event(
                        on_log,
                        LogLevel.DEBUG,
                        "Got result from chunk",
                        {"result": str(result)},
                    )
                    yield result

            except (KeyError, AttributeError) as e:
                _log_event(
                    on_log,
                    LogLevel.ERROR,
                    f"Error processing chunk in openai_structured_stream function: {str(e)}",
                    {
                        "error": str(e),
                        "chunk": str(chunk),
                        "error_type": type(e).__name__,
                    },
                )
                continue

        if buffer is not None:
            final_chunk = buffer.close()
            if final_chunk is not None:
                _log_event(
                    on_log,
                    LogLevel.DEBUG,
                    "Got final result from buffer close",
                    {"result": str(final_chunk)},
                )
                yield final_chunk

    except Exception as e:
        _log_event(
            on_log, LogLevel.ERROR, "Error in stream", {"error": str(e)}
        )
        _handle_stream_error(e, on_log)
    finally:
        if buffer:
            # Already closed in normal flow, only close here if exception occurred
            if not buffer._closed:
                buffer.close()
        if stream and hasattr(stream, "close"):
            try:
                stream.close()
            except Exception as e:
                _log_event(
                    on_log,
                    LogLevel.WARNING,
                    "Error closing stream",
                    {"error": str(e)},
                )


def _get_base_model(model: str) -> str:
    """Extract base model name from full model name.

    Args:
        model: The model name (e.g., 'gpt-4o', 'gpt-4o-2024-08-06')

    Returns:
        The base model name without version

    Example:
        >>> _get_base_model("gpt-4o-2024-08-06")
        'gpt-4o'
        >>> _get_base_model("o1")
        'o1'
    """
    if not isinstance(model, str):
        raise ValueError(f"Model name must be a string, got {type(model)}")
    match = MODEL_VERSION_PATTERN.match(model)
    if match:
        return match.group(1)
    return model


def get_context_window_limit(model_name: str) -> int:
    """Get the context window size for a model.

    Args:
        model_name: Name of the model to check

    Returns:
        int: Maximum context window size in tokens

    Raises:
        ModelNotSupportedError: If the model is not supported
    """
    try:
        registry = ModelRegistry.get_instance()
        capabilities = registry.get_capabilities(model_name)
        return capabilities.context_window
    except ModelNotSupportedError:
        # Default fallback value
        return 8_192
    except Exception as e:
        logger.warning(f"Error getting context window for {model_name}: {e}")
        return 8_192


def get_default_token_limit(model_name: str) -> int:
    """Get the default token limit for a model's output.

    Args:
        model_name: Name of the model to check

    Returns:
        int: Default max token limit for the model

    Raises:
        ModelNotSupportedError: If the model is not supported
    """
    try:
        registry = ModelRegistry.get_instance()
        capabilities = registry.get_capabilities(model_name)
        return capabilities.max_output_tokens
    except ModelNotSupportedError:
        # Default fallback value
        return 4_096
    except Exception as e:
        logger.warning(f"Error getting token limit for {model_name}: {e}")
        return 4_096


def _validate_token_limits(
    model: str, max_completion_tokens: Optional[int] = None
) -> None:
    """Validate token limit parameters for a model.

    Args:
        model: Model name
        max_completion_tokens: Maximum number of tokens for completion

    Raises:
        TokenLimitError: If token limits are invalid
        ModelNotSupportedError: If the model is not supported
    """
    if max_completion_tokens is None:
        return

    try:
        registry = ModelRegistry.get_instance()
        capabilities = registry.get_capabilities(model)

        # Convert to our own token limit error for API consistency
        if max_completion_tokens > capabilities.max_output_tokens:
            raise TokenLimitError(
                f"Invalid max_completion_tokens: {max_completion_tokens}. "
                f"Model {model} has a maximum of {capabilities.max_output_tokens} tokens.",
                requested_tokens=max_completion_tokens,
                model_limit=capabilities.max_output_tokens,
            )
    except ModelNotSupportedError:
        # For unsupported models, use more conservative defaults
        default_limit = get_default_token_limit(model)
        if max_completion_tokens > default_limit:
            raise TokenLimitError(
                f"Invalid max_completion_tokens: {max_completion_tokens}. "
                f"Using default limit of {default_limit} tokens for unknown model {model}.",
                requested_tokens=max_completion_tokens,
                model_limit=default_limit,
            )
    except Exception as e:
        logger.warning(f"Error validating token limits for {model}: {e}")
        # If we can't validate, pass through (better than blocking legitimate requests)
