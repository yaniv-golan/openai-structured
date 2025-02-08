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
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel, ValidationError
from typing_extensions import ParamSpec

# Local imports
from .buffer import StreamBuffer, StreamConfig
from .errors import (
    BufferOverflowError,
    EmptyResponseError,
    InvalidResponseFormatError,
    JSONParseError,
    ModelNotSupportedError,
    OpenAIClientError,
    StreamBufferError,
    StreamInterruptedError,
    StreamParseError,
    TokenLimitError,
)
from .model_version import ModelVersion

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
LogCallback = Callable[[int, str, dict[str, Any]], None]
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
        if model is None and len(args) > 1:
            model = args[1]  # model is the second argument after client

        # Check if this is an o1 or o3 model
        if model is None:
            base_model = ""
        else:
            if not isinstance(model, str):
                raise ValueError(
                    f"Model name must be a string, got {type(model)}"
                )
            base_model = _get_base_model(model)

        is_o1_model = base_model.startswith("o1")
        is_o3_model = base_model.startswith("o3")

        # Check if streaming is requested
        stream = kwargs.get("stream", True)  # Default to True since our functions use streaming
        if stream:
            if model == "o1-2024-12-17":
                raise OpenAIClientError(
                    "o1-2024-12-17 does not support streaming. Setting stream=True will "
                    "result in a 400 error with message: 'Unsupported value: 'stream' "
                    "does not support true with this model. Supported values are: false'. "
                    "Use o1-preview, o1-mini, or a different model if you need streaming "
                    "support."
                )
            elif model == "o3" or (is_o3_model and not ("mini" in model.lower())):
                raise OpenAIClientError(
                    "The main o3 model does not support streaming. Setting stream=True "
                    "will result in a 400 error. Use o3-mini or o3-mini-high if you "
                    "need streaming support."
                )

        # Get temperature with proper type casting
        temp_val = kwargs.get("temperature", DEFAULT_TEMPERATURE)
        temperature = float(
            temp_val
            if isinstance(temp_val, (int, float, str))
            else DEFAULT_TEMPERATURE
        )

        # Enforce fixed parameters for o1 and o3 models
        if is_o1_model or is_o3_model:
            model_name = "o1" if is_o1_model else "o3"
            if "temperature" in kwargs:
                raise OpenAIClientError(
                    f"{model_name} models have fixed parameters that cannot be modified. "
                    "Temperature adjustment is not supported."
                )
            if "top_p" in kwargs:
                raise OpenAIClientError(
                    f"{model_name} models have fixed parameters that cannot be modified. "
                    "Top-p adjustment is not supported."
                )
            if "frequency_penalty" in kwargs:
                raise OpenAIClientError(
                    f"{model_name} models have fixed parameters that cannot be modified. "
                    "Frequency penalty adjustment is not supported."
                )
            if "presence_penalty" in kwargs:
                raise OpenAIClientError(
                    f"{model_name} models have fixed parameters that cannot be modified. "
                    "Presence penalty adjustment is not supported."
                )
            # Set fixed values for o1 and o3 models
            kwargs["temperature"] = 1.0
            kwargs["top_p"] = 1.0
            kwargs["frequency_penalty"] = 0.0
            kwargs["presence_penalty"] = 0.0
        else:
            # Regular parameter validation for other models
            if not 0 <= temperature <= 2:
                raise OpenAIClientError("Temperature must be between 0 and 2")

            # Get top_p with proper type casting
            top_p_val = kwargs.get("top_p", 1.0)
            top_p = float(
                top_p_val if isinstance(top_p_val, (int, float, str)) else 1.0
            )
            if not 0 <= top_p <= 1:
                raise OpenAIClientError("Top-p must be between 0 and 1")

            # Get frequency and presence penalties with proper type casting
            for param in ["frequency_penalty", "presence_penalty"]:
                val = kwargs.get(param, 0.0)
                value = float(
                    val if isinstance(val, (int, float, str)) else 0.0
                )
                if not -2 <= value <= 2:
                    raise OpenAIClientError(
                        f"{param} must be between -2 and 2"
                    )

        return func(*args, **kwargs)

    return wrapper


def supports_structured_output(model_name: str) -> bool:
    """Check if a model supports structured output.

    This function validates whether a given model name supports structured output,
    handling both aliases and dated versions. For dated versions, it ensures they meet
    minimum version requirements.

    The function supports two types of model names:
    1. Aliases (e.g., "gpt-4o"): These are automatically resolved to the latest
       compatible version by OpenAI. We validate that the alias is supported.

    2. Dated versions (e.g., "gpt-4o-2024-08-06"): These specify an exact version.
       We validate that:
       a) The base model (e.g., "gpt-4o") is supported
       b) The date meets or exceeds our minimum version requirement

    If using a dated version newer than our minimum (e.g., "gpt-4o-2024-09-01"),
    it will be accepted as long as the base model is supported and the date is
    greater than or equal to our minimum version.

    Args:
        model_name: The model name to validate. Can be either:
                   - an alias (e.g., "gpt-4o")
                   - dated version (e.g., "gpt-4o-2024-08-06")
                   - newer version (e.g., "gpt-4o-2024-09-01")

    Returns:
        bool: True if the model supports structured output

    Examples:
        >>> supports_structured_output("gpt-4o")  # Alias
        True
        >>> supports_structured_output("gpt-4o-2024-08-06")  # Minimum version
        True
        >>> supports_structured_output("gpt-4o-2024-09-01")  # Newer version
        True
        >>> supports_structured_output("gpt-3.5-turbo")  # Unsupported model
        False
        >>> supports_structured_output("gpt-4o-2024-07-01")  # Too old
        False
    """
    # Check for exact matches (aliases)
    if model_name in OPENAI_API_SUPPORTED_MODELS:
        return True

    # Try to parse as a dated version
    match = MODEL_VERSION_PATTERN.match(model_name)
    if not match:
        return False

    base_model, version_str = match.groups()

    # Check if the base model has a minimum version requirement
    if base_model in OPENAI_API_SUPPORTED_MODELS:
        try:
            version = ModelVersion.from_string(version_str)
            min_version = OPENAI_API_SUPPORTED_MODELS[base_model]
            return version >= min_version
        except ValueError:
            return False

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
    temperature: float,
    max_tokens: Optional[int],
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stream: bool,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Create request parameters for the API call."""
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": cast(
            ResponseFormat,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                },
            },
        ),
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": stream,
    }

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


def _log(
    on_log: Optional[LogCallback],
    level: int,
    message: str,
    data: Optional[dict[str, Any]] = None,
) -> None:
    """Utility function for consistent logging with sensitive data redaction."""
    if on_log:
        safe_data = _redact_sensitive_data(data or {})
        on_log(level, message, safe_data)


def _handle_api_error(
    error: Exception, on_log: Optional[LogCallback] = None
) -> NoReturn:
    """Handle OpenAI API errors with enhanced logging."""
    if on_log:
        # First log basic error info
        _log(
            on_log,
            logging.ERROR,
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
                "status_code": getattr(
                    error, "e AsyncOpenAI or ensure y", None
                ),
                "request_id": getattr(error, "request_id", None),
                "should_retry": isinstance(
                    error,
                    (APIConnectionError, APITimeoutError, RateLimitError),
                ),
                "retry_after": getattr(error, "retry_after", None),
            }
            _log(on_log, logging.ERROR, "OpenAI API error details", error_data)

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
    temperature: float,
    max_tokens: Optional[int],
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    streaming: bool,
    output_schema: Type[BaseModel],
) -> None:
    """Log the start of an API request."""
    _log(
        on_log,
        logging.DEBUG,
        "Starting OpenAI request",
        {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
    temperature: float,
    max_tokens: Optional[int],
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stream: bool,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Prepare common request parameters and validate inputs."""
    _validate_request(model, client, expected_type)
    _validate_token_limits(model, max_tokens)
    messages = _create_chat_messages(system_prompt, user_prompt)
    schema = _get_schema(output_schema)

    return _create_request_params(
        model=model,
        messages=messages,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
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
        _log(
            on_log,
            logging.ERROR,
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
        _log(
            on_log,
            logging.ERROR,
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
        _log(
            on_log,
            logging.DEBUG,
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
            _log(on_log, logging.DEBUG, "Chunk missing 'choices' attribute")
            return None

        if not chunk.choices:
            _log(on_log, logging.DEBUG, "Chunk has no choices, skipping")
            return None

        choice = chunk.choices[0]
        _log(
            on_log,
            logging.DEBUG,
            "First choice details",
            {
                "index": getattr(choice, "index", None),
                "finish_reason": getattr(choice, "finish_reason", None),
            },
        )

        try:
            delta = choice.delta
            _log(
                on_log,
                logging.DEBUG,
                "Delta details",
                {
                    "role": getattr(delta, "role", None),
                    "content": getattr(delta, "content", None),
                    "function_call": getattr(delta, "function_call", None),
                    "tool_calls": getattr(delta, "tool_calls", None),
                },
            )
        except (KeyError, AttributeError) as e:
            _log(
                on_log,
                logging.ERROR,
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
            _log(
                on_log,
                logging.DEBUG,
                "Delta has no content",
                {
                    "role": getattr(delta, "role", None),
                    "function_call": getattr(delta, "function_call", None),
                    "tool_calls": getattr(delta, "tool_calls", None),
                },
            )
            return None

        _log(
            on_log,
            logging.DEBUG,
            "Processing chunk content",
            {"content": delta.content},
        )
        try:
            # Write the content to the buffer
            buffer.write(delta.content)
            _log(
                on_log,
                logging.DEBUG,
                "Buffer status after write",
                {"size_bytes": buffer.size, "content": buffer.getvalue()},
            )

            # Only try to parse if we have a complete JSON object
            current_content = buffer.getvalue()
            if current_content.strip().startswith(
                "{"
            ) and current_content.strip().endswith("}"):
                try:
                    _log(
                        on_log,
                        logging.DEBUG,
                        "Attempting to parse JSON",
                        {"content": current_content},
                    )
                    model_instance = output_schema.model_validate_json(
                        current_content
                    )
                    _log(
                        on_log,
                        logging.DEBUG,
                        "Successfully parsed model",
                        {"model": str(model_instance)},
                    )
                    buffer.reset()
                    return model_instance
                except ValidationError as e:
                    _log(
                        on_log,
                        logging.ERROR,
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
                        _log(
                            on_log,
                            logging.DEBUG,
                            "Buffer exceeded cleanup threshold after validation error, cleaning up",
                            {"size_bytes": buffer.size},
                        )
                        buffer.cleanup()
                except json.JSONDecodeError as e:
                    _log(
                        on_log,
                        logging.ERROR,
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
                        _log(
                            on_log,
                            logging.DEBUG,
                            "Buffer exceeded cleanup threshold after decode error, cleaning up",
                            {"size_bytes": buffer.size},
                        )
                        buffer.cleanup()

            # Only attempt cleanup if buffer is getting large
            if buffer.size > buffer.config.cleanup_threshold:
                _log(
                    on_log,
                    logging.DEBUG,
                    "Buffer exceeded cleanup threshold, cleaning up",
                    {"size_bytes": buffer.size},
                )
                buffer.cleanup()

        except (BufferOverflowError, StreamParseError) as e:
            _log(
                on_log,
                logging.ERROR,
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
        _log(
            on_log,
            logging.ERROR,
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
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
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
        max_tokens: Maximum tokens in response
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
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
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
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            False,
            timeout,
        )

        response = client.chat.completions.create(**params)

        if not response.choices or not response.choices[0].message.content:
            raise EmptyResponseError("OpenAI API returned an empty response.")

        _log(
            on_log,
            logging.DEBUG,
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
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
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
        max_tokens: Maximum tokens in response
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
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
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
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            False,
            timeout,
        )

        response = await client.chat.completions.create(**params)

        if not response.choices or not response.choices[0].message.content:
            raise EmptyResponseError("OpenAI API returned an empty response.")

        _log(
            on_log,
            logging.DEBUG,
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
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
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
        system_prompt: System message to guide the model's behavior
        user_prompt: User message containing the actual request
        temperature: Controls randomness (0.0-2.0, default: 0.7)
        max_tokens: Maximum tokens to generate (optional)
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
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
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
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            True,
            timeout,
        )

        buffer = StreamBuffer(
            config=stream_config or StreamConfig(), schema=output_schema
        )

        _log(on_log, logging.DEBUG, "Creating streaming completion")
        stream = await client.chat.completions.create(**params)
        _log(on_log, logging.DEBUG, "Stream created")

        async for chunk in stream:
            _log(
                on_log,
                logging.DEBUG,
                "Received chunk from API",
                {"chunk": str(chunk)},
            )

            if not hasattr(chunk, "choices") or not chunk.choices:
                _log(
                    on_log,
                    logging.DEBUG,
                    "Skipping chunk without choices",
                    {"chunk": str(chunk)},
                )
                continue

            choice = chunk.choices[0]
            if not hasattr(choice, "delta"):
                _log(
                    on_log,
                    logging.DEBUG,
                    "Skipping chunk without delta",
                    {"choice": str(choice)},
                )
                continue

            delta = choice.delta
            if not hasattr(delta, "content") or delta.content is None:
                _log(
                    on_log,
                    logging.DEBUG,
                    "Skipping chunk with no content",
                    {"delta": str(delta)},
                )
                continue

            _log(
                on_log,
                logging.DEBUG,
                "Processing chunk content",
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
                    _log(
                        on_log,
                        logging.DEBUG,
                        "Yielding parsed result",
                        {"result": str(result)},
                    )
                    yield result

            except (KeyError, AttributeError) as e:
                _log(
                    on_log,
                    logging.ERROR,
                    "Error processing chunk in async_openai_structured_stream function",
                    {
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                continue

        if buffer is not None:
            _log(on_log, logging.DEBUG, "Closing stream buffer")
            buffer.close()

    except Exception as e:
        _handle_stream_error(e, on_log)

    finally:
        if buffer:
            buffer.close()
        if stream and hasattr(stream, "close"):
            try:
                await stream.close()
            except Exception as e:
                _log(
                    on_log,
                    logging.WARNING,
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
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
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
        system_prompt: System message to guide the model's behavior
        user_prompt: User message containing the actual request
        temperature: Controls randomness (0.0-2.0, default: 0.7)
        max_tokens: Maximum tokens to generate (optional)
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
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
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
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            True,
            timeout,
        )

        buffer = StreamBuffer(
            config=stream_config or StreamConfig(), schema=output_schema
        )

        _log(on_log, logging.DEBUG, "Creating streaming completion")

        stream = client.chat.completions.create(**params)

        _log(on_log, logging.DEBUG, "Stream created")

        for chunk in stream:
            _log(
                on_log,
                logging.DEBUG,
                "Processing chunk",
                {"chunk": str(chunk)},
            )
            if not hasattr(chunk, "choices") or not chunk.choices:
                _log(on_log, logging.DEBUG, "Skipping chunk without choices")
                continue

            try:
                delta = chunk.choices[0].delta
                if not hasattr(delta, "content") or delta.content is None:
                    _log(
                        on_log, logging.DEBUG, "Skipping chunk without content"
                    )
                    continue

                _log(
                    on_log,
                    logging.DEBUG,
                    "Processing content",
                    {"content": delta.content},
                )
                result = buffer.process_stream_chunk(delta.content)
                if result is not None:
                    _log(
                        on_log,
                        logging.DEBUG,
                        "Got result from chunk",
                        {"result": str(result)},
                    )
                    yield result

            except (KeyError, AttributeError) as e:
                _log(
                    on_log,
                    logging.ERROR,
                    f"Error processing chunk in openai_structured_stream function: {str(e)}",
                    {
                        "error": str(e),
                        "chunk": str(chunk),
                        "error_type": type(e).__name__,
                    },
                )
                continue

        if buffer is not None:
            buffer.close()

    except Exception as e:
        _log(on_log, logging.ERROR, "Error in stream", {"error": str(e)})
        _handle_stream_error(e, on_log)
    finally:
        if buffer:
            buffer.close()
        if stream and hasattr(stream, "close"):
            try:
                stream.close()
            except Exception as e:
                _log(
                    on_log,
                    logging.WARNING,
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


def get_context_window_limit(model: str) -> int:
    """Get the total context window limit for a given model.

    Args:
        model: The model name (e.g., 'gpt-4o', 'o1', 'o1-mini', 'o3-mini')

    Returns:
        The total context window limit for the model in tokens

    Example:
        >>> get_context_window_limit("o1")
        200_000
        >>> get_context_window_limit("gpt-4o-2024-08-06")
        128_000
    """
    base_model = _get_base_model(model)
    for prefix, limit in MODEL_CONTEXT_WINDOWS.items():
        if base_model.startswith(prefix):
            return limit
    return 8_192  # Default fallback


def get_default_token_limit(model: str) -> int:
    """Get the default maximum output token limit for a given model.

    Args:
        model: The model name (e.g., 'gpt-4o', 'o1', 'o1-mini', 'o3-mini')

    Returns:
        The default maximum number of output tokens for the model

    Example:
        >>> get_default_token_limit("o1")
        100_000
        >>> get_default_token_limit("gpt-4o")
        16_384
    """
    base_model = _get_base_model(model)
    for prefix, limit in MODEL_OUTPUT_LIMITS.items():
        if base_model.startswith(prefix):
            return limit
    return 4_096  # Default fallback


def _validate_token_limits(model: str, max_tokens: Optional[int]) -> None:
    """Validate token limits for a model.

    Args:
        model: The model name
        max_tokens: The requested maximum number of tokens

    Raises:
        TokenLimitError: If the requested tokens exceed the model's limit
    """
    if max_tokens is not None:
        default_limit = get_default_token_limit(model)
        if max_tokens > default_limit:
            raise TokenLimitError(
                f"Requested {max_tokens} tokens exceeds model limit of {default_limit}",
                requested_tokens=max_tokens,
                model_limit=default_limit,
            )
