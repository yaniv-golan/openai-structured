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
    "EmptyResponseError",
    "JSONParseError",
]

# Standard library imports
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
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
import ijson  # type: ignore # missing stubs
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
from .errors import (
    BufferOverflowError,
    EmptyResponseError,
    InvalidResponseFormatError,
    JSONParseError,
    ModelNotSupportedError,
    OpenAIClientError,
    StreamBufferError,
    StreamInterruptedError,
)
from .model_version import ModelVersion

# Type variables and aliases
ClientT = TypeVar("ClientT", bound=Union[OpenAI, AsyncOpenAI])
LogCallback = Callable[[int, str, dict[str, Any]], None]
P = ParamSpec("P")
R = TypeVar("R")

# Constants
DEFAULT_TEMPERATURE = 0.2
MAX_BUFFER_SIZE = 1024 * 1024  # 1MB max buffer size
BUFFER_CLEANUP_THRESHOLD = (
    512 * 1024
)  # Clean up buffer if it exceeds 512KB without valid JSON
CHUNK_SIZE = 8192  # 8KB chunks for buffer management
DEFAULT_TIMEOUT = 60.0  # Default timeout in seconds

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
#
# When using aliases (e.g., "gpt-4o"), OpenAI will automatically use the latest
# compatible version. We validate that the model meets our minimum version
# requirements, but the actual version resolution is handled by OpenAI.
#
# Dated versions (recommended for production):
# - gpt-4o-2024-08-06: 128K context window, 16K max output tokens
# - gpt-4o-mini-2024-07-18: 128K context window, 16K max output tokens
# - o1-2024-12-17: 200K context window, 100K max output tokens
#
# Note: These limits may change as OpenAI updates their models

# Model version mapping - maps aliases to minimum supported versions
OPENAI_API_SUPPORTED_MODELS = {
    "gpt-4o": ModelVersion(2024, 8, 6),  # Minimum supported version
    "gpt-4o-mini": ModelVersion(2024, 7, 18),  # Minimum supported version
    "o1": ModelVersion(2024, 12, 17),  # Minimum supported version
}


def validate_parameters(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to validate common parameters."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Get temperature with proper type casting
        temp_val = kwargs.get("temperature", DEFAULT_TEMPERATURE)
        temperature = float(
            temp_val
            if isinstance(temp_val, (int, float, str))
            else DEFAULT_TEMPERATURE
        )
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
            value = float(val if isinstance(val, (int, float, str)) else 0.0)
            if not -2 <= value <= 2:
                raise OpenAIClientError(f"{param} must be between -2 and 2")

        return func(*args, **kwargs)

    return wrapper


@dataclass
class StreamConfig:
    """Configuration for stream behavior."""

    max_buffer_size: int = 1024 * 1024  # 1MB max buffer size
    cleanup_threshold: int = 512 * 1024  # Clean up at 512KB
    chunk_size: int = 8192  # 8KB chunks


class StreamParseError(StreamInterruptedError):
    """Raised when stream content cannot be parsed after multiple attempts."""

    def __init__(self, message: str, attempts: int, last_error: Exception):
        super().__init__(
            f"{message} after {attempts} attempts. Last error: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


@dataclass
class StreamBuffer:
    """Efficient buffer management for streaming responses."""

    chunks: List[str] = field(default_factory=list)
    total_bytes: int = 0
    cleanup_attempts: int = 0
    parse_errors: int = 0
    MAX_CLEANUP_ATTEMPTS: int = 3
    MAX_PARSE_ERRORS: int = 5
    _last_logged_size: int = 0
    LOG_SIZE_THRESHOLD: int = 100 * 1024
    config: StreamConfig = field(default_factory=StreamConfig)
    _current_content: Optional[str] = None
    _cleanup_stats: dict[str, Any] = field(default_factory=dict)

    def write(self, content: str) -> None:
        """Write content to the buffer with improved error handling."""
        if not content:
            return

        chunk_bytes = len(content.encode("utf-8"))
        new_total = self.total_bytes + chunk_bytes

        # Try cleanup even if we know it won't help, to maintain consistent behavior
        while new_total > self.config.max_buffer_size:
            if self.cleanup_attempts >= self.MAX_CLEANUP_ATTEMPTS:
                raise BufferOverflowError(
                    f"Buffer exceeded {self.config.max_buffer_size} bytes after "
                    f"{self.cleanup_attempts} cleanup attempts. "
                    f"Current size: {self.total_bytes}, "
                    f"Attempted to add: {chunk_bytes} bytes. "
                    f"Consider increasing max_buffer_size or adjusting cleanup_threshold."
                )
            self.cleanup()
            self.cleanup_attempts += 1
            new_total = self.total_bytes + chunk_bytes

        self.chunks.append(content)
        self.total_bytes = new_total
        self._current_content = None  # Invalidate cache

    def cleanup(self) -> None:
        """Clean up the buffer with improved JSON parsing strategy."""
        content = self.getvalue()
        if not content:
            return

        original_length = len(content)
        original_bytes = len(content.encode("utf-8"))

        try:
            # Use ijson to find the last complete object
            from io import StringIO

            parser = ijson.parse(StringIO(content))
            stack = []
            last_complete = 0
            current_pos = 0

            for prefix, event, value in parser:
                current_pos = parser.pos
                if event == "start_map":
                    stack.append(current_pos)
                elif event == "end_map":
                    if stack:
                        stack.pop()
                        if not stack:  # Complete object found
                            last_complete = current_pos + 1

            if last_complete > 0:
                self._update_buffer(content[last_complete:])
                self._log_cleanup_stats(
                    original_length,
                    original_bytes,
                    last_complete,
                    "ijson_parsing",
                )
                return

            # Fallback to pattern matching if ijson parsing fails
            patterns = ["}", '"]', "]"]
            positions = [content.rfind(p) for p in patterns]
            last_pos = max(pos for pos in positions if pos != -1)

            if last_pos > 0:
                self._update_buffer(content[last_pos + 1 :])
                self._log_cleanup_stats(
                    original_length,
                    original_bytes,
                    last_pos + 1,
                    "pattern_matching",
                )
                return

        except Exception as e:
            self._cleanup_stats.update(
                {"error": str(e), "error_type": type(e).__name__}
            )

        # If all strategies fail, increment error count
        self.parse_errors += 1
        if self.parse_errors >= self.MAX_PARSE_ERRORS:
            raise StreamParseError(
                "Failed to parse stream content",
                self.parse_errors,
                ValueError(
                    f"All cleanup strategies failed. Stats: {self._cleanup_stats}"
                ),
            )

    def _update_buffer(self, new_content: str) -> None:
        """Update buffer contents efficiently."""
        if new_content:
            self.chunks = [new_content]
            self.total_bytes = len(new_content.encode("utf-8"))
        else:
            self.chunks = []
            self.total_bytes = 0
        self._current_content = new_content
        self.cleanup_attempts = 0

    def reset(self) -> None:
        """Reset the buffer state while reusing the object."""
        self.chunks = []
        self.total_bytes = 0
        self.cleanup_attempts = 0
        self.parse_errors = 0
        self._last_logged_size = 0
        self._current_content = None

    def should_log_size(self) -> bool:
        """Determine if current size should be logged based on threshold."""
        # Only log significant changes to reduce overhead
        if self.total_bytes > self._last_logged_size + self.LOG_SIZE_THRESHOLD:
            self._last_logged_size = self.total_bytes
            return True
        return False

    def getvalue(self) -> str:
        """Get the current buffer contents as a string."""
        if self._current_content is None:
            self._current_content = "".join(self.chunks)
        return self._current_content

    def _log_cleanup_stats(
        self,
        original_length: int,
        original_bytes: int,
        cleanup_position: int,
        strategy: str,
    ) -> None:
        """Log statistics about buffer cleanup."""
        new_content = self.getvalue()
        new_length = len(new_content)
        new_bytes = len(new_content.encode("utf-8"))

        self._cleanup_stats.update(
            {
                "attempt": self.cleanup_attempts + 1,
                "strategy": strategy,
                "original_length": original_length,
                "original_bytes": original_bytes,
                "cleanup_position": cleanup_position,
                "chars_removed": original_length - new_length,
                "bytes_removed": original_bytes - new_bytes,
                "remaining_length": new_length,
                "remaining_bytes": new_bytes,
            }
        )

    def close(self) -> None:
        """Close the buffer and clean up resources."""
        self.reset()


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


def _validate_client_type(client: Any, expected_type: Type[ClientT]) -> None:
    """Validate that the client is of the expected type."""
    if not isinstance(client, expected_type):
        raise TypeError(
            f"Expected client of type {expected_type.__name__}, got {type(client).__name__}"
        )


def _validate_request(
    model: str,
    client: Union[OpenAI, AsyncOpenAI],
    expected_type: Type[ClientT],
) -> None:
    """Validate the request parameters."""
    _validate_client_type(client, expected_type)
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
        "max_tokens": max_tokens,
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
    return model_class.model_json_schema()


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
                "status_code": getattr(error, "status_code", None),
                "request_id": getattr(error, "request_id", None),
                "should_retry": isinstance(
                    error,
                    (APIConnectionError, APITimeoutError, RateLimitError),
                ),
                "retry_after": getattr(error, "retry_after", None),
            }
            _log(on_log, logging.ERROR, "OpenAI API error details", error_data)

    # Re-raise OpenAI errors as is, wrap others in OpenAIClientError
    if isinstance(error, OpenAIError):
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

    if isinstance(e, StreamParseError):
        _log(
            on_log,
            logging.ERROR,
            "Stream parsing failed",
            {"attempts": e.attempts, "error": str(e.last_error)},
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
    if not chunk.choices:
        return None

    delta = chunk.choices[0].delta
    if not delta.content:
        return None

    try:
        buffer.write(delta.content)

        # Only log significant size changes
        if buffer.should_log_size():
            _log(
                on_log,
                logging.DEBUG,
                "Buffer size increased significantly",
                {"size_bytes": buffer.total_bytes},
            )

        current_content = buffer.getvalue()
        try:
            model_instance = output_schema.model_validate_json(current_content)
            # Only log successful parse
            _log(on_log, logging.DEBUG, "Successfully parsed complete object")
            buffer.reset()
            return model_instance
        except (ValidationError, json.JSONDecodeError):
            # Only attempt cleanup if buffer is getting large
            if buffer.total_bytes > buffer.config.cleanup_threshold:
                buffer.cleanup()
    except (BufferOverflowError, StreamParseError) as e:
        # Only log critical errors
        _log(
            on_log,
            logging.ERROR,
            "Critical stream buffer error",
            {"error": str(e), "bytes": buffer.total_bytes},
        )
        raise

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

        buffer = StreamBuffer(config=stream_config or StreamConfig())

        _log(on_log, logging.DEBUG, "Creating streaming completion")

        stream = await client.chat.completions.create(**params)

        _log(on_log, logging.DEBUG, "Stream created")

        async for chunk in stream:
            result = _process_stream_chunk(
                chunk, buffer, output_schema, on_log
            )
            if result is not None:
                yield result

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

        buffer = StreamBuffer(config=stream_config or StreamConfig())

        _log(on_log, logging.DEBUG, "Creating streaming completion")

        stream = client.chat.completions.create(**params)

        _log(on_log, logging.DEBUG, "Stream created")

        for chunk in stream:
            result = _process_stream_chunk(
                chunk, buffer, output_schema, on_log
            )
            if result is not None:
                yield result

    except Exception as e:
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
