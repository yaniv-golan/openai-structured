"""Client for making structured OpenAI API calls."""

import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Deque,
    List,
    Optional,
    Type,
    cast,
)

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel, ValidationError

# Import custom exceptions
from .errors import (
    EmptyResponseError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
    ModelVersionError,
    OpenAIClientError,
)
from .model_version import ModelVersion

logger = logging.getLogger(__name__)

# Constants

# Regex pattern for model version extraction
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
    # Aliases map to minimum supported versions
    "gpt-4o": ModelVersion(2024, 8, 6),      # Minimum supported version
    "gpt-4o-mini": ModelVersion(2024, 7, 18), # Minimum supported version
    "o1": ModelVersion(2024, 12, 17),         # Minimum supported version
}

DEFAULT_TEMPERATURE = 0.2
MAX_BUFFER_SIZE = 1024 * 1024  # 1MB max buffer size
# Clean up buffer if it exceeds 512KB without valid JSON
BUFFER_CLEANUP_THRESHOLD = 512 * 1024
CHUNK_SIZE = 8192  # 8KB chunks for buffer management


@dataclass
class StreamBuffer:
    """Efficient buffer management for streaming responses."""

    chunks: Deque[str]
    total_bytes: int
    last_valid_json_pos: int

    def __init__(self) -> None:
        self.chunks = deque()
        self.total_bytes = 0
        self.last_valid_json_pos = 0

    def write(self, content: str) -> None:
        chunk_bytes = len(content.encode("utf-8"))
        if self.total_bytes + chunk_bytes > MAX_BUFFER_SIZE:
            raise BufferError(
                "Response exceeded maximum buffer size of "
                f"{MAX_BUFFER_SIZE} bytes"
            )
        self.chunks.append(content)
        self.total_bytes += chunk_bytes

    def getvalue(self) -> str:
        return "".join(self.chunks)

    def cleanup(self) -> None:
        """Attempt to clean up the buffer by finding"""
        """and removing processed JSON."""
        content = self.getvalue()
        try:
            # Find the last occurrence of '}'
            last_brace = content.rstrip().rfind("}")
            if last_brace > self.last_valid_json_pos:
                # Try to parse everything up to this point
                potential_json = content[: last_brace + 1]
                json.loads(potential_json)  # Just to validate
                # If successful, update the last valid position
                self.last_valid_json_pos = last_brace + 1
                # Keep only the content after the last valid JSON
                new_content = content[self.last_valid_json_pos :]
                self.chunks.clear()
                if new_content:
                    self.chunks.append(new_content)
                self.total_bytes = len(new_content.encode("utf-8"))
        except (json.JSONDecodeError, ValueError):
            pass  # Keep accumulating if no valid JSON found

    def close(self) -> None:
        self.chunks.clear()
        self.total_bytes = 0
        self.last_valid_json_pos = 0


class BufferOverflowError(BufferError):
    """Raised when the streaming buffer exceeds size limits."""

    pass


class JSONParseError(InvalidResponseFormatError):
    """Raised when JSON parsing fails."""

    pass


def supports_structured_output(model_name: str) -> bool:
    """Check if a model supports structured output.
    
    This function validates whether a given model name supports structured output,
    handling both aliases and dated versions. For dated versions, it ensures they meet
    minimum version requirements.
    
    Args:
        model_name: The model name, which can be either:
                   - an alias (e.g., "gpt-4o")
                   - dated version (e.g., "gpt-4o-2024-08-06")
                   - newer version (e.g., "gpt-4o-2024-09-01")
    
    Returns:
        bool: True if the model supports structured output
        
    Raises:
        ModelVersionError: If a dated version format is invalid or doesn't meet
                          minimum version requirements
        
    Examples:
        >>> supports_structured_output("gpt-4o")
        True
        >>> supports_structured_output("gpt-4o-2024-08-06")
        True
        >>> supports_structured_output("gpt-4o-2024-09-01")  # Newer version
        True
        >>> supports_structured_output("gpt-3.5-turbo")
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


def _create_chat_messages(
    system_prompt: str, user_prompt: str
) -> List[ChatCompletionMessageParam]:
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


def _log_request(logger_opt: Optional[logging.Logger], **kwargs: Any) -> None:
    if logger_opt:
        logger_opt.debug(
            "OpenAI API Request",
            extra={
                "payload": {
                    "component": "openai",
                    "operation": "structured_call",
                    **kwargs,
                }
            },
        )


def _log_response(logger_opt: Optional[logging.Logger], **kwargs: Any) -> None:
    if logger_opt:
        logger_opt.debug(
            "OpenAI API Response",
            extra={
                "payload": {
                    "component": "openai",
                    "operation": "structured_call",
                    **kwargs,
                }
            },
        )


def _log_error(
    logger_opt: Optional[logging.Logger],
    message: str,
    error: Exception,
    **kwargs: Any,
) -> None:
    if logger_opt:
        logger_opt.error(
            message,
            exc_info=True,
            extra={
                "payload": {
                    "component": "openai",
                    "operation": "structured_call",
                    "error": str(error),
                    **kwargs,
                }
            },
        )


def _parse_json_response(
    content: Optional[str], output_schema: Type[BaseModel]
) -> BaseModel:
    if not content:
        raise EmptyResponseError("OpenAI API returned empty response")
    try:
        return output_schema.model_validate_json(content)
    except ValidationError as e:
        # For Pydantic validation errors, include the validation error details
        raise InvalidResponseFormatError(
            f"Response validation failed: {e}\nReceived content (first 200 chars): {content[:200]}"
        ) from e
    except json.JSONDecodeError as e:
        # For JSON parsing errors, include position and snippet around the error
        error_pos = e.pos
        start = max(0, error_pos - 50)
        end = min(len(content), error_pos + 50)
        context = content[start:end]
        if start > 0:
            context = "..." + context
        if end < len(content):
            context = context + "..."

        raise InvalidResponseFormatError(
            f"Invalid JSON at position {error_pos}: {e.msg}\n"
            f"Context: {context}\n"
            f"Full response (first 200 chars): {content[:200]}"
        ) from e


def _log_debug(
    logger: Optional[logging.Logger],
    message: str,
) -> None:
    """Log a debug message if a logger is provided."""
    if logger is not None:
        logger.debug(message)


async def openai_structured_call(
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
    logger: Optional[logging.Logger] = None,
) -> BaseModel:
    """
    Make a structured call to the OpenAI API.

    Args:
        client: Initialized OpenAI client.
        model: OpenAI model name (e.g., "gpt-4o-2024-08-06").
        output_schema: Pydantic model defining the expected output structure.
        user_prompt: The user's request.
        system_prompt: System instructions for the model.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        top_p: Top-p sampling.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        logger: Optional custom logger.

    Returns:
        An instance of the `output_schema` Pydantic model.

    Raises:
        ModelNotSupportedError: If the model doesn't support structured output.
        ModelVersionError: If the model version is not supported.
        APIResponseError: For errors from the OpenAI API.
        OpenAIClientError: For general client-related errors.
        JSONParseError: If JSON parsing fails.
        EmptyResponseError: If the API returns an empty response.
    """

    if not supports_structured_output(model):
        raise ModelNotSupportedError(
            f"Model '{model}' does not support structured output."
        )

    messages = _create_chat_messages(system_prompt, user_prompt)

    _log_request(
        logger,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        streaming=False,
        output_schema=output_schema.__name__,
    )

    try:
        # Get the schema and add required name field
        schema = output_schema.model_json_schema()

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=cast(
                ResponseFormat,
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": schema,
                    },
                },
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=False,
        )

        if not response.choices or not response.choices[0].message.content:
            raise EmptyResponseError("OpenAI API returned an empty response.")

        try:
            model_instance = output_schema.model_validate_json(
                response.choices[0].message.content
            )
            return model_instance
        except ValidationError as e:
            raise InvalidResponseFormatError(
                f"Response validation failed: {e}",
                response_id=response.id,
            ) from e
        except json.JSONDecodeError as e:
            raise JSONParseError(
                f"Failed to parse JSON response: {e}",
                response_id=response.id,
            ) from e

    except (
        AuthenticationError,
        BadRequestError,
        APIConnectionError,
        InternalServerError,
        EmptyResponseError,
        InvalidResponseFormatError,
        JSONParseError,
    ) as e:
        _log_error(logger, "OpenAI API error during call.", e)
        raise e
    except Exception as e:
        _log_error(logger, "An unexpected error occurred during API call.", e)
        raise OpenAIClientError(
            f"Unexpected error during OpenAI API call: {e}"
        ) from e


async def openai_structured_stream(
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
    logger: Optional[logging.Logger] = None,
) -> AsyncGenerator[BaseModel, None]:
    """
    Asynchronously stream structured output from the OpenAI API.

    Args:
        client: Initialized OpenAI client.
        model: OpenAI model name (e.g., "gpt-4o-2024-08-06").
        output_schema: Pydantic model defining the expected output structure.
        user_prompt: The user's request.
        system_prompt: System instructions for the model.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        top_p: Top-p sampling.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        logger: Optional custom logger.

    Yields:
        Instances of the `output_schema` Pydantic model as
        they become available.

    Raises:
        ModelNotSupportedError: If the model doesn't support structured output.
        ModelVersionError: If the model version is not supported.
        APIResponseError: For errors from the OpenAI API.
        OpenAIClientError: For general client-related errors.
        BufferOverflowError: If the response exceeds the maximum buffer size.
        JSONParseError: If JSON parsing fails.
    """
    if not isinstance(client, AsyncOpenAI):
        raise TypeError("Streaming operations require AsyncOpenAI client")

    if not supports_structured_output(model):
        raise ModelNotSupportedError(
            f"Model '{model}' does not support structured output."
        )

    buffer = StreamBuffer()
    messages = _create_chat_messages(system_prompt, user_prompt)

    _log_request(
        logger,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        streaming=True,
        output_schema=output_schema.__name__,
    )

    try:
        # Get the schema and add required name field
        schema = output_schema.model_json_schema()
        _log_debug(logger, "Creating streaming completion...")

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=cast(
                ResponseFormat,
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": schema,
                    },
                },
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True,
        )

        _log_debug(logger, f"Stream object created: {type(stream)}")
        _log_debug(logger, "Starting stream iteration...")

        async for chunk in stream:
            _log_debug(logger, f"Received chunk: {chunk}")
            if chunk.choices and chunk.choices[0].delta.content is not None:
                try:
                    buffer.write(chunk.choices[0].delta.content)
                    current_content = buffer.getvalue()
                    _log_debug(
                        logger,
                        f"Current buffer content: " f"{current_content}",
                    )

                    # Try to parse and yield if valid JSON is formed
                    try:
                        model_instance = output_schema.model_validate_json(
                            current_content
                        )
                        _log_debug(
                            logger,
                            f"Successfully parsed " f"model: {model_instance}",
                        )
                        yield model_instance
                        # Reset buffer after successful parse
                        buffer = StreamBuffer()
                    except (ValidationError, json.JSONDecodeError) as e:
                        _log_debug(logger, f"Failed to parse JSON: {str(e)}")
                        # Check if we need to clean up the buffer
                        if buffer.total_bytes > BUFFER_CLEANUP_THRESHOLD:
                            buffer.cleanup()
                except BufferError as e:
                    raise BufferOverflowError(str(e))

    except (
        AuthenticationError,
        BadRequestError,
        APIConnectionError,
        InternalServerError,
        BufferOverflowError,
        JSONParseError,
    ) as e:
        _log_error(logger, "OpenAI API error during streaming.", e)
        raise e
    except Exception as e:
        _log_error(
            logger,
            "An unexpected error occurred during streaming.",
            e,
        )
        raise OpenAIClientError(
            "Unexpected error during OpenAI API streaming " f"call: {e}"
        ) from e
    finally:
        buffer.close()
