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
    NamedTuple,
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
    OpenAIClientError,
)

logger = logging.getLogger(__name__)

# Constants


class ModelVersion(NamedTuple):
    """Model version information."""

    year: int
    month: int
    day: int

    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(
            int(match.group(1)), int(match.group(2)), int(match.group(3))
        )

    def __str__(self) -> str:
        return f"{self.year}-{self.month:02d}-{self.day:02d}"

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return (self.year, self.month, self.day) >= (
            other.year,
            other.month,
            other.day,
        )


OPENAI_API_SUPPORTED_MODELS = {
    "gpt-4o": ModelVersion(2024, 8, 6),
    "gpt-4o-mini": ModelVersion(2024, 7, 18),
    "o1": ModelVersion(2024, 12, 17),
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


class ModelVersionError(ModelNotSupportedError):
    """Raised when the model version is not supported."""

    def __init__(self, model: str, min_version: ModelVersion) -> None:
        super().__init__(
            f"Model '{model}' version is not supported. "
            f"Minimum version required: {min_version}"
        )


class BufferOverflowError(BufferError):
    """Raised when the streaming buffer exceeds size limits."""

    pass


class JSONParseError(InvalidResponseFormatError):
    """Raised when JSON parsing fails."""

    pass


def _is_model_supported(model_name: str) -> bool:
    """Check if the model and its version are supported."""
    for base_model, min_version in OPENAI_API_SUPPORTED_MODELS.items():
        if base_model in model_name:
            try:
                version_str = model_name.split(f"{base_model}-")[-1]
                version = ModelVersion.from_string(version_str)
                return version >= min_version
            except (IndexError, ValueError):
                raise ModelVersionError(model_name, min_version)
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
    except (ValidationError, json.JSONDecodeError) as e:
        raise InvalidResponseFormatError(
            f"Invalid response format from OpenAI API: {e}"
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
        model: OpenAI model name (e.g., "gpt-4-0125-preview").
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

    if not _is_model_supported(model):
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

    if not _is_model_supported(model):
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
