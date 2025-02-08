import json
import logging
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from .errors import (
    BufferOverflowError,
    ClosedBufferError,
    StreamBufferError,
    StreamParseError,
)

# Buffer size constants
MAX_BUFFER_SIZE = 1024 * 1024  # 1MB
BUFFER_CLEANUP_THRESHOLD = MAX_BUFFER_SIZE // 2
LOG_SIZE_THRESHOLD = 100 * 1024  # 100KB

# Error handling constants
MAX_CLEANUP_ATTEMPTS = 3
MAX_PARSE_ERRORS = 5


@dataclass
class StreamConfig:
    """Configuration for stream buffer management."""

    def __init__(
        self,
        max_buffer_size: int = 1024 * 1024,  # 1MB
        cleanup_threshold: int = 512 * 1024,  # 500KB
        max_parse_errors: int = 3,
        chunk_size: int = 4096,  # 4KB
    ):
        self.max_buffer_size = max_buffer_size
        self.cleanup_threshold = cleanup_threshold
        self.max_parse_errors = max_parse_errors
        self.chunk_size = chunk_size


class BufferError(StreamBufferError):
    """Base class for buffer-related errors."""

    pass


class ParseError(BufferError):
    """Raised when JSON parsing fails repeatedly."""

    pass


class StreamBuffer:
    """Buffer for accumulating and validating stream chunks."""

    def __init__(
        self,
        config: StreamConfig,
        schema: Type[BaseModel],
        response_model: Optional[Type[BaseModel]] = None,
    ):
        self._buffer = StringIO()
        self._config = config
        self._schema = schema
        self._response_model = response_model
        self._parse_errors: List[Exception] = []
        self._partial_json = ""
        self._buffer_size = 0
        self._closed = False
        self._last_cleanup = 0

    @property
    def size(self) -> int:
        """Get the current size of the buffer in bytes."""
        return self._buffer_size

    @property
    def config(self) -> StreamConfig:
        """Get the buffer configuration."""
        return self._config

    def _log(
        self,
        on_log: Optional[Callable[[int, str, dict[str, Any]], None]],
        level: int,
        message: str,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a message with optional callback."""
        if on_log:
            on_log(level, message, data or {})

    def should_log_size(self) -> bool:
        """Check if buffer size should be logged."""
        return self._buffer_size >= self._config.max_buffer_size

    def write(self, content: str) -> None:
        """Write content to buffer, raising appropriate error if cannot write."""
        if self._closed:
            logging.debug("Attempting to write to closed buffer")
            closed_error = ClosedBufferError("Cannot write to a closed buffer")
            logging.debug(f"Raising error: {str(closed_error)!r}")
            raise closed_error

        content_bytes_len = len(content.encode("utf-8"))
        if (
            self._buffer_size + content_bytes_len
            > self._config.max_buffer_size
        ):
            logging.debug(
                f"Buffer size limit exceeded: {self._buffer_size + content_bytes_len} > {self._config.max_buffer_size}"
            )
            overflow_error = BufferOverflowError("Buffer size limit exceeded")
            logging.debug(f"Raising error: {str(overflow_error)!r}")
            raise overflow_error

        self._buffer.write(content)
        self._buffer_size += content_bytes_len

    def getvalue(self) -> str:
        """Get current buffer content."""
        return self._buffer.getvalue()

    def _reset_buffer(self) -> None:
        """Reset the buffer content and size."""
        self._buffer = StringIO()
        self._buffer_size = 0
        self._last_cleanup = 0

    def reset(self) -> None:
        """Reset buffer state."""
        self._reset_buffer()
        self._parse_errors = []

    def close(self) -> None:
        """Close and cleanup buffer."""
        self._closed = True
        self._reset_buffer()

    def _should_cleanup(self) -> bool:
        return (
            self._buffer_size - self._last_cleanup
            >= self._config.cleanup_threshold
            or len(self._parse_errors) > 0
            or self._closed
        )

    def _perform_cleanup(self) -> None:
        self._reset_buffer()
        self._last_cleanup = 0

    def extract_response(self) -> Optional[BaseModel]:
        """
        Attempt to extract a complete JSON object from the buffer and validate it.
        """
        if self._should_cleanup():
            self._perform_cleanup()

        buffer_content = self._buffer.getvalue()

        # Try to find a complete JSON object
        brace_count = 0
        in_string = False
        i = 0
        while i < len(buffer_content):
            char = buffer_content[i]

            # Handle escape sequences inside strings
            if in_string:
                if char == "\\":
                    # Skip the next character as it's escaped
                    i += 2
                    continue
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1

                    # Found a potential complete object
                    if brace_count == 0 and i > 0:
                        try:
                            json_str = buffer_content[: i + 1]
                            data = json.loads(json_str)
                            validated_data = self._schema.model_validate(data)

                            # If validation successful, update buffer with remaining content
                            remaining_content = buffer_content[i + 1 :]
                            self._reset_buffer()
                            if remaining_content:
                                self.write(remaining_content)

                            return validated_data
                        except json.JSONDecodeError as e:
                            self._parse_errors.append(e)
                        except ValidationError as e:
                            self._parse_errors.append(e)

                        # Check if maximum parse errors have been reached
                        if (
                            len(self._parse_errors)
                            >= self._config.max_parse_errors
                        ):
                            raise StreamParseError(
                                "Maximum parse errors reached",
                                attempts=len(self._parse_errors),
                                last_error=self._parse_errors[-1],
                            ) from self._parse_errors[-1]

                        # Reset buffer if there is an error to avoid partial JSON objects
                        self._reset_buffer()
                        return None

            i += 1

        return None

    def process_stream_chunk(
        self,
        content: str,
        on_log: Optional[Callable[[int, str, dict[str, Any]], None]] = None,
    ) -> Optional[Any]:
        """Process a stream chunk and return parsed content if complete."""
        try:
            self.write(content)

            if self.should_log_size():
                self._log(
                    on_log,
                    logging.DEBUG,
                    "Buffer size increased significantly",
                    {"size_bytes": self._buffer_size},
                )

            current_content = self.getvalue()
            if self._schema is not None:
                try:
                    result = self._schema.model_validate_json(current_content)
                    self._log(
                        on_log,
                        logging.DEBUG,
                        "Successfully parsed complete object",
                    )
                    self.reset()
                    return result
                except ValidationError as e:
                    self._parse_errors.append(e)
                    if (
                        len(self._parse_errors)
                        >= self._config.max_parse_errors
                    ):
                        self.reset()  # Clear buffer after max errors
                    elif self._buffer_size > self._config.cleanup_threshold:
                        self.cleanup()
                except json.JSONDecodeError as e:
                    self._parse_errors.append(e)
                    if (
                        len(self._parse_errors)
                        >= self._config.max_parse_errors
                    ):
                        self.reset()  # Clear buffer after max errors
                    elif self._buffer_size > self._config.cleanup_threshold:
                        self.cleanup()
            return None
        except BufferError as e:
            self._parse_errors.append(e)
            self._log(
                on_log,
                logging.ERROR,
                "Critical stream buffer error",
                {"error": str(e), "bytes": self._buffer_size},
            )
            raise

    def cleanup(self) -> None:
        """Clean up the buffer by resetting it."""
        self._reset_buffer()

    def _get_error_context(self, content: str, pos: Union[int, str]) -> str:
        """Get context around an error position."""
        if isinstance(pos, str):
            # For validation errors, find the position of the field in the JSON
            try:
                pos = content.find(f'"{pos}"')
            except (TypeError, ValueError):
                pos = 0
        else:
            pos = int(pos)

        start = max(0, pos - 20)
        end = min(len(content), pos + 20)
        return content[start:end]
