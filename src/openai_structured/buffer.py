import json
import logging
import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any, Callable, Dict, Optional, Type, Union

import ijson  # type: ignore # missing stubs
from pydantic import BaseModel, ValidationError

from .errors import StreamBufferError

# Buffer size constants
MAX_BUFFER_SIZE = 1024 * 1024  # 1MB
BUFFER_CLEANUP_THRESHOLD = MAX_BUFFER_SIZE // 2
LOG_SIZE_THRESHOLD = 100 * 1024  # 100KB

# Error handling constants
MAX_CLEANUP_ATTEMPTS = 3
MAX_PARSE_ERRORS = 5


@dataclass
class StreamConfig:
    """Configuration for stream behavior.

    Attributes:
        max_buffer_size: Maximum buffer size in bytes (default: 1MB)
        cleanup_threshold: Size at which to trigger cleanup (default: 512KB)
        chunk_size: Size of chunks for processing (default: 8KB)
        max_cleanup_attempts: Maximum number of cleanup attempts (default: 3)
        max_parse_errors: Maximum number of parse errors before failing (default: 5)
        log_size_threshold: Size change that triggers logging (default: 100KB)
    """

    max_buffer_size: int = MAX_BUFFER_SIZE  # 1MB max buffer size
    cleanup_threshold: int = BUFFER_CLEANUP_THRESHOLD  # Clean up at 512KB
    chunk_size: int = 8192  # 8KB chunks
    max_cleanup_attempts: int = MAX_CLEANUP_ATTEMPTS
    max_parse_errors: int = MAX_PARSE_ERRORS
    log_size_threshold: int = LOG_SIZE_THRESHOLD


class BufferError(StreamBufferError):
    """Base class for buffer-related errors."""

    pass


class BufferOverflowError(BufferError):
    """Raised when buffer size exceeds limits."""

    pass


class ParseError(BufferError):
    """Raised when JSON parsing fails repeatedly."""

    pass


class StreamBuffer:
    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        schema: Optional[Type[BaseModel]] = None,
    ):
        """Initialize the buffer with optional config and schema."""
        self.config = config or StreamConfig()
        if schema is not None and not issubclass(schema, BaseModel):
            raise ValueError("Schema must be a Pydantic model class")
        self.schema = schema
        self._buffer = StringIO()
        self.total_bytes = 0
        self._cleanup_stats: Dict[str, Union[None, int, str]] = {
            "strategy": None,
            "cleaned_bytes": 0,
        }
        self.parse_errors = 0
        self.cleanup_attempts = 0

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
        return self.total_bytes >= self.config.log_size_threshold

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
                    {"size_bytes": self.total_bytes},
                )

            current_content = self.getvalue()
            if self.schema is not None:
                try:
                    result = self.schema.model_validate_json(current_content)
                    self._log(
                        on_log,
                        logging.DEBUG,
                        "Successfully parsed complete object",
                    )
                    self.reset()
                    return result
                except ValidationError as e:
                    self._cleanup_stats.update(
                        {
                            "validation_error": str(e),
                            "content_length": len(current_content),
                        }
                    )
                    if self.total_bytes > self.config.cleanup_threshold:
                        self.cleanup()
                except json.JSONDecodeError as e:
                    self._cleanup_stats.update(
                        {
                            "json_error": str(e),
                            "error_position": int(e.pos),
                            "content_length": len(current_content),
                        }
                    )
                    if self.total_bytes > self.config.cleanup_threshold:
                        self.cleanup()
            return None
        except BufferError as e:
            self._cleanup_stats.update(
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "buffer_size": self.total_bytes,
                }
            )
            self._log(
                on_log,
                logging.ERROR,
                "Critical stream buffer error",
                {"error": str(e), "bytes": self.total_bytes},
            )
            raise

    def write(self, content: str) -> None:
        """Write content to the buffer with size checks and error handling."""
        try:
            content_bytes = len(content.encode())
            if self.total_bytes + content_bytes > self.config.max_buffer_size:
                raise BufferOverflowError(
                    f"Buffer would exceed max size of {self.config.max_buffer_size} bytes"
                )

            self._buffer.write(content)
            self.total_bytes += content_bytes

        except (IOError, OSError) as e:
            raise BufferError(f"Failed to write to buffer: {str(e)}")

    def getvalue(self) -> str:
        """Get the current buffer contents as a string with caching."""
        return self._buffer.getvalue()

    def cleanup(self) -> None:
        """Find and preserve the last complete JSON object using multiple strategies."""
        self.cleanup_attempts += 1
        if self.cleanup_attempts >= self.config.max_cleanup_attempts:
            raise ParseError(
                f"Exceeded maximum cleanup attempts ({self.config.max_cleanup_attempts})"
            )

        content = self.getvalue()
        if not content:
            return

        # Check if content contains JSON-like patterns anywhere
        content_stripped = content.strip()
        is_json_like = "{" in content_stripped or "[" in content_stripped

        original_length = len(content)
        original_bytes = len(content.encode("utf-8"))

        try:
            # Strategy 1: Use ijson for incremental parsing
            try:
                # Convert content to bytes and use BytesIO
                content_bytes = content.encode("utf-8")
                parser = ijson.parse(BytesIO(content_bytes))
                stack = []
                last_complete = 0
                current_pos = 0

                for _, event, _ in parser:
                    # Track nesting level without relying on position
                    if event in ("start_map", "start_array"):
                        stack.append(True)
                    elif event in ("end_map", "end_array"):
                        if stack:
                            stack.pop()
                            if not stack:  # Complete object found
                                # Use the current content up to this point
                                try:
                                    # Validate the content
                                    complete_content = content[:current_pos]
                                    json.loads(
                                        complete_content
                                    )  # Verify it's valid JSON
                                    if self.schema is not None:
                                        self.schema.model_validate_json(
                                            complete_content
                                        )
                                    last_complete = current_pos
                                    break
                                except (json.JSONDecodeError, ValidationError):
                                    continue
                    current_pos += 1

                if last_complete > 0:
                    self._update_buffer(content[last_complete:])
                    self._log_cleanup_stats(
                        original_length=original_length,
                        original_bytes=original_bytes,
                        cleanup_position=last_complete,
                        strategy="ijson_parsing",
                    )
                    return

            except ijson.JSONError as e:
                if is_json_like:
                    self._cleanup_stats.update(
                        {"json_error": str(e), "error_type": type(e).__name__}
                    )
                    self.parse_errors += 1
                    if (
                        self.cleanup_attempts == 1
                    ):  # Only raise on first attempt
                        raise ParseError(f"Failed to parse JSON content: {e}")
                    # On subsequent attempts, let max attempts check handle it

            # Strategy 2: Pattern matching as fallback
            if is_json_like:
                self.parse_errors += 1

                patterns = [
                    "}",  # Object end
                    "]",  # Array end
                    '"',  # String end
                    r"\btrue\b",  # Boolean true
                    r"\bfalse\b",  # Boolean false
                    r"\bnull\b",  # Null
                    r"\d+\.?\d*(?:[eE][+-]?\d+)?\s*(?:,|]|}|$)",  # Numbers
                ]

                # Try each pattern in order of complexity
                last_pos = -1
                for pattern in patterns:
                    matches = list(re.finditer(pattern, content))
                    if matches:
                        pos = matches[-1].end()
                        try:
                            complete_content = content[:pos]
                            # First validate JSON syntax
                            json.loads(complete_content)
                            # Then validate against schema if provided
                            if self.schema is not None:
                                self.schema.model_validate_json(
                                    complete_content
                                )
                            last_pos = pos
                            break
                        except json.JSONDecodeError as e:
                            context = self._get_error_context(content, e.pos)
                            self._cleanup_stats.update(
                                {
                                    "json_error": str(e),
                                    "error_position": int(e.pos),
                                    "error_context": context,
                                    "error_type": type(e).__name__,
                                }
                            )
                            continue
                        except ValidationError as e:
                            error_loc = e.errors()[0].get("loc", ["value"])[-1]
                            context = self._get_error_context(
                                complete_content, error_loc
                            )
                            self._cleanup_stats.update(
                                {
                                    "validation_error": str(e),
                                    "error_context": context,
                                    "error_type": type(e).__name__,
                                }
                            )
                            raise ParseError(
                                f"Schema validation failed: {e}\nContext: {context}"
                            )

                if last_pos > 0:
                    self._update_buffer(content[last_pos:])
                    self._log_cleanup_stats(
                        original_length=original_length,
                        original_bytes=original_bytes,
                        cleanup_position=last_pos,
                        strategy="pattern_matching",
                    )
                    return

                # If both strategies fail for JSON-like content, check max errors
                if self.parse_errors >= self.config.max_parse_errors:
                    context = self._get_error_context(
                        content, 0
                    )  # Show context from start
                    self._cleanup_stats.update(
                        {
                            "json_error": "Failed to find valid JSON content",
                            "error_context": context,
                            "error_type": "ParseError",
                        }
                    )
                    raise ParseError(
                        f"Failed to parse JSON content after {self.parse_errors} attempts.\nContext: {context}"
                    )

        except Exception as e:
            if isinstance(e, ParseError):
                raise e
            if isinstance(e, ValidationError):
                error_loc = e.errors()[0].get("loc", ["value"])[-1]
                context = self._get_error_context(content, error_loc)
                self._cleanup_stats.update(
                    {
                        "validation_error": str(e),
                        "error_context": context,
                        "error_type": type(e).__name__,
                    }
                )
                raise ParseError(
                    f"Schema validation failed: {e}\nContext: {context}"
                )

            self._cleanup_stats.update(
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_context": self._get_error_context(content, 0),
                }
            )
            raise ParseError(f"Error during cleanup: {e}")

    def _update_buffer(self, new_content: str) -> None:
        """Update buffer contents efficiently."""
        self._buffer.seek(0)
        self._buffer.truncate()
        self._buffer.write(new_content)
        self.total_bytes = len(new_content.encode("utf-8"))

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

    def reset(self) -> None:
        """Reset the buffer state while preserving configuration."""
        try:
            self._buffer.seek(0)
            self._buffer.truncate()
        except ValueError:  # Buffer is closed
            self._buffer = StringIO()  # Create new buffer
        self.total_bytes = 0
        self._cleanup_stats = {"strategy": None, "cleaned_bytes": 0}
        self.parse_errors = 0
        self.cleanup_attempts = 0

    def close(self) -> None:
        """Close the buffer and clean up resources."""
        if not self._buffer.closed:
            self._buffer.close()
        self.total_bytes = 0
        self._cleanup_stats = {"strategy": None, "cleaned_bytes": 0}
        self.parse_errors = 0
        self.cleanup_attempts = 0

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
