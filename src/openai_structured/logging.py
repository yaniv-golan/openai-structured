"""Standardized logging for OpenAI structured client."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel

# Type alias for log callback function
LogCallback = Callable[[int, str, Dict[str, Any]], None]

# Create module logger
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Standard log levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class LogEvent(Enum):
    """Standard event types for structured logging."""

    # Error events
    ERROR = "error"
    STREAM_ERROR = "stream_error"

    # Stream lifecycle events
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_COMPLETE = "stream_complete"

    # Request lifecycle events
    REQUEST_START = "request_start"
    REQUEST_COMPLETE = "request_complete"

    # Registry events
    MODEL_REGISTRY = "model_registry"  # For model registry operations

    # Validation events
    MODEL_VALIDATION = (
        "model_validation"  # For validating model properties/constraints
    )
    PARAMETER_VALIDATION = "parameter_validation"
    TOKEN_VALIDATION = "token_validation"


def _log(
    on_log: Optional[LogCallback],
    level: LogLevel,
    event: LogEvent,
    details: Dict[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Unified logging with structured data.

    Args:
        on_log: Optional callback for external logging
        level: Log level from LogLevel enum
        event: Event type from LogEvent enum
        details: Event-specific details
        extra: Optional additional context
    """
    if on_log:  # Only log if callback is provided
        data = {
            "event": event.value,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "openai_structured",
            **details,
        }
        if extra:
            data.update(extra)
        on_log(level.value, event.value, data)


def _get_request_details(
    model: str,
    output_schema: Type[BaseModel],
    params: Dict[str, Any],
    *,
    is_stream: bool = False,
) -> Dict[str, Any]:
    """Get standardized request details for logging.

    Args:
        model: Model name
        output_schema: Output schema class
        params: Request parameters
        is_stream: Whether this is a streaming request

    Returns:
        Dict with request details
    """
    # Filter sensitive or noisy data
    filtered_params = {
        k: v
        for k, v in params.items()
        if k
        not in {
            "client",
            "on_log",
            "stream_config",
            "system_prompt",
            "user_prompt",
        }
    }

    return {
        "model": model,
        "schema": output_schema.__name__,
        "is_stream": is_stream,
        "params": filtered_params,
    }


def _log_chunk(
    on_log: Optional[LogCallback],
    chunk_index: int,
    status: str,
    *,
    content: Optional[str] = None,
    error: Optional[Exception] = None,
    response_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log stream chunk processing.

    Args:
        on_log: Optional callback for external logging
        chunk_index: Index of current chunk
        status: Processing status (e.g., "received", "processed", "error")
        content: Optional chunk content
        error: Optional error if status is "error"
        response_id: Optional response ID
        extra: Optional additional context
    """
    details = {
        "chunk_index": chunk_index,
        "status": status,
    }

    if content:
        details["content"] = content
    if error:
        details["error"] = str(error)
        details["error_type"] = type(error).__name__
    if response_id:
        details["response_id"] = response_id
    if extra:
        details.update(extra)

    level = LogLevel.ERROR if error else LogLevel.DEBUG
    _log(on_log, level, LogEvent.STREAM_CHUNK, details)
