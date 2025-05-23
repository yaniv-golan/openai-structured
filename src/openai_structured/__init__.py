# src/openai_structured/__init__.py
"""
openai-structured
=================

A Python library for structured output from OpenAI's API.

:noindex:
"""
from importlib.metadata import version

try:
    __version__ = version("openai-structured")
except Exception:
    __version__ = "unknown"

# Import types from openai-model-registry library
from openai_model_registry import ModelVersion
from openai_model_registry.errors import (
    InvalidDateError,
    ModelFormatError,
    ModelNotSupportedError,
    ModelVersionError,
    VersionTooOldError,
)

from .client import (
    StreamConfig,
    async_openai_structured_call,
    async_openai_structured_stream,
    get_context_window_limit,
    get_default_token_limit,
    openai_structured_call,
    openai_structured_stream,
    supports_structured_output,
)
from .errors import (
    APIResponseError,
    BufferOverflowError,
    ClosedBufferError,
    ConnectionTimeoutError,
    EmptyResponseError,
    InvalidResponseFormatError,
    JSONParseError,
    OpenAIClientError,
    StreamBufferError,
    StreamInterruptedError,
    StreamParseError,
    TokenLimitError,
    TokenParameterError,
)

__all__ = [
    # Main functions
    "openai_structured_call",
    "openai_structured_stream",
    "async_openai_structured_call",
    "async_openai_structured_stream",
    "supports_structured_output",
    # Token limit functions
    "get_context_window_limit",
    "get_default_token_limit",
    # Configuration
    "StreamConfig",
    # Type hints
    "ModelVersion",
    # Exceptions
    "OpenAIClientError",
    "ModelNotSupportedError",
    "ModelVersionError",
    "ModelFormatError",
    "InvalidDateError",
    "VersionTooOldError",
    "APIResponseError",
    "InvalidResponseFormatError",
    "EmptyResponseError",
    "JSONParseError",
    "StreamInterruptedError",
    "StreamBufferError",
    "ClosedBufferError",
    "BufferOverflowError",
    "StreamParseError",
    "ConnectionTimeoutError",
    "TokenLimitError",
    "TokenParameterError",
]
