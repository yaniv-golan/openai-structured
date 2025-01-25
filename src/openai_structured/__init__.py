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

from .client import (
    StreamConfig,
    async_openai_structured_call,
    async_openai_structured_stream,
    openai_structured_call,
    openai_structured_stream,
    supports_structured_output,
)
from .errors import (
    APIResponseError,
    BufferOverflowError,
    EmptyResponseError,
    InvalidResponseFormatError,
    JSONParseError,
    ModelNotSupportedError,
    ModelVersionError,
    OpenAIClientError,
    StreamBufferError,
    StreamInterruptedError,
    StreamParseError,
)
from .model_version import ModelVersion, parse_model_version

__all__ = [
    # Main functions
    "openai_structured_call",
    "openai_structured_stream",
    "async_openai_structured_call",
    "async_openai_structured_stream",
    "supports_structured_output",
    # Configuration
    "StreamConfig",
    # Type hints
    "ModelVersion",
    # Utility functions
    "parse_model_version",
    # Exceptions
    "OpenAIClientError",
    "ModelNotSupportedError",
    "ModelVersionError",
    "APIResponseError",
    "InvalidResponseFormatError",
    "EmptyResponseError",
    "JSONParseError",
    "StreamInterruptedError",
    "StreamBufferError",
    "StreamParseError",
    "BufferOverflowError",
]
