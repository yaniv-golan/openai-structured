# src/openai_structured/__init__.py
"""
openai-structured
=================

A Python library for structured output from OpenAI's API.

:noindex:
"""
__version__ = "0.6.0"  # Follow Semantic Versioning

from .cli.cli import ExitCode, main
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
    # CLI
    "main",
    "ExitCode",
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
