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

from .cli import (
    ExitCode,
    ProgressContext,
    create_dynamic_model,
    estimate_tokens_for_chat,
    get_context_window_limit,
    get_default_token_limit,
    main,
    supports_structured_output,
    validate_template_placeholders,
    validate_token_limits,
)
from .client import (
    StreamConfig,
    async_openai_structured_call,
    async_openai_structured_stream,
    openai_structured_call,
    openai_structured_stream,
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
    "ExitCode",
    "main",
    "create_dynamic_model",
    "validate_template_placeholders",
    "estimate_tokens_for_chat",
    "get_context_window_limit",
    "get_default_token_limit",
    "validate_token_limits",
    "ProgressContext",
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
