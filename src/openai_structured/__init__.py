# src/openai_structured/__init__.py
"""
openai-structured
=================

A Python library for structured output from OpenAI's API.

:noindex:
"""
__version__ = "0.5.0"  # Follow Semantic Versioning

from .client import (
    openai_structured_call,
    openai_structured_stream,
    supports_structured_output,
)
from .errors import (
    BufferOverflowError,
    EmptyResponseError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
    ModelVersionError,
    OpenAIClientError,
)

__all__ = [
    "openai_structured_call",
    "openai_structured_stream",
    "supports_structured_output",
    "OpenAIClientError",
    "ModelNotSupportedError",
    "ModelVersionError",
    "EmptyResponseError",
    "InvalidResponseFormatError",
    "BufferOverflowError",
]
