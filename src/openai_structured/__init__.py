# src/openai_structured/__init__.py
"""
openai-structured
=================

A Python library for structured output from OpenAI's API.

:noindex:
"""
__version__ = "0.2.0"  # Follow Semantic Versioning

from .client import openai_structured_call, openai_structured_stream
from .errors import (
    APIResponseError,
    BufferOverflowError,
    EmptyResponseError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
    OpenAIClientError,
    StreamProcessingError,
)

__all__ = [
    "openai_structured_call",
    "openai_structured_stream",
    "APIResponseError",
    "EmptyResponseError",
    "InvalidResponseFormatError",
    "ModelNotSupportedError",
    "OpenAIClientError",
    "StreamProcessingError",
    "BufferOverflowError",
]
