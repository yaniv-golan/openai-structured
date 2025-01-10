# src/openai_structured/__init__.py
"""
openai-structured
=================

A Python library for structured output from OpenAI's API.

:noindex:
"""
__version__ = "0.1.0"  # Follow Semantic Versioning

from .client import openai_structured_call, openai_structured_stream
from .errors import (
    APIResponseError,
    EmptyResponseError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
    OpenAIClientError,
)
