"""Example schemas and usage patterns for openai-structured.

This module provides example schemas and patterns that demonstrate how to use
the openai-structured library effectively.

Example Schemas
-------------
- SimpleMessage: Basic schema with a single message field
- SentimentMessage: More complex schema with validation and field descriptions

Usage
-----
>>> from openai_structured.examples.schemas import SimpleMessage
>>> 
>>> # Use in your code
>>> result = openai_structured(
...     client=client,
...     model="gpt-4o",
...     output_schema=SimpleMessage,
...     user_prompt="Hello"
... )
>>> print(result.message)
"""

from openai_structured.examples.schemas import SimpleMessage, SentimentMessage

__all__ = ["SimpleMessage", "SentimentMessage"]
