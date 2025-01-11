# src/openai_structured/errors.py
from typing import Optional


class OpenAIClientError(Exception):
    """Base class for exceptions in the OpenAI client."""

    pass


class ModelNotSupportedError(OpenAIClientError):
    """Raised when the provided model is not supported."""

    pass


class APIResponseError(OpenAIClientError):
    """Raised for errors in the API response."""

    def __init__(
        self,
        message: str,
        response_id: Optional[str] = None,
        content: Optional[str] = None,
    ):
        super().__init__(message)
        self.response_id = response_id
        self.content = content


class InvalidResponseFormatError(APIResponseError):
    """Raised if the API doesn't provide the expected JSON format"""

    pass


class EmptyResponseError(APIResponseError):
    """Raised if the API returns an empty response"""

    pass


class StreamProcessingError(OpenAIClientError):
    """Raised when an error occurs during stream processing."""

    pass


class BufferOverflowError(OpenAIClientError):
    """Raised when the buffer size exceeds the maximum allowed size."""

    pass
