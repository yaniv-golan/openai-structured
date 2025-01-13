# src/openai_structured/errors.py
from typing import Optional

from .model_version import ModelVersion


class OpenAIClientError(Exception):
    """Base class for exceptions in the OpenAI client."""

    pass


class ModelNotSupportedError(OpenAIClientError):
    """Raised when the provided model is not supported."""

    pass


class ModelVersionError(ModelNotSupportedError):
    """Raised when the model version is not supported."""

    def __init__(self, model: str, min_version: ModelVersion) -> None:
        super().__init__(
            f"Model '{model}' version is not supported. "
            f"Minimum version required: {min_version}"
        )


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
    """Raised if the API doesn't provide the expected JSON format."""

    pass


class EmptyResponseError(APIResponseError):
    """Raised if the API returns an empty response."""

    pass


class JSONParseError(InvalidResponseFormatError):
    """Raised when JSON parsing fails."""

    pass


class StreamInterruptedError(OpenAIClientError):
    """Raised when a stream is interrupted unexpectedly."""

    pass


class StreamBufferError(StreamInterruptedError):
    """Base class for stream buffer related errors."""

    pass


class StreamParseError(StreamInterruptedError):
    """Raised when stream content cannot be parsed after multiple attempts."""

    def __init__(self, message: str, attempts: int, last_error: Exception):
        super().__init__(
            f"{message} after {attempts} attempts. Last error: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


class BufferOverflowError(StreamBufferError):
    """Raised when the buffer exceeds size limits."""

    pass
