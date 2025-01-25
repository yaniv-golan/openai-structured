# src/openai_structured/errors.py
from typing import Optional

from .model_version import ModelVersion


class OpenAIClientError(Exception):
    """Base class for exceptions in the OpenAI client."""

    pass


class CLIError(Exception):
    """Base class for CLI-related exceptions."""

    pass


class VariableError(CLIError):
    """Base class for variable-related errors."""

    pass


class VariableNameError(VariableError):
    """Raised when a variable name is invalid."""

    pass


class VariableValueError(VariableError):
    """Raised when a variable value is invalid."""

    pass


class InvalidJSONError(VariableError):
    """Raised when JSON variable value is invalid."""

    pass


class PathError(CLIError):
    """Base class for path-related errors."""

    pass


class FileNotFoundError(PathError):
    """Raised when a file is not found."""

    pass


class DirectoryNotFoundError(PathError):
    """Raised when a directory is not found."""

    pass


class PathSecurityError(PathError):
    """Raised when a path is outside the allowed directory."""

    pass


class TaskTemplateError(CLIError):
    """Base class for task template errors."""

    pass


class TaskTemplateVariableError(TaskTemplateError):
    """Raised when a template uses undefined variables."""

    pass


class TaskTemplateSyntaxError(TaskTemplateError):
    """Raised when a template has invalid syntax."""

    pass


class SchemaError(CLIError):
    """Base class for schema-related errors."""

    pass


class SchemaValidationError(SchemaError):
    """Raised when schema validation fails."""

    pass


class SchemaFileError(SchemaError):
    """Raised when there are issues with the schema file."""

    pass


class ModelNotSupportedError(OpenAIClientError):
    """Raised when a model is not supported."""

    def __init__(self, message: str, model: Optional[str] = None):
        super().__init__(message)
        self.model = model


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

    def __init__(self, message: str, chunk_index: Optional[int] = None):
        super().__init__(message)
        self.chunk_index = chunk_index


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


class ConnectionTimeoutError(OpenAIClientError):
    """Raised when a connection times out during testing.
    
    This error simulates network timeouts in test scenarios.
    """
    def __init__(self, message: str = "Request timed out", timeout: float = None):
        self.timeout = timeout
        super().__init__(message)
