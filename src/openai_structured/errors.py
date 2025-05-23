# src/openai_structured/errors.py
from typing import Optional


class OpenAIClientError(Exception):
    """Base class for exceptions in the OpenAI client.

    This is the root exception class for all errors that can occur when using
    the OpenAI client. All other error classes in this module inherit from this.

    Examples:
        >>> try:
        ...     result = openai_structured_call(...)
        ... except OpenAIClientError as e:
        ...     print(f"OpenAI client error: {e}")
    """

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
    """Raised when there's a security concern with a path."""

    pass


class TaskTemplateError(CLIError):
    """Base class for task template-related errors."""

    pass


class TaskTemplateVariableError(TaskTemplateError):
    """Raised when there's an issue with a task template variable."""

    pass


class TaskTemplateSyntaxError(TaskTemplateError):
    """Raised when there's a syntax error in a task template."""

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


class APIResponseError(OpenAIClientError):
    """Base class for API response-related errors."""

    def __init__(
        self,
        message: str,
        response_id: Optional[str] = None,
        content: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
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


class StreamBufferError(OpenAIClientError):
    """Base class for stream buffer related errors."""

    pass


class ClosedBufferError(StreamBufferError):
    """Raised when attempting to write to a closed buffer."""

    pass


class BufferOverflowError(StreamBufferError):
    """Raised when the buffer exceeds size limits."""

    def __init__(self, message: str):
        super().__init__(message)


class StreamParseError(StreamInterruptedError):
    """Raised when stream content cannot be parsed after multiple attempts."""

    def __init__(self, message: str, attempts: int, last_error: Exception):
        super().__init__(
            message
            + " after "
            + str(attempts)
            + " attempts. Last error: "
            + str(last_error)
        )
        self.attempts = attempts
        self.last_error = last_error


class ConnectionTimeoutError(OpenAIClientError):
    """Raised when a request times out."""

    def __init__(
        self,
        message: str = "Request timed out",
        timeout: Optional[float] = None,
    ):
        super().__init__(message)
        self.timeout = timeout


class TokenLimitError(OpenAIClientError):
    """Raised when token limits are exceeded."""

    def __init__(
        self,
        message: str,
        requested_tokens: Optional[int] = None,
        model_limit: Optional[int] = None,
    ):
        super().__init__(message)
        self.requested_tokens = requested_tokens
        self.model_limit = model_limit


class TokenParameterError(OpenAIClientError):
    """Raised when both max_output_tokens and max_completion_tokens are used.

    These parameters are mutually exclusive as they control the same functionality.
    Only one should be used in a request.

    Examples:
        >>> try:
        ...     client.complete(
        ...         "gpt-4o",
        ...         max_output_tokens=100,
        ...         max_completion_tokens=100
        ...     )
        ... except TokenParameterError as e:
        ...     print(f"Token error: {e}")  # "Token error: Cannot use both max_output_tokens and max_completion_tokens"
    """

    def __init__(self, model: str):
        self.model = model
        self.message = (
            "Cannot specify both 'max_output_tokens' and 'max_completion_tokens' parameters.\n"
            "These parameters are mutually exclusive as they control the same functionality.\n"
            "Choose one:\n"
            "- max_output_tokens (recommended)\n"
            "- max_completion_tokens (legacy)"
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
