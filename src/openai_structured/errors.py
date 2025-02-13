# src/openai_structured/errors.py
import re
from typing import Any, Optional


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
    """Raised when a model is not supported by the client.

    This error indicates that the requested model is not in the registry of
    supported models. This is different from version-related errors, which
    indicate that the model exists but the specific version is invalid.

    Examples:
        >>> try:
        ...     registry.get_capabilities("unsupported-model")
        ... except ModelNotSupportedError as e:
        ...     print(f"Model {e.model} is not supported")
    """

    def __init__(self, message: str, model: Optional[str] = None):
        super().__init__(message)
        self.model = model
        self.message = message

    def __str__(self) -> str:
        return self.message


class ModelVersionError(OpenAIClientError):
    """Base class for version-related errors.

    This class serves as the base for all errors related to model versions.
    It inherits directly from OpenAIClientError (not ModelNotSupportedError)
    because version errors are distinct from model support errors:

    1. ModelNotSupportedError means the model itself is not supported
    2. ModelVersionError means the model exists but there's a version-specific issue:
       - InvalidVersionFormatError: The version string format is wrong
       - InvalidDateError: The date components are invalid
       - VersionTooOldError: The version is older than the minimum supported

    Examples:
        >>> try:
        ...     registry.get_capabilities("gpt-4o-2024-07-01")
        ... except ModelVersionError as e:
        ...     print(f"Version error for model {e.model}: {e}")
    """

    def __init__(self, model: str, min_version: Any):
        super().__init__(
            "Model " + model + " requires minimum version " + str(min_version)
        )
        self.model = model
        self.min_version = min_version


class InvalidVersionFormatError(ModelVersionError):
    """Raised when a model version string is not in the correct format.

    The version string must be in the format: <model>-YYYY-MM-DD
    where:
    - <model> is the base model name (e.g., "gpt-4o")
    - YYYY is a four-digit year (2000 or later)
    - MM is a two-digit month (01-12)
    - DD is a two-digit day (01-31)

    Examples:
        >>> try:
        ...     ModelVersion.parse_version_string("gpt-4o-2024")  # Missing month/day
        ... except InvalidVersionFormatError as e:
        ...     print(f"Invalid format: {e}")  # "Invalid format: Version must be in format: <model>-YYYY-MM-DD"

        >>> try:
        ...     ModelVersion.parse_version_string("gpt-4o-2024-13-01")  # Invalid month
        ... except InvalidVersionFormatError as e:
        ...     print(f"Invalid format: {e}")  # "Invalid format: Month must be between 1 and 12"
    """

    def __init__(self, model: str, reason: str):
        super().__init__(model, None)
        self.model = model
        self.reason = reason
        self.message = (
            "Invalid version format for model " + model + ": " + reason
        )

    def __str__(self) -> str:
        return self.message


class InvalidDateError(ModelVersionError):
    """Raised when a model version has invalid date components.

    This error occurs when the date components in a model version string
    are invalid, such as invalid month/day values or incorrect format.

    Examples:
        >>> try:
        ...     registry.get_capabilities("gpt-4o-2024-13-01")  # Invalid month
        ... except InvalidDateError as e:
        ...     print(f"Invalid date: {e}")  # "Invalid date: Model gpt-4o-2024-13-01 has invalid version 2024-13-01"
    """

    def __init__(self, model: str, version: str, message: str):
        super().__init__(model, version)
        self.model = model
        self.version = version
        self.message = message

    def __str__(self) -> str:
        return self.message


class VersionTooOldError(ModelVersionError):
    """Raised when a model version is older than the minimum supported version.

    Each model in the registry can specify a minimum supported version. This error
    occurs when trying to use a version that is older than the minimum.

    Examples:
        >>> try:
        ...     # If gpt-4o requires version 2024-08-06 or later
        ...     registry.get_capabilities("gpt-4o-2024-07-01")
        ... except VersionTooOldError as e:
        ...     print(f"Version too old: {e}")  # "Version too old: Model gpt-4o version 2024-07-01 is too old"
        ...     # "(minimum supported version is 2024-08-06)"
    """

    def __init__(self, model: str, version: str, min_version: str):
        super().__init__(model, min_version)
        self.model = model
        self.min_version = min_version
        self.version = version
        # Extract base model using regex pattern matching
        version_match = re.match(r"^(.*)-\d{4}-\d{2}-\d{2}$", model)
        base_model = (
            version_match.group(1) if version_match else model.split("-")[0]
        )
        self.message = (
            "Model '" + model + "' version " + version + " is too old.\n"
            "Minimum supported version: " + min_version + "\n"
            "Note: Use the alias '"
            + base_model
            + "' to always get the latest version"
        )

    def __str__(self) -> str:
        return self.message


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
