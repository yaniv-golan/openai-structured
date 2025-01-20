"""Custom error classes for CLI error handling."""

from pathlib import Path
from typing import Optional

class CLIError(Exception):
    """Base class for all CLI errors."""
    pass

class VariableError(CLIError):
    """Base class for variable-related errors."""
    pass

class VariableNameError(VariableError):
    """Raised when a variable name is invalid or empty."""
    pass

class VariableValueError(VariableError):
    """Raised when a variable value is invalid or missing."""
    pass

class InvalidJSONError(VariableError):
    """Raised when JSON parsing fails for a variable value."""
    pass

class PathError(CLIError):
    """Base class for path-related errors."""
    pass

class FileNotFoundError(PathError):
    """Raised when a specified file does not exist."""
    pass

class DirectoryNotFoundError(PathError):
    """Raised when a specified directory does not exist."""
    pass

class PathSecurityError(Exception):
    """Error raised for security violations in path access.
    
    Provides standardized error messages for different types of security violations:
    - Access denied (general)
    - Outside allowed directories
    - Directory traversal attempts
    - Invalid path access
    """
    
    @classmethod
    def access_denied(cls, path: Path, reason: Optional[str] = None) -> 'PathSecurityError':
        """Create access denied error.
        
        Args:
            path: Path that was denied
            reason: Optional reason for denial
            
        Returns:
            PathSecurityError with standardized message
        """
        msg = f"Access denied: {path}"
        if reason:
            msg += f" - {reason}"
        return cls(msg)
    
    @classmethod
    def outside_allowed(cls, path: Path, base_dir: Optional[Path] = None) -> 'PathSecurityError':
        """Create error for path outside allowed directories.
        
        Args:
            path: Path that was outside
            base_dir: Optional base directory for context
            
        Returns:
            PathSecurityError with standardized message
        """
        msg = f"Access denied: {path} is outside allowed directories"
        if base_dir:
            msg += f" (base: {base_dir})"
        return cls(msg)
    
    @classmethod
    def traversal_attempt(cls, path: Path) -> 'PathSecurityError':
        """Create error for directory traversal attempt.
        
        Args:
            path: Path that attempted traversal
            
        Returns:
            PathSecurityError with standardized message
        """
        return cls(f"Access denied: {path} - directory traversal not allowed")

class TaskTemplateError(CLIError):
    """Base class for task template-related errors."""
    pass

class TaskTemplateSyntaxError(TaskTemplateError):
    """Raised when a task template has invalid syntax."""
    pass

class TaskTemplateVariableError(TaskTemplateError):
    """Raised when a task template uses undefined variables."""
    pass

class SystemPromptError(TaskTemplateError):
    """Raised when there are issues with system prompt loading or processing."""
    pass

class SchemaError(CLIError):
    """Base class for schema-related errors."""
    pass

class SchemaFileError(SchemaError):
    """Raised when a schema file is invalid or inaccessible."""
    pass

class SchemaValidationError(SchemaError):
    """Raised when a schema fails validation."""
    pass 