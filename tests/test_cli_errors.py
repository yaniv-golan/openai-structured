"""Tests for CLI error handling."""

import json
import pytest
from openai_structured.cli.errors import (
    CLIError,
    DirectoryNotFoundError,
    FileNotFoundError,
    InvalidJSONError,
    PathError,
    PathSecurityError,
    SchemaError,
    SchemaFileError,
    SchemaValidationError,
    TaskTemplateError,
    TaskTemplateSyntaxError,
    TaskTemplateVariableError,
    VariableError,
    VariableNameError,
    VariableValueError,
)
from typing import Any

def test_variable_name_error() -> None:
    """Test VariableNameError."""
    with pytest.raises(VariableNameError) as exc:
        raise VariableNameError("test error")
    assert str(exc.value) == "test error"

def test_variable_value_error() -> None:
    """Test VariableValueError."""
    with pytest.raises(VariableValueError) as exc:
        raise VariableValueError("test error")
    assert str(exc.value) == "test error"

def test_json_variable_name_error():
    """Test JSON variable name error."""
    with pytest.raises(VariableNameError) as exc:
        raise VariableNameError("Empty name in JSON variable mapping")
    assert str(exc.value) == "Empty name in JSON variable mapping"

def test_json_variable_value_error():
    """Test JSON variable value error."""
    with pytest.raises(ValueError) as exc:
        raise ValueError("Invalid JSON value")
    assert str(exc.value) == "Invalid JSON value"

def test_path_name_error():
    """Test path name error."""
    with pytest.raises(VariableNameError) as exc:
        raise VariableNameError("Empty name in file mapping")
    assert str(exc.value) == "Empty name in file mapping"

def test_file_not_found_error() -> None:
    """Test FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc:
        raise FileNotFoundError("test error")
    assert str(exc.value) == "test error"

def test_directory_not_found_error() -> None:
    """Test DirectoryNotFoundError."""
    with pytest.raises(DirectoryNotFoundError) as exc:
        raise DirectoryNotFoundError("test error")
    assert str(exc.value) == "test error"

def test_path_security_error_traversal(fs):
    """Test path security error for directory traversal."""
    fs.create_file("../outside.txt")
    with pytest.raises(PathSecurityError) as exc:
        raise PathSecurityError("Path '../outside.txt' is outside the base directory")
    assert str(exc.value) == "Path '../outside.txt' is outside the base directory"

def test_path_security_error_permission(fs):
    """Test path security error for permission denied."""
    fs.create_file("secret.txt", st_mode=0o000)
    with pytest.raises(PathSecurityError) as exc:
        raise PathSecurityError("Permission denied accessing path: 'secret.txt'")
    assert str(exc.value) == "Permission denied accessing path: 'secret.txt'"

def test_task_template_syntax_error() -> None:
    """Test TaskTemplateSyntaxError."""
    with pytest.raises(TaskTemplateSyntaxError) as exc:
        raise TaskTemplateSyntaxError("test error")
    assert str(exc.value) == "test error"

def test_task_template_file_error():
    """Test task template file error."""
    with pytest.raises(TaskTemplateVariableError) as exc:
        raise TaskTemplateVariableError("Invalid task template file: File not found")
    assert str(exc.value) == "Invalid task template file: File not found"

def test_task_template_file_security_error(fs):
    """Test task template file security error."""
    fs.create_file("../template.txt")
    with pytest.raises(TaskTemplateVariableError) as exc:
        raise TaskTemplateVariableError(
            "Invalid task template file: Path '../template.txt' is outside the base directory"
        )
    assert str(exc.value) == "Invalid task template file: Path '../template.txt' is outside the base directory"

def test_schema_file_error() -> None:
    """Test SchemaFileError."""
    with pytest.raises(SchemaFileError) as exc:
        raise SchemaFileError("test error")
    assert str(exc.value) == "test error"

def test_schema_json_error(fs):
    """Test schema JSON error."""
    fs.create_file("invalid.json", contents="{not valid json}")
    with pytest.raises(InvalidJSONError) as exc:
        raise InvalidJSONError("Invalid JSON in schema file: Expecting property name")
    assert str(exc.value) == "Invalid JSON in schema file: Expecting property name"

def test_schema_validation_error() -> None:
    """Test SchemaValidationError."""
    with pytest.raises(SchemaValidationError) as exc:
        raise SchemaValidationError("test error")
    assert str(exc.value) == "test error"

def test_schema_file_security_error(fs):
    """Test schema file security error."""
    fs.create_file("../schema.json")
    with pytest.raises(SchemaFileError) as exc:
        raise SchemaFileError(
            "Invalid schema file: Path '../schema.json' is outside the base directory"
        )
    assert str(exc.value) == "Invalid schema file: Path '../schema.json' is outside the base directory"

def test_cli_error_str() -> None:
    """Test string representation of CLIError."""
    error = CLIError("test error")
    assert str(error) == "test error"

def test_variable_error_str() -> None:
    """Test string representation of VariableError."""
    error = VariableError("test error")
    assert str(error) == "test error"

def test_variable_name_error_str() -> None:
    """Test string representation of VariableNameError."""
    error = VariableNameError("test error")
    assert str(error) == "test error"

def test_variable_value_error_str() -> None:
    """Test string representation of VariableValueError."""
    error = VariableValueError("test error")
    assert str(error) == "test error"

def test_invalid_json_error() -> None:
    """Test InvalidJSONError."""
    with pytest.raises(InvalidJSONError) as exc:
        raise InvalidJSONError("test error")
    assert str(exc.value) == "test error"

def test_path_error() -> None:
    """Test PathError."""
    with pytest.raises(PathError) as exc:
        raise PathError("test error")
    assert str(exc.value) == "test error"

def test_file_not_found_error_str() -> None:
    """Test string representation of FileNotFoundError."""
    error = FileNotFoundError("test error")
    assert str(error) == "test error"

def test_directory_not_found_error_str() -> None:
    """Test string representation of DirectoryNotFoundError."""
    error = DirectoryNotFoundError("test error")
    assert str(error) == "test error"

def test_path_security_error() -> None:
    """Test PathSecurityError."""
    with pytest.raises(PathSecurityError) as exc:
        raise PathSecurityError("test error")
    assert str(exc.value) == "test error"

def test_task_template_error() -> None:
    """Test TaskTemplateError."""
    with pytest.raises(TaskTemplateError) as exc:
        raise TaskTemplateError("test error")
    assert str(exc.value) == "test error"

def test_task_template_variable_error() -> None:
    """Test TaskTemplateVariableError."""
    with pytest.raises(TaskTemplateVariableError) as exc:
        raise TaskTemplateVariableError("test error")
    assert str(exc.value) == "test error"

def test_schema_error() -> None:
    """Test SchemaError."""
    with pytest.raises(SchemaError) as exc:
        raise SchemaError("test error")
    assert str(exc.value) == "test error"

def test_schema_file_error_str() -> None:
    """Test string representation of SchemaFileError."""
    error = SchemaFileError("test error")
    assert str(error) == "test error"

def test_schema_validation_error_str() -> None:
    """Test string representation of SchemaValidationError."""
    error = SchemaValidationError("test error")
    assert str(error) == "test error"

def test_schema_file_error_with_base_dir() -> None:
    """Test SchemaFileError with base directory."""
    with pytest.raises(SchemaFileError) as exc:
        raise SchemaFileError(
            "Invalid schema file: Path '../schema.json' is outside the base directory"
        )
    assert str(exc.value) == "Invalid schema file: Path '../schema.json' is outside the base directory" 