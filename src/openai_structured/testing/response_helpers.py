"""Response helpers for testing.

This module provides utilities for generating mock responses with schema validation
and support for nested Pydantic models.
"""

from typing import Any, Dict, List, Type, TypeVar, Union
from unittest.mock import MagicMock

from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

def create_structured_response(
    output_schema: Type[T],
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    validate: bool = True
) -> MagicMock:
    """Create a mock response that matches a Pydantic schema.
    
    Args:
        output_schema: The Pydantic model class to validate against
        data: Dict or list of dicts to convert to response
        validate: Whether to validate against schema before creating response
        
    Returns:
        MagicMock configured with valid schema response
        
    Raises:
        ValidationError: If validate=True and data doesn't match schema
    """
    # Handle single dict vs list of dicts
    if isinstance(data, dict):
        if validate:
            # Validate against schema
            output_schema(**data)  # Will raise ValidationError if invalid
        json_str = output_schema(**data).model_dump_json()
    else:
        if validate:
            # Validate each item
            for d in data:
                output_schema(**d)
        json_str = [output_schema(**d).model_dump_json() for d in data]
    
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=json_str))]
    return mock

def create_invalid_response(
    output_schema: Type[T],
    error_type: str = "missing_field",
    field_path: str = None
) -> MagicMock:
    """Create a mock response that fails schema validation.
    
    Args:
        output_schema: The Pydantic model class to validate against
        error_type: Type of validation error to simulate:
            - "missing_field": Required field is missing
            - "wrong_type": Field has incorrect type
            - "invalid_enum": Invalid enum value
            - "pattern_mismatch": String doesn't match pattern
            - "nested_error": Error in nested model
        field_path: Dot notation path to field with error (e.g. "user.address.city")
        
    Returns:
        MagicMock configured to trigger schema validation error
    """
    error_patterns = {
        "missing_field": lambda schema: {
            k: "test" for k, v in schema.model_fields.items() 
            if k != list(schema.model_fields.keys())[0]  # Omit first field
        },
        "wrong_type": lambda schema: {
            k: 123 if v.annotation == str else "wrong"
            for k, v in schema.model_fields.items()
        },
        "invalid_enum": lambda schema: {
            k: "INVALID" if hasattr(v.annotation, "__members__") else "test"
            for k, v in schema.model_fields.items()
        },
        "pattern_mismatch": lambda schema: {
            k: "invalid!" if getattr(v, "pattern", None) else "test"
            for k, v in schema.model_fields.items()
        }
    }
    
    # Get invalid data based on error type
    if error_type == "nested_error" and field_path:
        # Handle nested model errors
        parts = field_path.split(".")
        data = {}
        current = data
        for i, part in enumerate(parts[:-1]):
            current[part] = {}
            current = current[part]
        current[parts[-1]] = "invalid"
    else:
        pattern = error_patterns.get(error_type, error_patterns["missing_field"])
        data = pattern(output_schema)
    
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=str(data)))]
    return mock 