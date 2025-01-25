"""Tests for testing utilities.

This module verifies the functionality of the testing utilities provided by
openai_structured.testing.
"""

import pytest
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

from openai_structured.testing import (
    create_structured_response,
    create_invalid_response,
    create_structured_stream_response,
    create_invalid_stream_response,
    create_error_response,
    create_rate_limit_response,
    create_timeout_response,
    MockAPIError,
)

# Test Models
class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class Address(BaseModel):
    street: str
    city: str
    zip_code: str = Field(pattern=r"^\d{5}$")

class User(BaseModel):
    name: str
    age: int
    status: Status
    address: Address
    tags: Optional[List[str]] = None

# Test Data
valid_user = {
    "name": "Test User",
    "age": 30,
    "status": "active",
    "address": {
        "street": "123 Test St",
        "city": "Test City",
        "zip_code": "12345"
    }
}

# Response Helper Tests
def test_create_structured_response_valid():
    """Test creating valid structured responses."""
    mock = create_structured_response(User, valid_user)
    assert mock.choices[0].message.content is not None
    # Should not raise ValidationError
    User.model_validate_json(mock.choices[0].message.content)

def test_create_structured_response_invalid():
    """Test validation catches invalid data."""
    invalid_user = {**valid_user, "age": "not a number"}
    with pytest.raises(ValueError):  # Pydantic raises ValueError for validation errors
        create_structured_response(User, invalid_user)

def test_create_invalid_response_missing_field():
    """Test generating response with missing required field."""
    mock = create_invalid_response(User, error_type="missing_field")
    content = mock.choices[0].message.content
    with pytest.raises(ValueError):
        User.model_validate_json(content)

def test_create_invalid_response_wrong_type():
    """Test generating response with wrong field type."""
    mock = create_invalid_response(User, error_type="wrong_type")
    content = mock.choices[0].message.content
    with pytest.raises(ValueError):
        User.model_validate_json(content)

def test_create_invalid_response_nested_error():
    """Test generating response with nested model error."""
    mock = create_invalid_response(
        User,
        error_type="nested_error",
        field_path="address.zip_code"
    )
    content = mock.choices[0].message.content
    with pytest.raises(ValueError):
        User.model_validate_json(content)

# Stream Helper Tests
def test_create_structured_stream_response():
    """Test creating valid streaming response."""
    mock = create_structured_stream_response(User, valid_user)
    chunks = list(mock)
    assert len(chunks) > 0
    # Combine chunks to verify final result
    combined = "".join(c.choices[0].delta.content for c in chunks)
    assert '"status": "processing"' in combined
    assert '"result"' in combined

def test_create_invalid_stream_response_malformed():
    """Test malformed JSON in stream."""
    mock = create_invalid_stream_response(error_type="malformed")
    chunks = list(mock)
    assert any("invalid json" in c.choices[0].delta.content for c in chunks)

def test_create_invalid_stream_response_timeout():
    """Test timeout error in stream."""
    mock = create_invalid_stream_response(error_type="timeout")
    with pytest.raises(Exception):  # Should raise timeout error
        list(mock)

# Error Helper Tests
def test_create_error_response_string():
    """Test creating error response from string."""
    mock = create_error_response("Test error", status_code=400)
    with pytest.raises(MockAPIError) as exc:
        mock()
    assert exc.value.status_code == 400
    assert str(exc.value) == "Test error"

def test_create_rate_limit_response():
    """Test rate limit simulation."""
    mock = create_rate_limit_response(max_requests=2)
    # First two calls should succeed
    mock()
    mock()
    # Third call should raise rate limit error
    with pytest.raises(MockAPIError) as exc:
        mock()
    assert exc.value.status_code == 429

def test_create_timeout_response():
    """Test timeout simulation."""
    mock = create_timeout_response(timeout_after=1.0)
    with pytest.raises(TimeoutError):
        mock() 