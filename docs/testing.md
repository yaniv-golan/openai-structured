# Testing Guide

This guide covers testing utilities provided by openai-structured and best practices for testing applications that use the library.

## Overview

The testing utilities are designed to help you:

- Mock OpenAI API responses with schema validation
- Test streaming responses and error handling
- Simulate various error conditions
- Test async code paths

## Installation

The testing utilities are included with openai-structured:

```bash
pip install openai-structured
```

## Basic Usage

### Response Mocking

The simplest way to mock responses is using `create_structured_response`:

```python
from unittest.mock import MagicMock
from openai_structured.testing import create_structured_response
from your_app.schemas import UserProfile

def test_get_user_profile():
    # Create mock client
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_structured_response(
        output_schema=UserProfile,
        data={
            "name": "Test User",
            "age": 30,
            "email": "test@example.com"
        }
    )
    
    # Test your code
    result = get_user_profile(client=mock_client)
    assert result.name == "Test User"
    assert result.age == 30
```

### Streaming Responses

For streaming APIs, use `create_structured_stream_response`:

```python
from openai_structured.testing import create_structured_stream_response

def test_stream_user_profiles():
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_structured_stream_response(
        output_schema=UserProfile,
        data=[
            {"name": "User 1", "age": 30},
            {"name": "User 2", "age": 25}
        ]
    )
    
    results = list(stream_user_profiles(client=mock_client))
    assert len(results) == 2
    assert results[0].name == "User 1"
    assert results[1].name == "User 2"
```

## Error Testing

### Schema Validation Errors

Test how your code handles invalid responses:

```python
from openai_structured.testing import create_invalid_response

def test_invalid_response_handling():
    mock_client = MagicMock()
    
    # Test missing required field
    mock_client.chat.completions.create = create_invalid_response(
        output_schema=UserProfile,
        error_type="missing_field"
    )
    with pytest.raises(ValidationError):
        get_user_profile(client=mock_client)
    
    # Test wrong field type
    mock_client.chat.completions.create = create_invalid_response(
        output_schema=UserProfile,
        error_type="wrong_type"
    )
    with pytest.raises(ValidationError):
        get_user_profile(client=mock_client)
```

### API Errors

Simulate various API errors:

```python
from openai_structured.testing import (
    create_error_response,
    create_rate_limit_response,
    create_timeout_response
)

def test_error_handling():
    mock_client = MagicMock()
    
    # Test API error
    mock_client.chat.completions.create = create_error_response(
        "Internal Server Error",
        status_code=500
    )
    with pytest.raises(Exception):
        get_user_profile(client=mock_client)
    
    # Test rate limiting
    mock_client.chat.completions.create = create_rate_limit_response(
        max_requests=3,
        reset_after=60
    )
    # First 3 calls succeed
    get_user_profile(client=mock_client)
    get_user_profile(client=mock_client)
    get_user_profile(client=mock_client)
    # Fourth call raises rate limit error
    with pytest.raises(RateLimitError):
        get_user_profile(client=mock_client)
    
    # Test timeout
    mock_client.chat.completions.create = create_timeout_response(
        timeout_after=5.0
    )
    with pytest.raises(TimeoutError):
        get_user_profile(client=mock_client)
```

### Stream Errors

Test error handling in streaming responses:

```python
from openai_structured.testing import create_invalid_stream_response

def test_stream_error_handling():
    mock_client = MagicMock()
    
    # Test malformed JSON
    mock_client.chat.completions.create = create_invalid_stream_response(
        error_type="malformed"
    )
    with pytest.raises(JSONDecodeError):
        list(stream_user_profiles(client=mock_client))
    
    # Test incomplete response
    mock_client.chat.completions.create = create_invalid_stream_response(
        error_type="incomplete"
    )
    with pytest.raises(ValidationError):
        list(stream_user_profiles(client=mock_client))
```

## Async Testing

For async code, use AsyncMock:

```python
from unittest.mock import AsyncMock
import pytest

@pytest.mark.asyncio
async def test_async_stream():
    mock_client = AsyncMock()
    mock_client.chat.completions.create = create_structured_stream_response(
        output_schema=UserProfile,
        data=[
            {"name": "User 1", "age": 30},
            {"name": "User 2", "age": 25}
        ]
    )
    
    results = []
    async for profile in stream_user_profiles_async(client=mock_client):
        results.append(profile)
    
    assert len(results) == 2
    assert all(isinstance(r, UserProfile) for r in results)
```

## Testing Complex Schemas

### Nested Models

Test validation of nested data structures:

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class User(BaseModel):
    name: str
    age: int
    address: Address

def test_nested_schema():
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_structured_response(
        output_schema=User,
        data={
            "name": "Test User",
            "age": 30,
            "address": {
                "street": "123 Test St",
                "city": "Test City",
                "zip_code": "12345"
            }
        }
    )
    
    result = get_user(client=mock_client)
    assert result.address.city == "Test City"
```

### Nested Validation Errors

Test handling of validation errors in nested structures:

```python
def test_nested_validation_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_invalid_response(
        output_schema=User,
        error_type="nested_error",
        field_path="address.zip_code"
    )
    
    with pytest.raises(ValidationError) as exc:
        get_user(client=mock_client)
    assert "address.zip_code" in str(exc.value)
```

## Best Practices

1. **Use Schema Validation**
   - Always validate responses against your schemas
   - Test both valid and invalid data
   - Include edge cases in your test data

2. **Test Error Handling**
   - Test all error scenarios your code should handle
   - Include API errors, validation errors, and timeouts
   - Verify error messages and status codes

3. **Stream Testing**
   - Test both complete and partial responses
   - Verify correct handling of malformed chunks
   - Test interruption and timeout scenarios

4. **Async Testing**
   - Use AsyncMock for async code
   - Test both success and error paths
   - Verify correct async context management

5. **Complex Schemas**
   - Test nested model validation
   - Verify handling of optional fields
   - Test array and enum fields

6. **Rate Limiting**
   - Test rate limit detection
   - Verify retry logic if implemented
   - Test rate limit reset behavior
