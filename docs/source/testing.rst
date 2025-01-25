Testing Guide
============

This guide covers how to test applications that use the openai-structured library.

Overview
-------

The library provides comprehensive testing utilities in the ``openai_structured.testing`` module:

.. code-block:: python

    from openai_structured.testing import (
        # Response helpers
        create_structured_response,    # Create schema-validated responses
        create_invalid_response,       # Generate invalid responses
        
        # Stream helpers
        create_structured_stream_response,  # Create schema-validated streams
        create_invalid_stream_response,     # Generate invalid streams
        
        # Error helpers
        create_error_response,         # Simulate API errors
        create_rate_limit_response,    # Simulate rate limiting
        create_timeout_response,       # Simulate timeouts
    )

Basic Response Testing
-------------------

To test basic structured responses:

.. code-block:: python

    from unittest.mock import MagicMock
    from openai_structured import openai_structured
    from openai_structured.testing import create_structured_response
    from openai_structured.examples.schemas import SimpleMessage

    def test_basic_response():
        # Create mock client with schema validation
        mock_client = MagicMock()
        mock_client.chat.completions.create = create_structured_response(
            output_schema=SimpleMessage,
            data={"message": "test"}
        )
        
        # Test your code
        result = openai_structured(
            client=mock_client,
            model="gpt-4",
            output_schema=SimpleMessage,
            user_prompt="test"
        )
        assert isinstance(result, SimpleMessage)
        assert result.message == "test"

Stream Testing
------------

For testing streaming responses:

.. code-block:: python

    from openai_structured import openai_structured_stream
    from openai_structured.testing import create_structured_stream_response
    from openai_structured.examples.schemas import SimpleMessage

    def test_stream():
        # Create mock client with streaming response
        mock_client = MagicMock()
        mock_client.chat.completions.create = create_structured_stream_response(
            output_schema=SimpleMessage,
            data=[
                {"message": "part1"},  # First chunk
                {"message": "part2"}   # Second chunk
            ]
        )
        
        # Test streaming
        results = list(openai_structured_stream(
            client=mock_client,
            model="gpt-4",
            output_schema=SimpleMessage,
            user_prompt="test"
        ))
        assert len(results) == 2

Error Testing
-----------

Test various error scenarios:

.. code-block:: python

    from openai_structured.testing import (
        create_invalid_response,
        create_error_response,
        create_rate_limit_response,
        create_timeout_response
    )

    def test_error_handling():
        mock_client = MagicMock()
        
        # Test schema validation errors
        mock_client.chat.completions.create = create_invalid_response(
            output_schema=SimpleMessage,
            error_type="missing_field"  # or "wrong_type", "nested_error"
        )
        with pytest.raises(ValidationError):
            result = openai_structured(...)
        
        # Test API errors
        mock_client.chat.completions.create = create_error_response(
            "Internal Server Error",
            status_code=500
        )
        with pytest.raises(Exception):
            result = openai_structured(...)
        
        # Test rate limiting
        mock_client.chat.completions.create = create_rate_limit_response(
            max_requests=3,
            reset_after=60
        )
        # First 3 calls succeed
        result1 = openai_structured(...)
        result2 = openai_structured(...)
        result3 = openai_structured(...)
        # Fourth call raises rate limit error
        with pytest.raises(RateLimitError):
            result4 = openai_structured(...)
        
        # Test timeouts
        mock_client.chat.completions.create = create_timeout_response(
            timeout_after=5.0
        )
        with pytest.raises(TimeoutError):
            result = openai_structured(...)

Async Testing
-----------

For testing async code:

.. code-block:: python

    import pytest
    from unittest.mock import AsyncMock
    from openai_structured import async_openai_structured_stream
    from openai_structured.testing import create_structured_stream_response

    @pytest.mark.asyncio
    async def test_async_stream():
        # Create async mock client
        mock_client = AsyncMock()
        mock_client.chat.completions.create = create_structured_stream_response(
            output_schema=SimpleMessage,
            data=[
                {"message": "part1"},
                {"message": "part2"}
            ]
        )
        
        # Test async streaming
        results = []
        async for chunk in async_openai_structured_stream(
            client=mock_client,
            model="gpt-4",
            output_schema=SimpleMessage,
            user_prompt="test"
        ):
            results.append(chunk)
        
        assert len(results) == 2
        assert all(isinstance(r, SimpleMessage) for r in results)

Complex Schema Testing
-------------------

Testing with nested models and validation:

.. code-block:: python

    from pydantic import BaseModel, Field

    class Address(BaseModel):
        street: str
        city: str
        zip_code: str = Field(pattern=r"^\d{5}$")

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
        
        result = openai_structured(...)
        assert result.address.city == "Test City"

    def test_nested_validation_error():
        mock_client = MagicMock()
        mock_client.chat.completions.create = create_invalid_response(
            output_schema=User,
            error_type="nested_error",
            field_path="address.zip_code"
        )
        
        with pytest.raises(ValidationError) as exc:
            result = openai_structured(...)
        assert "address.zip_code" in str(exc.value)

Best Practices
------------

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
