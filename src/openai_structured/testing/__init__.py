"""Testing utilities for openai-structured.

This module provides utilities for testing applications that use openai-structured.
It includes mock clients and response helpers.

Testing Guidelines
----------------

1. Basic Response Mocking
------------------------
To mock a basic structured response:

    >>> from unittest.mock import MagicMock
    >>> from openai_structured.examples.schemas import SimpleMessage
    >>> from openai_structured.testing import create_structured_response
    >>> 
    >>> # Create mock client with validated response
    >>> mock_client = MagicMock()
    >>> mock_client.chat.completions.create = create_structured_response(
    ...     output_schema=SimpleMessage,
    ...     data={"message": "test"}
    ... )
    >>> 
    >>> # Use in your test
    >>> result = openai_structured(
    ...     client=mock_client,
    ...     model="gpt-4o",
    ...     output_schema=SimpleMessage,
    ...     user_prompt="test"
    ... )
    >>> assert isinstance(result, SimpleMessage)

2. Stream Response Mocking
-------------------------
To mock streaming responses:

    >>> from openai_structured.testing import create_structured_stream_response
    >>> 
    >>> # Create mock client with streaming response
    >>> mock_client = MagicMock()
    >>> mock_client.chat.completions.create = create_structured_stream_response(
    ...     output_schema=SimpleMessage,
    ...     data={"message": "test"}
    ... )
    >>> 
    >>> # Use in your test
    >>> results = list(openai_structured_stream(
    ...     client=mock_client,
    ...     model="gpt-4o",
    ...     output_schema=SimpleMessage,
    ...     user_prompt="test"
    ... ))

3. Error Handling Testing
------------------------
To test error scenarios:

    >>> from openai_structured.testing import create_error_response
    >>> 
    >>> # Test network timeout
    >>> mock_client = MagicMock()
    >>> mock_client.chat.completions.create = create_timeout_response(
    ...     timeout_after=5.0
    ... )
    >>> 
    >>> # Test rate limiting
    >>> mock_client.chat.completions.create = create_rate_limit_response(
    ...     max_requests=3,
    ...     reset_after=60
    ... )
    >>> 
    >>> # Test schema validation errors
    >>> mock_client.chat.completions.create = create_invalid_response(
    ...     output_schema=SimpleMessage,
    ...     error_type="missing_field"
    ... )

4. Async Testing
---------------
For async code, use AsyncMock:

    >>> from unittest.mock import AsyncMock
    >>> 
    >>> @pytest.mark.asyncio
    >>> async def test_async_stream():
    ...     mock_client = AsyncMock()
    ...     mock_client.chat.completions.create = create_structured_stream_response(
    ...         output_schema=SimpleMessage,
    ...         data={"message": "test"}
    ...     )
    ...     
    ...     async for chunk in async_openai_structured_stream(...):
    ...         assert isinstance(chunk, SimpleMessage)

Best Practices
-------------
1. Use schema validation to catch issues early
2. Test both success and error scenarios
3. For streams, test partial and complete responses
4. Use appropriate error types for different scenarios
5. Test nested schema validation
6. Verify rate limiting and timeout handling
"""

from .error_helpers import (
    create_error_response,
    create_rate_limit_response,
    create_timeout_response,
    MockAPIError,
)
from .response_helpers import (
    create_structured_response,
    create_invalid_response,
)
from .stream_helpers import (
    create_structured_stream_response,
    create_invalid_stream_response,
)

__all__ = [
    # Response helpers
    "create_structured_response",
    "create_invalid_response",
    
    # Stream helpers
    "create_structured_stream_response",
    "create_invalid_stream_response",
    
    # Error helpers
    "create_error_response",
    "create_rate_limit_response",
    "create_timeout_response",
    "MockAPIError",
]
