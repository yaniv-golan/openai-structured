"""Tests for testing utilities."""
import pytest
import logging
from unittest.mock import MagicMock

from openai_structured import openai_structured_stream
from openai_structured.testing import (
    create_structured_stream_response,
    create_error_response,
    create_invalid_stream_response,
    create_rate_limit_response
)
from openai_structured.examples.schemas import SimpleMessage
from openai_structured.errors import (
    StreamInterruptedError,
    ConnectionTimeoutError
)

logger = logging.getLogger(__name__)

def test_create_structured_stream_response():
    """Test stream response creation."""
    # Create a mock response with multiple chunks
    response = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=[
            {"message": "part1"},
            {"message": "part2"}
        ]
    )
    
    # Verify response structure
    assert hasattr(response, '__iter__')  # Should be iterable
    chunks = list(response)
    
    # Each message is split into 4 chunks:
    # 1. {"status": "processing", 
    # 2. "result": 
    # 3. actual content
    # 4. }
    assert len(chunks) == 8  # 2 messages * 4 chunks each
    
    # Get the actual content from the mock
    content1 = chunks[2].choices[0].delta.content  # Third chunk of first message
    content2 = chunks[6].choices[0].delta.content  # Third chunk of second message
    
    # Verify chunk content matches schema (compact JSON)
    assert '"message":"part1"' in content1
    assert '"message":"part2"' in content2


def test_create_error_response():
    """Test error response creation."""
    # Create a mock error response
    error = ConnectionError("Test error")
    mock = create_error_response(error)
    
    # The mock should be a MagicMock
    assert isinstance(mock, MagicMock)
    
    # When used in streaming, it should raise StreamInterruptedError
    with pytest.raises(StreamInterruptedError) as exc_info:
        list(openai_structured_stream(
            client=MagicMock(chat=MagicMock(completions=MagicMock(create=mock))),
            model="gpt-4o",
            output_schema=SimpleMessage,
            system_prompt="You are a test assistant.",
            user_prompt="test"
        ))
    assert "Test error" in str(exc_info.value)


def test_create_invalid_stream_response():
    """Test invalid stream response creation."""
    # Test malformed JSON
    response = create_invalid_stream_response(error_type="malformed")
    chunks = list(response)
    assert len(chunks) > 0
    assert not all('"message":' in c.choices[0].delta.content for c in chunks)

    # Test timeout
    response = create_invalid_stream_response(error_type="timeout")
    with pytest.raises(ConnectionTimeoutError) as exc:
        list(response)
    assert "timed out" in str(exc.value).lower()


def test_create_rate_limit_response():
    """Test rate limit response creation."""
    mock = create_rate_limit_response(max_requests=2, reset_after=60)
    
    # First two calls should succeed
    result1 = mock()
    assert isinstance(result1, MagicMock)
    
    result2 = mock()
    assert isinstance(result2, MagicMock)
    
    # Third call should raise rate limit error
    with pytest.raises(Exception) as exc:  # Should be specific rate limit error
        mock()
    assert "rate limit" in str(exc.value).lower()


def test_stream_integration():
    """Test integration of stream helpers with openai_structured_stream."""
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    logger.debug("Creating mock response")
    # Create mock client with proper structure
    mock_create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=[
            {"message": "Hello"},
            {"message": "World"}
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    
    logger.debug("Starting stream test")
    # Use the mock in streaming
    results = []
    for result in openai_structured_stream(
        client=mock_client,
        model="gpt-4o-2024-08-06",  # Use specific version that's guaranteed to work
        output_schema=SimpleMessage,
        system_prompt="You are a test assistant.",
        user_prompt="test"
    ):
        logger.debug(f"Got result: {result}")
        logger.debug(f"Result type: {type(result)}")
        logger.debug(f"Result dict: {result.model_dump()}")
        results.append(result)
    logger.debug(f"Stream complete, got {len(results)} results")
    
    # Verify results
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert all(isinstance(r, SimpleMessage) for r in results)
    assert [r.message for r in results] == ["Hello", "World"]


def test_error_integration():
    """Test integration of error helpers with openai_structured_stream."""
    # Create mock client with error response
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_error_response(
        ConnectionError("Network error")
    )
    
    # Verify error handling
    with pytest.raises(StreamInterruptedError) as exc_info:
        list(openai_structured_stream(
            client=mock_client,
            model="gpt-4o",
            output_schema=SimpleMessage,
            system_prompt="You are a test assistant.",
            user_prompt="test"
        ))
    assert "Network error" in str(exc_info.value) 