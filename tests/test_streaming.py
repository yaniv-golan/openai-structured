"""Unit tests for streaming functionality."""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from openai_structured import openai_structured_stream, async_openai_structured_stream
from openai_structured.errors import StreamInterruptedError
from openai_structured.testing import (
    create_structured_stream_response,
    create_error_response,
    create_invalid_stream_response
)
from openai_structured.examples.schemas import SimpleMessage
from openai_structured.buffer import StreamConfig
from openai_structured.errors import StreamParseError

def test_incremental_parsing(mock_openai_sync_client: MagicMock) -> None:
    """Test incremental JSON parsing in stream."""
    mock_openai_sync_client.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data={"message": "hello"}
    )
    
    results = list(openai_structured_stream(
        client=mock_openai_sync_client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        system_prompt="You are a test assistant.",
        user_prompt="test"
    ))
    
    assert len(results) == 1
    assert isinstance(results[0], SimpleMessage)
    assert results[0].message == "hello"

def test_stream_error_handling(mock_openai_sync_client: MagicMock) -> None:
    """Test stream error handling."""
    mock_openai_sync_client.chat.completions.create = create_error_response(
        "Stream interrupted",
        status_code=500
    )
    
    with pytest.raises(StreamInterruptedError):
        list(openai_structured_stream(
            client=mock_openai_sync_client,
            model="gpt-4o",
            output_schema=SimpleMessage,
            system_prompt="You are a test assistant.",
            user_prompt="test"
        ))

def test_stream_invalid_json(mock_openai_sync_client: MagicMock) -> None:
    """Test handling of invalid JSON in stream."""
    mock_openai_sync_client.chat.completions.create = create_invalid_stream_response(
        error_type="malformed"
    )
    
    with pytest.raises(ValueError):  # or specific JSON parse error
        list(openai_structured_stream(
            client=mock_openai_sync_client,
            model="gpt-4o",
            output_schema=SimpleMessage,
            system_prompt="You are a test assistant.",
            user_prompt="test"
        ))

@pytest.mark.asyncio
async def test_async_stream(mock_openai_async_client: AsyncMock) -> None:
    """Test async streaming."""
    mock_openai_async_client.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=[
            {"message": "part1"},
            {"message": "part2"}
        ]
    )
    
    results = []
    async for result in async_openai_structured_stream(
        client=mock_openai_async_client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        system_prompt="You are a test assistant.",
        user_prompt="test"
    ):
        results.append(result)
    
    assert len(results) == 2
    assert all(isinstance(r, SimpleMessage) for r in results)
    assert results[0].message == "part1"
    assert results[1].message == "part2"

@pytest.mark.asyncio
async def test_async_stream_error(mock_openai_async_client: AsyncMock) -> None:
    """Test async stream error handling."""
    mock_openai_async_client.chat.completions.create = create_error_response(
        "Async stream error",
        status_code=500
    )
    
    with pytest.raises(StreamInterruptedError):
        async for _ in async_openai_structured_stream(
            client=mock_openai_async_client,
            model="gpt-4o",
            output_schema=SimpleMessage,
            system_prompt="You are a test assistant.",
            user_prompt="test"
        ):
            pass

def test_stream_rate_limit(mock_openai_sync_client: MagicMock) -> None:
    """Test rate limit handling in streams."""
    from openai_structured.testing import create_rate_limit_response
    
    mock_openai_sync_client.chat.completions.create = create_rate_limit_response(
        max_requests=2,
        reset_after=60
    )
    
    # First two calls should succeed
    results1 = list(openai_structured_stream(
        client=mock_openai_sync_client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        system_prompt="You are a test assistant.",
        user_prompt="test1"
    ))
    assert len(results1) > 0
    
    results2 = list(openai_structured_stream(
        client=mock_openai_sync_client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        system_prompt="You are a test assistant.",
        user_prompt="test2"
    ))
    assert len(results2) > 0
    
    # Third call should raise rate limit error
    with pytest.raises(Exception) as exc:  # Replace with specific rate limit error
        list(openai_structured_stream(
            client=mock_openai_sync_client,
            model="gpt-4o",
            output_schema=SimpleMessage,
            system_prompt="You are a test assistant.",
            user_prompt="test3"
        ))
    assert "rate limit" in str(exc.value).lower()

def test_stream_sync_chunks(mock_openai_sync_client: MagicMock) -> None:
    """Test sync streaming with multiple chunks."""
    mock_openai_sync_client.chat.completions.create.return_value = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=[
            {"message": "chunk1"},
            {"message": "chunk2"}
        ]
    )
    
    result = []
    for chunk in openai_structured_stream(
        client=mock_openai_sync_client,
        model="gpt-4o",
        system_prompt="You are a test assistant.",
        user_prompt="test",
        output_schema=SimpleMessage,
    ):
        result.append(chunk)
    
    assert len(result) == 2
    assert all(isinstance(r, SimpleMessage) for r in result)
    assert [r.message for r in result] == ["chunk1", "chunk2"]

def test_stream_sync_error_handling(mock_openai_sync_client: MagicMock) -> None:
    """Test sync streaming with error response."""
    mock_openai_sync_client.chat.completions.create = create_error_response("Test error")
    
    with pytest.raises(StreamInterruptedError):
        for _ in openai_structured_stream(
            client=mock_openai_sync_client,
            model="gpt-4o",
            system_prompt="You are a test assistant.",
            user_prompt="test",
            output_schema=SimpleMessage,
        ):
            pass

@pytest.mark.asyncio
async def test_async_stream_chunks(mock_openai_async_client: MagicMock) -> None:
    """Test async streaming with multiple chunks."""
    async def async_create(*args, **kwargs):
        return create_structured_stream_response(
            output_schema=SimpleMessage,
            data=[
                {"message": "chunk1"},
                {"message": "chunk2"}
            ]
        )
    mock_openai_async_client.chat.completions.create = AsyncMock(side_effect=async_create)
    
    result = []
    async for chunk in async_openai_structured_stream(
        client=mock_openai_async_client,
        model="gpt-4o",
        system_prompt="You are a test assistant.",
        user_prompt="test",
        output_schema=SimpleMessage,
    ):
        result.append(chunk)
    
    assert len(result) == 2
    assert all(isinstance(r, SimpleMessage) for r in result)
    assert [r.message for r in result] == ["chunk1", "chunk2"]

@pytest.mark.asyncio
async def test_async_stream_error_handling(mock_openai_async_client: MagicMock) -> None:
    """Test async streaming with error response."""
    async def async_error(*args, **kwargs):
        raise ConnectionError("Test error")
    mock_openai_async_client.chat.completions.create = AsyncMock(side_effect=async_error)
    
    with pytest.raises(StreamInterruptedError):
        async for _ in async_openai_structured_stream(
            client=mock_openai_async_client,
            model="gpt-4o",
            system_prompt="You are a test assistant.",
            user_prompt="test",
            output_schema=SimpleMessage,
        ):
            pass

def test_create_stream_response():
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
    assert len(chunks) == 2
    
    # Get the actual content from the mock
    content1 = chunks[0].choices[0].delta.content
    content2 = chunks[1].choices[0].delta.content
    
    # Verify chunk content
    assert content1 == '{"message": "part1"}'
    assert content2 == '{"message": "part2"}'


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


def test_stream_integration():
    """Test integration of stream helpers with openai_structured_stream."""
    # Create mock client with streaming response
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=[
            {"message": "Hello"},
            {"message": "World"}
        ]
    )
    
    # Use the mock in streaming
    results = list(openai_structured_stream(
        client=mock_client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        system_prompt="You are a test assistant.",
        user_prompt="test"
    ))
    
    # Verify results
    assert len(results) == 2
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

@pytest.mark.asyncio
async def test_async_stream_integration():
    """Test async streaming with buffer integration."""
    config = StreamConfig(
        max_buffer_size=1024,
        cleanup_threshold=512,
        max_parse_errors=3
    )
    
    test_data = [
        {"message": "Hello"},
        {"message": "World"}
    ]
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=test_data,
        is_async=True,
        stream_config=config
    )
    
    results = []
    async for result in async_openai_structured_stream(
        client=mock_client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        system_prompt="Test",
        user_prompt="Test",
        stream_config=config
    ):
        results.append(result)
        
    assert len(results) == 2
    assert all(isinstance(r, SimpleMessage) for r in results)
    assert [r.message for r in results] == ["Hello", "World"]
    
    # Verify cleanup
    stream = mock_client.chat.completions.create.return_value
    assert stream._closed
    assert stream._buffer.closed

@pytest.mark.asyncio
async def test_async_stream_validation_error():
    """Test async streaming with validation errors."""
    config = StreamConfig(max_parse_errors=2)
    
    # Invalid data that will fail validation
    test_data = [
        {"wrong_field": "invalid"},
        {"message": "valid"}
    ]
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=test_data,
        is_async=True,
        stream_config=config
    )
    
    with pytest.raises(StreamParseError) as exc_info:
        async for _ in async_openai_structured_stream(
            client=mock_client,
            model="gpt-4o",
            output_schema=SimpleMessage,
            system_prompt="Test",
            user_prompt="Test",
            stream_config=config
        ):
            pass
    
    assert "Validation failed" in str(exc_info.value)
    assert exc_info.value.attempts <= config.max_parse_errors

@pytest.mark.asyncio
async def test_async_stream_buffer_cleanup():
    """Test async stream buffer cleanup on close."""
    config = StreamConfig(
        max_buffer_size=1024,
        cleanup_threshold=512
    )
    
    test_data = [{"message": "test"}]
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=test_data,
        is_async=True,
        stream_config=config
    )
    
    stream = mock_client.chat.completions.create.return_value
    assert not stream._closed
    assert not stream._buffer.closed
    
    results = []
    async for result in async_openai_structured_stream(
        client=mock_client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        system_prompt="Test",
        user_prompt="Test",
        stream_config=config
    ):
        results.append(result)
    
    assert stream._closed
    assert stream._buffer.closed
    assert len(results) == 1

@pytest.mark.asyncio
async def test_async_stream_metadata():
    """Test async stream response metadata."""
    test_data = [{"message": "test"}]
    response_id = "test_123"
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=test_data,
        is_async=True,
        response_id=response_id
    )
    
    async for result in async_openai_structured_stream(
        client=mock_client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        system_prompt="Test",
        user_prompt="Test"
    ):
        assert result.id == response_id
        assert result.model == "gpt-4o"
        assert result.object == "chat.completion.chunk"
        assert result.choices[0].delta.role == "assistant"
        assert isinstance(result.created, int)

@pytest.mark.asyncio
async def test_async_stream_early_close():
    """Test async stream early closure."""
    test_data = [
        {"message": "part1"},
        {"message": "part2"}
    ]
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=test_data,
        is_async=True
    )
    
    stream = mock_client.chat.completions.create.return_value
    await stream.close()
    
    with pytest.raises(StreamInterruptedError) as exc_info:
        async for _ in stream:
            pass
    
    assert "Stream closed before iteration" in str(exc_info.value)
    assert stream._closed
    assert stream._buffer.closed 