import pytest
from pydantic import BaseModel

from openai_structured.buffer import ParseError, StreamBuffer, StreamConfig


def test_buffer_close() -> None:
    """Test that buffer resources are properly closed."""
    buffer = StreamBuffer()
    buffer.write("test content")
    buffer.close()

    # Verify buffer is closed
    assert buffer._buffer.closed
    assert buffer.total_bytes == 0
    assert buffer.parse_errors == 0
    assert buffer.cleanup_attempts == 0


def test_buffer_reset_state() -> None:
    """Test that all state is properly reset."""
    buffer = StreamBuffer()

    # Perform operations that modify state
    buffer.write("test content")
    buffer.cleanup()  # This will increment cleanup_attempts

    # Force a parse error
    buffer.write("{invalid json")
    try:
        buffer.cleanup()
    except ParseError:
        pass

    # Verify state before reset
    assert buffer.total_bytes > 0
    assert buffer.parse_errors > 0
    assert buffer.cleanup_attempts > 0

    # Reset and verify all counters
    buffer.reset()
    assert buffer.total_bytes == 0
    assert buffer.parse_errors == 0
    assert buffer.cleanup_attempts == 0


def test_invalid_schema_type() -> None:
    """Test that invalid schema types are rejected."""

    class NotAPydanticModel:
        pass

    with pytest.raises(
        ValueError, match="Schema must be a Pydantic model class"
    ):
        StreamBuffer(schema=NotAPydanticModel)


def test_schema_validation_error_context() -> None:
    """Test that schema validation errors include proper context."""

    class TestModel(BaseModel):  # type: ignore[misc]
        value: int

    buffer = StreamBuffer(schema=TestModel)
    buffer.write('{"value": "not an integer"}')

    with pytest.raises(ParseError) as exc_info:
        buffer.cleanup()

    # Verify error context is included
    assert "Context:" in str(exc_info.value)


def test_cleanup_attempts_tracking() -> None:
    """Test that cleanup attempts are properly tracked and limited."""
    buffer = StreamBuffer(config=StreamConfig(max_cleanup_attempts=2))

    # Write invalid content
    buffer.write("{invalid")

    # First attempt
    with pytest.raises(ParseError):
        buffer.cleanup()
    assert buffer.cleanup_attempts == 1

    # Second attempt should fail with max attempts error
    with pytest.raises(ParseError, match="Exceeded maximum cleanup attempts"):
        buffer.cleanup()


def test_error_stats_consistency() -> None:
    """Test that error statistics are consistently tracked."""
    buffer = StreamBuffer()

    # Test JSON decode error stats
    buffer.write("{invalid")
    try:
        buffer.cleanup()
    except ParseError:
        pass

    assert "json_error" in buffer._cleanup_stats
    assert "error_type" in buffer._cleanup_stats

    # Test validation error stats
    class TestModel(BaseModel):  # type: ignore[misc]
        value: int

    buffer = StreamBuffer(schema=TestModel)
    buffer.write('{"value": "not an integer"}')

    try:
        buffer.cleanup()
    except ParseError:
        pass

    assert "validation_error" in buffer._cleanup_stats
    assert "error_context" in buffer._cleanup_stats
