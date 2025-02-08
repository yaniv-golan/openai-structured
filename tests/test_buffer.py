"""Test buffer functionality."""

from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from openai_structured.buffer import StreamBuffer, StreamConfig
from openai_structured.errors import ClosedBufferError
from openai_structured.examples.schemas import SimpleMessage


# Test models
class ComplexMessage(BaseModel):
    meta: Dict[str, Any]
    items: List[Any]


@pytest.fixture
def default_stream_config() -> StreamConfig:
    return StreamConfig(
        max_buffer_size=1024, cleanup_threshold=512, max_parse_errors=3
    )


# 1. Basic Buffer Operations
def test_buffer_write_read(default_stream_config: StreamConfig) -> None:
    """Test basic buffer write and read."""
    buf = StreamBuffer(default_stream_config, SimpleMessage)
    buf.write('{"message": "test"}')
    assert buf.getvalue() == '{"message": "test"}'


# 2. JSON Parsing and Validation
def test_buffer_valid_json(default_stream_config: StreamConfig) -> None:
    """Test processing valid JSON."""
    buf = StreamBuffer(default_stream_config, SimpleMessage)
    buf.write('{"message": "test"}')
    result = buf.process_stream_chunk("")
    assert isinstance(result, SimpleMessage)
    assert result.message == "test"


def test_buffer_complex_json(default_stream_config: StreamConfig) -> None:
    """Test processing complex JSON in chunks."""
    buf = StreamBuffer(default_stream_config, SimpleMessage)
    buf.write('{"message": "')
    assert buf.process_stream_chunk("") is None
    buf.write('test"}')
    result = buf.process_stream_chunk("")
    assert isinstance(result, SimpleMessage)
    assert result.message == "test"


# 3. Error Handling and Cleanup
def test_buffer_invalid_json(default_stream_config: StreamConfig) -> None:
    """Test handling invalid JSON."""
    buf = StreamBuffer(default_stream_config, SimpleMessage)
    buf.write('{"message": invalid}')
    # Process chunk to trigger validation
    assert buf.process_stream_chunk("") is None
    # Buffer should be cleaned up after max parse errors
    for _ in range(default_stream_config.max_parse_errors + 1):
        buf.process_stream_chunk("")
    assert buf.getvalue() == ""


def test_buffer_cleanup(default_stream_config: StreamConfig) -> None:
    """Test buffer cleanup after threshold."""
    config = StreamConfig(cleanup_threshold=10)
    buf = StreamBuffer(config, SimpleMessage)
    buf.write('{"message": "test"}')
    result = buf.process_stream_chunk("")
    assert isinstance(result, SimpleMessage)
    assert buf.getvalue() == ""  # Buffer should be cleaned up


def test_buffer_parse_errors(default_stream_config: StreamConfig) -> None:
    """Test parse error accumulation."""
    buf = StreamBuffer(default_stream_config, SimpleMessage)
    buf.write('{"wrong": "format"}')
    # Process chunk to trigger validation
    assert buf.process_stream_chunk("") is None
    # Buffer should be cleaned up after max parse errors
    for _ in range(default_stream_config.max_parse_errors + 1):
        buf.process_stream_chunk("")
    assert buf.getvalue() == ""


# 4. Buffer Size Limits and Thresholds
def test_buffer_cleanup_threshold(default_stream_config: StreamConfig) -> None:
    """Test buffer cleanup threshold."""
    config = StreamConfig(cleanup_threshold=10)
    buf = StreamBuffer(config, SimpleMessage)
    buf.write('{"message": "test"}')
    assert len(buf.getvalue()) > 0
    buf.cleanup()
    assert len(buf.getvalue()) == 0


def test_buffer_closed_state(default_stream_config: StreamConfig) -> None:
    """Test that writing to a closed buffer raises ClosedBufferError."""
    buf = StreamBuffer(default_stream_config, SimpleMessage)
    buf.write('{"message": "test"}')  # Initial write should succeed
    buf.close()

    with pytest.raises(ClosedBufferError) as exc_info:
        buf.write('{"message": "more"}')
    assert str(exc_info.value) == "Cannot write to a closed buffer"

    # Verify buffer was properly cleaned up
    assert buf.getvalue() == ""
    assert buf.size == 0
