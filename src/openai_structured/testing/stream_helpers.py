"""Stream response helpers for testing.

This module provides utilities for generating mock stream responses that match
expected schemas and simulate various streaming scenarios.
"""

from typing import Any, Dict, Iterator, List, Type, TypeVar, Union, AsyncIterator, Optional
import logging
from unittest.mock import MagicMock, AsyncMock
import json
import time

from openai import OpenAIError
from pydantic import BaseModel, ValidationError

from ..errors import ConnectionTimeoutError, StreamInterruptedError, StreamParseError
from ..buffer import StreamBuffer, StreamConfig

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)

class AsyncStreamResponse:
    """Production-ready async stream mock with full buffer integration."""

    def __init__(
        self,
        chunks: List[MagicMock],
        schema: Type[BaseModel],
        response_id: Optional[str] = None,
        stream_config: Optional[StreamConfig] = None,
    ):
        """Initialize async stream response.

        Args:
            chunks: List of mock chunks to yield
            schema: Schema to validate against
            response_id: Optional response ID for metadata
            stream_config: Optional stream configuration
        """
        self._chunks = chunks
        self._buffer = StreamBuffer(config=stream_config or StreamConfig(), schema=schema)
        self._closed = False
        self._response_id = response_id
        self._chunk_index = 0

    def __aiter__(self):
        """Return self as async iterator."""
        return self

    async def __anext__(self):
        """Get next chunk asynchronously."""
        if self._closed:
            raise StreamInterruptedError("Stream closed before iteration")

        if self._chunk_index >= len(self._chunks):
            await self.close()
            raise StopAsyncIteration

        chunk = self._chunks[self._chunk_index]
        self._chunk_index += 1

        # Set response metadata on chunk
        if self._response_id:
            chunk.id = self._response_id
            chunk.object = 'chat.completion.chunk'
            chunk.created = int(time.time())
            chunk.model = 'gpt-4o'

        try:
            if not hasattr(chunk.choices[0].delta, 'content') or chunk.choices[0].delta.content is None:
                return chunk

            content = chunk.choices[0].delta.content
            result = self._buffer.process_stream_chunk(content)
            
            if result is not None:
                # Set response metadata on the result if supported
                for attr in ['id', 'model', 'object', 'created']:
                    if hasattr(result, attr) and hasattr(chunk, attr):
                        setattr(result, attr, getattr(chunk, attr))
                return result
            
            return chunk

        except (KeyError, json.JSONDecodeError) as e:
            self._buffer.close()
            raise StreamInterruptedError(f"Stream interrupted: {e}") from e
        except ValidationError as e:
            if self._buffer.parse_errors >= self._buffer.config.max_parse_errors:
                self._buffer.close()
                raise StreamParseError(
                    "Validation failed",
                    self._buffer.parse_errors,
                    e
                )
            return await self.__anext__()

    async def close(self):
        """Full async cleanup with buffer finalization."""
        if not self._closed:
            self._closed = True
            if self._buffer:
                try:
                    final_content = self._buffer.getvalue()
                    if final_content:
                        self._buffer.schema.model_validate_json(final_content)
                except ValidationError as e:
                    raise StreamParseError(
                        "Final validation failed",
                        self._buffer.parse_errors,
                        e
                    )
                finally:
                    self._buffer.close()

def create_structured_stream_response(
    output_schema: Type[BaseModel],
    data: Union[dict, List[dict]],
    *,
    is_async: bool = False,
    response_id: str = "mock_123",
    stream_config: Optional[StreamConfig] = None,
) -> Union[MagicMock, AsyncMock]:
    """Create protocol-compliant stream response.

    Args:
        output_schema: Schema to validate against
        data: Data to stream (single dict or list of dicts)
        is_async: Whether to return async mock
        response_id: Response ID for tracking
        stream_config: Optional stream configuration

    Returns:
        Mock configured with valid schema response
    """
    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]
        
    logger.debug(f"Creating mock chunks for data: {data}")
    chunks = []
    for i, item in enumerate(data):
        chunk = MagicMock()
        chunk.id = response_id
        chunk.object = 'chat.completion.chunk'
        chunk.created = 1677652288  # Fixed timestamp for testing
        chunk.model = 'gpt-4o'
        
        delta = MagicMock()
        delta.content = json.dumps(item)
        delta.role = 'assistant'
        
        choice = MagicMock()
        choice.delta = delta
        choice.index = i
        choice.finish_reason = None if i < len(data)-1 else 'stop'
        
        chunk.choices = [choice]
        logger.debug(f"Created chunk {i}: {chunk}")
        chunks.append(chunk)

    if is_async:
        logger.debug("Creating async response")
        async_response = AsyncStreamResponse(
            chunks=chunks,
            schema=output_schema,
            response_id=response_id,
            stream_config=stream_config
        )
        mock = AsyncMock()
        mock.__aiter__ = AsyncMock(return_value=async_response)
        mock.__anext__ = AsyncMock(side_effect=async_response.__anext__)
        return mock
    else:
        logger.debug("Creating sync response")
        response = MagicMock()
        def mock_iter():
            logger.debug("Starting sync iteration")
            for i, chunk in enumerate(chunks):
                logger.debug(f"Yielding chunk {i}: {chunk}")
                yield chunk
        response.__iter__ = MagicMock(side_effect=mock_iter)
        return response

def create_invalid_stream_response(
    error_type: str = "malformed",
    is_async: bool = False,
    delay: float = None,
    status_code: int = None
) -> Union[MagicMock, AsyncMock]:
    """Create a mock stream response that simulates various error conditions.

    Args:
        error_type: Type of error to simulate:
            - "malformed": Invalid JSON format
            - "incomplete": Partial JSON response
            - "invalid_format": Wrong message format
            - "empty": Empty response
            - "interrupted": Connection interruption
            - "timeout": Network timeout
            - "rate_limit": Rate limit error
            - "partial_json": Incomplete JSON chunks
        is_async: Whether to create an async response
        delay: Optional delay before error occurs
        status_code: Optional HTTP status code

    Returns:
        MagicMock configured to simulate the specified error
    """
    if error_type == "rate_limit":
        raise OpenAIError("Rate limit exceeded")

    error_patterns = {
        "malformed": [
            '{"status": "error"',  # Missing closing brace
            'invalid json here',
        ],
        "incomplete": [
            '{"status": "processing",',
            '"result": {"message": "test"',  # Missing closing braces
        ],
        "invalid_format": [
            '{"wrong_key": "wrong_value"}',
        ],
        "empty": [],
        "timeout": ConnectionTimeoutError("Request timed out"),
        "rate_limit": Exception("Rate limit exceeded"),
        "partial_json": [
            '{"status": "processing",',
            '"result": {"mess',
            'age": "test"}}',
        ]
    }

    if error_type in ["timeout", "rate_limit"]:
        mock = MagicMock()
        mock.__iter__.side_effect = error_patterns[error_type]
        return mock

    # Create chunks with invalid content
    chunks = []
    for content in error_patterns.get(error_type, []):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = content
        chunks.append(chunk)

    if is_async:
        mock = AsyncMock()
        mock.return_value = AsyncStreamResponse(
            chunks=chunks,
            schema=BaseModel,  # Use BaseModel as fallback
            response_id="error_test"
        )
        return mock
    else:
        response = MagicMock()
        response.__iter__ = MagicMock(return_value=iter(chunks))
        return response 