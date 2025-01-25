"""Test response handling functionality."""

from typing import Optional, Type
from unittest.mock import MagicMock, AsyncMock

import pytest

from openai_structured.errors import (
    EmptyResponseError,
    InvalidResponseFormatError,
    JSONParseError,
)
from openai_structured.client import (
    openai_structured_call,
    async_openai_structured_call,
)
from openai_structured.examples.schemas import SimpleMessage

# Test cases for response handling
RESPONSE_TEST_CASES = [
    ('{"message": "test"}', True, None),  # valid
    ("invalid{json", False, InvalidResponseFormatError),  # invalid json
    ("", False, EmptyResponseError),  # empty
    ('{"wrong_field": "test"}', False, InvalidResponseFormatError),  # schema mismatch
]

# Default test parameters
DEFAULT_TEST_PARAMS = {
    "model": "gpt-4o",
    "output_schema": SimpleMessage,
    "system_prompt": "You are a test assistant.",
    "user_prompt": "test",
}

@pytest.mark.parametrize(
    "input_data,is_valid,expected_error",
    RESPONSE_TEST_CASES,
    ids=["valid", "invalid_json", "empty", "schema_mismatch"]
)
def test_response_handling(
    mock_openai_sync_client: MagicMock,
    input_data: str,
    is_valid: bool,
    expected_error: Optional[Type[Exception]]
) -> None:
    """Test response handling with various inputs."""
    # Mock the response
    mock_response = MagicMock()
    if input_data:
        mock_response.choices = [MagicMock(message=MagicMock(content=input_data))]
    else:
        # For empty response test
        mock_response.choices = []
    mock_openai_sync_client.chat.completions.create.return_value = mock_response

    if not is_valid:
        with pytest.raises(expected_error):  # type: ignore
            openai_structured_call(
                client=mock_openai_sync_client,
                **DEFAULT_TEST_PARAMS
            )
    else:
        result = openai_structured_call(
            client=mock_openai_sync_client,
            **DEFAULT_TEST_PARAMS
        )
        assert isinstance(result, SimpleMessage)
        assert result.message == "test"


@pytest.mark.asyncio
async def test_async_response_handling(
    mock_openai_async_client: MagicMock,
) -> None:
    """Test async response handling."""
    # Create async mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"message": "test"}'))]
    
    # Make create method async
    mock_openai_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await async_openai_structured_call(
        client=mock_openai_async_client,
        **DEFAULT_TEST_PARAMS
    )

    assert isinstance(result, SimpleMessage)
    assert result.message == "test" 