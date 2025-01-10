# tests/test_client.py
from unittest.mock import MagicMock

import pytest
from openai import OpenAI
from pydantic import BaseModel

from openai_structured.client import openai_structured_call
from openai_structured.errors import ModelNotSupportedError


class MockOutput(BaseModel):
    name: str
    age: int


def test_openai_structured_call_success(mocker):
    # Create a more complete mock structure
    mock_completion = MagicMock()
    mock_message = MagicMock(content='{"name": "Alice", "age": 30}')
    mock_completion.choices = [MagicMock(message=mock_message)]
    mock_completion.id = "test_id"
    mock_completion.model = "gpt-4o"
    mock_completion.usage = MagicMock(
        dict=lambda: {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
    )

    # Create the chat completions mock
    mock_chat = MagicMock()
    mock_chat.completions = MagicMock()
    mock_chat.completions.create = MagicMock(return_value=mock_completion)

    # Create the client mock with the chat attribute
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = mock_chat

    result = openai_structured_call(
        client=mock_client,
        model="gpt-4o-2024-08-06",
        output_schema=MockOutput,
        user_prompt="Tell me about a person.",
        system_prompt="Extract person information.",
    )

    assert isinstance(result, MockOutput)
    assert result.name == "Alice"
    assert result.age == 30


def test_openai_structured_call_model_not_supported():
    client = OpenAI()
    with pytest.raises(ModelNotSupportedError):
        openai_structured_call(
            client=client,
            model="unsupported-model",
            output_schema=MockOutput,
            user_prompt="...",
            system_prompt="...",
        )
