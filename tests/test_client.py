# tests/test_client.py
from typing import List
from unittest.mock import MagicMock

import pytest
from openai import OpenAI
from pydantic import BaseModel

from openai_structured.client import (openai_structured_call,
                                      openai_structured_stream)
from openai_structured.errors import APIResponseError, ModelNotSupportedError


class MockOutput(BaseModel):
    name: str
    age: int


def test_openai_structured_call_success(mocker):
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content='{"name": "Alice", "age": 30}'))
    ]
    mock_client.chat.completions.create.return_value.id = "test_id"
    mock_client.chat.completions.create.return_value.model = "gpt-4o"
    mock_client.chat.completions.create.return_value.usage = MagicMock(
        dict=lambda: {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )

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
