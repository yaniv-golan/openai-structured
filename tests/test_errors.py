# tests/test_errors.py
import pytest

from openai_structured.errors import (
    APIResponseError,
    EmptyResponseError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
    OpenAIClientError,
)


def test_custom_errors():
    with pytest.raises(OpenAIClientError):
        raise OpenAIClientError("Test client error")

    with pytest.raises(ModelNotSupportedError):
        raise ModelNotSupportedError("Test model not supported")

    with pytest.raises(APIResponseError):
        raise APIResponseError("Test API error")

    with pytest.raises(InvalidResponseFormatError):
        raise InvalidResponseFormatError("Test invalid format")

    with pytest.raises(EmptyResponseError):
        raise EmptyResponseError("Test empty response")
