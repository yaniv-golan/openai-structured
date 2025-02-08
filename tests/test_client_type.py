# tests/test_client_type.py
import pytest
from openai import AsyncOpenAI, OpenAI

from openai_structured.client import _validate_client_type


def test_official_sync_client() -> None:
    """Test validation of official OpenAI sync client."""
    sync_client = OpenAI()
    _validate_client_type(sync_client, "sync")  # Should not raise
    with pytest.raises(TypeError) as exc:
        _validate_client_type(sync_client, "async")
    assert "Async client required but got sync client" in str(exc.value)


def test_official_async_client() -> None:
    """Test validation of official AsyncOpenAI client."""
    async_client = AsyncOpenAI()
    _validate_client_type(async_client, "async")  # Should not raise
    with pytest.raises(TypeError) as exc:
        _validate_client_type(async_client, "sync")
    assert "Sync client required but got async client" in str(exc.value)
