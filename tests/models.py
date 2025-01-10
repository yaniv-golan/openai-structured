"""Test models for structured output."""

from pydantic import BaseModel


class MockResponseModel(BaseModel):
    """Model for mocked API responses in tests."""

    value: str
