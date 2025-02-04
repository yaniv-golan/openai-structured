"""Tests for example schemas."""

import pytest
from pydantic import ValidationError

from openai_structured.examples.schemas import SentimentMessage, SimpleMessage


def test_simple_message_valid() -> None:
    """Test SimpleMessage with valid data."""
    data = {"message": "Hello, world!"}
    msg = SimpleMessage(**data)
    assert msg.message == "Hello, world!"


def test_simple_message_invalid() -> None:
    """Test SimpleMessage with invalid data."""
    with pytest.raises(ValidationError):
        SimpleMessage(**{"wrong_field": "test"})


def test_sentiment_message_valid() -> None:
    """Test SentimentMessage with valid data."""
    data = {"message": "I love this!", "sentiment": "positive"}
    msg = SentimentMessage(**data)
    assert msg.message == "I love this!"
    assert msg.sentiment == "positive"


def test_sentiment_message_invalid_sentiment() -> None:
    """Test SentimentMessage with invalid sentiment."""
    with pytest.raises(ValidationError):
        SentimentMessage(
            **{
                "message": "test",
                "sentiment": "invalid",  # Must be positive/negative/neutral/mixed
            }
        )


def test_sentiment_message_case_insensitive() -> None:
    """Test SentimentMessage accepts case-insensitive sentiments."""
    variations = ["POSITIVE", "negative", "Neutral", "MIXED"]
    for sentiment in variations:
        msg = SentimentMessage(message="test", sentiment=sentiment)
        assert msg.sentiment == sentiment
