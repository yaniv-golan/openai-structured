"""Tests for the CLI implementation."""

import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from openai_structured.cli import (
    ExitCode,
    create_dynamic_model,
    estimate_tokens_for_chat,
    get_context_window_limit,
    get_default_token_limit,
    main,
    validate_template_placeholders,
)
from openai_structured.errors import StreamInterruptedError, StreamParseError


class SampleOutputSchema(BaseModel):
    """Schema used for testing CLI output."""

    field: str


def test_create_dynamic_model() -> None:
    """Test creating a dynamic model from JSON schema."""
    schema = {
        "type": "object",
        "properties": {
            "string_field": {"type": "string"},
            "int_field": {"type": "integer"},
            "float_field": {"type": "number"},
            "bool_field": {"type": "boolean"},
            "array_field": {"type": "array"},
            "object_field": {"type": "object"},
        },
    }

    model = create_dynamic_model(schema)
    assert issubclass(model, BaseModel)


def test_validate_template_placeholders() -> None:
    """Test template placeholder validation."""
    available_files = {"file1", "file2"}

    # Valid template
    validate_template_placeholders("Test {file1} and {file2}", available_files)

    # Missing placeholder
    with pytest.raises(
        ValueError, match="Template placeholders missing files"
    ):
        validate_template_placeholders("Test {file3}", available_files)


def test_estimate_tokens_for_chat() -> None:
    """Test token estimation for chat messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]

    with patch("tiktoken.get_encoding") as mock_get_encoding:
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Mock token IDs
        mock_get_encoding.return_value = mock_encoding

        tokens = estimate_tokens_for_chat(messages, "gpt-4o")
        assert tokens > 0


def test_get_context_window_limit() -> None:
    """Test getting context window limits."""
    assert get_context_window_limit("o1") == 200_000
    assert get_context_window_limit("gpt-4o") == 128_000
    assert get_context_window_limit("gpt-4o-mini") == 128_000
    assert get_context_window_limit("unknown") == 8_192


def test_get_default_token_limit() -> None:
    """Test getting default token limits."""
    assert get_default_token_limit("o1") == 100_000
    assert get_default_token_limit("gpt-4o") == 16_384
    assert get_default_token_limit("gpt-4o-mini") == 16_384
    assert get_default_token_limit("unknown") == 4_096


@pytest.mark.asyncio
async def test_cli_validation_error() -> None:
    """Test CLI with validation error."""
    with (
        patch(
            "sys.argv",
            [
                "cli",
                "--system-prompt",
                "test",
                "--template",
                "test {missing}",  # Missing file mapping
                "--schema-file",
                "schema.json",
            ],
        ),
        patch.object(Path, "open", create=True),
        patch("json.load", return_value={}),
    ):
        result = await main()
        assert result == ExitCode.VALIDATION_ERROR


@pytest.mark.asyncio
async def test_cli_usage_error() -> None:
    """Test CLI with usage error."""
    with (
        patch("sys.argv", ["cli"]),  # Missing required arguments
        pytest.raises(SystemExit) as exc_info,
    ):
        await main()
        assert (
            exc_info.value.code == 2
        )  # argparse exits with status code 2 for usage errors


@pytest.mark.asyncio
async def test_cli_io_error() -> None:
    """Test CLI with IO error."""
    with (
        patch(
            "sys.argv",
            [
                "cli",
                "--system-prompt",
                "test",
                "--template",
                "test {input}",
                "--file",
                "input=nonexistent.txt",
                "--schema-file",
                "schema.json",
            ],
        ),
        patch.object(Path, "open", side_effect=OSError),
    ):
        result = await main()
        assert result == ExitCode.IO_ERROR


@pytest.mark.asyncio
async def test_cli_streaming_success() -> None:
    """Test successful streaming CLI execution."""
    schema = {
        "type": "object",
        "properties": {"field": {"type": "string"}},
        "required": ["field"],
    }

    # Create mock stream responses
    mock_responses = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content='{"field": '))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content='"test"'))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="}"))]),
    ]

    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        for response in mock_responses:
            yield response

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream())

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    # Create mock encoding
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3]
    mock_get_encoding = MagicMock(return_value=mock_encoding)

    with (
        patch("builtins.open", create=True),
        patch("json.load", return_value=schema),
        patch("tiktoken.get_encoding", mock_get_encoding),
        patch("openai_structured.cli.AsyncOpenAI", return_value=mock_client),
        patch.object(
            sys,
            "argv",
            [
                "cli",
                "--model",
                "gpt-4o",
                "--system-prompt",
                "You are a helpful assistant",
                "--template",
                "Process this: {input}",
                "--file",
                "input=test.txt",
                "--schema-file",
                "schema.json",
            ],
        ),
    ):
        result = await main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio
async def test_cli_streaming_interruption() -> None:
    """Test CLI with stream interruption."""
    schema = {
        "type": "object",
        "properties": {"field": {"type": "string"}},
    }

    # Create mock stream that raises interruption
    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        yield MagicMock(
            choices=[MagicMock(delta=MagicMock(content='{"field": '))]
        )
        raise StreamInterruptedError("Stream interrupted")

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream())

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    with (
        patch(
            "sys.argv",
            [
                "cli",
                "--system-prompt",
                "test",
                "--template",
                "test {input}",
                "--file",
                "input=test.txt",
                "--schema-file",
                "schema.json",
                "--model",
                "gpt-4o",
            ],
        ),
        patch.object(Path, "open", create=True),
        patch("json.load", return_value=schema),
        patch("builtins.open", create=True),
        patch("openai_structured.cli.AsyncOpenAI", return_value=mock_client),
        patch("tiktoken.get_encoding") as mock_get_encoding,
    ):
        # Mock tiktoken
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding

        result = await main()
        assert result == ExitCode.API_ERROR


@pytest.mark.asyncio
async def test_cli_streaming_parse_error() -> None:
    """Test CLI with stream parse error."""
    schema = {
        "type": "object",
        "properties": {"field": {"type": "string"}},
    }

    # Create mock stream that raises parse error
    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        yield MagicMock(
            choices=[MagicMock(delta=MagicMock(content="invalid json"))]
        )
        raise StreamParseError(
            "Failed to parse stream", 3, ValueError("Invalid JSON")
        )

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream())

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    with (
        patch(
            "sys.argv",
            [
                "cli",
                "--system-prompt",
                "test",
                "--template",
                "test {input}",
                "--file",
                "input=test.txt",
                "--schema-file",
                "schema.json",
                "--model",
                "gpt-4o",
            ],
        ),
        patch.object(Path, "open", create=True),
        patch("json.load", return_value=schema),
        patch("builtins.open", create=True),
        patch("openai_structured.cli.AsyncOpenAI", return_value=mock_client),
        patch("tiktoken.get_encoding") as mock_get_encoding,
    ):
        # Mock tiktoken
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding

        result = await main()
        assert result == ExitCode.API_ERROR


@pytest.mark.asyncio
async def test_cli_streaming_with_output_file() -> None:
    """Test streaming CLI with output file."""
    schema = {
        "type": "object",
        "properties": {"field": {"type": "string"}},
    }

    # Create mock stream responses
    mock_responses = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content='{"field": '))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content='"test"'))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="}"))]),
    ]

    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        for response in mock_responses:
            yield response

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream())

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    with (
        patch(
            "sys.argv",
            [
                "cli",
                "--system-prompt",
                "test",
                "--template",
                "test {input}",
                "--file",
                "input=test.txt",
                "--schema-file",
                "schema.json",
                "--model",
                "gpt-4o",
                "--output-file",
                "output.json",
            ],
        ),
        patch.object(Path, "open", create=True),
        patch("json.load", return_value=schema),
        patch("builtins.open", create=True),
        patch("openai_structured.cli.AsyncOpenAI", return_value=mock_client),
        patch("tiktoken.get_encoding") as mock_get_encoding,
        patch.object(Path, "mkdir", create=True),
    ):
        # Mock tiktoken
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding

        result = await main()
        assert result == ExitCode.SUCCESS


@pytest.mark.asyncio
async def test_cli_api_error() -> None:
    """Test CLI with API error."""
    schema = {
        "type": "object",
        "properties": {"field": {"type": "string"}},
    }

    # Create a debug handler to capture all log messages
    debug_logs = []

    def debug_handler(*args: Any, **kwargs: Any) -> None:
        debug_logs.append(" ".join(str(arg) for arg in args))

    def error_handler(*args: Any, **kwargs: Any) -> None:
        debug_logs.append(" ".join(str(arg) for arg in args))

    # Create mock stream that raises API error
    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        # First yield a mock response to simulate stream start
        yield MagicMock(
            choices=[MagicMock(delta=MagicMock(content='{"field": '))]
        )
        # Then raise the API error
        raise BadRequestError(
            message="Invalid request",
            response=MagicMock(
                status=400,
                headers={},
                text='{"error": {"message": "Invalid request", "type": "invalid_request_error"}}',
            ),
            body={
                "error": {
                    "message": "Invalid request",
                    "type": "invalid_request_error",
                }
            },
        )

    # Create mock completions
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream())

    # Create mock chat
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    # Create mock client
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat

    with (
        patch(
            "sys.argv",
            [
                "cli",
                "--system-prompt",
                "test",
                "--template",
                "test {input}",
                "--file",
                "input=test.txt",
                "--schema-file",
                "schema.json",
                "--model",
                "gpt-4o",
                "--api-key",
                "test-key",
                "--verbose",  # Enable verbose logging
            ],
        ),
        patch.object(Path, "open", create=True),
        patch("json.load", return_value=schema),
        patch("builtins.open", create=True),
        patch("openai_structured.cli.AsyncOpenAI", return_value=mock_client),
        patch("tiktoken.get_encoding") as mock_get_encoding,
        patch.dict(os.environ, {}, clear=True),
        patch("logging.getLogger") as mock_get_logger,
    ):
        # Set up logging to capture all messages
        mock_logger = MagicMock()
        mock_logger.debug.side_effect = debug_handler
        mock_logger.error.side_effect = error_handler
        mock_get_logger.return_value = mock_logger

        # Mock tiktoken
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding

        result = await main()
        assert result == ExitCode.API_ERROR

        # Print logs for debugging
        print("\nDebug logs:")
        for log in debug_logs:
            print(f"  {log}")

        # Check for both possible error messages
        error_found = any(
            "API error: Invalid request" in log
            or "Stream interrupted: Invalid request" in log
            for log in debug_logs
        )
        assert (
            error_found
        ), f"Expected error message not found in logs: {debug_logs}"
