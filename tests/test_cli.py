"""Tests for the CLI implementation."""

import os
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from openai_structured.cli import (
    ExitCode,
    _main,
    create_dynamic_model,
    estimate_tokens_for_chat,
    get_context_window_limit,
    get_default_token_limit,
    main,
    read_file,
    render_template,
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
    # Valid template
    template = """
    {{ code | remove_comments | dedent }}
    {{ text | wrap(80) | indent(4) }}
    """
    validate_template_placeholders(template, {"code", "text"})

    # Missing placeholder
    with pytest.raises(ValueError, match="missing files"):
        validate_template_placeholders(template, {"code"})

    # Invalid syntax
    with pytest.raises(ValueError, match="Invalid template syntax"):
        validate_template_placeholders("{% if x %}", set())  # Missing endif


def test_validate_template_placeholders_extended() -> None:
    """Test validate_template_placeholders with extended features."""
    # Test valid template with functions and options
    template = """
    {{ read_file('test.txt', encoding='utf-8', use_cache=True) }}
    {{ process_code(input, 'python', 'plain') }}
    {{ estimate_tokens(input, model='gpt-4') }}
    """
    validate_template_placeholders(template, {"input"})  # Should not raise
    
    # Test valid template with filters and options
    template = """
    {{ input | process_code('python', 'html') }}
    {{ input | estimate_tokens(model='gpt-4') }}
    """
    validate_template_placeholders(template, {"input"})  # Should not raise
    
    # Test valid template with nested functions and options
    template = """
    {{ process_code(read_file('test.txt', encoding='utf-8'), 'python', 'html') }}
    {{ estimate_tokens(read_file('test.txt'), model='gpt-4') }}
    """
    validate_template_placeholders(template, set())  # Should not raise
    
    # Test invalid syntax
    template = "{{ {% }}"  # Invalid syntax
    with pytest.raises(ValueError) as exc_info:
        validate_template_placeholders(template, set())
    assert "Invalid template syntax" in str(exc_info.value)
    
    # Test undefined variables
    template = "{{ undefined_var }}"
    with pytest.raises(ValueError) as exc_info:
        validate_template_placeholders(template, set())
    assert "Template placeholders missing files" in str(exc_info.value)


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
                "test {{ missing }}",  # Missing file mapping
                "--schema-file",
                "schema.json",
                "--api-key",
                "dummy-key",  # Add API key to prevent auth error
            ],
        ),
        patch.object(Path, "open", create=True),
        patch("json.load", return_value={}),
        patch(
            "openai_structured.cli.AsyncOpenAI",
            side_effect=Exception("Should not be called"),
        ),
        patch("sys.stdin.isatty", return_value=True),
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
        patch("sys.stdin.isatty", return_value=True),
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
        patch("sys.stdin.isatty", return_value=True),
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
        patch("sys.stdin.isatty", return_value=True),
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
        patch("sys.stdin.isatty", return_value=True),
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
        patch("sys.stdin.isatty", return_value=True),
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


@pytest.mark.asyncio
async def test_stdin_handling() -> None:
    """Test stdin handling in CLI."""
    # Test stdin from pipe
    mock_responses = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content='"test input"'))])
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
        patch("sys.stdin", StringIO("test input")),
        patch(
            "sys.argv",
            [
                "ostruct",
                "--system-prompt",
                "test",
                "--template",
                "{{ stdin }}",
                "--schema-file",
                "schema.json",
                "--model",
                "gpt-4o",
                "--api-key",
                "test-key",
            ],
        ),
        patch("builtins.open", mock_open(read_data='{"type": "string"}')),
        patch("sys.stdin.isatty", return_value=False),
        patch("openai_structured.cli.AsyncOpenAI", return_value=mock_client),
        patch("tiktoken.get_encoding") as mock_get_encoding,
    ):
        # Mock tiktoken
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding

        result = await _main()
        assert result == ExitCode.SUCCESS


def test_read_file_security() -> None:
    """Test read_file security checks."""
    current_time = time.time()

    # Test path traversal attempt
    with (
        patch("builtins.open", mock_open(read_data="test content")),
        patch("os.path.getmtime", return_value=current_time),
    ):
        with pytest.raises(ValueError, match="Access denied"):
            read_file("../outside.txt")
    
    # Test valid path
    with (
        patch("builtins.open", mock_open(read_data="test content")),
        patch("os.path.getmtime", return_value=current_time),
    ):
        content = read_file("test.txt")
        assert content == "test content"
    
    # Test encoding error by simulating a file that raises UnicodeDecodeError when read
    mock = mock_open()
    handle = mock.return_value
    handle.read.side_effect = UnicodeDecodeError(
        "utf-8", b"test", 0, 1, "invalid byte"
    )
    
    with (
        patch("builtins.open", mock),
        patch("os.path.getmtime", return_value=current_time),
    ):
        with pytest.raises(OSError, match="Failed to read file"):
            read_file(
                "test.txt", use_cache=False
            )  # Disable caching for this test


def test_template_filters() -> None:
    """Test template filters."""
    template = """
    {%- set code = code | process_code -%}
    {{ code }}
    {{ text | wrap | indent }}
    """

    context = {
        "code": "def test():\n    # comment\n    pass",
        "text": "This is a long text that needs to be wrapped and indented properly for formatting purposes.",
    }

    result = render_template(template, context)
    assert "def test():" in result
    assert "pass" in result
    assert "# comment" not in result
    assert "    This is a long text" in result
