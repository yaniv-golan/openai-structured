"""
Tests for the CLI implementation.
"""

import errno
import json
import os
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import jinja2
import pytest
import pytest_asyncio
from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel
from pyfakefs.fake_filesystem_unittest import Patcher

from openai_structured.cli import (
    ExitCode,
    _main,
    cli_main,
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


async def mock_stream_response(content: str = '{"field": "test"}') -> AsyncGenerator[ChatCompletionChunk, None]:
    """
    Mock OpenAI API stream response for testing.
    Yields one chunk containing the specified content.
    """
    response = ChatCompletionChunk(
        id="test_chunk",
        model="gpt-4o",
        object="chat.completion.chunk",
        created=123,
        choices=[
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": "stop",
            }
        ],
    )
    yield response


@pytest_asyncio.fixture
async def mock_openai_stream() -> AsyncMock:
    """Create a mock OpenAI client with streaming response."""
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_stream_response())

    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat = mock_chat
    return mock_client


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

    # Additional validation checks could go here...
    # (omitted for brevity)

# ----------------------------------------------------------------------------------
# Placeholder for additional tests between lines ~77 and ~803
# ----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stdin_handling(fs: Patcher, mock_openai_stream: AsyncMock) -> None:
    """Test stdin handling in CLI."""
    # Create test schema file
    fs.create_file("schema.json", contents='{"type": "string"}')

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
        patch("sys.stdin.isatty", return_value=False),
        patch(
            "openai_structured.cli.cli.AsyncOpenAI",
            return_value=mock_openai_stream,
        ),
        patch("tiktoken.get_encoding") as mock_get_encoding,
    ):
        # Mock tiktoken
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding

        result = await _main()
        assert result == ExitCode.SUCCESS

# ----------------------------------------------------------------------------------
# Placeholder for additional tests between lines ~860 and ~949
# ----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cli_dry_run(fs: Patcher, mock_openai_stream: AsyncMock) -> None:
    """Test CLI dry run mode."""
    # Setup mock data and environment
    system_prompt = "DRY RUN test"
    template_content = f"""---
system_prompt: {system_prompt}
---
Hello, {{{{ file1 }}}}"""
    info_messages: List[str] = []

    def info_handler(*args: Any, **kwargs: Any) -> None:
        """Capture info-level log messages."""
        message = " ".join(str(arg) for arg in args)
        info_messages.append(message)
        print(f"INFO: {message}")  # Debug output

    # Create test files in the fake filesystem
    fs.create_file("input.txt", contents="test content")
    fs.create_file("schema.json", contents='{"type": "object", "properties": {"field": {"type": "string"}}}')
    fs.create_file("test.txt", contents="sample text file")
    fs.create_file("template.txt", contents=template_content)

    with (
        patch("sys.argv", [
            "ostruct",
            "--template",
            "template.txt",  # Use template file
            "--template-is-file",  # Indicate that template is a file
            "--schema-file",
            "schema.json",
            "--file",
            "file1=input.txt",
            "--dry-run",
            "--validate-schema",
            "--api-key",
            "test-key",
        ]),
        patch("sys.stdin.isatty", return_value=True),
        patch("logging.getLogger") as mock_logger,
        patch("tiktoken.get_encoding") as mock_get_encoding,
        patch("openai_structured.cli.cli.AsyncOpenAI", return_value=mock_openai_stream),
    ):
        # Mock tiktoken
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Mock token IDs
        mock_get_encoding.return_value = mock_encoding

        mock_logger.return_value.info = info_handler
        mock_logger.return_value.debug = lambda *args, **kwargs: None
        mock_logger.return_value.error = lambda *args, **kwargs: None

        exit_code = await main()
        assert exit_code == ExitCode.SUCCESS

        print("\nAll captured messages:")
        for msg in info_messages:
            print(f"  {msg}")

        # Verify dry run output
        assert any("DRY RUN MODE" in msg for msg in info_messages), "Missing DRY RUN MODE message"
        assert any("System Prompt:" in msg for msg in info_messages), "Missing System Prompt message"
        assert any("User Prompt:" in msg for msg in info_messages), "Missing User Prompt message"
        assert any("test content" in msg for msg in info_messages), "Missing test content in messages"

# ----------------------------------------------------------------------------------
# Placeholder for additional tests between lines ~1004 and ~1048
# ----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cli_template_system_prompt(fs: Patcher, mock_openai_stream: AsyncMock) -> None:
    """Test system prompt from template content."""
    # Setup mock data
    schema = {
        "type": "object",
        "properties": {"field": {"type": "string"}},
        "required": ["field"],
    }

    # Create test files in the fake filesystem
    fs.create_file("template.txt", contents="---\nsystem_prompt: You are analyzing {{ language }} code.\n---\n")
    fs.create_file("test.txt", contents="test content")
    fs.create_file("schema.json", contents=json.dumps(schema))

    debug_logs: List[str] = []

    def debug_handler(*args: Any, **kwargs: Any) -> None:
        debug_logs.append(" ".join(str(arg) for arg in args))

    with (
        patch(
            "sys.argv",
            [
                "cli",
                "--template",
                "template.txt",
                "--file",
                "input=test.txt",
                "--value",
                "language=python",
                "--schema-file",
                "schema.json",
                "--model",
                "gpt-4o",
                "--verbose",
            ],
        ),
        patch("openai_structured.cli.cli.AsyncOpenAI", return_value=mock_openai_stream),
        patch("tiktoken.get_encoding") as mock_get_encoding,
        patch("sys.stdin.isatty", return_value=True),
        patch("logging.getLogger") as mock_get_logger,
    ):
        mock_logger = MagicMock()
        mock_logger.debug.side_effect = debug_handler
        mock_get_logger.return_value = mock_logger

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding

        result = await main()
        assert result == ExitCode.SUCCESS

        print("\nDebug logs:")
        for log in debug_logs:
            print(f"  {log}")

        # Verify system prompt is set from template.yaml frontmatter
        calls = mock_openai_stream.chat.completions.create.mock_calls
        assert len(calls) == 1
        call_kwargs = calls[0].kwargs
        assert call_kwargs["messages"][0]["content"] == "You are analyzing python code."


# ----------------------------------------------------------------------------------
# Placeholder for additional tests between lines ~1107 and ~1188
# ----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cli_multiple_files(fs: Patcher, mock_openai_stream: AsyncMock) -> None:
    """Test CLI with multiple file inputs."""
    schema = {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
        },
    }

    # Create test files in the fake filesystem
    fs.create_file("test1.py", contents="def test1(): pass")
    fs.create_file("test2.py", contents="def test2(): pass")
    fs.create_file("test3.py", contents="def test3(): pass")
    fs.create_file("test.txt", contents="test content")
    fs.create_file("schema.json", contents=json.dumps(schema))
    
    # Create docs directory with some files
    fs.create_dir("docs")
    fs.create_file("docs/doc1.txt", contents="doc1 content")
    fs.create_file("docs/doc2.txt", contents="doc2 content")

    debug_messages: List[str] = []

    def debug_handler(*args: Any, **kwargs: Any) -> None:
        """Capture debug messages."""
        debug_messages.append(" ".join(str(arg) for arg in args))

    with (
        patch(
            "sys.argv",
            [
                "cli",
                "--template",
                "template.txt",
                "--file",
                "single=test.txt",
                "--files",
                "tests=*.py",
                "--dir",
                "docs=docs",
                "--recursive",
                "--ext",
                ".py,.txt",
                "--value",
                "language=python",
                "--schema-file",
                "schema.json",
                "--model",
                "gpt-4o",
                "--verbose",
            ],
        ),
        patch("openai_structured.cli.cli.AsyncOpenAI", return_value=mock_openai_stream),
        patch("tiktoken.get_encoding") as mock_get_encoding,
        patch("sys.stdin.isatty", return_value=True),
        patch("logging.getLogger") as mock_get_logger,
    ):
        # Set up logging to capture all messages
        mock_logger = MagicMock()
        mock_logger.debug.side_effect = debug_handler
        mock_get_logger.return_value = mock_logger

        # Mock tiktoken
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding

        result = await main()
        assert result == ExitCode.SUCCESS

        print("\nDebug messages:")
        for msg in debug_messages:
            print(f"  {msg}")

        # Verify API call
        calls = mock_openai_stream.chat.completions.create.mock_calls
        assert len(calls) == 1
        call_kwargs = calls[0].kwargs

        # Verify messages
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # Find debug message containing the file mappings
        file_mappings_msg = next(
            (msg for msg in debug_messages if "File mappings for context" in msg),
            "",
        )
        assert file_mappings_msg, "No debug message found with file mappings"

        # Verify file mappings were processed
        assert "single" in file_mappings_msg
        assert "tests" in file_mappings_msg
        assert "docs" in file_mappings_msg
        assert "language" in file_mappings_msg
