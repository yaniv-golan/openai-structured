"""Tests for the CLI implementation."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIStatusError
from pydantic import BaseModel

from openai_structured.cli import (
    create_dynamic_model,
    validate_json_schema,
    validate_response,
    validate_template,
    estimate_tokens_for_chat,
    get_default_token_limit,
    _main,
)
from openai_structured.errors import ModelNotSupportedError


@pytest.fixture
def temp_files(tmp_path):
    """Create temporary test files."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("Content of file 1")
    file2 = tmp_path / "file2.txt"
    file2.write_text("Content of file 2")
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps({
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "score": {"type": "integer"}
        },
        "required": ["summary", "score"]
    }))
    return {
        "file1": file1,
        "file2": file2,
        "schema": schema_file
    }


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_chat = AsyncMock()
    mock_completions = AsyncMock()
    mock_client.chat = mock_chat
    mock_chat.completions = mock_completions
    
    # Set up a default response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"summary": "Test summary", "score": 5}'
    mock_completions.create.return_value = mock_response
    
    return mock_client


def test_create_dynamic_model():
    """Test creating a Pydantic model from JSON schema."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "scores": {"type": "array"},
            "metadata": {"type": "object"}
        }
    }
    model = create_dynamic_model(schema)
    assert issubclass(model, BaseModel)
    assert model.model_fields["name"].annotation == str
    assert model.model_fields["age"].annotation == int
    assert model.model_fields["scores"].annotation == list
    assert model.model_fields["metadata"].annotation == dict


def test_validate_template():
    """Test template validation."""
    template = "Compare {file1} with {file2}"
    files = {"file1", "file2"}
    validate_template(template, files)  # Should not raise

    with pytest.raises(ValueError, match="missing files"):
        validate_template(template, {"file1"})


def test_estimate_tokens():
    """Test token estimation."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ]
    tokens = estimate_tokens_for_chat(messages, "gpt-4")
    assert tokens > 0


def test_get_default_token_limit():
    """Test default token limits for different models."""
    assert get_default_token_limit("o1") == 100_000
    assert get_default_token_limit("gpt-4o") == 16_384
    assert get_default_token_limit("gpt-4o-mini") == 16_384
    assert get_default_token_limit("unknown-model") == 4_096


def test_get_context_window_limit():
    """Test context window limits for different models."""
    assert get_context_window_limit("o1") == 200_000
    assert get_context_window_limit("gpt-4o") == 128_000
    assert get_context_window_limit("gpt-4o-mini") == 128_000
    assert get_context_window_limit("unknown-model") == 8_192


def test_validate_token_limits():
    """Test token limit validation."""
    # Should not raise for valid token counts
    validate_token_limits("gpt-4o", total_tokens=1000, max_token_limit=16_384)
    validate_token_limits("o1", total_tokens=50_000, max_token_limit=100_000)
    
    # Should raise for exceeding context window
    with pytest.raises(ValueError, match="exceed model's context window limit"):
        validate_token_limits("gpt-4o", total_tokens=130_000)
    
    # Should raise when not enough room for output
    with pytest.raises(ValueError, match="Only .* tokens remaining"):
        validate_token_limits("gpt-4o", total_tokens=120_000, max_token_limit=16_384)


def test_cli_token_limits(mock_openai_client, temp_files):
    """Test CLI handling of token limits."""
    schema_file = temp_files["schema.json"]
    input_file = temp_files["input.txt"]
    
    # Test o1 model with large token count
    with pytest.raises(SystemExit) as exc_info:
        _main([
            "--system-prompt", "x" * 190_000,  # Very large prompt
            "--template", "{file1}",
            "--file", f"file1={input_file}",
            "--schema-file", schema_file,
            "--model", "o1",
        ])
    assert exc_info.value.code == 1
    
    # Test gpt-4o with context window limit
    with pytest.raises(SystemExit) as exc_info:
        _main([
            "--system-prompt", "x" * 120_000,  # Large prompt
            "--template", "{file1}",
            "--file", f"file1={input_file}",
            "--schema-file", schema_file,
            "--model", "gpt-4o",
        ])
    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_cli_basic(temp_files, mock_openai_client):
    """Test basic CLI functionality."""
    with patch("sys.argv", [
        "cli.py",
        "--system-prompt", "Analyze files",
        "--template", "Compare {file1} with {file2}",
        "--file", f"file1={temp_files['file1']}",
        "--file", f"file2={temp_files['file2']}",
        "--schema-file", str(temp_files["schema"]),
        "--model", "gpt-4o-2024-08-06",
        "--api-key", "test-key"
    ]), patch("openai_structured.cli.AsyncOpenAI", return_value=mock_openai_client):
        await _main()  # Should complete successfully


@pytest.mark.asyncio
async def test_cli_token_limit(temp_files, mock_openai_client):
    """Test token limit handling."""
    with patch("sys.argv", [
        "cli.py",
        "--system-prompt", "Analyze files",
        "--template", "Compare {file1} with {file2}",
        "--file", f"file1={temp_files['file1']}",
        "--file", f"file2={temp_files['file2']}",
        "--schema-file", str(temp_files["schema"]),
        "--model", "gpt-4o-2024-08-06",
        "--max-token-limit", "1",
        "--api-key", "test-key"
    ]), patch("openai_structured.cli.AsyncOpenAI", return_value=mock_openai_client):
        with pytest.raises(SystemExit) as exc_info:
            await _main()
        assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_cli_rate_limit(temp_files, mock_openai_client):
    """Test rate limit error handling."""
    mock_response = MagicMock()
    mock_response.status = 429
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    mock_openai_client.chat.completions.create.side_effect = APIStatusError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}}
    )
    with patch("sys.argv", [
        "cli.py",
        "--system-prompt", "Analyze files",
        "--template", "Compare {file1} with {file2}",
        "--file", f"file1={temp_files['file1']}",
        "--file", f"file2={temp_files['file2']}",
        "--schema-file", str(temp_files["schema"]),
        "--model", "gpt-4o-2024-08-06",
        "--api-key", "test-key"
    ]), patch("openai_structured.cli.AsyncOpenAI", return_value=mock_openai_client):
        with pytest.raises(SystemExit) as exc_info:
            await _main()
        assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_cli_model_not_supported(temp_files, mock_openai_client):
    """Test unsupported model error handling."""
    with patch("sys.argv", [
        "cli.py",
        "--system-prompt", "Analyze files",
        "--template", "Compare {file1} with {file2}",
        "--file", f"file1={temp_files['file1']}",
        "--file", f"file2={temp_files['file2']}",
        "--schema-file", str(temp_files["schema"]),
        "--model", "gpt-3.5-turbo",
        "--api-key", "test-key"
    ]), patch("openai_structured.cli.AsyncOpenAI", return_value=mock_openai_client):
        with pytest.raises(SystemExit) as exc_info:
            await _main()
        assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_cli_api_error(temp_files, mock_openai_client):
    """Test API error handling."""
    mock_request = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = APIConnectionError(request=mock_request)
    with patch("sys.argv", [
        "cli.py",
        "--system-prompt", "Analyze files",
        "--template", "Compare {file1} with {file2}",
        "--file", f"file1={temp_files['file1']}",
        "--file", f"file2={temp_files['file2']}",
        "--schema-file", str(temp_files["schema"]),
        "--model", "gpt-4o-2024-08-06",
        "--api-key", "test-key"
    ]), patch("openai_structured.cli.AsyncOpenAI", return_value=mock_openai_client):
        with pytest.raises(SystemExit) as exc_info:
            await _main()
        assert exc_info.value.code == 1


def test_cli_subprocess(temp_files):
    """Test CLI using subprocess."""
    import subprocess
    
    result = subprocess.run([
        sys.executable,
        "-m", "openai_structured.cli",
        "--system-prompt", "Analyze files",
        "--template", "Compare {file1} with {file2}",
        "--file", f"file1={temp_files['file1']}",
        "--file", f"file2={temp_files['file2']}",
        "--schema-file", str(temp_files["schema"]),
        "--model", "gpt-4",
        "--api-key", "invalid-key"  # This should fail
    ], capture_output=True, text=True)
    
    assert result.returncode == 1
    assert "Model not supported" in result.stderr


def test_cli_subprocess_stdin(temp_files):
    """Test CLI with stdin input."""
    import subprocess
    
    process = subprocess.Popen([
        sys.executable,
        "-m", "openai_structured.cli",
        "--system-prompt", "Analyze text",
        "--template", "Analyze {stdin}",
        "--schema-file", str(temp_files["schema"]),
        "--model", "gpt-4",
        "--api-key", "invalid-key"
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout, stderr = process.communicate(input="Test input")
    assert process.returncode == 1
    assert "Model not supported" in stderr


def test_cli_subprocess_token_limit(temp_files):
    """Test CLI token limit using subprocess."""
    import subprocess
    
    result = subprocess.run([
        sys.executable,
        "-m", "openai_structured.cli",
        "--system-prompt", "Analyze files",
        "--template", "Compare {file1} with {file2}",
        "--file", f"file1={temp_files['file1']}",
        "--file", f"file2={temp_files['file2']}",
        "--schema-file", str(temp_files["schema"]),
        "--model", "gpt-4",
        "--max-token-limit", "1",  # Very low limit
        "--api-key", "test-key"
    ], capture_output=True, text=True)
    
    assert result.returncode == 1
    assert "exceeds limit" in result.stderr 