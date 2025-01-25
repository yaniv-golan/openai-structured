"""Test configuration and fixtures."""

import json
from typing import Dict, Generator, Any
from unittest.mock import MagicMock, AsyncMock

import pytest
from dotenv import load_dotenv
from openai import OpenAI
from pyfakefs.fake_filesystem import FakeFilesystem

from openai_structured.testing import (
    create_structured_response,
    create_structured_stream_response,
)
from openai_structured.examples.schemas import SimpleMessage

pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line(
        "markers",
        "no_collect: mark class to not be collected by pytest",
    )
    config.addinivalue_line(
        "markers",
        "live: mark test as a live test that should use real API key",
    )


@pytest.fixture(autouse=True)  # type: ignore[misc]
def env_setup(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set up environment variables for testing."""
    # Load .env file for live tests
    if "live" in request.keywords:
        load_dotenv()
    # Only set test key for non-live tests
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture  # type: ignore[misc]
def fs(fs: FakeFilesystem) -> Generator[FakeFilesystem, None, None]:
    """Create a fake filesystem for testing.

    This fixture is automatically used by tests that have an fs parameter.
    It provides a clean filesystem for each test, preventing interference
    between tests.

    Args:
        fs: The pyfakefs fixture

    Returns:
        The pyfakefs FakeFilesystem object
    """
    # pyfakefs already sets up common system paths
    # We can add any additional setup here if needed in the future
    yield fs


@pytest.fixture  # type: ignore[misc]
def mock_openai_sync_client() -> MagicMock:
    """Create a mock OpenAI sync client."""
    mock = MagicMock()
    mock.chat.completions.create = create_structured_response(
        output_schema=SimpleMessage,
        data={"message": "test"}
    )
    return mock


@pytest.fixture  # type: ignore[misc]
def mock_openai_async_client() -> AsyncMock:
    """Create a mock OpenAI async client."""
    mock = AsyncMock()
    mock.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data={"message": "test"}
    )
    return mock


@pytest.fixture  # type: ignore[misc]
def mock_openai_sync_stream_client() -> MagicMock:
    """Create a mock OpenAI sync client for streaming."""
    mock = MagicMock()
    mock.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=[
            {"message": "part1"},
            {"message": "part2"}
        ]
    )
    return mock


@pytest.fixture  # type: ignore[misc]
def mock_openai_async_stream_client() -> AsyncMock:
    """Create a mock OpenAI async client for streaming."""
    mock = AsyncMock()
    mock.chat.completions.create = create_structured_stream_response(
        output_schema=SimpleMessage,
        data=[
            {"message": "part1"},
            {"message": "part2"}
        ]
    )
    return mock


@pytest.fixture  # type: ignore[misc]
def test_files(fs: FakeFilesystem) -> Dict[str, str]:
    """Create test files for testing.

    This fixture creates a set of test files in a temporary directory
    and returns their paths.

    Args:
        fs: The pyfakefs fixture

    Returns:
        A dictionary mapping file names to their paths
    """
    base_dir = "/Users/yaniv/Documents/code/openai-structured/tests/test_files"
    fs.create_dir(base_dir)

    # Create test files
    files = {
        "base_dir": base_dir,
        "input": f"{base_dir}/input.txt",
        "template": f"{base_dir}/template.txt",
        "template_no_prompt": f"{base_dir}/template_no_prompt.txt",
        "schema": f"{base_dir}/schema.json",
        "system_prompt": f"{base_dir}/system.txt",
    }

    # Create input file
    fs.create_file(files["input"], contents="Test input file")

    # Create template with YAML frontmatter
    fs.create_file(
        files["template"],
        contents=(
            "---\n"
            "system_prompt: You are a test assistant using YAML frontmatter.\n"
            "---\n"
            "Process input: {{ input }}"
        ),
    )

    # Create template without YAML frontmatter
    fs.create_file(
        files["template_no_prompt"], contents="Process input: {{ input }}"
    )

    # Create schema file
    schema_content = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "status": {"type": "string"},
        },
        "required": ["result", "status"],
    }
    fs.create_file(files["schema"], contents=json.dumps(schema_content))

    # Create system prompt file
    fs.create_file(
        files["system_prompt"],
        contents="You are a test assistant from a file.",
    )

    return files
