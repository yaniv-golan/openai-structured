"""Test configuration and fixtures."""

# Standard library imports
import asyncio

# Third-party imports
import pytest
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pyfakefs.fake_filesystem_unittest import Patcher
from typing import Dict
from unittest.mock import MagicMock

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


@pytest.fixture(autouse=True)
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


@pytest.fixture
def fs(fs: Patcher) -> Patcher:
    """Create a fake filesystem for testing.
    
    This fixture is automatically used by tests that have an fs parameter.
    It provides a clean filesystem for each test, preventing interference
    between tests.
    
    Args:
        fs: The pyfakefs fixture
        
    Returns:
        The pyfakefs Patcher object
    """
    # pyfakefs already sets up common system paths
    # We can add any additional setup here if needed in the future
    return fs


class MockResponse(BaseModel):
    """Mock response for testing."""

    message: str
    sentiment: str


@pytest.fixture
def mock_openai_client() -> OpenAI:
    """Create a mock OpenAI client for testing."""
    return OpenAI(api_key="test-key", base_url="http://localhost:8000")


@pytest.fixture
def test_files(fs: Patcher) -> Dict[str, str]:
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
        contents="""---
system_prompt: You are a test assistant using YAML frontmatter.
---
Process input: {{ input }}"""
    )
    
    # Create template without YAML frontmatter
    fs.create_file(
        files["template_no_prompt"],
        contents="Process input: {{ input }}"
    )
    
    # Create schema file
    fs.create_file(
        files["schema"],
        contents='{"type": "object", "properties": {"result": {"type": "string"}, "status": {"type": "string"}}, "required": ["result", "status"]}'
    )
    
    # Create system prompt file
    fs.create_file(
        files["system_prompt"],
        contents="You are a test assistant from a file."
    )
    
    return files
