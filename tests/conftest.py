"""Test configuration and fixtures."""

# Standard library imports

# Third-party imports
import pytest
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pyfakefs.fake_filesystem_unittest import Patcher

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
def mock_response() -> MockResponse:
    """Create a mock response for testing."""
    return MockResponse(message="Hello!", sentiment="positive")
