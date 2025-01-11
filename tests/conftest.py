"""Test configuration and fixtures."""

import asyncio
from typing import Generator

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# TODO(#XX): Create tracking issue for pytest-asyncio event loop migration
# - Monitor pytest-asyncio's built-in loop management capabilities
# - Test streaming behavior with built-in management
# - Verify proper cleanup of StreamBuffer resources
# - Plan migration once equivalent guarantees are provided

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


@pytest_asyncio.fixture(scope="function")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create and provide a new event loop for each test.

    Note: While pytest-asyncio suggests deprecating custom event loop fixtures,
    we maintain this implementation for several critical reasons:

    1. Streaming Resource Management:
       - Ensures proper cleanup of StreamBuffer instances
       - Handles partial JSON state in streaming responses
       - Maintains memory cleanup between test runs

    2. Test Isolation:
       - Function scope provides clean state for each test
       - Prevents streaming state leakage between tests
       - Critical for error handling scenarios

    3. Explicit Cleanup:
       - Ensures running loops are properly stopped
       - Guarantees loop closure even after streaming errors
       - Maintains consistent state for mock and live API tests

    Future Migration:
    This implementation will be maintained until pytest-asyncio's built-in
    loop management can provide equivalent guarantees for streaming operations.
    Track: [Add issue link for migration tracking]
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    if loop.is_running():
        loop.stop()
    if not loop.is_closed():
        loop.close()
