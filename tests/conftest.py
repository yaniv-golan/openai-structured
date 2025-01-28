"""Test configuration and fixtures."""

from typing import Generator

import pytest
from dotenv import load_dotenv
from pyfakefs.fake_filesystem import FakeFilesystem

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
