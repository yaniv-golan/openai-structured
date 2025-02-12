"""Test configuration and fixtures."""

import pytest
from dotenv import load_dotenv

from openai_structured.model_registry import ModelRegistry

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
    """Set up environment variables for testing.

    Args:
        request: Pytest fixture request
        monkeypatch: Pytest monkeypatch fixture
    """
    # Load .env file for live tests
    if "live" in request.keywords:
        load_dotenv()
    # Only set test key for non-live tests
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def registry() -> ModelRegistry:
    """Create a model registry for testing.

    Returns:
        ModelRegistry: A model registry instance for testing
    """
    return ModelRegistry()
