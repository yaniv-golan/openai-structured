# tests/conftest.py
# Add any pytest configuration here.

"""Test configuration for pytest."""
import os
import pytest


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
def env_setup(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment variables for testing."""
    # Only set test key for non-live tests
    if 'live' not in request.keywords:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
