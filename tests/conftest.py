# tests/conftest.py
# Add any pytest configuration here.

"""Test configuration for pytest."""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line(
        "markers",
        "no_collect: mark class to not be collected by pytest",
    )


@pytest.fixture(autouse=True)
def env_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
