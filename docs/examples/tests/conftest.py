"""Shared test fixtures for example tests."""

from pathlib import Path

import pytest

from openai_structured.model_registry import ModelRegistry
from openai_structured.testing import create_test_registry


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
        # Use live configuration files
        live_config_dir = (
            Path(__file__).parent.parent.parent / "tests" / "live_config"
        )
        monkeypatch.setenv(
            "MODEL_REGISTRY_PATH", str(live_config_dir / "models.yml")
        )
        monkeypatch.setenv(
            "PARAMETER_CONSTRAINTS_PATH",
            str(live_config_dir / "parameter_constraints.yml"),
        )
    # Use test configuration files for non-live tests
    else:
        test_dir = (
            Path(__file__).parent.parent.parent
            / "src"
            / "openai_structured"
            / "testing"
            / "templates"
        )
        monkeypatch.setenv(
            "MODEL_REGISTRY_PATH", str(test_dir / "test_models.yml")
        )
        monkeypatch.setenv(
            "PARAMETER_CONSTRAINTS_PATH",
            str(test_dir / "test_constraints.yml"),
        )


@pytest.fixture
def test_registry():
    """Pytest fixture that provides a test registry.

    This fixture creates a fresh test registry for each test,
    ensuring tests are isolated and don't affect each other.

    Example:
        >>> def test_model_capabilities(test_registry):
        ...     capabilities = test_registry.get_capabilities("test-model")
        ...     assert capabilities.context_window == 4096
    """
    registry = create_test_registry()
    yield registry
    ModelRegistry.cleanup()
