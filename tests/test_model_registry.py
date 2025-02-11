"""Tests for the model registry functionality."""

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Type

import pytest
import yaml

from openai_structured.errors import (
    InvalidDateError,
    InvalidVersionFormatError,
    ModelNotSupportedError,
    ModelVersionError,
    OpenAIClientError,
    VersionTooOldError,
)
from openai_structured.model_registry import ModelCapabilities, ModelRegistry
from openai_structured.model_version import ModelVersion


@pytest.fixture
def temp_config():
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml") as f:
        test_config = {
            "gpt-4o": {
                "context_window": 128000,
                "max_output_tokens": 16000,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": {
                    "temperature": {"min": 0.0, "max": 2.0},
                    "top_p": {"min": 0.0, "max": 1.0},
                    "frequency_penalty": {"min": -2.0, "max": 2.0},
                    "presence_penalty": {"min": -2.0, "max": 2.0},
                    "max_completion_tokens": {"min": 1, "max": 16000},
                },
                "aliases": ["gpt-4o-latest"],
                "min_version": {
                    "year": 2024,
                    "month": 8,
                    "day": 6,
                },
            },
            "test-model": {
                "context_window": 4096,
                "max_output_tokens": 2048,
                "supports_structured": True,
                "supports_streaming": False,
                "supported_parameters": {
                    "temperature": {"min": 0.0, "max": 2.0},
                    "frequency_penalty": {"min": -2.0, "max": 2.0},
                },
                "aliases": ["test-latest"],
                "min_version": None,
            },
        }
        yaml.safe_dump(test_config, f)
        f.flush()
        yield Path(f.name)


@pytest.fixture
def registry(temp_config, monkeypatch):
    """Create a registry instance with test config."""
    # Reset singleton instance
    ModelRegistry._instance = None

    # Use test config
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(temp_config))

    return ModelRegistry()


def test_model_capabilities_validation():
    """Test ModelCapabilities validation."""
    # Valid capabilities
    caps = ModelCapabilities(
        context_window=4096,
        max_output_tokens=2048,
        supports_structured=True,
        supports_streaming=True,
        supported_parameters={
            "temperature": {"min": 0.0, "max": 2.0},
            "top_p": {"min": 0.0, "max": 1.0},
        },
        aliases={"test-latest"},
        min_version=ModelVersion(2024, 1, 1),
    )
    assert caps.context_window == 4096
    assert caps.max_output_tokens == 2048
    assert caps.supports_structured
    assert caps.supports_streaming
    assert "temperature" in caps.supported_parameters
    assert "top_p" in caps.supported_parameters
    assert caps.aliases == {"test-latest"}
    assert caps.min_version == ModelVersion(2024, 1, 1)

    # Test parameter validation
    with pytest.raises(OpenAIClientError, match="not supported"):
        caps.validate_parameter("invalid_param", 0.5)

    with pytest.raises(OpenAIClientError, match="must be between"):
        caps.validate_parameter("temperature", 2.5)

    # Valid parameter values
    caps.validate_parameter("temperature", 0.5)
    caps.validate_parameter("top_p", 0.9)


def test_parameter_support(registry):
    """Test parameter support validation."""
    # Test GPT-4o parameters
    gpt4o = registry.get_capabilities("gpt-4o")
    gpt4o.validate_parameter("temperature", 0.5)
    gpt4o.validate_parameter("top_p", 0.9)

    with pytest.raises(OpenAIClientError):
        gpt4o.validate_parameter("temperature", 2.5)  # Too high

    with pytest.raises(OpenAIClientError):
        gpt4o.validate_parameter("top_p", -0.1)  # Too low

    # Test test-model parameters
    test_model = registry.get_capabilities("test-model")
    test_model.validate_parameter("temperature", 0.5)

    with pytest.raises(OpenAIClientError):
        test_model.validate_parameter("top_p", 0.9)  # Not supported


def test_get_capabilities(registry):
    """Test getting model capabilities."""
    # Test exact match
    caps = registry.get_capabilities("gpt-4o")
    assert caps.context_window == 128000
    assert caps.max_output_tokens == 16000
    assert caps.supports_structured
    assert caps.supports_streaming
    assert "temperature" in caps.supported_parameters
    assert "gpt-4o-latest" in caps.aliases
    assert caps.min_version == ModelVersion(2024, 8, 6)

    # Test alias resolution
    caps = registry.get_capabilities("gpt-4o-latest")
    assert caps.context_window == 128000

    # Test dated version
    caps = registry.get_capabilities("gpt-4o-2024-08-06")
    assert caps.context_window == 128000

    # Test newer version
    caps = registry.get_capabilities("gpt-4o-2024-09-01")
    assert caps.context_window == 128000

    # Test unsupported model
    with pytest.raises(ModelNotSupportedError):
        registry.get_capabilities("unsupported-model")

    # Test too old version
    with pytest.raises(ModelVersionError):
        registry.get_capabilities("gpt-4o-2024-07-01")


def test_registry_singleton(registry):
    """Test that ModelRegistry is a singleton."""
    registry2 = ModelRegistry()
    assert registry is registry2


def test_registry_cleanup(registry):
    """Test that cleanup properly resets the registry."""
    # Store initial state
    initial_models = registry.models.copy()

    # Verify we have the same instance
    registry2 = ModelRegistry()
    assert registry is registry2
    assert registry2.models == initial_models

    # Clean up
    ModelRegistry.cleanup()

    # Get new instance
    new_registry = ModelRegistry()

    # Verify it's a different instance
    assert new_registry is not registry

    # Verify models are reloaded
    assert new_registry.models is not initial_models
    assert new_registry.models == initial_models  # But content should be same

    # Clean up after test
    ModelRegistry.cleanup()


def test_supports_structured_output(registry):
    """Test checking structured output support."""
    assert registry.supports_structured_output("gpt-4o")
    assert registry.supports_structured_output("gpt-4o-2024-08-06")
    assert registry.supports_structured_output("gpt-4o-2024-09-01")
    assert not registry.supports_structured_output("unsupported-model")
    with pytest.raises(ModelVersionError):
        registry.supports_structured_output("gpt-4o-2024-07-01")


def test_supports_streaming(registry):
    """Test checking streaming support."""
    assert registry.supports_streaming("gpt-4o")
    assert not registry.supports_streaming("test-model")


def test_get_context_window(registry):
    """Test getting context window size."""
    assert registry.get_context_window("gpt-4o") == 128000
    assert registry.get_context_window("test-model") == 4096


def test_get_max_output_tokens(registry):
    """Test getting maximum output tokens."""
    assert registry.get_max_output_tokens("gpt-4o") == 16000
    assert registry.get_max_output_tokens("test-model") == 2048


def test_invalid_config(monkeypatch, tmp_path):
    """Test handling of invalid configuration."""
    # Create invalid config
    config_path = tmp_path / "invalid.yml"
    config_path.write_text("invalid: yaml: content")

    # Reset singleton
    ModelRegistry._instance = None

    # Use invalid config
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(config_path))

    # Should fall back to defaults
    registry = ModelRegistry()
    assert registry.supports_structured_output("gpt-4o")


@pytest.fixture
def requests_mock():
    """Mock requests for testing."""
    try:
        import requests_mock as rm

        with rm.Mocker() as m:
            yield m
    except ImportError:
        pytest.skip("requests_mock not installed")


def test_refresh_from_remote(registry, requests_mock):
    """Test refreshing registry from remote source."""
    # Mock successful response
    new_config = {
        "new-model": {
            "context_window": 8192,
            "max_output_tokens": 4096,
            "supports_structured": True,
            "supports_streaming": True,
            "supported_parameters": {
                "temperature": {"min": 0.0, "max": 2.0},
            },
            "aliases": ["new-latest"],
            "min_version": {
                "year": 2024,
                "month": 1,
                "day": 1,
            },
        }
    }

    requests_mock.get(
        "https://raw.githubusercontent.com/yaniv-golan/openai-structured/main/src/openai_structured/config/models.yml",
        text=yaml.safe_dump(new_config),
    )

    # Test successful refresh
    assert registry.refresh_from_remote()
    assert registry.supports_structured_output("new-model")

    # Test failed refresh
    requests_mock.get(
        "https://raw.githubusercontent.com/yaniv-golan/openai-structured/main/src/openai_structured/config/models.yml",
        status_code=404,
    )
    assert not registry.refresh_from_remote()


def test_missing_config(monkeypatch, tmp_path):
    """Test handling of missing configuration file."""
    # Reset singleton
    ModelRegistry._instance = None

    # Point to non-existent config
    non_existent = tmp_path / "does_not_exist.yml"
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(non_existent))

    # Should use fallbacks
    registry = ModelRegistry()

    # Basic capability checks
    assert registry.supports_structured_output("gpt-4o")
    assert registry.get_context_window("gpt-4o") == 128000
    assert registry.get_max_output_tokens("gpt-4o") == 16000


def test_fallback_matches_models_yml():
    """Test that fallback models match exactly with models.yml."""
    # Load models.yml
    config_path = (
        Path(__file__).parent.parent
        / "src"
        / "openai_structured"
        / "config"
        / "models.yml"
    )
    with open(config_path) as f:
        models_yml = yaml.safe_load(f)

    # Get fallbacks
    fallbacks = ModelRegistry._fallback_models

    # Compare each model
    for model_name, yml_config in models_yml.items():
        assert (
            model_name in fallbacks
        ), f"Model {model_name} missing from fallbacks"
        fallback = fallbacks[model_name]

        # Compare all fields
        assert fallback["context_window"] == yml_config["context_window"]
        assert fallback["max_output_tokens"] == yml_config["max_output_tokens"]
        assert (
            fallback["supports_structured"]
            == yml_config["supports_structured"]
        )
        assert (
            fallback["supports_streaming"] == yml_config["supports_streaming"]
        )

        # Compare supported parameters
        yml_params = yml_config["supported_parameters"]
        fallback_params = fallback["supported_parameters"]
        assert set(yml_params.keys()) == set(fallback_params.keys())

        for param, constraints in yml_params.items():
            assert fallback_params[param] == constraints

        # Compare aliases and min_version
        assert set(fallback["aliases"]) == set(yml_config["aliases"])
        if yml_config["min_version"]:
            assert fallback["min_version"] == ModelVersion(
                yml_config["min_version"]["year"],
                yml_config["min_version"]["month"],
                yml_config["min_version"]["day"],
            )
        else:
            assert fallback["min_version"] is None


def test_fallback_parameter_validation(monkeypatch, tmp_path):
    """Test parameter validation using fallback models."""
    # Reset singleton and use non-existent config to force fallbacks
    ModelRegistry._instance = None
    monkeypatch.setenv(
        "MODEL_REGISTRY_PATH", str(tmp_path / "nonexistent.yml")
    )
    registry = ModelRegistry()

    # Test gpt-4o parameters
    gpt4o = registry.get_capabilities("gpt-4o")

    # Test numeric constraints
    gpt4o.validate_parameter("temperature", 0.5)  # Valid
    with pytest.raises(OpenAIClientError, match="must be between"):
        gpt4o.validate_parameter("temperature", 2.5)  # Too high

    gpt4o.validate_parameter("frequency_penalty", -1.5)  # Valid
    with pytest.raises(OpenAIClientError, match="must be between"):
        gpt4o.validate_parameter("frequency_penalty", -3.0)  # Too low

    # Test o1 parameters
    o1 = registry.get_capabilities("o1")

    # Test enum constraints
    o1.validate_parameter("reasoning_effort", "low")  # Valid
    with pytest.raises(OpenAIClientError, match="Must be one of"):
        o1.validate_parameter("reasoning_effort", "invalid")  # Invalid value

    # Test unsupported parameters
    with pytest.raises(OpenAIClientError, match="not supported"):
        o1.validate_parameter("temperature", 0.5)  # Not supported by o1


def test_fallback_version_validation(
    monkeypatch, tmp_path, version_test_cases
):
    """Test version validation using fallback models."""
    # Reset singleton and use non-existent config to force fallbacks
    ModelRegistry._instance = None
    monkeypatch.setenv(
        "MODEL_REGISTRY_PATH", str(tmp_path / "nonexistent.yml")
    )
    registry = ModelRegistry()

    # Test valid versions
    registry.get_capabilities(
        "gpt-4o-2024-08-06"
    )  # Minimum version - should work
    registry.get_capabilities(
        "gpt-4o-2024-09-01"
    )  # Newer version - should work
    registry.get_capabilities(
        "o3-mini-2025-01-31"
    )  # Minimum version - should work
    registry.get_capabilities(
        "o3-mini-2025-02-01"
    )  # Newer version - should work

    # Test invalid versions using shared test cases
    for category, cases in version_test_cases.items():
        for model_name, expected_error in cases:
            with pytest.raises(expected_error):
                registry.get_capabilities(model_name)


@pytest.mark.parametrize(
    "model_name,expected_error",
    [
        # Invalid format tests
        ("gpt-4o-2024", InvalidVersionFormatError),  # Incomplete
        ("gpt-4o-2024-08", InvalidVersionFormatError),  # Missing day
        (
            "gpt-4o-2024-08-06-extra",
            InvalidVersionFormatError,
        ),  # Extra components
        ("gpt-4o-20x4-08-06", InvalidVersionFormatError),  # Non-numeric year
        ("model-201", InvalidVersionFormatError),  # False positive check
        (
            "test-2024-08-06-invalid",
            InvalidVersionFormatError,
        ),  # Invalid suffix
        # Invalid date tests
        ("gpt-4o-2024-13-01", InvalidDateError),  # Invalid month
        ("gpt-4o-2024-12-32", InvalidDateError),  # Invalid day
        ("gpt-4o-2024-02-30", InvalidDateError),  # Invalid Feb date
        ("gpt-4o-1999-12-31", InvalidDateError),  # Year too old
        # Version too old tests
        ("gpt-4o-2024-07-01", VersionTooOldError),  # Before min version
        (
            "o3-mini-2025-01-30",
            VersionTooOldError,
        ),  # Before o3-mini min version
    ],
)
def test_invalid_version_formats(registry, model_name, expected_error):
    """Test various invalid version formats and dates."""
    with pytest.raises(expected_error):
        registry.get_capabilities(model_name)


def test_version_validation_with_capabilities(registry, version_test_cases):
    """Test version validation in conjunction with capabilities."""
    # Test exact minimum version
    caps = registry.get_capabilities("gpt-4o-2024-08-06")
    assert caps.context_window == 128000

    # Test newer version
    caps = registry.get_capabilities("gpt-4o-2024-09-01")
    assert caps.context_window == 128000

    # Test version too old
    for model_name, _ in version_test_cases["too_old"]:
        with pytest.raises(VersionTooOldError) as exc_info:
            registry.get_capabilities(model_name)
        assert "too old" in str(exc_info.value)
        if "gpt-4o" in model_name:
            assert "2024-08-06" in str(exc_info.value)
        elif "o3-mini" in model_name:
            assert "2025-01-31" in str(exc_info.value)


@pytest.fixture
def version_test_cases() -> Dict[str, List[Tuple[str, Type[Exception]]]]:
    """Fixture providing common version validation test cases.

    Returns:
        Dict with categories of test cases, each containing a list of
        (model_name, expected_error) tuples.
    """
    return {
        "invalid_format": [
            ("gpt-4o-2024", InvalidVersionFormatError),  # Incomplete
            ("gpt-4o-2024-08", InvalidVersionFormatError),  # Missing day
            (
                "gpt-4o-2024-08-06-extra",
                InvalidVersionFormatError,
            ),  # Extra components
            (
                "gpt-4o-20x4-08-06",
                InvalidVersionFormatError,
            ),  # Non-numeric year
            ("model-201", InvalidVersionFormatError),  # False positive check
            (
                "test-2024-08-06-invalid",
                InvalidVersionFormatError,
            ),  # Invalid suffix
        ],
        "invalid_date": [
            ("gpt-4o-2024-13-01", InvalidDateError),  # Invalid month
            ("gpt-4o-2024-12-32", InvalidDateError),  # Invalid day
            ("gpt-4o-2024-02-30", InvalidDateError),  # Invalid Feb date
            ("gpt-4o-1999-12-31", InvalidDateError),  # Year too old
        ],
        "too_old": [
            ("gpt-4o-2024-07-01", VersionTooOldError),  # Before min version
            (
                "o3-mini-2025-01-30",
                VersionTooOldError,
            ),  # Before o3-mini min version
        ],
    }


def test_version_comparison(registry):
    """Test version comparison logic."""
    v1 = ModelVersion.from_string("2024-08-06")
    v2 = ModelVersion.from_string("2024-08-07")
    v3 = ModelVersion.from_string("2024-09-01")
    v4 = ModelVersion.from_string("2025-01-01")

    # Test regular version comparisons
    assert v1 < v2 < v3 < v4
    assert v4 > v3 > v2 > v1
    assert v1 == ModelVersion.from_string("2024-08-06")
    assert v1 != v2

    # Test None comparisons
    assert not (v1 < None)  # Version is never less than None
    assert not (v1 is None)  # Version is never equal to None
    assert v1 > None  # Version is always greater than None
    assert v1 >= None  # Version is always greater than or equal to None
    assert None < v1  # None is always less than a version
    assert None <= v1  # None is always less than or equal to a version
