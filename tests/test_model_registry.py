"""Tests for the model registry functionality."""

import os
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Set, Type, cast
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from openai_structured.errors import (
    InvalidDateError,
    ModelNotSupportedError,
    OpenAIClientError,
    VersionTooOldError,
)
from openai_structured.model_registry import (
    EnumConstraint,
    FixedParameterSet,
    ModelCapabilities,
    ModelRegistry,
    ModelVersion,
    NumericConstraint,
    ParameterReference,
    RegistryUpdateStatus,
)


@pytest.fixture
def registry(
    tmp_path: Path, monkeypatch: Any
) -> Generator[ModelRegistry, None, None]:
    """Create a test registry with temporary config files."""
    # Store original env vars
    original_registry_path = os.environ.get("MODEL_REGISTRY_PATH")
    original_constraints_path = os.environ.get("PARAMETER_CONSTRAINTS_PATH")

    # Cleanup any existing registry first
    ModelRegistry.cleanup()

    # Create parameter constraints file
    constraints_path = tmp_path / "parameter_constraints.yml"
    constraints = {
        "numeric_constraints": {
            "temperature": {
                "type": "numeric",
                "min_value": 0.0,
                "max_value": 2.0,
                "description": "Test numeric parameter",
                "allow_float": True,
                "allow_int": True,
            },
            "top_p": {
                "type": "numeric",
                "min_value": 0.0,
                "max_value": 1.0,
                "description": "Test numeric parameter",
                "allow_float": True,
                "allow_int": True,
            },
            "max_completion_tokens": {
                "type": "numeric",
                "min_value": 1,
                "max_value": None,
                "description": "Maximum completion tokens",
                "allow_float": False,
                "allow_int": True,
            },
        },
        "enum_constraints": {
            "reasoning_effort": {
                "type": "enum",
                "allowed_values": ["low", "medium", "high"],
                "description": "Test enum parameter",
            }
        },
    }
    constraints_path.write_text(yaml.dump(constraints))

    # Create model registry file
    registry_path = tmp_path / "models.yml"
    models = {
        "version": "1.0.0",
        "dated_models": {
            "gpt-4o-2024-08-06": {
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": [
                    {"ref": "numeric_constraints.temperature"},
                    {"ref": "numeric_constraints.top_p"},
                    {"ref": "numeric_constraints.max_completion_tokens"},
                ],
                "description": "Test model with full capabilities",
                "min_version": {"year": 2024, "month": 8, "day": 6},
            },
            "test-model-2024-01-01": {
                "context_window": 4096,
                "max_output_tokens": 2048,
                "supports_structured": True,
                "supports_streaming": False,
                "supported_parameters": [
                    {"ref": "numeric_constraints.temperature"}
                ],
                "description": "Basic test model",
                "min_version": {"year": 2024, "month": 1, "day": 1},
            },
            "o1-2024-12-17": {
                "context_window": 200000,
                "max_output_tokens": 100000,
                "supports_structured": True,
                "supports_streaming": False,
                "supported_parameters": [
                    {"ref": "numeric_constraints.temperature"},
                    {"ref": "numeric_constraints.top_p"},
                    {"ref": "enum_constraints.reasoning_effort"},
                ],
                "description": "Test model with enum parameters",
                "min_version": {"year": 2024, "month": 12, "day": 17},
            },
            "o3-mini-2025-01-31": {
                "context_window": 200000,
                "max_output_tokens": 100000,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": [
                    {"ref": "numeric_constraints.max_completion_tokens"},
                    {"ref": "enum_constraints.reasoning_effort"},
                ],
                "description": "Test model with token parameters",
                "min_version": {"year": 2025, "month": 1, "day": 31},
            },
        },
        "aliases": {
            "gpt-4o": "gpt-4o-2024-08-06",
            "test-model": "test-model-2024-01-01",
            "o1": "o1-2024-12-17",
            "o3-mini": "o3-mini-2025-01-31",
        },
    }
    registry_path.write_text(yaml.dump(models))

    # Set environment variables
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("PARAMETER_CONSTRAINTS_PATH", str(constraints_path))

    # Create and return registry
    registry = ModelRegistry()
    yield registry

    # Cleanup
    ModelRegistry.cleanup()

    # Restore original env vars
    if original_registry_path:
        monkeypatch.setenv("MODEL_REGISTRY_PATH", original_registry_path)
    else:
        monkeypatch.delenv("MODEL_REGISTRY_PATH", raising=False)

    if original_constraints_path:
        monkeypatch.setenv(
            "PARAMETER_CONSTRAINTS_PATH", original_constraints_path
        )
    else:
        monkeypatch.delenv("PARAMETER_CONSTRAINTS_PATH", raising=False)


def test_model_capabilities_validation(registry: ModelRegistry) -> None:
    """Test ModelCapabilities validation."""
    # Get capabilities for test model
    caps = registry.get_capabilities("test-model")

    # Test basic properties
    assert caps.context_window == 4096
    assert caps.max_output_tokens == 2048
    assert caps.supports_structured
    assert not caps.supports_streaming
    assert "test-model" in caps.aliases
    assert caps.min_version == ModelVersion(2024, 1, 1)

    # Test parameter validation
    with pytest.raises(OpenAIClientError, match="not supported"):
        caps.validate_parameter("invalid_param", 0.5)

    with pytest.raises(OpenAIClientError, match="must be between"):
        caps.validate_parameter("temperature", 2.5)

    # Valid parameter values
    caps.validate_parameter("temperature", 0.5)


def test_parameter_validation(registry: ModelRegistry) -> None:
    """Test parameter validation for different models."""
    # Test gpt-4o parameters
    gpt4o = registry.get_capabilities("gpt-4o")

    # Valid parameters
    gpt4o.validate_parameter("temperature", 0.5)
    gpt4o.validate_parameter("top_p", 0.9)
    gpt4o.validate_parameter("max_completion_tokens", 100)

    # Invalid parameters
    with pytest.raises(OpenAIClientError, match="must be between"):
        gpt4o.validate_parameter("temperature", 2.5)

    with pytest.raises(OpenAIClientError, match="must be between"):
        gpt4o.validate_parameter("top_p", -0.1)

    with pytest.raises(OpenAIClientError, match="must be an integer"):
        gpt4o.validate_parameter("max_completion_tokens", 50.5)

    with pytest.raises(OpenAIClientError, match="not supported"):
        gpt4o.validate_parameter("unsupported_param", 1.0)

    # Test o1 parameters
    o1 = registry.get_capabilities("o1")

    # Valid parameters
    o1.validate_parameter("temperature", 0.5)
    o1.validate_parameter("reasoning_effort", "low")

    # Invalid parameters
    with pytest.raises(OpenAIClientError, match="Invalid value"):
        o1.validate_parameter("reasoning_effort", "invalid")

    # Test test-model parameters
    test_model = registry.get_capabilities("test-model")

    # Valid parameters
    test_model.validate_parameter("temperature", 0.5)

    # Invalid parameters
    with pytest.raises(OpenAIClientError, match="not supported"):
        test_model.validate_parameter("top_p", 0.9)

    with pytest.raises(OpenAIClientError, match="not supported"):
        test_model.validate_parameter("reasoning_effort", "low")


def test_parameter_constraints() -> None:
    """Test parameter constraint validation."""
    # Test numeric constraints
    numeric = NumericConstraint(
        min_value=0.0,
        max_value=2.0,
        description="Test numeric parameter",
    )
    assert numeric.type == "numeric"
    assert numeric.allow_int is True
    assert numeric.allow_float is True

    # Test enum constraints
    enum = EnumConstraint(
        allowed_values=["low", "medium", "high"],
        description="Test enum parameter",
    )
    assert enum.type == "enum"
    assert "low" in enum.allowed_values


def test_parameter_reference() -> None:
    """Test parameter reference validation."""
    ref = ParameterReference(
        ref="numeric_constraints.temperature",
        max_value=1.5,  # Override max value
    )
    assert ref.ref == "numeric_constraints.temperature"
    assert ref.max_value == 1.5


def test_fixed_parameter_set() -> None:
    """Test fixed parameter set validation."""
    fixed = FixedParameterSet(
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        description="Test fixed parameters",
    )
    assert fixed.temperature == 1.0
    assert fixed.top_p == 1.0
    assert fixed.frequency_penalty == 0.0
    assert fixed.presence_penalty == 0.0
    assert fixed.description == "Test fixed parameters"


def test_get_capabilities(registry: ModelRegistry) -> None:
    """Test getting model capabilities."""
    # Test exact match (dated model)
    caps = registry.get_capabilities("gpt-4o-2024-08-06")
    assert caps.context_window == 128000
    assert caps.max_output_tokens == 16384
    assert caps.supports_structured
    assert caps.supports_streaming
    assert any(
        param.ref == "numeric_constraints.temperature"
        for param in caps.supported_parameters
    )
    assert caps.min_version == ModelVersion(2024, 8, 6)

    # Test alias resolution
    caps = registry.get_capabilities("gpt-4o")
    assert caps.context_window == 128000
    assert "gpt-4o" in caps.aliases

    # Test newer version
    caps = registry.get_capabilities("gpt-4o-2024-09-01")
    assert caps.context_window == 128000

    # Test unsupported model
    with pytest.raises(ModelNotSupportedError):
        registry.get_capabilities("unsupported-model")

    # Test too old version
    with pytest.raises(VersionTooOldError):
        registry.get_capabilities("gpt-4o-2024-07-01")


def test_registry_singleton(registry: ModelRegistry) -> None:
    """Test that ModelRegistry is a singleton."""
    registry2 = ModelRegistry()
    assert registry is registry2


def test_registry_cleanup(registry: ModelRegistry) -> None:
    """Test that cleanup properly resets the registry."""
    # Store initial state
    initial_capabilities = registry._capabilities.copy()

    # Verify we have the same instance
    registry2 = ModelRegistry()
    assert registry is registry2
    assert registry2._capabilities == initial_capabilities

    # Clean up
    ModelRegistry.cleanup()

    # Get new instance
    new_registry = ModelRegistry()

    # Verify it's a different instance
    assert new_registry is not registry

    # Verify capabilities are reloaded
    assert new_registry._capabilities is not initial_capabilities
    assert (
        new_registry._capabilities == initial_capabilities
    )  # But content should be same

    # Clean up after test
    ModelRegistry.cleanup()


def test_supports_structured_output(registry: ModelRegistry) -> None:
    """Test checking structured output support."""
    assert registry.supports_structured_output("gpt-4o")
    assert registry.supports_structured_output("gpt-4o-2024-08-06")
    assert registry.supports_structured_output("gpt-4o-2024-09-01")
    assert not registry.supports_structured_output("unsupported-model")
    with pytest.raises(VersionTooOldError):
        registry.supports_structured_output("gpt-4o-2024-07-01")


def test_supports_streaming(registry: ModelRegistry) -> None:
    """Test checking streaming support."""
    assert registry.supports_streaming("gpt-4o")
    assert not registry.supports_streaming("test-model")


def test_get_context_window(registry: ModelRegistry) -> None:
    """Test getting context window size."""
    assert registry.get_context_window("gpt-4o") == 128000
    assert registry.get_context_window("test-model") == 4096


def test_get_max_output_tokens(registry: ModelRegistry) -> None:
    """Test getting maximum output tokens."""
    assert registry.get_max_output_tokens("gpt-4o") == 16384
    assert registry.get_max_output_tokens("test-model") == 2048


def test_invalid_config(monkeypatch: Any, tmp_path: Path) -> None:
    """Test handling of invalid configuration."""
    # Create invalid config
    config_path = tmp_path / "invalid.yml"
    config_path.write_text("invalid: yaml: content")

    # Reset singleton
    ModelRegistry.cleanup()

    # Use invalid config
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(config_path))

    # Should fall back to defaults
    registry = ModelRegistry()
    assert registry.supports_structured_output("gpt-4o")


def test_fallback_matches_models_yml() -> None:
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

    # Compare version
    assert fallbacks["version"] == models_yml["version"]

    # Compare dated models
    models_yml_dated = cast(Dict[str, Any], models_yml["dated_models"])
    fallbacks_dated = cast(Dict[str, Any], fallbacks["dated_models"])
    for model_name, yml_config in models_yml_dated.items():
        assert (
            model_name in fallbacks_dated
        ), f"Model {model_name} missing from fallbacks"
        fallback = fallbacks_dated[model_name]

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
        assert fallback["description"] == yml_config["description"]

        # Compare supported parameters
        yml_params = yml_config["supported_parameters"]
        fallback_params = fallback["supported_parameters"]

        # Convert lists to sets for comparison
        yml_param_refs = {param["ref"] for param in yml_params}
        fallback_param_refs = {param["ref"] for param in fallback_params}
        assert yml_param_refs == fallback_param_refs

    # Compare aliases
    assert fallbacks["aliases"] == models_yml["aliases"]


@pytest.mark.parametrize(
    "model_name,expected_error",
    [
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
        # Model not supported tests
        ("unsupported-model", ModelNotSupportedError),  # Unknown model
        ("gpt-5", ModelNotSupportedError),  # Unknown model with no version
    ],
)
def test_invalid_version_formats(
    registry: ModelRegistry, model_name: str, expected_error: Type[Exception]
) -> None:
    """Test various invalid version formats and dates."""
    with pytest.raises(expected_error):
        registry.get_capabilities(model_name)


def test_version_validation_with_capabilities(registry: ModelRegistry) -> None:
    """Test version validation in conjunction with capabilities."""
    # Test exact minimum version
    caps = registry.get_capabilities("gpt-4o-2024-08-06")
    assert caps.context_window == 128000

    # Test newer version
    caps = registry.get_capabilities("gpt-4o-2024-09-01")
    assert caps.context_window == 128000

    # Test version too old
    with pytest.raises(VersionTooOldError) as exc_info:
        registry.get_capabilities("gpt-4o-2024-07-01")
    assert "too old" in str(exc_info.value)
    assert "2024-08-06" in str(exc_info.value)

    with pytest.raises(VersionTooOldError) as exc_info:
        registry.get_capabilities("o3-mini-2025-01-30")
    assert "too old" in str(exc_info.value)
    assert "2025-01-31" in str(exc_info.value)


def test_version_comparison() -> None:
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


def test_numeric_parameter_validation() -> None:
    """Test numeric parameter validation."""
    caps = ModelCapabilities(
        openai_model_name="test-model",
        context_window=4096,
        max_output_tokens=2048,
        supported_parameters=[
            ParameterReference(
                ref="numeric_constraints.temperature", max_value=None
            ),
            ParameterReference(
                ref="numeric_constraints.top_p", max_value=None
            ),
            ParameterReference(
                ref="numeric_constraints.max_completion_tokens", max_value=None
            ),
        ],
    )

    # Test valid values
    caps.validate_parameter("temperature", 0.5)  # Float in range
    caps.validate_parameter("temperature", 1)  # Integer in range
    caps.validate_parameter("top_p", 0.5)  # Valid float
    caps.validate_parameter("max_completion_tokens", 50)  # Valid integer

    # Test invalid values
    with pytest.raises(OpenAIClientError, match="must be between"):
        caps.validate_parameter("temperature", 2.5)  # Out of range

    with pytest.raises(OpenAIClientError, match="must be a number"):
        caps.validate_parameter("temperature", "0.5")  # Wrong type

    with pytest.raises(OpenAIClientError, match="must be between"):
        caps.validate_parameter("top_p", 1.5)  # Out of range

    with pytest.raises(OpenAIClientError, match="must be an integer"):
        caps.validate_parameter(
            "max_completion_tokens", 50.5
        )  # Float not allowed


def test_enum_parameter_validation() -> None:
    """Test enum parameter validation."""
    caps = ModelCapabilities(
        openai_model_name="test-model",
        context_window=4096,
        max_output_tokens=2048,
        supported_parameters=[
            ParameterReference(
                ref="enum_constraints.reasoning_effort", max_value=None
            ),
        ],
    )

    # Test valid values
    caps.validate_parameter("reasoning_effort", "low")
    caps.validate_parameter("reasoning_effort", "medium")
    caps.validate_parameter("reasoning_effort", "high")

    # Test invalid values
    with pytest.raises(OpenAIClientError, match="must be a string"):
        caps.validate_parameter("reasoning_effort", 1)  # Wrong type

    with pytest.raises(OpenAIClientError, match="Invalid value"):
        caps.validate_parameter(
            "reasoning_effort", "invalid"
        )  # Not in allowed values


def test_legacy_constraint_conversion() -> None:
    """Test conversion of legacy constraint formats."""
    # Create capabilities with legacy format
    caps = ModelCapabilities(
        openai_model_name="test-model",
        context_window=4096,
        max_output_tokens=2048,
        supported_parameters=[
            ParameterReference(
                ref="numeric_constraints.temperature", max_value=None
            ),
            ParameterReference(
                ref="enum_constraints.reasoning_effort", max_value=None
            ),
        ],
    )

    # Get the constraints from the registry
    registry = ModelRegistry.get_instance()

    # Get the actual constraints
    temp_constraint = registry.get_parameter_constraint(
        "numeric_constraints.temperature"
    )
    assert isinstance(temp_constraint, NumericConstraint)
    assert temp_constraint.min_value == 0.0
    assert temp_constraint.max_value == 2.0

    effort_constraint = registry.get_parameter_constraint(
        "enum_constraints.reasoning_effort"
    )
    assert isinstance(effort_constraint, EnumConstraint)
    assert set(effort_constraint.allowed_values) == {"low", "medium", "high"}

    # Test validation still works
    caps.validate_parameter("temperature", 0.5)
    caps.validate_parameter("reasoning_effort", "medium")

    with pytest.raises(OpenAIClientError):
        caps.validate_parameter("temperature", 2.5)

    with pytest.raises(OpenAIClientError):
        caps.validate_parameter("reasoning_effort", "invalid")


@pytest.mark.parametrize(
    "model_name,expected_window",
    [
        ("test-model", 4096),
        ("gpt-4o-2024-08-06", 128000),  # Valid versioned model
        ("gpt-4o-special", None),  # Invalid model with hyphen
    ],
)
def test_hyphen_handling(
    registry: ModelRegistry, model_name: str, expected_window: Optional[int]
) -> None:
    """Test hyphenated model name handling."""
    if expected_window is None:
        with pytest.raises(ModelNotSupportedError):
            registry.get_context_window(model_name)
    else:
        assert registry.get_context_window(model_name) == expected_window


@pytest.mark.live
def test_live_model_capabilities() -> None:
    """Test model capabilities with live OpenAI API."""
    registry = ModelRegistry()

    # Test gpt-4o capabilities
    caps = registry.get_capabilities("gpt-4o")
    assert caps.context_window == 128000
    assert caps.max_output_tokens == 16384
    assert caps.supports_structured
    assert caps.supports_streaming
    assert "gpt-4o" in caps.aliases

    # Test o1 capabilities
    caps = registry.get_capabilities("o1")
    assert caps.context_window == 200000
    assert caps.max_output_tokens == 100000
    assert caps.supports_structured
    assert caps.supports_streaming
    assert "o1" in caps.aliases


@pytest.mark.live
def test_live_parameter_validation() -> None:
    """Test parameter validation with live OpenAI API."""
    registry = ModelRegistry()

    # Test gpt-4o parameters
    gpt4o = registry.get_capabilities("gpt-4o")
    used_params: Set[str] = set()

    # Test numeric parameters
    gpt4o.validate_parameter("temperature", 0.7, used_params=used_params)
    gpt4o.validate_parameter("top_p", 0.9, used_params=used_params)

    # Test token parameters
    gpt4o.validate_parameter(
        "max_output_tokens", 1000, used_params=used_params
    )
    with pytest.raises(OpenAIClientError):
        gpt4o.validate_parameter(
            "max_completion_tokens", 1000, used_params=used_params
        )

    # Test o1 parameters
    o1 = registry.get_capabilities("o1")
    used_params = set()

    # Test enum parameters
    o1.validate_parameter(
        "reasoning_effort", "medium", used_params=used_params
    )
    with pytest.raises(OpenAIClientError):
        o1.validate_parameter(
            "reasoning_effort", "invalid", used_params=used_params
        )


@pytest.mark.live
def test_live_version_validation() -> None:
    """Test version validation with live OpenAI API."""
    registry = ModelRegistry()

    # Test valid versions
    assert (
        registry.get_capabilities("gpt-4o-2024-08-06").context_window == 128000
    )
    assert registry.get_capabilities("o1-2024-12-17").context_window == 200000

    # Test too old versions
    with pytest.raises(VersionTooOldError):
        registry.get_capabilities("gpt-4o-2024-08-05")  # One day too old
    with pytest.raises(VersionTooOldError):
        registry.get_capabilities("o1-2024-12-16")  # One day too old

    # Test invalid versions
    with pytest.raises(InvalidDateError):
        registry.get_capabilities("gpt-4o-2024-13-01")  # Invalid month
    with pytest.raises(ModelNotSupportedError):
        registry.get_capabilities("unsupported-model")


def test_refresh_from_remote_updated(registry: ModelRegistry) -> None:
    """Test refresh_from_remote when there are updates."""
    with patch("requests.get") as mock_get:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = yaml.dump(
            {
                "version": 1,
                "dated_models": {
                    "test-model-2024-01-01": {
                        "context_window": 8000,
                        "max_output_tokens": 2000,
                        "supports_structured": True,
                        "supports_streaming": True,
                    }
                },
                "aliases": {
                    "test-model": "test-model-2024-01-01",
                },
            }
        )
        mock_get.return_value = mock_response

        # Call the method
        result = registry.refresh_from_remote()

        # Verify results
        assert result.success
        assert result.status == RegistryUpdateStatus.UPDATED
        assert "Registry updated successfully" in result.message
        assert mock_get.called


def test_refresh_from_remote_not_modified(registry: ModelRegistry) -> None:
    """Test refresh_from_remote when content hasn't changed (304 Not Modified)."""
    with patch("requests.get") as mock_get:
        # Create a mock response for 304 Not Modified
        mock_response = MagicMock()
        mock_response.status_code = 304
        mock_get.return_value = mock_response

        # Call the method
        result = registry.refresh_from_remote()

        # Verify results
        assert result.success
        assert result.status == RegistryUpdateStatus.ALREADY_CURRENT
        assert "already up to date" in result.message
        assert mock_get.called

        # Check that If-Modified-Since header was sent (this assumes the file exists)
        calls = mock_get.call_args_list
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert "headers" in kwargs
        config_path = registry._config_path
        if config_path is not None and os.path.exists(config_path):
            assert "If-Modified-Since" in kwargs["headers"]


def test_refresh_from_remote_force(registry: ModelRegistry) -> None:
    """Test refresh_from_remote with force=True."""
    # Mock a proper YAML response that matches the structure expected by the registry
    yaml_content = """
models:
  test-model:
    context_window: 4096
    max_output_tokens: 2048
    supports_structured: true
    supports_streaming: true
    supported_parameters: []
    description: "Test model"
aliases:
  test-alias: test-model
"""

    with patch("requests.get") as mock_get:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = yaml_content
        # Add some headers that would be included in the metadata
        mock_response.headers = {
            "ETag": '"abc123"',
            "Last-Modified": "Wed, 01 Jan 2024 00:00:00 GMT",
        }
        mock_get.return_value = mock_response

        # Mock json.dump to avoid character-by-character writing
        with patch("json.dump") as mock_json_dump:
            # Also mock the file write operation for the YAML content
            with patch("builtins.open", mock_open()) as mock_file:
                # Additional mock for _load_capabilities to prevent parsing errors
                with patch.object(registry, "_load_capabilities"):
                    # Call the method with force=True
                    result = registry.refresh_from_remote(force=True)

                    # Verify results
                    assert result.success
                    assert result.status == RegistryUpdateStatus.UPDATED

                    # Check that If-Modified-Since header was not sent even if the file exists
                    calls = mock_get.call_args_list
                    assert len(calls) == 1
                    args, kwargs = calls[0]
                    assert "headers" in kwargs
                    # Force should result in empty headers
                    assert not kwargs["headers"]

                    # Verify the YAML content was written
                    mock_file().write.assert_called_with(yaml_content)

                    # Verify json.dump was called (for metadata)
                    assert mock_json_dump.called

                    # If we want to be more specific, we can check the metadata
                    # Get the first positional arg (metadata dict)
                    metadata = mock_json_dump.call_args[0][0]
                    assert "etag" in metadata
                    assert "last_modified" in metadata


def test_refresh_from_remote_error_cases(registry: ModelRegistry) -> None:
    """Test refresh_from_remote error handling."""

    # Test 404 Not Found
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = registry.refresh_from_remote()
        assert not result.success
        assert result.status == RegistryUpdateStatus.NOT_FOUND

    # Test network error
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Network error")

        result = registry.refresh_from_remote()
        assert not result.success
        assert result.status == RegistryUpdateStatus.UNKNOWN_ERROR
        assert "Unexpected error" in result.message


def test_refresh_from_remote_import_error() -> None:
    """Test refresh_from_remote when requests module is not available."""
    # Use patch.dict to modify sys.modules
    with patch.dict("sys.modules", {"requests": None}):
        # Need a fresh instance that will try to import requests
        ModelRegistry.cleanup()
        registry = ModelRegistry()

        # Call the method which should handle the ImportError
        result = registry.refresh_from_remote()

        # Check results
        assert not result.success
        assert result.status == RegistryUpdateStatus.IMPORT_ERROR
        assert "Could not import requests module" in result.message


def test_check_for_updates_no_update(registry: ModelRegistry) -> None:
    """Test check_for_updates when no update is available."""
    with patch("requests.get") as mock_get:
        # Create a mock response for 304 Not Modified
        mock_response = MagicMock()
        mock_response.status_code = 304
        mock_get.return_value = mock_response

        # Call the method
        result = registry.check_for_updates()

        # Verify results
        assert result.success
        assert result.status == RegistryUpdateStatus.ALREADY_CURRENT
        assert "already up to date" in result.message
        assert mock_get.called

        # Check that If-Modified-Since header was sent if the file exists
        calls = mock_get.call_args_list
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert "headers" in kwargs
        assert "Range" in kwargs["headers"]
        assert kwargs["headers"]["Range"] == "bytes=0-0"
        config_path = registry._config_path
        if config_path is not None and os.path.exists(config_path):
            assert "If-Modified-Since" in kwargs["headers"]


def test_check_for_updates_available(registry: ModelRegistry) -> None:
    """Test check_for_updates when an update is available."""
    with patch("requests.get") as mock_get:
        # Create a mock response for 200 OK (update available)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Call the method
        result = registry.check_for_updates()

        # Verify results
        assert result.success
        assert result.status == RegistryUpdateStatus.UPDATE_AVAILABLE
        assert "update is available" in result.message
        assert mock_get.called

        # Test with 206 Partial Content as well
        mock_response.status_code = 206
        result = registry.check_for_updates()
        assert result.success
        assert result.status == RegistryUpdateStatus.UPDATE_AVAILABLE


def test_check_for_updates_error_cases(registry: ModelRegistry) -> None:
    """Test check_for_updates error handling."""

    # Test 404 Not Found
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = registry.check_for_updates()
        assert not result.success
        assert result.status == RegistryUpdateStatus.NOT_FOUND

    # Test network error
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Network error")

        result = registry.check_for_updates()
        assert not result.success
        assert result.status == RegistryUpdateStatus.UNKNOWN_ERROR
        assert "Unexpected error" in result.message
