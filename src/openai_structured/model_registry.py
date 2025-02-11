"""Model registry for managing OpenAI model capabilities and versions.

This module provides a centralized registry for OpenAI model capabilities,
including context windows, token limits, and supported features. It handles
both model aliases and dated versions, with version validation and fallback
support.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml
from pydantic import BaseModel, Field, ValidationError

from .errors import (
    InvalidVersionFormatError,
    ModelNotSupportedError,
    ModelVersionError,
    OpenAIClientError,
    VersionTooOldError,
)
from .logging import LogEvent, LogLevel, _log
from .model_version import ModelVersion


class ModelCapabilities(BaseModel):
    """Model capabilities and constraints."""

    context_window: int = Field(
        ..., description="Maximum context window size in tokens", gt=0
    )
    max_output_tokens: int = Field(
        ..., description="Maximum number of output tokens", gt=0
    )
    supports_structured: bool = Field(
        default=True,
        description="Whether the model supports structured output",
    )
    supports_streaming: bool = Field(
        default=True,
        description="Whether the model supports streaming responses",
    )
    supported_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters supported by this model with their constraints",
    )
    aliases: Set[str] = Field(
        default_factory=set, description="Alternative names for this model"
    )
    min_version: Optional[ModelVersion] = Field(
        default=None, description="Minimum supported version for this model"
    )

    def validate_parameter(self, param_name: str, value: Any) -> None:
        """Validate that a parameter is supported and its value is valid.

        Args:
            param_name: Name of the parameter to validate
            value: Value to validate

        Raises:
            OpenAIClientError: If parameter is not supported or value is invalid
        """
        if param_name not in self.supported_parameters:
            raise OpenAIClientError(
                f"Parameter '{param_name}' is not supported by this model"
            )

        constraints = self.supported_parameters[param_name]

        # Handle numeric parameters with min/max constraints
        if (
            isinstance(constraints, dict)
            and "min" in constraints
            and "max" in constraints
        ):
            if not isinstance(value, (int, float)):
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be a number, got {type(value)}"
                )
            if value < constraints["min"] or value > constraints["max"]:
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be between {constraints['min']} and {constraints['max']}"
                )
            return

        # Handle list of allowed values
        if isinstance(constraints, list):
            if value not in constraints:
                valid_values = ", ".join(map(str, constraints))
                raise OpenAIClientError(
                    f"Invalid value '{value}' for parameter '{param_name}'. Must be one of: {valid_values}"
                )
            return

        raise OpenAIClientError(
            f"Invalid constraints format for parameter '{param_name}'"
        )

    def __str__(self) -> str:
        """Human readable representation of model capabilities."""
        features = []
        if self.supports_structured:
            features.append("structured")
        if self.supports_streaming:
            features.append("streaming")

        version_info = (
            f", min_version={self.min_version}" if self.min_version else ""
        )
        params_info = (
            f", supported_params={list(self.supported_parameters.keys())}"
            if self.supported_parameters
            else ""
        )

        return (
            f"ModelCapabilities(context={self.context_window:,}, "
            f"max_output={self.max_output_tokens:,}, "
            f"features=[{', '.join(features)}]{version_info}{params_info})"
        )


# Fallback capabilities for known models
# This is auto-updated by CI/CD from the models.yml config
# AUTO-GENERATED FALLBACK START
_fallback_models = {
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
        "aliases": {"gpt-4o-latest"},
        "min_version": ModelVersion(2024, 8, 6),
    },
    "gpt-4o-mini": {
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
        "aliases": {"gpt-4o-mini-latest"},
        "min_version": ModelVersion(2024, 7, 18),
    },
    "o1": {
        "context_window": 200000,
        "max_output_tokens": 100000,
        "supports_structured": True,
        "supports_streaming": True,
        "supported_parameters": {
            "reasoning_effort": ["low", "medium", "high"],
            "max_completion_tokens": {"min": 1, "max": 100000},
        },
        "aliases": {"o1-latest"},
        "min_version": ModelVersion(2024, 12, 17),
    },
    "o3-mini": {
        "context_window": 200000,
        "max_output_tokens": 100000,
        "supports_structured": True,
        "supports_streaming": True,
        "supported_parameters": {
            "reasoning_effort": ["low", "medium", "high"],
            "max_completion_tokens": {"min": 1, "max": 100000},
        },
        "aliases": {"o3-mini-latest"},
        "min_version": ModelVersion(2025, 1, 31),
    },
}
# AUTO-GENERATED FALLBACK END


class ModelRegistry:
    """Registry for OpenAI model capabilities and version requirements."""

    _instance = None
    _fallback_models = _fallback_models

    def __new__(cls):
        """Singleton pattern to ensure one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
        return cls._instance

    @classmethod
    def cleanup(cls) -> None:
        """Clean up registry resources and reset singleton instance.

        This method should be called when you want to:
        1. Release resources held by the registry
        2. Force a fresh load of the model configuration
        3. Reset the singleton instance

        This is particularly useful in:
        - Test environments where you need a fresh registry
        - Long-running applications that need to refresh state
        - Before application shutdown to ensure proper cleanup

        Example:
            >>> registry = ModelRegistry()
            >>> # Use registry...
            >>> ModelRegistry.cleanup()  # Clean up resources
            >>> new_registry = ModelRegistry()  # Fresh instance
        """
        if cls._instance is not None:
            # Clear the models dictionary to release memory
            if hasattr(cls._instance, "models"):
                cls._instance.models.clear()

            # Log the cleanup
            _log(
                on_log=None,
                level=LogLevel.DEBUG,
                event=LogEvent.MODEL_REGISTRY,
                details={
                    "action": "cleanup",
                    "source": "model_registry",
                },
            )

            # Reset the singleton instance
            cls._instance = None

    def _load_models(self) -> None:
        """Load models from config with fallback safety."""
        self.models: Dict[str, ModelCapabilities] = {}
        config_path = self._get_config_path()

        try:
            if config_path.exists():
                with open(config_path) as f:
                    file_models = yaml.safe_load(f)
                self.models = self._validate_models(file_models)
                _log(
                    on_log=None,
                    level=LogLevel.DEBUG,
                    event=LogEvent.MODEL_REGISTRY,
                    details={
                        "action": "load_models",
                        "source": "model_registry",
                        "path": str(config_path),
                        "models": list(self.models.keys()),
                    },
                )

            # Merge fallbacks only for missing models
            for model_name, capabilities in self._fallback_models.items():
                if model_name not in self.models:
                    self.models[model_name] = ModelCapabilities(**capabilities)

        except (ValidationError, yaml.YAMLError) as e:
            _log(
                on_log=None,
                level=LogLevel.ERROR,
                event=LogEvent.ERROR,
                details={
                    "action": "load_models",
                    "source": "model_registry",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback": "using default models",
                },
            )
            self.models = {
                name: ModelCapabilities(**vals)
                for name, vals in self._fallback_models.items()
            }

    def refresh_from_remote(self, url: Optional[str] = None) -> bool:
        """Explicit refresh from trusted source.

        Args:
            url: Optional custom URL for the model config. If not provided,
                uses the URL from MODEL_REGISTRY_REMOTE environment variable
                or falls back to the default GitHub URL.

        Returns:
            bool: True if refresh was successful, False otherwise.
        """
        import requests  # Import here to avoid dependency if not needed

        remote_url = url or os.getenv(
            "MODEL_REGISTRY_REMOTE",
            "https://raw.githubusercontent.com/yaniv-golan/"
            "openai-structured/main/src/openai_structured/config/models.yml",
        )

        try:
            response = requests.get(
                remote_url, timeout=10, allow_redirects=False
            )
            response.raise_for_status()

            # Validate before persisting
            temp_models = yaml.safe_load(response.content)
            validated = self._validate_models(temp_models)

            config_path = self._get_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w") as f:
                yaml.safe_dump(
                    {k: v.model_dump() for k, v in validated.items()},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )

            self._load_models()
            _log(
                on_log=None,
                level=LogLevel.INFO,
                event=LogEvent.MODEL_REGISTRY,
                details={
                    "action": "refresh_registry",
                    "source": "model_registry",
                    "url": remote_url,
                    "models": list(validated.keys()),
                    "status": "success",
                },
            )
            return True

        except Exception as e:
            _log(
                on_log=None,
                level=LogLevel.ERROR,
                event=LogEvent.ERROR,
                details={
                    "action": "refresh_registry",
                    "source": "model_registry",
                    "url": remote_url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed",
                },
            )
            return False

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities with alias resolution and version validation.

        Args:
            model_name: The model name or alias to look up

        Returns:
            ModelCapabilities for the requested model

        Raises:
            ModelNotSupportedError: If the model is not found
            InvalidVersionFormatError: If the version format is invalid
            InvalidDateError: If the version date is invalid
            VersionTooOldError: If the version is too old
        """
        _log(
            on_log=None,
            level=LogLevel.DEBUG,
            event=LogEvent.MODEL_REGISTRY,
            details={
                "action": "get_capabilities",
                "model": model_name,
                "available_models": list(self.models.keys()),
            },
        )

        # First try exact match
        if model_name in self.models:
            _log(
                on_log=None,
                level=LogLevel.DEBUG,
                event=LogEvent.MODEL_REGISTRY,
                details={
                    "action": "get_capabilities",
                    "model": model_name,
                    "match_type": "exact",
                },
            )
            return self.models[model_name]

        # Then check aliases
        _log(
            on_log=None,
            level=LogLevel.DEBUG,
            event=LogEvent.MODEL_REGISTRY,
            details={
                "action": "get_capabilities",
                "model": model_name,
                "match_type": "alias_check",
            },
        )
        for name, caps in self.models.items():
            if model_name in caps.aliases:
                _log(
                    on_log=None,
                    level=LogLevel.DEBUG,
                    event=LogEvent.MODEL_REGISTRY,
                    details={
                        "action": "get_capabilities",
                        "model": model_name,
                        "match_type": "alias",
                        "matched_model": name,
                    },
                )
                return caps

        # Finally try to parse as versioned model
        try:
            base_model, version_str = ModelVersion.parse_version_string(
                model_name
            )
        except InvalidVersionFormatError:
            # Check if it looks like a versioned model attempt
            # Match patterns that look like they're trying to be a version:
            # - Full YYYY-MM-DD pattern but with invalid components
            # - Partial YYYY-MM pattern
            # - Just YYYY pattern
            # - Any sequence that looks like a year (full or partial)
            # - Non-numeric year-like patterns (e.g. 20x4)
            if re.search(
                r"(?:\d{2,4}|\d{2}x\d|\d{3}x)(?:-\d{2}(?:-\d{2})?)?",
                model_name,
            ):
                # Has date-like pattern, treat as version format error
                raise
            # Not trying to be a versioned model, treat as unsupported
            raise ModelNotSupportedError(
                f"Model {model_name} not found in registry", model=model_name
            ) from None

        _log(
            on_log=None,
            level=LogLevel.DEBUG,
            event=LogEvent.MODEL_REGISTRY,
            details={
                "action": "parse_version",
                "base_model": base_model,
                "version": version_str,
            },
        )

        # Find the model by name or alias
        for name, caps in self.models.items():
            if name == base_model or base_model in caps.aliases:
                if caps.min_version:
                    # Let InvalidDateError propagate up
                    version = ModelVersion.from_string(version_str)
                    if version >= caps.min_version:
                        _log(
                            on_log=None,
                            level=LogLevel.DEBUG,
                            event=LogEvent.MODEL_REGISTRY,
                            details={
                                "action": "version_check",
                                "model": model_name,
                                "version": str(version),
                                "min_version": str(caps.min_version),
                                "status": "valid",
                            },
                        )
                        return caps

                    _log(
                        on_log=None,
                        level=LogLevel.DEBUG,
                        event=LogEvent.MODEL_REGISTRY,
                        details={
                            "action": "version_check",
                            "model": model_name,
                            "version": str(version),
                            "min_version": str(caps.min_version),
                            "status": "too_old",
                        },
                    )
                    raise VersionTooOldError(
                        model_name, str(version), str(caps.min_version)
                    )
                else:
                    _log(
                        on_log=None,
                        level=LogLevel.DEBUG,
                        event=LogEvent.MODEL_REGISTRY,
                        details={
                            "action": "version_check",
                            "model": model_name,
                            "status": "no_version_requirement",
                        },
                    )
                    return caps

        _log(
            on_log=None,
            level=LogLevel.DEBUG,
            event=LogEvent.MODEL_REGISTRY,
            details={
                "action": "get_capabilities",
                "model": model_name,
                "status": "not_found",
            },
        )
        raise ModelNotSupportedError(
            f"Model {model_name} not found in registry", model=model_name
        )

    def supports_structured_output(self, model_name: str) -> bool:
        """Check if a model supports structured output.

        Args:
            model_name: The model name or alias to check

        Returns:
            bool: True if the model supports structured output, False otherwise

        Raises:
            ModelVersionError: If the model version is too old
        """
        _log(
            on_log=None,
            level=LogLevel.DEBUG,
            event=LogEvent.MODEL_REGISTRY,
            details={
                "action": "check_structured_support",
                "model": model_name,
            },
        )
        try:
            result = self.get_capabilities(model_name).supports_structured
            _log(
                on_log=None,
                level=LogLevel.DEBUG,
                event=LogEvent.MODEL_REGISTRY,
                details={
                    "action": "check_structured_support",
                    "model": model_name,
                    "result": result,
                },
            )
            return result
        except ModelVersionError as e:
            _log(
                on_log=None,
                level=LogLevel.DEBUG,
                event=LogEvent.MODEL_REGISTRY,
                details={
                    "action": "check_structured_support",
                    "model": model_name,
                    "error": str(e),
                    "error_type": "ModelVersionError",
                },
            )
            raise  # Re-raise the version error
        except ModelNotSupportedError:
            _log(
                on_log=None,
                level=LogLevel.DEBUG,
                event=LogEvent.MODEL_REGISTRY,
                details={
                    "action": "check_structured_support",
                    "model": model_name,
                    "error_type": "ModelNotSupportedError",
                    "result": False,
                },
            )
            return False

    def supports_streaming(self, model_name: str) -> bool:
        """Check if a model supports streaming.

        Args:
            model_name: The model name or alias to check

        Returns:
            bool: True if the model supports streaming, False otherwise

        Raises:
            ModelVersionError: If the model version is too old
        """
        try:
            return self.get_capabilities(model_name).supports_streaming
        except ModelNotSupportedError:
            return False

    def get_context_window(self, model_name: str) -> int:
        """Get the context window size for a model.

        Args:
            model_name: The model name or alias to check

        Returns:
            int: The context window size in tokens

        Raises:
            ModelNotSupportedError: If the model is not found
            ModelVersionError: If the model version is too old
        """
        return self.get_capabilities(model_name).context_window

    def get_max_output_tokens(self, model_name: str) -> int:
        """Get the maximum output tokens for a model.

        Args:
            model_name: The model name or alias to check

        Returns:
            int: The maximum output tokens

        Raises:
            ModelNotSupportedError: If the model is not found
            ModelVersionError: If the model version is too old
        """
        return self.get_capabilities(model_name).max_output_tokens

    # Private helpers
    def _get_config_path(self) -> Path:
        """Get the path to the model config file."""
        default_path = Path(__file__).parent / "config" / "models.yml"
        return Path(os.getenv("MODEL_REGISTRY_PATH", default_path))

    def _validate_models(
        self, data: Dict[str, Any]
    ) -> Dict[str, ModelCapabilities]:
        """Validate model data and convert to ModelCapabilities instances."""
        validated = {}
        for name, values in data.items():
            try:
                # Convert any sets to lists for logging
                log_values = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in values.items()
                }
                _log(
                    on_log=None,
                    level=LogLevel.DEBUG,
                    event=LogEvent.MODEL_VALIDATION,
                    details={
                        "action": "validate_model",
                        "model": name,
                        "values": log_values,
                    },
                )
                validated[name] = ModelCapabilities(**values)
            except ValidationError as e:
                _log(
                    on_log=None,
                    level=LogLevel.ERROR,
                    event=LogEvent.MODEL_VALIDATION,
                    details={
                        "action": "validate_model",
                        "model": name,
                        "error": str(e),
                        "error_type": "ValidationError",
                    },
                )
                raise
        return validated
