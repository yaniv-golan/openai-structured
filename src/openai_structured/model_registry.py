"""Model registry for managing OpenAI model capabilities and versions.

This module provides a centralized registry for OpenAI model capabilities,
including context windows, token limits, and supported features. It handles
both model aliases and dated versions, with version validation and fallback
support.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel, Field, ValidationError

from .errors import (
    InvalidDateError,
    ModelNotSupportedError,
    ModelVersionError,
    OpenAIClientError,
    TokenParameterError,
    VersionTooOldError,
)
from .logging import LogEvent, LogLevel, _log
from .model_version import ModelVersion

# Create module logger
logger = logging.getLogger(__name__)


def _default_log_callback(level: int, event: str, data: Dict) -> None:
    """Default logging callback that uses the standard logging module."""
    logger.log(level, f"{event}: {data}")


class ParameterConstraint(BaseModel):
    """Base class for parameter constraints."""

    type: str
    description: str = ""


class NumericConstraint(ParameterConstraint):
    """Constraints for numeric parameters."""

    type: str = "numeric"
    min_value: float
    max_value: Optional[float]  # Allow null for dynamic max values
    allow_int: bool = True
    allow_float: bool = True


class EnumConstraint(ParameterConstraint):
    """Constraints for enum parameters."""

    type: str = "enum"
    allowed_values: List[str]


class FixedParameterSet(BaseModel):
    """Fixed parameter values for models."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    description: str = ""


class ParameterReference(BaseModel):
    """Reference to a parameter constraint with optional overrides."""

    ref: str
    max_value: Optional[float] = None  # For overriding numeric constraints


class ModelCapabilities(BaseModel):
    """Model capabilities and constraints."""

    model_name: str = Field(
        ...,
        description="The model name used to create this capabilities instance",
    )
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
    supported_parameters: List[ParameterReference] = Field(
        default_factory=list,
        description="References to parameter constraints with optional overrides",
    )
    aliases: Set[str] = Field(
        default_factory=set, description="Alternative names for this model"
    )
    min_version: Optional[ModelVersion] = Field(
        default=None, description="Minimum supported version for this model"
    )

    def validate_parameter(
        self,
        param_name: str,
        value: Any,
        *,
        used_params: Optional[Set[str]] = None,
    ) -> None:
        """Validate that a parameter is supported and its value is valid.

        Args:
            param_name: Name of the parameter to validate
            value: Value to validate
            used_params: Optional set of parameters already used in this request

        Raises:
            OpenAIClientError: If parameter is not supported or value is invalid
            TokenParameterError: If both token parameters are used
        """
        # Special handling for token parameters
        if param_name in ("max_output_tokens", "max_completion_tokens"):
            # Check if either token parameter is already used
            other_param = (
                "max_completion_tokens"
                if param_name == "max_output_tokens"
                else "max_output_tokens"
            )
            if used_params is not None and other_param in used_params:
                raise TokenParameterError(model=self.model_name)

            # Add to used parameters
            if used_params is not None:
                used_params.add(param_name)

            # Validate integer type for token parameters
            if not isinstance(value, int):
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be an integer"
                )

            # Both parameters use the same limit
            if value > self.max_output_tokens:
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must not exceed {self.max_output_tokens}"
                )
            return

        # Find the parameter reference
        param_ref = None
        for ref in self.supported_parameters:
            ref_parts = ref.ref.split(".")
            if len(ref_parts) == 2 and ref_parts[1] == param_name:
                param_ref = ref
                break

        if param_ref is None:
            raise OpenAIClientError(
                f"Parameter '{param_name}' is not supported by this model"
            )

        # Get the constraint from the registry
        constraint = ModelRegistry.get_instance().get_parameter_constraint(
            param_ref.ref
        )

        if isinstance(constraint, NumericConstraint):
            # Validate numeric type
            if not isinstance(value, (int, float)):
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be a number, got {type(value)}"
                )

            # Validate integer/float requirements
            if isinstance(value, float) and not constraint.allow_float:
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be an integer"
                )
            if isinstance(value, int) and not constraint.allow_int:
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be a float"
                )

            # Use override max_value if specified
            max_value = param_ref.max_value or constraint.max_value
            if max_value is None:
                max_value = self.max_output_tokens

            # Validate range
            if value < constraint.min_value or value > max_value:
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be between "
                    f"{constraint.min_value} and {max_value}"
                )

        elif isinstance(constraint, EnumConstraint):
            # Validate type
            if not isinstance(value, str):
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be a string, got {type(value)}"
                )

            # Validate allowed values
            if value not in constraint.allowed_values:
                raise OpenAIClientError(
                    f"Invalid value '{value}' for parameter '{param_name}'. "
                    f"Must be one of: {', '.join(map(str, constraint.allowed_values))}"
                )


class ModelRegistry:
    """Registry for model capabilities and validation."""

    _instance = None
    _config_path = None
    _constraints_path = None

    def __new__(cls) -> "ModelRegistry":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the model registry."""
        if not hasattr(self, "_initialized"):
            self._capabilities: Dict[str, ModelCapabilities] = {}
            self._constraints: Dict[
                str, Union[NumericConstraint, EnumConstraint]
            ] = {}
            self._fixed_parameters: Dict[str, FixedParameterSet] = {}

            # Get paths from environment or defaults
            self._config_path = os.getenv(
                "MODEL_REGISTRY_PATH",
                str(Path(__file__).parent / "config" / "models.yml"),
            )
            self._constraints_path = os.getenv(
                "PARAMETER_CONSTRAINTS_PATH",
                str(
                    Path(__file__).parent
                    / "config"
                    / "parameter_constraints.yml"
                ),
            )

            self._load_constraints()
            self._load_capabilities()
            self._initialized = True

    def _load_constraints(self) -> None:
        """Load parameter constraints from YAML."""
        try:
            with open(self._constraints_path) as f:
                data = yaml.safe_load(f)

            # Load numeric constraints
            for name, config in data.get("numeric_constraints", {}).items():
                key = f"numeric_constraints.{name}"
                self._constraints[key] = NumericConstraint(**config)

            # Load enum constraints
            for name, config in data.get("enum_constraints", {}).items():
                key = f"enum_constraints.{name}"
                self._constraints[key] = EnumConstraint(**config)

            # Load fixed parameter sets
            for name, config in data.get("fixed_parameter_sets", {}).items():
                key = f"fixed_parameter_sets.{name}"
                self._fixed_parameters[key] = FixedParameterSet(**config)

        except FileNotFoundError:
            _log(
                _default_log_callback,
                LogLevel.WARNING,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Parameter constraints file not found, using defaults",
                    "path": self._constraints_path,
                },
            )
        except Exception as e:
            _log(
                _default_log_callback,
                LogLevel.ERROR,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Failed to load parameter constraints",
                    "error": str(e),
                },
            )
            raise

    def _load_capabilities(self) -> None:
        """Load model capabilities from YAML."""
        try:
            with open(self._config_path) as f:
                data = yaml.safe_load(f)

            # Load dated models
            dated_models = data.get("dated_models", {})
            for model, config in dated_models.items():
                try:
                    # Convert min_version dict to ModelVersion instance if present
                    if "min_version" in config:
                        config["min_version"] = ModelVersion(
                            **config["min_version"]
                        )
                    config["model_name"] = model  # Add model name from key
                    self._capabilities[model] = ModelCapabilities(**config)
                except ValidationError as e:
                    _log(
                        _default_log_callback,
                        LogLevel.ERROR,
                        LogEvent.MODEL_REGISTRY,
                        {
                            "message": f"Failed to load capabilities for model {model}",
                            "error": str(e),
                        },
                    )

            # Load aliases
            aliases = data.get("aliases", {})
            for alias, target in aliases.items():
                if target in self._capabilities:
                    # Copy capabilities from target and add this alias
                    caps = self._capabilities[target]
                    caps.aliases.add(alias)
                    # Create new capabilities instance for alias with its own name
                    alias_caps = ModelCapabilities(
                        model_name=alias,
                        context_window=caps.context_window,
                        max_output_tokens=caps.max_output_tokens,
                        supports_structured=caps.supports_structured,
                        supports_streaming=caps.supports_streaming,
                        supported_parameters=caps.supported_parameters,
                        aliases=caps.aliases,
                        min_version=caps.min_version,
                    )
                    self._capabilities[alias] = alias_caps
                else:
                    _log(
                        _default_log_callback,
                        LogLevel.ERROR,
                        LogEvent.MODEL_REGISTRY,
                        {
                            "message": f"Alias target {target} not found for alias {alias}"
                        },
                    )

        except (FileNotFoundError, yaml.YAMLError) as e:
            _log(
                _default_log_callback,
                LogLevel.WARNING,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Failed to load model registry config, using fallbacks",
                    "error": str(e),
                    "path": self._config_path,
                },
            )
            # Load fallback models
            fallbacks = self._fallback_models

            # Load dated models from fallbacks
            for model, config in fallbacks.get("dated_models", {}).items():
                try:
                    if "min_version" in config:
                        config["min_version"] = ModelVersion(
                            **config["min_version"]
                        )
                    config["model_name"] = model  # Add model name from key
                    self._capabilities[model] = ModelCapabilities(**config)
                except ValidationError as e:
                    _log(
                        _default_log_callback,
                        LogLevel.ERROR,
                        LogEvent.MODEL_REGISTRY,
                        {
                            "message": f"Failed to load fallback for model {model}",
                            "error": str(e),
                        },
                    )

            # Load aliases from fallbacks
            for alias, target in fallbacks.get("aliases", {}).items():
                if target in self._capabilities:
                    # Copy capabilities from target and add this alias
                    caps = self._capabilities[target]
                    caps.aliases.add(alias)
                    # Create new capabilities instance for alias with its own name
                    alias_caps = ModelCapabilities(
                        model_name=alias,
                        context_window=caps.context_window,
                        max_output_tokens=caps.max_output_tokens,
                        supports_structured=caps.supports_structured,
                        supports_streaming=caps.supports_streaming,
                        supported_parameters=caps.supported_parameters,
                        aliases=caps.aliases,
                        min_version=caps.min_version,
                    )
                    self._capabilities[alias] = alias_caps

    def supports_structured_output(self, model: str) -> bool:
        """Check if a model supports structured output."""
        try:
            capabilities = self.get_capabilities(model)
            return capabilities.supports_structured
        except ModelNotSupportedError:
            return False

    def supports_streaming(self, model: str) -> bool:
        """Check if a model supports streaming."""
        try:
            capabilities = self.get_capabilities(model)
            return capabilities.supports_streaming
        except (ModelNotSupportedError, ModelVersionError):
            return False

    def get_context_window(self, model: str) -> int:
        """Get the context window size for a model."""
        capabilities = self.get_capabilities(model)
        return capabilities.context_window

    def get_max_output_tokens(self, model: str) -> int:
        """Get the maximum output tokens for a model."""
        capabilities = self.get_capabilities(model)
        return capabilities.max_output_tokens

    def refresh_from_remote(self) -> bool:
        """Refresh registry from remote source.

        Returns:
            bool: True if refresh was successful, False otherwise
        """
        try:
            import requests

            url = (
                "https://raw.githubusercontent.com/yaniv-golan/"
                "openai-structured/main/src/openai_structured/config/models.yml"
            )
            response = requests.get(url)
            if response.status_code == 200:
                with open(self._config_path, "w") as f:
                    f.write(response.text)
                self._load_capabilities()
                return True
            return False
        except Exception:
            return False

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a model.

        Args:
            model: Model name, which can be:
                  - Dated model (e.g. "gpt-4o-2024-08-06")
                  - Alias (e.g. "gpt-4o")
                  - Versioned model (e.g. "gpt-4o-2024-09-01")

        Returns:
            ModelCapabilities for the requested model

        Raises:
            ModelNotSupportedError: If the model is not supported
            InvalidDateError: If the date components are invalid
            VersionTooOldError: If the version is older than minimum supported
        """
        # First check for exact match (dated model or alias)
        if model in self._capabilities:
            return self._capabilities[model]

        # Check if this is a versioned model (exact format: base-YYYY-MM-DD)
        version_match = re.match(r"^(.*)-(\d{4}-\d{2}-\d{2})$", model)
        if not version_match:
            # Not a dated model, alias, or properly formatted version - reject
            raise ModelNotSupportedError(
                f"Model {model} is not supported. Available models: "
                f"{', '.join(sorted(m for m in self._capabilities.keys() if not m.endswith('-latest')))}"
            )

        # Extract base model and version
        base_model, version_str = version_match.groups()

        # Find matching base model or alias
        base_caps = None
        for name, caps in self._capabilities.items():
            if name == base_model or base_model in caps.aliases:
                base_caps = caps
                break

        if not base_caps:
            raise ModelNotSupportedError(
                f"Base model {base_model} is not supported. Available models: "
                f"{', '.join(sorted(m for m in self._capabilities.keys() if not m.endswith('-latest')))}"
            )

        # Validate version format and components
        try:
            version = ModelVersion.from_string(version_str)
            if base_caps.min_version and version < base_caps.min_version:
                raise VersionTooOldError(
                    model=model,
                    version=str(version),
                    min_version=str(base_caps.min_version),
                )
            return base_caps
        except ValueError as e:
            raise InvalidDateError(model, version_str, str(e))

    def get_parameter_constraint(
        self, ref: str
    ) -> Union[NumericConstraint, EnumConstraint]:
        """Get a parameter constraint by reference."""
        if ref not in self._constraints:
            raise KeyError(f"Parameter constraint {ref} not found")
        return self._constraints[ref]

    def get_fixed_parameters(self, ref: str) -> FixedParameterSet:
        """Get fixed parameters by reference."""
        if ref not in self._fixed_parameters:
            raise KeyError(f"Fixed parameter set {ref} not found")
        return self._fixed_parameters[ref]

    # Fallback models provide default capabilities when the models.yml config is missing or invalid.
    # They are automatically updated by the update-fallbacks.yml GitHub workflow when:
    # 1. Changes are pushed to models.yml in main/next branches
    # 2. The workflow is manually triggered
    #
    # The update process:
    # 1. Reads the latest models.yml configuration
    # 2. Converts the YAML structure to match the registry format
    # 3. Updates the section between the AUTO-GENERATED markers below
    # 4. Creates a PR with the changes for review
    #
    # The structure matches models.yml:
    # - version: Schema version
    # - dated_models: Specific dated versions with full capabilities
    # - aliases: Latest/stable version aliases pointing to dated versions
    #
    # DO NOT EDIT THIS SECTION MANUALLY - Changes will be overwritten
    # Use models.yml for configuration changes instead

    # AUTO-GENERATED FALLBACK START
    _fallback_models = {
        "version": "1.0.0",
        "dated_models": {
            "gpt-4o-2024-08-06": {
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": [
                    {
                        "ref": "numeric_constraints.temperature",
                        "max_value": None,
                    },
                    {"ref": "numeric_constraints.top_p", "max_value": None},
                    {
                        "ref": "numeric_constraints.frequency_penalty",
                        "max_value": None,
                    },
                    {
                        "ref": "numeric_constraints.presence_penalty",
                        "max_value": None,
                    },
                    {
                        "ref": "numeric_constraints.max_completion_tokens",
                        "max_value": None,
                    },
                ],
                "description": "Initial release with 16k output support",
            },
            "gpt-4o-mini-2024-07-18": {
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": [
                    {
                        "ref": "numeric_constraints.temperature",
                        "max_value": None,
                    },
                    {"ref": "numeric_constraints.top_p", "max_value": None},
                    {
                        "ref": "numeric_constraints.frequency_penalty",
                        "max_value": None,
                    },
                    {
                        "ref": "numeric_constraints.presence_penalty",
                        "max_value": None,
                    },
                    {
                        "ref": "numeric_constraints.max_completion_tokens",
                        "max_value": None,
                    },
                ],
                "description": "First release of mini variant",
            },
            "o1-2024-12-17": {
                "context_window": 200000,
                "max_output_tokens": 100000,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": [
                    {
                        "ref": "numeric_constraints.max_completion_tokens",
                        "max_value": None,
                    },
                    {
                        "ref": "enum_constraints.reasoning_effort",
                        "max_value": None,
                    },
                ],
                "description": "Initial preview release",
            },
            "o3-mini-2025-01-31": {
                "context_window": 200000,
                "max_output_tokens": 100000,
                "supports_structured": True,
                "supports_streaming": True,
                "supported_parameters": [
                    {
                        "ref": "numeric_constraints.max_completion_tokens",
                        "max_value": None,
                    },
                    {
                        "ref": "enum_constraints.reasoning_effort",
                        "max_value": None,
                    },
                ],
                "description": "First o3-series model",
            },
        },
        "aliases": {
            "gpt-4o": "gpt-4o-2024-08-06",
            "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
            "o1": "o1-2024-12-17",
            "o3-mini": "o3-mini-2025-01-31",
        },
    }
    # AUTO-GENERATED FALLBACK END

    @classmethod
    def cleanup(cls) -> None:
        """Clean up the registry instance."""
        cls._instance = None
        cls._config_path = None
        cls._constraints_path = None

    @staticmethod
    def _parse_version(model: str) -> Optional[ModelVersion]:
        """Parse version from a model name."""
        # ... rest of the method remains unchanged ...
        pass

    @property
    def models(self) -> Dict[str, ModelCapabilities]:
        """Get all registered models.

        This property exists for backward compatibility with tests.
        """
        return self._capabilities
