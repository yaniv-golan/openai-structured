"""Model registry for managing OpenAI model capabilities and versions.

This module provides a centralized registry for OpenAI model capabilities,
including context windows, token limits, and supported features. It handles
both model aliases and dated versions, with version validation and fallback
support.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, cast

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

__all__ = [
    "ModelRegistry",
    "ModelCapabilities",
    "NumericConstraint",
    "EnumConstraint",
    "ParameterReference",
    "FixedParameterSet",
    "ModelVersion",
    "RegistryUpdateStatus",
    "RegistryUpdateResult",
]

# Create module logger
logger = logging.getLogger(__name__)


def _default_log_callback(
    level: int, event: str, data: Dict[str, Any]
) -> None:
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
            supported_params = [
                ref.ref.split(".")[1] for ref in self.supported_parameters
            ]
            raise OpenAIClientError(
                f"Parameter '{param_name}' is not supported by model '{self.model_name}'.\n"
                f"Supported parameters: {', '.join(sorted(supported_params))}"
            )

        # Get the constraint from the registry
        constraint = ModelRegistry.get_instance().get_parameter_constraint(
            param_ref.ref
        )

        if isinstance(constraint, NumericConstraint):
            # Validate numeric type
            if not isinstance(value, (int, float)):
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be a number, got {type(value).__name__}.\n"
                    "Allowed types: "
                    + (
                        "float and integer"
                        if constraint.allow_float and constraint.allow_int
                        else (
                            "float only"
                            if constraint.allow_float
                            else "integer only"
                        )
                    )
                )

            # Validate integer/float requirements
            if isinstance(value, float) and not constraint.allow_float:
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be an integer, got float {value}.\n"
                    f"Description: {constraint.description}"
                )
            if isinstance(value, int) and not constraint.allow_int:
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be a float, got integer {value}.\n"
                    f"Description: {constraint.description}"
                )

            # Use override max_value if specified
            max_value = param_ref.max_value or constraint.max_value
            if max_value is None:
                max_value = self.max_output_tokens

            # Validate range
            if value < constraint.min_value or value > max_value:
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be between {constraint.min_value} and {max_value}.\n"
                    f"Description: {constraint.description}\n"
                    f"Current value: {value}"
                )

        elif isinstance(constraint, EnumConstraint):
            # Validate type
            if not isinstance(value, str):
                raise OpenAIClientError(
                    f"Parameter '{param_name}' must be a string, got {type(value).__name__}.\n"
                    f"Description: {constraint.description}"
                )

            # Validate allowed values
            if value not in constraint.allowed_values:
                raise OpenAIClientError(
                    f"Invalid value '{value}' for parameter '{param_name}'.\n"
                    f"Description: {constraint.description}\n"
                    f"Allowed values: {', '.join(map(str, sorted(constraint.allowed_values)))}"
                )


class RegistryUpdateStatus(Enum):
    """Status of a registry update operation."""

    UPDATED = "updated"
    ALREADY_CURRENT = "already_current"
    NOT_FOUND = "not_found"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    IMPORT_ERROR = "import_error"
    UNKNOWN_ERROR = "unknown_error"
    UPDATE_AVAILABLE = "update_available"


@dataclass
class RegistryUpdateResult:
    """Result of a registry update operation."""

    success: bool
    status: RegistryUpdateStatus
    message: str
    url: Optional[str] = None
    error: Optional[Exception] = None


class ModelRegistry:
    """Registry for model capabilities and validation."""

    _instance: Optional["ModelRegistry"] = None
    _config_path: Optional[str] = None
    _constraints_path: Optional[str] = None

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

            if not self._config_path or not self._constraints_path:
                raise ValueError("Registry paths not set")

            self._load_constraints()
            self._load_capabilities()
            self._initialized = True

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load model configuration from file.

        Returns:
            Optional[Dict[str, Any]]: The loaded configuration data, or None if loading failed
        """
        if not self._config_path:
            raise ValueError("Config path not set")

        try:
            with open(self._config_path, "r") as f:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    return None
                return data
        except FileNotFoundError:
            _log(
                _default_log_callback,
                LogLevel.WARNING,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Model registry config file not found, using defaults",
                    "path": self._config_path,
                },
            )
            return None
        except Exception as e:
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
            return None

    def _load_constraints(self) -> None:
        """Load parameter constraints from file."""
        if not self._constraints_path:
            raise ValueError("Constraints path not set")

        try:
            with open(self._constraints_path, "r") as f:
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
            data = self._load_config()
            if not data:
                # Load fallback models if no data was loaded
                data = self._fallback_models

            # Load dated models
            dated_models = cast(Dict[str, Any], data.get("dated_models", {}))
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
            aliases = cast(Dict[str, str], data.get("aliases", {}))
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
        except Exception as e:
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
            data = self._fallback_models
            # Load dated models from fallbacks
            dated_models = cast(Dict[str, Any], data.get("dated_models", {}))
            for model, config in dated_models.items():
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
            aliases = cast(Dict[str, str], data.get("aliases", {}))
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

    def _get_cache_metadata_path(self) -> Optional[str]:
        """Get the path to the cache metadata file for the registry."""
        if not self._config_path:
            return None

        return f"{self._config_path}.meta"

    def _save_cache_metadata(self, metadata: Dict[str, str]) -> None:
        """Save cache metadata to a file."""
        metadata_path = self._get_cache_metadata_path()
        if not metadata_path:
            return

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            _log(
                _default_log_callback,
                LogLevel.DEBUG,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Saved cache metadata",
                    "path": metadata_path,
                    "metadata": metadata,
                },
            )
        except (OSError, IOError) as e:
            _log(
                _default_log_callback,
                LogLevel.WARNING,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Could not save cache metadata",
                    "error": str(e),
                },
            )

    def _load_cache_metadata(self) -> Dict[str, str]:
        """Load cache metadata from file."""
        metadata_path = self._get_cache_metadata_path()
        try:
            if metadata_path is None or not os.path.exists(metadata_path):
                return {}

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Ensure we're returning the expected type
            if not isinstance(metadata, dict):
                _log(
                    _default_log_callback,
                    LogLevel.WARNING,
                    LogEvent.MODEL_REGISTRY,
                    {
                        "message": "Cache metadata is not a dictionary",
                        "path": metadata_path,
                    },
                )
                return {}

            # Convert all values to strings to match the return type
            string_metadata: Dict[str, str] = {
                k: str(v) for k, v in metadata.items()
            }

            _log(
                _default_log_callback,
                LogLevel.DEBUG,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Loaded cache metadata",
                    "path": metadata_path,
                    "metadata": string_metadata,
                },
            )
            return string_metadata
        except (OSError, IOError, json.JSONDecodeError) as e:
            _log(
                _default_log_callback,
                LogLevel.WARNING,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Could not load cache metadata",
                    "error": str(e),
                },
            )
            return {}

    def _get_conditional_headers(self, force: bool = False) -> Dict[str, str]:
        """Get headers for conditional requests."""
        headers: Dict[str, str] = {}

        if force:
            return headers

        # Try to load from metadata file first (for ETag)
        metadata = self._load_cache_metadata()
        if "etag" in metadata:
            headers["If-None-Match"] = metadata["etag"]

        # Add If-Modified-Since as a fallback
        if self._config_path and os.path.exists(self._config_path):
            try:
                file_mtime = os.path.getmtime(self._config_path)
                last_modified = time.strftime(
                    "%a, %d %b %Y %H:%M:%S GMT", time.gmtime(file_mtime)
                )
                headers["If-Modified-Since"] = last_modified

                # Debug log the file modification time
                _log(
                    _default_log_callback,
                    LogLevel.DEBUG,
                    LogEvent.MODEL_REGISTRY,
                    {
                        "message": "Local file mtime information",
                        "file_path": self._config_path,
                        "file_mtime_timestamp": file_mtime,
                        "file_mtime_human": last_modified,
                    },
                )
            except (OSError, ValueError) as e:
                # If we can't get file time, just continue without the header
                _log(
                    _default_log_callback,
                    LogLevel.WARNING,
                    LogEvent.MODEL_REGISTRY,
                    {
                        "message": "Could not get file modification time",
                        "error": str(e),
                    },
                )

        return headers

    def refresh_from_remote(
        self, url: Optional[str] = None, force: bool = False
    ) -> RegistryUpdateResult:
        """Refresh registry from remote source using conditional requests.

        Args:
            url: Optional custom URL to fetch configuration from.
                If not provided, uses the default GitHub repository URL.
            force: If True, bypasses the conditional check and forces an update.

        Returns:
            RegistryUpdateResult with status and details
        """
        try:
            import requests
        except ImportError as e:
            return RegistryUpdateResult(
                success=False,
                status=RegistryUpdateStatus.IMPORT_ERROR,
                message="Could not import requests module",
                error=e,
            )

        # Set up the URL and headers
        config_url = url or (
            "https://raw.githubusercontent.com/yaniv-golan/"
            "openai-structured/main/src/openai_structured/config/models.yml"
        )

        headers = self._get_conditional_headers(force)

        _log(
            _default_log_callback,
            LogLevel.INFO,
            LogEvent.MODEL_REGISTRY,
            {
                "message": "Fetching model registry configuration",
                "url": config_url,
                "conditional": bool(
                    headers
                ),  # True if any conditional headers are present
                "headers": headers,
            },
        )

        try:
            # Make the request
            response = requests.get(config_url, headers=headers)

            # Log response headers
            _log(
                _default_log_callback,
                LogLevel.DEBUG,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Received response from server",
                    "status_code": response.status_code,
                    "response_headers": dict(response.headers),
                },
            )

            # Handle different response statuses
            if response.status_code == 304:  # Not Modified
                _log(
                    _default_log_callback,
                    LogLevel.INFO,
                    LogEvent.MODEL_REGISTRY,
                    {
                        "message": "Registry is already up to date (not modified)"
                    },
                )
                return RegistryUpdateResult(
                    success=True,
                    status=RegistryUpdateStatus.ALREADY_CURRENT,
                    message="Registry is already up to date",
                    url=config_url,
                )

            elif response.status_code == 200:  # OK
                if self._config_path is None:
                    return RegistryUpdateResult(
                        success=False,
                        status=RegistryUpdateStatus.UNKNOWN_ERROR,
                        message="Config path not set",
                        url=config_url,
                    )

                try:
                    # Ensure directory exists
                    os.makedirs(
                        os.path.dirname(self._config_path), exist_ok=True
                    )

                    # Write content
                    with open(self._config_path, "w") as f:
                        f.write(response.text)

                    # Save ETag and Last-Modified
                    metadata = {}
                    if "ETag" in response.headers:
                        metadata["etag"] = response.headers["ETag"]
                    if "Last-Modified" in response.headers:
                        metadata["last_modified"] = response.headers[
                            "Last-Modified"
                        ]

                    # Save the metadata
                    self._save_cache_metadata(metadata)

                    # Also update file mtime if Last-Modified header exists
                    if "Last-Modified" in response.headers:
                        try:
                            last_modified_str = response.headers[
                                "Last-Modified"
                            ]
                            last_modified_time = time.strptime(
                                last_modified_str, "%a, %d %b %Y %H:%M:%S GMT"
                            )
                            last_modified_timestamp = time.mktime(
                                last_modified_time
                            )
                            os.utime(
                                self._config_path,
                                (
                                    last_modified_timestamp,
                                    last_modified_timestamp,
                                ),
                            )

                            _log(
                                _default_log_callback,
                                LogLevel.DEBUG,
                                LogEvent.MODEL_REGISTRY,
                                {
                                    "message": "Updated file modification time from server header",
                                    "last_modified_from_server": last_modified_str,
                                    "timestamp_applied": last_modified_timestamp,
                                },
                            )
                        except (ValueError, OSError) as e:
                            _log(
                                _default_log_callback,
                                LogLevel.WARNING,
                                LogEvent.MODEL_REGISTRY,
                                {
                                    "message": "Could not update file modification time from response header",
                                    "error": str(e),
                                },
                            )

                    # Reload capabilities
                    self._load_capabilities()

                    _log(
                        _default_log_callback,
                        LogLevel.INFO,
                        LogEvent.MODEL_REGISTRY,
                        {"message": "Successfully updated model registry"},
                    )

                    return RegistryUpdateResult(
                        success=True,
                        status=RegistryUpdateStatus.UPDATED,
                        message="Registry updated successfully",
                        url=config_url,
                    )
                except (OSError, IOError) as e:
                    _log(
                        _default_log_callback,
                        LogLevel.ERROR,
                        LogEvent.MODEL_REGISTRY,
                        {
                            "message": f"Could not write to {self._config_path}",
                            "error": str(e),
                        },
                    )
                    return RegistryUpdateResult(
                        success=False,
                        status=RegistryUpdateStatus.PERMISSION_ERROR,
                        message=f"Could not write to {self._config_path}",
                        error=e,
                        url=config_url,
                    )

            elif response.status_code == 404:  # Not Found
                _log(
                    _default_log_callback,
                    LogLevel.ERROR,
                    LogEvent.MODEL_REGISTRY,
                    {
                        "message": "Registry not found",
                        "status_code": response.status_code,
                    },
                )
                return RegistryUpdateResult(
                    success=False,
                    status=RegistryUpdateStatus.NOT_FOUND,
                    message=f"Registry not found at {config_url}",
                    url=config_url,
                )

            else:  # Other error codes
                _log(
                    _default_log_callback,
                    LogLevel.ERROR,
                    LogEvent.MODEL_REGISTRY,
                    {
                        "message": "Failed to fetch configuration",
                        "status_code": response.status_code,
                        "response": response.text,
                    },
                )
                return RegistryUpdateResult(
                    success=False,
                    status=RegistryUpdateStatus.NETWORK_ERROR,
                    message=f"HTTP error {response.status_code}: {response.text}",
                    url=config_url,
                )

        except requests.RequestException as e:
            _log(
                _default_log_callback,
                LogLevel.ERROR,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Network error when refreshing model registry",
                    "error": str(e),
                },
            )
            return RegistryUpdateResult(
                success=False,
                status=RegistryUpdateStatus.NETWORK_ERROR,
                message=f"Network error: {str(e)}",
                error=e,
                url=config_url,
            )

        except Exception as e:
            _log(
                _default_log_callback,
                LogLevel.ERROR,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Failed to refresh model registry",
                    "error": str(e),
                },
            )
            return RegistryUpdateResult(
                success=False,
                status=RegistryUpdateStatus.UNKNOWN_ERROR,
                message=f"Unexpected error: {str(e)}",
                error=e,
                url=config_url,
            )

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
            available_models = sorted(
                m
                for m in self._capabilities.keys()
                if not m.endswith("-latest")
            )
            dated_models = [
                m
                for m in available_models
                if re.match(r"^.*-\d{4}-\d{2}-\d{2}$", m)
            ]
            alias_models = [
                m
                for m in available_models
                if not re.match(r"^.*-\d{4}-\d{2}-\d{2}$", m)
            ]
            error_msg = (
                f"Model '{model}' is not supported.\n"
                f"Available models:\n"
                f"- Dated models: {', '.join(dated_models)}\n"
                f"- Aliases: {', '.join(alias_models)}\n"
                f"Note: For dated models, use format: base-YYYY-MM-DD (e.g. gpt-4o-2024-08-06)"
            )
            raise ModelNotSupportedError(error_msg)

        # Extract base model and version
        base_model, version_str = version_match.groups()

        # Find matching base model or alias
        base_caps = None
        for name, caps in self._capabilities.items():
            if name == base_model or base_model in caps.aliases:
                base_caps = caps
                break

        if not base_caps:
            available_bases = {
                name.split("-")[0]
                for name in self._capabilities.keys()
                if not name.endswith("-latest")
            }
            raise ModelNotSupportedError(
                f"Base model '{base_model}' is not supported.\n"
                f"Available base models: {', '.join(sorted(available_bases))}\n"
                f"Note: Base model names are case-sensitive"
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
            raise InvalidDateError(
                model=model,
                version=version_str,
                message=(
                    f"Invalid date format in model version: {str(e)}\n"
                    f"Use format: YYYY-MM-DD (e.g. 2024-08-06)"
                ),
            )

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
                "min_version": {
                    "year": 2024,
                    "month": 8,
                    "day": 6,
                },
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
                "min_version": {
                    "year": 2024,
                    "month": 7,
                    "day": 18,
                },
            },
            "gpt-4.5-preview-2025-02-27": {
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
                "description": "GPT-4.5 preview release",
                "min_version": {
                    "year": 2025,
                    "month": 2,
                    "day": 27,
                },
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
                "min_version": {
                    "year": 2024,
                    "month": 12,
                    "day": 17,
                },
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
                "min_version": {
                    "year": 2025,
                    "month": 1,
                    "day": 31,
                },
            },
        },
        "aliases": {
            "gpt-4o": "gpt-4o-2024-08-06",
            "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
            "gpt-4.5-preview": "gpt-4.5-preview-2025-02-27",
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
        """Get a read-only view of registered models."""
        return dict(self._capabilities)

    def check_for_updates(
        self, url: Optional[str] = None
    ) -> RegistryUpdateResult:
        """Check if an updated registry is available without downloading it.

        This method performs a lightweight request to check if a newer version of the
        registry is available. This allows applications to notify users about available
        updates without automatically downloading them.

        Args:
            url: Optional custom URL to check for updates.
                If not provided, uses the default GitHub repository URL.

        Returns:
            RegistryUpdateResult with status indicating if an update is available
        """
        try:
            import requests
        except ImportError as e:
            return RegistryUpdateResult(
                success=False,
                status=RegistryUpdateStatus.IMPORT_ERROR,
                message="Could not import requests module",
                error=e,
            )

        # Set up the URL and headers
        config_url = url or (
            "https://raw.githubusercontent.com/yaniv-golan/"
            "openai-structured/main/src/openai_structured/config/models.yml"
        )

        # Get conditional headers
        headers = self._get_conditional_headers()

        # Add Range header to request only the first byte
        # This makes the request lightweight while still using GET
        # which has more consistent behavior with refresh_from_remote
        headers["Range"] = "bytes=0-0"

        _log(
            _default_log_callback,
            LogLevel.INFO,
            LogEvent.MODEL_REGISTRY,
            {
                "message": "Checking for registry updates",
                "url": config_url,
                "conditional": bool(headers),
                "headers": headers,
            },
        )

        try:
            # Use GET with Range instead of HEAD to ensure consistent behavior
            # with refresh_from_remote
            response = requests.get(config_url, headers=headers)

            # Log response headers
            _log(
                _default_log_callback,
                LogLevel.DEBUG,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Received check response from server",
                    "status_code": response.status_code,
                    "response_headers": dict(response.headers),
                },
            )

            # Handle different response statuses
            if response.status_code == 304:  # Not Modified
                _log(
                    _default_log_callback,
                    LogLevel.INFO,
                    LogEvent.MODEL_REGISTRY,
                    {
                        "message": "Registry is already up to date (not modified)"
                    },
                )
                return RegistryUpdateResult(
                    success=True,
                    status=RegistryUpdateStatus.ALREADY_CURRENT,
                    message="Registry is already up to date",
                    url=config_url,
                )

            elif response.status_code in (200, 206):  # OK or Partial Content
                # A 200/206 response means the resource has been modified or is available

                # Update the metadata for next time, without modifying the file
                metadata = {}
                if "ETag" in response.headers:
                    metadata["etag"] = response.headers["ETag"]
                if "Last-Modified" in response.headers:
                    metadata["last_modified"] = response.headers[
                        "Last-Modified"
                    ]

                # Save the metadata for future checks
                self._save_cache_metadata(metadata)

                _log(
                    _default_log_callback,
                    LogLevel.INFO,
                    LogEvent.MODEL_REGISTRY,
                    {"message": "Registry update is available"},
                )
                return RegistryUpdateResult(
                    success=True,
                    status=RegistryUpdateStatus.UPDATE_AVAILABLE,
                    message="Registry update is available",
                    url=config_url,
                )

            elif response.status_code == 404:  # Not Found
                _log(
                    _default_log_callback,
                    LogLevel.ERROR,
                    LogEvent.MODEL_REGISTRY,
                    {
                        "message": "Registry not found",
                        "status_code": response.status_code,
                    },
                )
                return RegistryUpdateResult(
                    success=False,
                    status=RegistryUpdateStatus.NOT_FOUND,
                    message=f"Registry not found at {config_url}",
                    url=config_url,
                )

            else:  # Other error codes
                _log(
                    _default_log_callback,
                    LogLevel.ERROR,
                    LogEvent.MODEL_REGISTRY,
                    {
                        "message": "Failed to check for updates",
                        "status_code": response.status_code,
                    },
                )
                return RegistryUpdateResult(
                    success=False,
                    status=RegistryUpdateStatus.NETWORK_ERROR,
                    message=f"HTTP error {response.status_code}",
                    url=config_url,
                )

        except requests.RequestException as e:
            _log(
                _default_log_callback,
                LogLevel.ERROR,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Network error when checking for updates",
                    "error": str(e),
                },
            )
            return RegistryUpdateResult(
                success=False,
                status=RegistryUpdateStatus.NETWORK_ERROR,
                message=f"Network error: {str(e)}",
                error=e,
                url=config_url,
            )

        except Exception as e:
            _log(
                _default_log_callback,
                LogLevel.ERROR,
                LogEvent.MODEL_REGISTRY,
                {
                    "message": "Failed to check for updates",
                    "error": str(e),
                },
            )
            return RegistryUpdateResult(
                success=False,
                status=RegistryUpdateStatus.UNKNOWN_ERROR,
                message=f"Unexpected error: {str(e)}",
                error=e,
                url=config_url,
            )
