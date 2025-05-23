"""Testing utilities for openai-structured library.

This module provides utilities for testing code that uses the openai-structured library,
including test registry creation, model capabilities mocking, and parameter constraint
utilities.
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Union

import pytest
import yaml
from openai_model_registry import (
    ConstraintNotFoundError,
    EnumConstraint,
    InvalidDateError,
    ModelCapabilities,
    ModelNotSupportedError,
    ModelRegistry,
    NumericConstraint,
    ParameterReference,
    VersionTooOldError,
)
from openai_model_registry.deprecation import DeprecationInfo


def create_test_registry(
    model_config: Optional[Dict[str, Any]] = None,
    constraints_config: Optional[Dict[str, Any]] = None,
) -> ModelRegistry:
    """Create a test registry with custom configuration.

    Args:
        model_config: Optional dictionary with model configurations
        constraints_config: Optional dictionary with parameter constraints

    Returns:
        ModelRegistry: Configured test registry

    Example:
        >>> registry = create_test_registry({
        ...     "test-model": {
        ...         "context_window": 4096,
        ...         "supported_parameters": [
        ...             {"ref": "numeric_constraints.temperature"}
        ...         ]
        ...     }
        ... })
    """
    # Clean up any existing registry
    ModelRegistry.cleanup()

    # Create a custom subclass of ModelCapabilities with the methods we need
    class TestModelCapabilities(ModelCapabilities):
        def __init__(
            self,
            model_name: str,
            openai_model_name: str,
            context_window: int,
            max_output_tokens: int,
            supports_structured: bool,
            supports_streaming: bool,
            supported_parameters: List[ParameterReference],
            registry: "TestModelRegistry",
        ) -> None:
            super().__init__(
                model_name=model_name,
                openai_model_name=openai_model_name,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                deprecation=DeprecationInfo(
                    status="active",
                    deprecates_on=None,
                    sunsets_on=None,
                    replacement=None,
                    migration_guide=None,
                    reason="Test model for openai-structured library",
                ),
                supports_structured=supports_structured,
                supports_streaming=supports_streaming,
                supported_parameters=supported_parameters,
            )
            self._registry = registry

        def get_constraint(self, ref: str) -> Any:
            """Get constraint by reference.

            Args:
                ref: Constraint reference in format 'group.name'

            Returns:
                Constraint object or None if not found
            """
            if not ref or "." not in ref:
                return None

            group, name = ref.split(".", 1)
            if (
                group in self._registry._constraints
                and name in self._registry._constraints[group]
            ):
                return self._registry._constraints[group][name]

            return None

        def validate_parameter(
            self, name: str, value: Any, used_params: Optional[Set[str]] = None
        ) -> None:
            """Validate a parameter against constraints.

            Args:
                name: Parameter name
                value: Parameter value
                used_params: Optional set of already used parameters

            Raises:
                Various validation errors depending on the parameter
            """
            from openai_structured.errors import (
                OpenAIClientError,
                TokenParameterError,
            )

            # Track used parameters if requested
            if used_params is not None:
                used_params.add(name)

            # Special case for token parameters
            if (
                name == "max_completion_tokens"
                and used_params
                and "max_output_tokens" in used_params
            ):
                raise TokenParameterError(
                    "Cannot specify both 'max_output_tokens' and 'max_completion_tokens' parameters. "
                    "Choose one: max_output_tokens (recommended) or max_completion_tokens (legacy)."
                )

            # Find matching parameter reference
            param_ref = next(
                (
                    p
                    for p in self.supported_parameters
                    if p.ref.split(".")[-1] == name
                ),
                None,
            )

            if not param_ref:
                # Handle unsupported parameter
                supported_params = ", ".join(
                    sorted(
                        p.ref.split(".")[-1] for p in self.supported_parameters
                    )
                )
                raise OpenAIClientError(
                    f"Parameter '{name}' is not supported by model '{self.model_name}'. "
                    f"Supported parameters: {supported_params}"
                )

            constraint = self.get_constraint(param_ref.ref)
            if not constraint:
                raise ConstraintNotFoundError(
                    f"Constraint reference '{param_ref.ref}' not found for parameter '{name}'",
                    ref=param_ref.ref,
                )

            # Numeric constraint validation
            if isinstance(constraint, NumericConstraint):
                if not (
                    (constraint.allow_int and isinstance(value, int))
                    or (constraint.allow_float and isinstance(value, float))
                ):
                    allowed_types = []
                    if constraint.allow_int:
                        allowed_types.append("int")
                    if constraint.allow_float:
                        allowed_types.append("float")

                    raise OpenAIClientError(
                        f"Parameter '{name}' must be a number, got {type(value).__name__}. "
                        f"Allowed types: {', '.join(allowed_types)}."
                    )

                # Range check
                if value < constraint.min_value or (
                    constraint.max_value is not None
                    and value > constraint.max_value
                ):
                    max_str = (
                        str(constraint.max_value)
                        if constraint.max_value is not None
                        else "unlimited"
                    )
                    raise OpenAIClientError(
                        f"Parameter '{name}' must be between {constraint.min_value} and {max_str}. "
                        f"Description: {constraint.description}. Current value: {value}"
                    )

            # Enum constraint validation
            elif isinstance(constraint, EnumConstraint):
                if value not in constraint.allowed_values:
                    raise OpenAIClientError(
                        f"Invalid value '{value}' for parameter '{name}'. "
                        f"Description: {constraint.description}. "
                        f"Allowed values: {', '.join(constraint.allowed_values)}."
                    )

    # Create a custom subclass of ModelRegistry with the methods we need
    class TestModelRegistry(ModelRegistry):
        def __init__(self) -> None:
            # Skip parent initialization
            self._capabilities: Dict[str, ModelCapabilities] = {}
            self._constraints: Dict[
                str, Dict[str, Union[NumericConstraint, EnumConstraint]]
            ] = {}
            self._aliases: Dict[str, str] = {}

        def get_capabilities(self, model: str) -> ModelCapabilities:
            """Get the capabilities for a model.

            Args:
                model: Model name

            Returns:
                ModelCapabilities for the requested model

            Raises:
                ModelNotSupportedError: If the model is not supported
                InvalidDateError: If the date components are invalid
                VersionTooOldError: If the version is older than minimum supported
            """
            # Check aliases first
            if model in self._aliases:
                model = self._aliases[model]

            if model in self._capabilities:
                return self._capabilities[model]

            # Handle dated model requests by returning base model capabilities
            if model == "gpt-4o-2024-08-06":
                return self._capabilities.get(
                    "gpt-4o", self._capabilities["test-model"]
                )

            # Handle special test cases for each error type
            if model == "unsupported-model":
                raise ModelNotSupportedError(
                    f"Model '{model}' is not supported. Available models: test-model, gpt-4o. "
                    f"Dated models: test-model-2024-01-01, gpt-4o-2024-08-06. "
                    f"Aliases: test-model, gpt-4o. "
                    f"Note: For dated models, use format: base-YYYY-MM-DD",
                    model=model,
                    available_models=["test-model", "gpt-4o"],
                )
            elif model == "invalid-2024-08-06":
                raise ModelNotSupportedError(
                    "Base model 'invalid' is not supported. Available base models: "
                    "test-model, gpt-4o, o1. Note: Base model names are case-sensitive",
                    model=model,
                    available_models=["test-model", "gpt-4o", "o1"],
                )
            elif model == "gpt-4o-2024-07-01":
                raise VersionTooOldError(
                    f"Model '{model}' version 2024-07-01 is too old. "
                    f"Minimum supported version: 2024-08-06. "
                    f"Note: Use the alias 'gpt-4o' to always get the latest version",
                    model=model,
                    min_version="2024-08-06",
                    alias="gpt-4o",
                )
            elif model == "test-gpt4o-2024-07-01":
                raise VersionTooOldError(
                    f"Model '{model}' version 2024-07-01 is too old. "
                    f"Minimum supported version: 2024-08-06. "
                    f"Note: Use the alias 'test-gpt4o' to always get the latest version",
                    model=model,
                    min_version="2024-08-06",
                    alias="test-gpt4o",
                )
            elif model == "gpt-4o-2024-13-01":
                raise InvalidDateError(
                    f"Invalid date format in model version: {model}. "
                    f"Use format: YYYY-MM-DD (e.g. 2024-08-06)"
                )
            else:
                available_models = list(self._capabilities.keys()) + list(
                    self._aliases.keys()
                )
                raise ModelNotSupportedError(
                    f"Model '{model}' not found. Available models: {', '.join(available_models)}",
                    model=model,
                    available_models=available_models,
                )

    # Create constraints programmatically
    registry = TestModelRegistry()

    # Define default constraints
    default_constraints = {
        "numeric_constraints": {
            "temperature": NumericConstraint(
                min_value=0.0,
                max_value=2.0,
                description="Controls randomness in the output",
                allow_float=True,
                allow_int=True,
            ),
            "top_p": NumericConstraint(
                min_value=0.0,
                max_value=1.0,
                description="Controls diversity via nucleus sampling",
                allow_float=True,
                allow_int=True,
            ),
            "max_completion_tokens": NumericConstraint(
                min_value=1,
                max_value=None,  # Will be set by model's max_output_tokens
                description="Maximum number of tokens to generate",
                allow_float=False,
                allow_int=True,
            ),
        },
        "enum_constraints": {
            "reasoning_effort": EnumConstraint(
                allowed_values=["low", "medium", "high"],
                description="Controls the model's reasoning depth",
            ),
        },
    }

    # Merge with custom constraints if provided
    if constraints_config:
        for group_name, group_constraints in constraints_config.items():
            if group_name not in default_constraints:
                default_constraints[group_name] = {}
            for (
                constraint_name,
                constraint_config,
            ) in group_constraints.items():
                if constraint_config.get("type") == "numeric":
                    default_constraints[group_name][constraint_name] = (
                        NumericConstraint(
                            min_value=constraint_config["min_value"],
                            max_value=constraint_config["max_value"],
                            description=constraint_config.get(
                                "description", ""
                            ),
                            allow_float=constraint_config.get(
                                "allow_float", True
                            ),
                            allow_int=constraint_config.get("allow_int", True),
                        )
                    )
                elif constraint_config.get("type") == "enum":
                    default_constraints[group_name][constraint_name] = (
                        EnumConstraint(
                            allowed_values=constraint_config["allowed_values"],
                            description=constraint_config.get(
                                "description", ""
                            ),
                        )
                    )

    registry._constraints = default_constraints

    # Define default model capabilities
    default_models = {
        "test-model": {
            "openai_model_name": "test-model",
            "context_window": 4096,
            "max_output_tokens": 2048,
            "supports_structured": True,
            "supports_streaming": True,
            "supported_parameters": [
                {"ref": "numeric_constraints.temperature"},
                {"ref": "numeric_constraints.top_p"},
                {"ref": "numeric_constraints.max_completion_tokens"},
            ],
        },
        "test-o1": {
            "openai_model_name": "test-o1",
            "context_window": 200000,
            "max_output_tokens": 100000,
            "supports_structured": True,
            "supports_streaming": True,
            "supported_parameters": [
                {"ref": "enum_constraints.reasoning_effort"},
            ],
        },
        "gpt-4o": {
            "openai_model_name": "gpt-4o",
            "context_window": 128000,
            "max_output_tokens": 16384,
            "supports_structured": True,
            "supports_streaming": True,
            "supported_parameters": [
                {"ref": "numeric_constraints.temperature"},
                {"ref": "numeric_constraints.top_p"},
                {"ref": "numeric_constraints.max_completion_tokens"},
            ],
        },
        "test-gpt4o": {
            "openai_model_name": "test-gpt4o",
            "context_window": 128000,
            "max_output_tokens": 16384,
            "supports_structured": True,
            "supports_streaming": True,
            "supported_parameters": [
                {"ref": "numeric_constraints.temperature"},
                {"ref": "numeric_constraints.top_p"},
            ],
        },
    }

    # Process custom model config if provided
    if model_config:
        # Handle dated models
        if "dated_models" in model_config:
            for model_name, model_spec in model_config["dated_models"].items():
                default_models[model_name] = model_spec

        # Handle aliases
        if "aliases" in model_config:
            registry._aliases.update(model_config["aliases"])

        # Handle direct model definitions
        for key, value in model_config.items():
            if key not in ["dated_models", "aliases"] and isinstance(
                value, dict
            ):
                default_models[key] = value

    # Create model capabilities for all models
    for model_name, model_spec in default_models.items():
        param_refs = [
            ParameterReference(**param)
            for param in model_spec["supported_parameters"]
        ]

        capabilities = TestModelCapabilities(
            model_name=model_name,
            openai_model_name=model_spec.get("openai_model_name", model_name),
            context_window=model_spec["context_window"],
            max_output_tokens=model_spec["max_output_tokens"],
            supports_structured=model_spec.get("supports_structured", True),
            supports_streaming=model_spec.get("supports_streaming", True),
            supported_parameters=param_refs,
            registry=registry,
        )

        registry._capabilities[model_name] = capabilities

    return registry


def get_test_capabilities(
    model_name: str = "test-model",
    openai_model_name: str = "test-model",
    context_window: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    supports_structured: bool = True,
    supports_streaming: bool = True,
    supported_parameters: Optional[List[Dict[str, Any]]] = None,
) -> ModelCapabilities:
    """Get pre-configured test capabilities.

    Args:
        model_name: Model identifier in registry (default: "test-model")
        openai_model_name: Name of the model (default: "test-model")
        context_window: Optional context window size
        max_output_tokens: Optional maximum output tokens
        supports_structured: Whether model supports structured output
        supports_streaming: Whether model supports streaming responses
        supported_parameters: Optional list of supported parameters

    Returns:
        ModelCapabilities: Configured test capabilities

    Example:
        >>> capabilities = get_test_capabilities(
        ...     model_name="test-model-2024-01-01",
        ...     openai_model_name="my-model",
        ...     context_window=8192,
        ...     supported_parameters=[
        ...         {"ref": "numeric_constraints.temperature"}
        ...     ]
        ... )
    """
    # Use default values if not provided
    context_window = context_window or 4096
    max_output_tokens = max_output_tokens or 2048
    supported_parameters = supported_parameters or [
        {"ref": "numeric_constraints.temperature"}
    ]

    # Convert parameter dictionaries to ParameterReference objects
    param_refs = [
        ParameterReference(**param) for param in supported_parameters
    ]

    return ModelCapabilities(
        model_name=model_name,
        openai_model_name=openai_model_name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        deprecation=DeprecationInfo(
            status="active",
            deprecates_on=None,
            sunsets_on=None,
            replacement=None,
            migration_guide=None,
            reason="Test model for openai-structured library",
        ),
        supports_structured=supports_structured,
        supports_streaming=supports_streaming,
        supported_parameters=param_refs,
    )


def create_numeric_constraint(
    min_value: float,
    max_value: float,
    description: str = "",
    allow_int: bool = True,
    allow_float: bool = True,
) -> NumericConstraint:
    """Create a numeric parameter constraint.

    Args:
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        description: Optional constraint description
        allow_int: Whether to allow integer values
        allow_float: Whether to allow float values

    Returns:
        NumericConstraint: Configured numeric constraint

    Example:
        >>> constraint = create_numeric_constraint(0.0, 2.0)
    """
    return NumericConstraint(
        min_value=min_value,
        max_value=max_value,
        description=description,
        allow_int=allow_int,
        allow_float=allow_float,
    )


def create_enum_constraint(
    allowed_values: List[str],
    description: str = "",
) -> EnumConstraint:
    """Create an enum parameter constraint.

    Args:
        allowed_values: List of allowed values
        description: Optional constraint description

    Returns:
        EnumConstraint: Configured enum constraint

    Example:
        >>> constraint = create_enum_constraint(["low", "medium", "high"])
    """
    return EnumConstraint(
        allowed_values=allowed_values,
        description=description,
    )


def _create_temp_config(template_path: Path, config: Dict[str, Any]) -> Path:
    """Create a temporary configuration file.

    Args:
        template_path: Path to template file
        config: Configuration dictionary to merge

    Returns:
        Path: Path to temporary configuration file
    """
    # Read template if it exists
    if template_path.exists():
        with open(template_path) as f:
            template = yaml.safe_load(f)
    else:
        template = {}

    # Merge config with template
    merged = {**template, **config}

    # Create temporary file
    temp_path = template_path.parent / f"temp_{template_path.name}"
    with open(temp_path, "w") as f:
        yaml.dump(merged, f)

    return temp_path


@pytest.fixture
def test_registry() -> Generator[ModelRegistry, None, None]:
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
    # Cleanup after test
    ModelRegistry.cleanup()
