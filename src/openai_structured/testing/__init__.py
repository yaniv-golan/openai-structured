"""Testing utilities for openai-structured library.

This module provides utilities for testing code that uses the openai-structured library,
including test registry creation, model capabilities mocking, and parameter constraint
utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pytest
import yaml

from ..model_registry import (
    EnumConstraint,
    ModelCapabilities,
    ModelRegistry,
    NumericConstraint,
    ParameterReference,
)


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

    # Get template paths
    template_dir = Path(__file__).parent / "templates"
    models_template = template_dir / "test_models.yml"
    constraints_template = template_dir / "test_constraints.yml"

    # Set environment variables for registry paths
    os.environ["MODEL_REGISTRY_PATH"] = str(
        models_template
        if model_config is None
        else _create_temp_config(models_template, model_config)
    )
    os.environ["PARAMETER_CONSTRAINTS_PATH"] = str(
        constraints_template
        if constraints_config is None
        else _create_temp_config(constraints_template, constraints_config)
    )

    # Create and return registry
    return ModelRegistry()


def get_test_capabilities(
    model_name: str = "test-model",
    context_window: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    supported_parameters: Optional[List[Dict[str, Any]]] = None,
) -> ModelCapabilities:
    """Get pre-configured test capabilities.

    Args:
        model_name: Name of the model (default: "test-model")
        context_window: Optional context window size
        max_output_tokens: Optional maximum output tokens
        supported_parameters: Optional list of supported parameters

    Returns:
        ModelCapabilities: Configured test capabilities

    Example:
        >>> capabilities = get_test_capabilities(
        ...     model_name="my-model",
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
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_structured=True,
        supports_streaming=True,
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
        type="numeric",
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
        type="enum",
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
