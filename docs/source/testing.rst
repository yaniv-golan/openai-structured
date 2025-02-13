.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

Testing Guide
============

This guide explains how to effectively test code that uses the openai-structured library.

Architecture Overview
------------------

The openai-structured library uses a layered architecture for parameter validation and model capabilities:

1. **Model Registry**: Central registry that manages model capabilities and parameter constraints
2. **Model Capabilities**: Defines what each model supports (context window, parameters, etc.)
3. **Parameter Constraints**: Defines validation rules for parameters (ranges, allowed values)

These components work together to ensure proper parameter validation and model compatibility.

Testing Components
---------------

Model Registry Testing
~~~~~~~~~~~~~~~~~~~~

The library provides a test registry that can be used in your tests:

.. code-block:: python

    def test_model_capabilities():
        # Get capabilities for a model
        registry = ModelRegistry.get_instance()
        capabilities = registry.get_capabilities("gpt-4o")

        # Test model properties
        assert capabilities.context_window == 128000
        assert capabilities.max_output_tokens == 16384
        assert capabilities.supports_structured

Parameter Validation Testing
~~~~~~~~~~~~~~~~~~~~~~~~~

When testing parameter validation, you can use the model capabilities directly:

.. code-block:: python

    def test_parameter_validation():
        registry = ModelRegistry.get_instance()
        capabilities = registry.get_capabilities("gpt-4o")

        # Test valid parameters
        capabilities.validate_parameter("temperature", 0.7)
        capabilities.validate_parameter("top_p", 0.9)

        # Test invalid parameters
        with pytest.raises(OpenAIClientError, match="must be between"):
            capabilities.validate_parameter("temperature", 2.5)

Common Testing Patterns
--------------------

Testing Model Support
~~~~~~~~~~~~~~~~~~~

Test if a model supports structured output:

.. code-block:: python

    def test_model_support():
        assert supports_structured_output("gpt-4o")
        assert not supports_structured_output("unsupported-model")

Testing Parameter Limits
~~~~~~~~~~~~~~~~~~~~~~

Test parameter validation with different models:

.. code-block:: python

    def test_parameter_limits():
        # Test GPT-4 parameters
        gpt4o = ModelRegistry.get_instance().get_capabilities("gpt-4o")
        gpt4o.validate_parameter("temperature", 0.5)  # Valid

        # Test O1 parameters
        o1 = ModelRegistry.get_instance().get_capabilities("o1")
        o1.validate_parameter("reasoning_effort", "medium")  # Valid

        with pytest.raises(OpenAIClientError):
            o1.validate_parameter("reasoning_effort", "invalid")

Testing Token Limits
~~~~~~~~~~~~~~~~~~

Test token limit validation:

.. code-block:: python

    def test_token_limits():
        from openai_structured.client import _validate_token_limits

        # Test valid limits
        _validate_token_limits("gpt-4o", 16000)  # Under limit

        # Test invalid limits
        with pytest.raises(TokenLimitError):
            _validate_token_limits("gpt-4o", 16385)  # Over limit

Error Handling
------------

Common Error Types
~~~~~~~~~~~~~~~~

1. **OpenAIClientError**: Base error for client-side issues
2. **TokenLimitError**: Raised when token limits are exceeded
3. **ModelNotSupportedError**: Raised for unsupported models
4. **VersionTooOldError**: Raised when model version is too old

Testing Error Cases
~~~~~~~~~~~~~~~~~

.. code-block:: python

    def test_error_handling():
        registry = ModelRegistry.get_instance()

        # Test unsupported model
        with pytest.raises(ModelNotSupportedError) as exc_info:
            registry.get_capabilities("unsupported-model")
        assert "Model 'unsupported-model' is not supported" in str(exc_info.value)
        assert "Available models:" in str(exc_info.value)
        assert "Dated models:" in str(exc_info.value)
        assert "Aliases:" in str(exc_info.value)

        # Test old version
        with pytest.raises(VersionTooOldError) as exc_info:
            registry.get_capabilities("gpt-4o-2024-07-01")
        assert "Model 'gpt-4o-2024-07-01' version 2024-07-01 is too old" in str(exc_info.value)
        assert "Minimum supported version:" in str(exc_info.value)
        assert "Use the alias 'gpt-4o' to always get the latest version" in str(exc_info.value)

        # Test parameter validation
        capabilities = registry.get_capabilities("gpt-4o")
        with pytest.raises(OpenAIClientError) as exc_info:
            capabilities.validate_parameter("reasoning_effort", "invalid")
        assert "Invalid value 'invalid' for parameter 'reasoning_effort'" in str(exc_info.value)
        assert "Description:" in str(exc_info.value)
        assert "Allowed values:" in str(exc_info.value)

Best Practices
------------

1. **Use Instance Methods**: Always use ``ModelRegistry.get_instance()`` to get the registry
2. **Test Both Success and Failure**: Verify both valid and invalid cases
3. **Check Error Messages**: Verify error messages match expectations and include helpful guidance
4. **Test Model Versions**: Test both aliases and dated versions
5. **Validate Parameters**: Test parameter validation for each model type
6. **Verify Error Details**: Check that error messages include all necessary information:
   - Available models and aliases for unsupported models
   - Format guidance for invalid dates
   - Latest alias suggestions for old versions
   - Parameter descriptions and allowed values

Common Pitfalls
-------------

1. **Missing Registry Instance**: Always use ``get_instance()``
2. **Incorrect Parameter Names**: Parameter names are case-sensitive
3. **Wrong Error Types**: Use specific error types for assertions
4. **Version Format**: Model versions must be YYYY-MM-DD format
5. **Parameter Types**: Numeric parameters must be float or int

Example Test Suite
---------------

Here's a complete example test suite:

.. code-block:: python

    import pytest
    from openai_structured import (
        ModelRegistry,
        OpenAIClientError,
        TokenLimitError,
        ModelNotSupportedError,
        VersionTooOldError,
    )

    class TestModelValidation:
        def setup_method(self):
            self.registry = ModelRegistry.get_instance()

        def test_model_capabilities(self):
            capabilities = self.registry.get_capabilities("gpt-4o")
            assert capabilities.context_window == 128000
            assert capabilities.supports_structured

        def test_parameter_validation(self):
            capabilities = self.registry.get_capabilities("gpt-4o")

            # Test valid parameters
            capabilities.validate_parameter("temperature", 0.7)
            capabilities.validate_parameter("top_p", 0.9)

            # Test invalid parameters
            with pytest.raises(OpenAIClientError):
                capabilities.validate_parameter("temperature", 2.5)

        def test_token_limits(self):
            with pytest.raises(TokenLimitError):
                capabilities = self.registry.get_capabilities("gpt-4o")
                capabilities.validate_parameter(
                    "max_completion_tokens", 16385
                )

        def test_model_versions(self):
            # Test valid version
            self.registry.get_capabilities("gpt-4o-2024-08-06")

            # Test invalid version
            with pytest.raises(VersionTooOldError):
                self.registry.get_capabilities("gpt-4o-2024-07-01")
