Model Registry
=============

The model registry is a central component that manages OpenAI model capabilities, version requirements, and parameter validation.

Configuration
------------

The model registry uses two main configuration files:

1. ``models.yml``: Defines model capabilities and version requirements
2. ``parameter_constraints.yml``: Defines parameter validation rules

Model Capabilities
----------------

Each model in the registry has the following capabilities:

- ``context_window``: Maximum context window size in tokens
- ``max_output_tokens``: Maximum number of output tokens
- ``supports_structured``: Whether the model supports structured output
- ``supports_streaming``: Whether the model supports streaming responses
- ``supported_parameters``: List of supported parameters with constraints
- ``min_version``: Minimum supported version for the model

Version Validation
---------------

The registry supports both dated models and aliases:

- Dated models (e.g., ``gpt-4o-2024-08-06``): Specific versions with fixed capabilities
- Aliases (e.g., ``gpt-4o``): Point to the latest stable version
- Version validation ensures compatibility:
  - Validates date format (YYYY-MM-DD)
  - Checks against minimum supported version
  - Handles version comparison and fallbacks

Parameter Validation
-----------------

The registry provides comprehensive parameter validation:

Supported Parameters
~~~~~~~~~~~~~~~~

Different models support different parameters:

- GPT-4 models (gpt-4o and gpt-4o-mini):
    - temperature
    - top_p
    - frequency_penalty
    - presence_penalty
    - max_completion_tokens

- o1 and o3 models:
    - max_completion_tokens
    - reasoning_effort

.. note::
    o1 and o3 models do not support temperature, top_p, frequency_penalty, or presence_penalty parameters.
    Attempting to use these parameters with o1 or o3 models will raise an OpenAIClientError.

Numeric Parameters
~~~~~~~~~~~~~~~~

.. code-block:: python

    {
        "temperature": {
            "type": "numeric",
            "min_value": 0.0,
            "max_value": 2.0,
            "allow_float": true,
            "allow_int": false
        }
    }

Enum Parameters
~~~~~~~~~~~~~

.. code-block:: python

    {
        "reasoning_effort": {
            "type": "enum",
            "allowed_values": ["low", "medium", "high"]
        }
    }


Error Handling
------------

The registry provides specific error types for different validation scenarios:

- ``ModelNotSupportedError``: Model not found in registry (includes available models and aliases)
- ``InvalidDateError``: Invalid date format in model version (includes format guidance)
- ``VersionTooOldError``: Model version older than minimum supported (includes latest alias suggestion)
- ``TokenParameterError``: Invalid token-related parameter (includes parameter guidance)
- ``OpenAIClientError``: Base class for all registry errors

Example Usage
-----------

Basic Capability Check
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from openai_structured import ModelRegistry

    registry = ModelRegistry.get_instance()

    # Check model capabilities
    caps = registry.get_capabilities("gpt-4o-2024-08-06")
    print(f"Context window: {caps.context_window}")
    print(f"Supports streaming: {caps.supports_streaming}")

Parameter Validation
~~~~~~~~~~~~~~~~~

.. code-block:: python

    try:
        # Validate parameters
        caps.validate_parameter("temperature", 0.7)
        caps.validate_parameter("reasoning_effort", "medium")
    except OpenAIClientError as e:
        # Error message examples:
        # "Invalid value 'high' for parameter 'reasoning_effort'. Description: Controls the model's reasoning depth. Allowed values: low, medium, high"
        # "Parameter 'temperature' must be between 0.0 and 2.0. Description: Controls randomness in the output"
        print(f"Parameter validation failed: {e}")

Version Validation
~~~~~~~~~~~~~~~

.. code-block:: python

    try:
        # Check version compatibility
        caps = registry.get_capabilities("gpt-4o-2024-07-01")
    except VersionTooOldError as e:
        # Error message example:
        # "Model 'gpt-4o-2024-07-01' version 2024-07-01 is too old.
        # Minimum supported version: 2024-08-06
        # Note: Use the alias 'gpt-4o' to always get the latest version"
        print(f"Version too old: {e}")
    except InvalidDateError as e:
        # Error message example:
        # "Invalid date format in model version: Month must be between 1 and 12
        # Use format: YYYY-MM-DD (e.g. 2024-08-06)"
        print(f"Invalid date format: {e}")

Configuration
-----------

Custom Registry Path
~~~~~~~~~~~~~~~~~

You can specify custom paths for the registry configuration:

.. code-block:: bash

    export MODEL_REGISTRY_PATH=/path/to/models.yml
    export PARAMETER_CONSTRAINTS_PATH=/path/to/constraints.yml

Fallback Behavior
~~~~~~~~~~~~~~

The registry includes built-in fallback configurations when the main configuration files are unavailable:

1. Attempts to load from specified paths
2. Falls back to built-in configuration if files are missing
3. Maintains core functionality even without external configuration

Updating Registry
~~~~~~~~~~~~~~

The registry can be updated from the official repository using the command line tool:

.. code-block:: bash

    # Basic update with confirmation prompt
    openai-structured-refresh

    # Update with verbose output showing available models
    openai-structured-refresh -v

    # Update from custom URL without confirmation
    openai-structured-refresh -f --url https://example.com/models.yml

    # Validate current configuration without updating
    openai-structured-refresh --validate

Command Options
^^^^^^^^^^^^^

-v, --verbose      Show detailed information about available models
-f, --force        Skip confirmation prompt
--url TEXT         Custom config URL for fetching model configurations
--validate         Validate current configuration without updating

The refresh command will:

1. Download the latest model configurations from the official repository (or custom URL)
2. Validate the configuration format and values
3. Update your local ``models.yml`` file
4. Reload the registry with the new configurations

When using ``--verbose``, you'll see detailed information about each model:

.. code-block:: text

    Available models:
    - gpt-4o-2024-08-06
      Context window: 128000
      Max output tokens: 16384
      Supports streaming: True

    - o1-2024-12-17
      Context window: 200000
      Max output tokens: 100000
      Supports streaming: False

You can also update the registry programmatically:

.. code-block:: python

    from openai_structured import ModelRegistry

    registry = ModelRegistry.get_instance()
    if registry.refresh_from_remote():
        print("Registry updated successfully")

Command Line Utilities
-------------------

The library provides command line utilities for managing the model registry:

Update Registry
~~~~~~~~~~~~~~~~~~~

The ``openai-structured-refresh`` command (implemented in ``scripts/update_registry.py``) provides a user-friendly way to update and validate the model registry:

.. code-block:: bash

    # Basic update with confirmation prompt
    openai-structured-refresh

    # Update with verbose output showing available models
    openai-structured-refresh -v

    # Update from custom URL without confirmation
    openai-structured-refresh -f --url https://example.com/models.yml

    # Validate current configuration without updating
    openai-structured-refresh --validate

Command Options
^^^^^^^^^^^^^

-v, --verbose      Show detailed information about available models
-f, --force        Skip confirmation prompt
--url TEXT         Custom config URL for fetching model configurations
--validate         Validate current configuration without updating

Update Fallback Models
~~~~~~~~~~~~~~~~~~~

The ``scripts/update_fallbacks.py`` script updates the fallback models in ``model_registry.py`` to match the configuration in ``models.yml``:

.. code-block:: bash

    python scripts/update_fallbacks.py

This script:

1. Reads the current ``models.yml`` configuration
2. Generates Python code for fallback models
3. Updates the fallback section in ``model_registry.py``
4. Maintains proper indentation and formatting

The script is used in two ways:

1. Automatically via GitHub Actions:
   - Triggered when ``models.yml`` changes in main/next branches
   - Creates a PR with the updates
   - Labels the PR as "automated pr" and "dependencies"

2. Manually by developers:
   - Run locally to test changes
   - Verify fallback models match configuration
   - Debug configuration issues

Error Handling:

- Validates file existence
- Reports clear error messages
- Exits with status code 1 on failure

Example workflow:

1. Update ``models.yml`` with new model:

   .. code-block:: yaml

       dated_models:
         new-model-2024-03-01:
           context_window: 128000
           max_output_tokens: 16384
           supports_structured: true
           supports_streaming: true
           supported_parameters:
             - ref: numeric_constraints.temperature
             - ref: numeric_constraints.top_p

2. Run update script:

   .. code-block:: bash

       python scripts/update_fallbacks.py

3. Verify changes in ``model_registry.py``

Generate Default Models
~~~~~~~~~~~~~~~~~~~~

The model registry automatically generates default model configurations when external configuration files are unavailable:

1. Built-in fallbacks provide core model support
2. Ensures library works without external files
3. Matches the structure in ``models.yml``

To update the default models:

1. Modify ``models.yml`` with new configuration
2. Run the update script
3. Commit changes to both files
