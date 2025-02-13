.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

.. openai-structured documentation master file

Welcome to openai-structured's documentation!
=========================================

openai-structured is a Python library that provides a structured approach to interfacing with OpenAI's Structured Outputs using Pydantic models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   model_registry
   examples
   api
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Version Compatibility
------------------

* Python Support
    - Python 3.9+: Full support
    - Python 3.8: Limited support (no TypedDict)
    - Python 3.7 and below: Not supported

* API Versions
    - OpenAI API: v2024-02-15 or later
    - JSON Schema: Draft 7
    - Pydantic: v2.0+

Key Features
----------

* **OpenAI Structured Outputs**
    - Full support for OpenAI's structured output feature
    - JSON Schema validation and Pydantic integration
    - Streaming and non-streaming APIs
    - Comprehensive error handling

* **Streaming Support**
    - Real-time response processing via ``async_openai_structured_stream``
    - Memory-efficient buffer management with configurable limits
    - Automatic cleanup and resource management
    - Progress visibility through debug logging

* **Schema Validation**
    - JSON Schema Draft 7 support
    - Pydantic model integration
    - Real-time validation during streaming
    - Comprehensive error handling

* **Error Recovery**
    - Stream interruption handling
    - Buffer overflow protection
    - Parse error recovery with retries
    - Validation error handling
    - Automatic resource cleanup

* **Model Support**
    - Version validation with strict format checking
    - Token limits and context window management
    - Model aliases with minimum version requirements
    - Clear error messages for version mismatches

Quick Example
------------

.. code-block:: python

    from openai_structured import async_openai_structured_stream, StreamConfig
    from openai_structured.errors import StreamBufferError, ValidationError

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["summary", "key_points"]
    }

    async def analyze_text():
        try:
            async for chunk in async_openai_structured_stream(
                messages=[
                    {"role": "system", "content": "You are an expert analyst."},
                    {"role": "user", "content": "Analyze this text: ..."}
                ],
                schema=schema,
                stream_config=StreamConfig(
                    max_buffer_size=1024 * 1024,  # 1MB
                    cleanup_threshold=512 * 1024   # 512KB
                )
            ):
                print(chunk)
        except StreamBufferError as e:
            print(f"Buffer overflow: {e}")
        except ValidationError as e:
            print(f"Validation error: {e}")

Installation
-----------

Using pip:

.. code-block:: bash

    pip install openai-structured

Using Poetry:

.. code-block:: bash

    poetry add openai-structured

Supported Models
--------------

Production Models (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``gpt-4o-2024-08-06``: GPT-4 with OpenAI Structured Outputs
    * 128K context window
    * 16K output tokens
    * Full JSON schema support
    * Minimum version: 2024-08-06

* ``gpt-4o-mini-2024-07-18``: Smaller GPT-4 variant with OpenAI Structured Outputs
    * 128K context window
    * 16K output tokens
    * Minimum version: 2024-07-18

* ``o1-2024-12-17``: Optimized for OpenAI Structured Outputs
    * 200K context window
    * 100K output tokens
    * Limited parameter support (only max_completion_tokens and reasoning_effort)
    * Minimum version: 2024-12-17

* ``o3-mini-2025-01-31``: Mini variant optimized for OpenAI Structured Outputs
    * 200K context window
    * 100K output tokens
    * Limited parameter support (only max_completion_tokens and reasoning_effort)
    * Minimum version: 2025-01-31

Development Aliases
~~~~~~~~~~~~~~~~

* ``gpt-4o``: Latest GPT-4 model with OpenAI Structured Outputs
    * Maps to most recent compatible version
    * Minimum version: 2024-08-06

* ``gpt-4o-mini``: Latest GPT-4 mini variant with OpenAI Structured Outputs
    * Maps to most recent compatible version
    * Minimum version: 2024-07-18

* ``o1``: Latest model with OpenAI Structured Outputs
    * Maps to most recent compatible version
    * Limited parameter support (only max_completion_tokens and reasoning_effort)
    * Minimum version: 2024-12-17

* ``o3-mini``: Latest mini variant with OpenAI Structured Outputs
    * Maps to most recent compatible version
    * Limited parameter support (only max_completion_tokens and reasoning_effort)
    * Minimum version: 2025-01-31

.. note::
    Use dated versions in production for stability.
    Aliases automatically use the latest compatible version.
    o1 and o3 models only support max_completion_tokens and reasoning_effort parameters.
    Actual output tokens may be slightly less due to invisible reasoning tokens.

Contributing
-----------

We welcome contributions! Please see :doc:`contributing` for guidelines.

License
-------

Copyright (c) 2025 Yaniv Golan. All rights reserved.
