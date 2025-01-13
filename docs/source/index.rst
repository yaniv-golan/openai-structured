.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

openai-structured
===============

A Python library and command-line tool for making OpenAI API calls with structured output, featuring streaming support and efficient buffer management. The library provides both a Python API and a CLI command `ostruct` for extracting structured data using OpenAI models.

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

Documentation
-----------

.. toctree::
   :maxdepth: 2

   quickstart
   cli
   api
   examples
   contributing
   installation

Supported Models
--------------

Production Models (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``gpt-4o-2024-08-06``: GPT-4 with structured output
    * 128K context window
    * 16K output tokens
    * Full JSON schema support
    * Minimum version: 2024-08-06

* ``gpt-4o-mini-2024-07-18``: Smaller GPT-4 variant
    * 128K context window
    * 16K output tokens
    * Minimum version: 2024-07-18

* ``o1-2024-12-17``: Optimized for structured data
    * 200K context window
    * 100K output tokens
    * Minimum version: 2024-12-17

Development Aliases
~~~~~~~~~~~~~~~~~

* ``gpt-4o``: Latest GPT-4 with structured output
    * Maps to most recent compatible version
    * Minimum version: 2024-08-06

* ``gpt-4o-mini``: Latest GPT-4 mini variant
    * Maps to most recent compatible version
    * Minimum version: 2024-07-18

* ``o1``: Latest optimized model
    * Maps to most recent compatible version
    * Minimum version: 2024-12-17

Contributing
-----------

We welcome contributions! Please see :doc:`contributing` for guidelines.

License
-------

Copyright (c) 2025 Yaniv Golan. All rights reserved.
