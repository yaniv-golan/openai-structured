Installation
============

System Requirements
-----------------

- Any operating system (Linux, macOS, Windows)
- Python 3.9 or higher

Dependencies
-----------

Required Packages
~~~~~~~~~~~~~~

The following packages will be automatically installed:

- ``openai>=1.12.0``: OpenAI Python SDK
- ``pydantic>=2.6.3``: Data validation using Python type annotations
- ``jsonschema>=4.23.0``: JSON Schema validation
- ``aiohttp>=3.11.11``: Async HTTP client
- ``typing-extensions>=4.9.0``: Enhanced typing support

Optional Dependencies
~~~~~~~~~~~~~~~~~

For development and testing:

- ``pytest>=8.3.4``: For running tests
- ``black>=24.8.0``: For code formatting
- ``flake8>=6.0``: For linting
- ``mypy>=1.0``: For type checking
- ``sphinx>=7.0``: For building documentation

Installation Methods
------------------

Using pip
~~~~~~~~~

The recommended way to install openai-structured:

.. code-block:: bash

   pip install "openai-structured>=2.0.0"

For development installation with all optional dependencies:

.. code-block:: bash

   pip install "openai-structured[dev]>=2.0.0"

From Source
~~~~~~~~~~

For development or to get the latest version:

.. code-block:: bash

   git clone https://github.com/yaniv-golan/openai-structured.git
   cd openai-structured
   poetry install

Configuration
------------

The library uses the OpenAI client which looks for the ``OPENAI_API_KEY`` environment variable.
You can set this in your environment:

.. code-block:: bash

   export OPENAI_API_KEY=your-api-key-here

Or in Python:

.. code-block:: python

   from openai import OpenAI
   client = OpenAI(api_key="your-api-key-here")  # If not using environment variable

Verifying Installation
--------------------

You can verify the installation by running this simple test:

.. code-block:: python

   from openai import OpenAI
   from openai_structured import openai_structured_call
   from pydantic import BaseModel

   # Define a simple model
   class Greeting(BaseModel):
       message: str

   # Create a client (requires OPENAI_API_KEY)
   client = OpenAI()

   # Try a simple call
   try:
       result = openai_structured_call(
           client=client,
           model="gpt-4o-2024-08-06",  # Default model for structured output
           output_schema=Greeting,
           user_prompt="Say hello",
           system_prompt="Respond with a greeting in JSON format"
       )
       print(f"Installation verified successfully! Got: {result.message}")
   except Exception as e:
       print(f"Error: {e}")

Troubleshooting
-------------

Common Issues
~~~~~~~~~~~

1. **ImportError**: Make sure you've installed both ``openai`` and ``pydantic``
2. **ModuleNotFoundError**: Verify you've installed ``openai-structured``
3. **APIError**: Check your OpenAI API key is set correctly
4. **VersionError**: Ensure you have compatible versions of dependencies

Getting Help
~~~~~~~~~~

If you encounter issues:

1. Check the :doc:`examples` section for proper usage
2. Visit our `GitHub Issues <https://github.com/yaniv-golan/openai-structured/issues>`_
3. Ensure your dependencies are up to date
4. Try updating to the latest version:

   .. code-block:: bash

      pip install --upgrade openai-structured

# Example configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-2024-08-06  # Default model for testing
LOG_LEVEL=INFO
