.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

Command Line Interface
=====================

The ``oai-structured-cli`` command line tool allows you to easily make structured OpenAI API calls using JSON schemas, with support for multiple input files, stdin reading, and token counting.

Installation
------------

The CLI is automatically installed when you install the package::

    pip install openai-structured

Basic Usage
----------

Here's a basic example::

    oai-structured-cli \
      --system-prompt "You are a helpful assistant" \
      --template "Summarize the doc: {file1}" \
      --schema-file schema.json \
      --file file1=doc1.txt

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~

* ``--system-prompt TEXT``
    The system role prompt text (e.g., "You are an expert in legal analysis")

* ``--template TEXT``
    The main user prompt template with placeholders like ``{file1}``, ``{stdin}``

* ``--schema-file PATH``
    Path to a JSON schema file that defines the structure of the OpenAI response

Optional Arguments
~~~~~~~~~~~~~~~~~

* ``--file NAME=PATH``
    Repeatable. Associates file contents with a name for template substitution.
    Example: ``--file doc1=contract.txt --file doc2=terms.txt``

* ``--model MODEL``
    OpenAI model to use (default: "gpt-4o-2024-08-06")

* ``--max-token-limit INT``
    Maximum allowed tokens. Set to 0 or negative to disable check.
    Default is model-dependent (8192 for GPT-4 models)

* ``--output-file PATH``
    Write JSON output to this file instead of stdout

* ``--log-level LEVEL``
    Logging level: DEBUG, INFO, WARNING, or ERROR (default: INFO)

* ``--api-key KEY``
    OpenAI API key. Overrides OPENAI_API_KEY environment variable if provided

* ``--validate-schema``
    Enable validation of the JSON schema and response

Model Support
------------

The following models support structured output:

* ``gpt-4o-2024-08-06`` (default): Latest GPT-4 model with structured output support
* ``gpt-4o-mini-2024-07-18``: Smaller, faster GPT-4 model
* ``o1-2024-12-17``: Optimized model for structured output

.. warning::
    Other models like ``gpt-4`` or ``gpt-3.5-turbo`` do not support structured output and will result in an error.

Template Validation
------------------

The CLI validates templates before making API calls. A template must:

1. Reference only defined file names::

    # Valid - both files are provided
    --template "Compare {file1} with {file2}" \
    --file file1=a.txt --file file2=b.txt

    # Invalid - file2 is missing
    --template "Compare {file1} with {file2}" \
    --file file1=a.txt

2. Use ``{stdin}`` only when input is provided on stdin::

    # Valid - stdin is provided
    echo "test" | oai-structured-cli --template "Analyze {stdin}"

    # Invalid - no stdin provided
    oai-structured-cli --template "Analyze {stdin}"

Examples
--------

Multiple Files with Schema
~~~~~~~~~~~~~~~~~~~~~~~~

Compare two documents with a custom response schema::

    # schema.json
    {
      "type": "object",
      "properties": {
        "differences": {
          "type": "array",
          "items": { "type": "string" }
        },
        "similarity_score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        }
      },
      "required": ["differences", "similarity_score"]
    }

    oai-structured-cli \
      --system-prompt "You are a legal AI." \
      --template "Compare the docs for conflicts. docA={docA}, docB={docB}" \
      --schema-file schema.json \
      --file docA=contract1.txt \
      --file docB=contract2.txt

Using stdin
~~~~~~~~~~

Process data from stdin with a summary schema::

    # summary_schema.json
    {
      "type": "object",
      "properties": {
        "title": { "type": "string" },
        "key_points": {
          "type": "array",
          "items": { "type": "string" }
        }
      },
      "required": ["title", "key_points"]
    }

    cat report.txt | oai-structured-cli \
      --system-prompt "Analyze the following report" \
      --template "Please summarize this: {stdin}" \
      --schema-file summary_schema.json

Environment Variables
-------------------

The CLI supports the following environment variables:

1. ``OPENAI_API_KEY`` (required if --api-key not provided)
   OpenAI API key for authentication

2. ``LOG_LEVEL`` (optional)
   Default logging level (DEBUG, INFO, WARNING, ERROR)
   Can be overridden by --log-level

Environment variables take precedence over command-line arguments for security reasons.

API Key Configuration
--------------------

The CLI supports two ways to provide your OpenAI API key:

1. Environment variable (recommended)::

       export OPENAI_API_KEY="your-key-here"
       oai-structured-cli ...

2. Command line argument (less secure)::

       oai-structured-cli --api-key "your-key-here" ...

.. warning::
    Using ``--api-key`` on the command line is less secure as the key might appear in shell history or process listings.
    Prefer using the ``OPENAI_API_KEY`` environment variable.

Alternative Security Approaches
-----------------------------

Besides environment variables and command-line arguments, there are several more secure ways to handle API keys:

1. **Configuration Files**::

       # ~/.config/oai-structured-cli/config
       OPENAI_API_KEY=your-key-here

2. **Secret Managers**:
   * HashiCorp Vault
   * AWS Secrets Manager
   * Azure Key Vault
   * Google Cloud Secret Manager

3. **Docker Secrets** (if running in containers)

Token Limits
-----------

The CLI automatically counts tokens in your prompts using ``tiktoken`` and enforces limits:

* Default limit for GPT-4 models: 8192 tokens
* Default limit for other models: 4096 tokens

You can override these limits with ``--max-token-limit`` or disable checking by setting it to 0::

    # Custom limit
    oai-structured-cli --max-token-limit 2000 ...

    # Disable limit checking
    oai-structured-cli --max-token-limit 0 ...

Response Format
--------------

The CLI outputs JSON responses in a consistent format:

1. **Success Response**::

    {
      "field1": "value1",
      "field2": 123,
      "field3": ["item1", "item2"]
    }

2. **String Response**::
    If the model returns a string instead of a JSON object, the CLI will attempt to parse it as JSON.
    If parsing fails, an error is returned.

3. **Output File**::
    When using --output-file, the JSON is written with 2-space indentation::

        {
          "field1": "value1",
          "field2": 123
        }

Schema Validation
----------------

The CLI supports validation of both the schema file and the OpenAI response using the ``--validate-schema`` flag.

Schema File Validation
~~~~~~~~~~~~~~~~~~~~~

When ``--validate-schema`` is enabled, the CLI validates that your schema file is a valid JSON Schema::

    # valid_schema.json
    {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description": "A brief summary of the document"
        },
        "key_points": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "minItems": 1,
          "description": "List of key points from the document"
        }
      },
      "required": ["summary", "key_points"]
    }

Common validation errors::

    Invalid JSON Schema: 'type' is a required property
    Invalid JSON Schema: 'items' is required for array types
    Invalid JSON Schema: unknown property format 'datetime'

Response Validation
~~~~~~~~~~~~~~~~~~

The CLI also validates the OpenAI response against your schema::

    # Example validation errors
    Response validation errors:
    At summary: 'summary' is a required property
    At key_points/0: 'test' is not of type 'number'
    At confidence: 0.5 is less than the minimum of 0.8

.. note::
    Schema validation requires the ``jsonschema`` package. If not installed, validation
    is skipped with a warning message.

Exit Codes
----------

The CLI uses the following exit codes:

* ``0``: Success - Command completed successfully
* ``1``: Error - Command failed. Specific cases include:
    - File reading errors
    - Invalid template
    - Token limit exceeded
    - API authentication failure
    - Model not supported
    - Network connectivity issues
    - OpenAI server errors
    - Schema validation errors
    - JSON parsing errors

Error Handling
-------------

The CLI handles various error conditions with specific error messages:

1. **File Errors**::

    Cannot read schema.json: No such file or directory
    Cannot read input.txt: Permission denied

2. **Template Errors**::

    Template placeholders missing files: file2
    Template references {stdin} but no input provided on stdin

3. **Token Limits**::

    Prompt requires 9000 tokens, exceeds limit of 8192

4. **API Errors**::

    API error: Could not connect to OpenAI API
    Rate limit exceeded: Please retry after 20s
    Model not supported: gpt-3.5-turbo does not support structured output

5. **Schema Errors**::

    Invalid JSON Schema: 'type' is a required property
    Response validation errors:
    At summary: 'summary' is a required property

All errors are logged with appropriate messages and result in non-zero exit codes.

Repository Integration
--------------------

The ``oai-structured-cli`` tool is fully integrated with the repository:

* **Location**: ``src/openai_structured/cli.py``
* **Entry Point**: Defined in ``pyproject.toml`` as ``oai-structured-cli = "openai_structured.cli:main"``
* **Testing**: Comprehensive test suite in ``tests/test_cli.py``
* **CI/CD**: Integrated with GitHub Actions workflows
* **Documentation**: This documentation is built and deployed automatically

For development:

1. Clone the repository::

    git clone https://github.com/your-org/openai-structured.git
    cd openai-structured

2. Install in development mode::

    poetry install

3. Run tests::

    poetry run pytest tests/test_cli.py 