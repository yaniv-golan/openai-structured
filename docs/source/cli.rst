.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

Command Line Interface
-------------------

The ``ostruct`` command provides a CLI for working with `OpenAI Structured Outputs <https://platform.openai.com/docs/guides/function-calling>`_ using OpenAI models.

Arguments
--------

Required:

* ``--system-prompt TEXT``: System prompt for the model.
* ``--template TEXT``: Template with {file} placeholders for input substitution.
* ``--schema-file PATH``: JSON Schema file defining the expected response structure.

Optional:

* ``--file KEY=PATH``: File to read as input, can be specified multiple times.
* ``--output-file PATH``: Write JSON output to file instead of stdout.
* ``--model TEXT``: Model to use (default: gpt-4o-2024-08-06).
* ``--temperature FLOAT``: Temperature for sampling (default: 0.0).
* ``--max-tokens INTEGER``: Maximum tokens to generate (defaults to model-specific limit).
* ``--validate-schema``: Validate response against schema.
* ``--verbose``: Enable verbose logging.

Optional Arguments
~~~~~~~~~~~~~~~~~

* ``--file NAME=PATH``
    Repeatable. Associates file contents with a name for template substitution.
    Example: ``--file doc1=contract.txt --file doc2=terms.txt``

* ``--output-file PATH``
    Write JSON output to file instead of stdout.
    - Creates parent directories if needed
    - Writes output as it's received
    - Supports both relative and absolute paths
    - Overwrites existing file if present

* ``--api-key TEXT``
    OpenAI API key. Overrides OPENAI_API_KEY environment variable.
    - Securely handled and not logged
    - Not visible in process list
    - Can be provided via environment variable for better security

* ``--model TEXT``
    OpenAI model with Structured Outputs support (default: "gpt-4o-2024-08-06")

* ``--temperature FLOAT``
    Temperature for sampling (default: 0.0)
    - Higher: More creative, varied outputs
    - Lower: More focused, deterministic outputs
    - Range: 0.0 to 2.0

* ``--max-tokens INT``
    Maximum tokens to generate. Set to 0 or negative to disable token limit checks.
    Defaults to model-specific limit.
    - Affects response length and cost
    - Higher limits increase memory usage

* ``--top-p FLOAT``
    Top-p sampling parameter (default: 1.0)
    - Controls output diversity
    - Lower values: More focused
    - Range: 0.0 to 1.0

* ``--frequency-penalty FLOAT``
    Frequency penalty parameter (default: 0.0)
    - Controls repetition across all generated tokens
    - Higher: Less repetition
    - Range: -2.0 to 2.0

* ``--presence-penalty FLOAT``
    Presence penalty parameter (default: 0.0)
    - Controls repetition based on token presence
    - Higher: More topic changes
    - Range: -2.0 to 2.0

* ``--timeout FLOAT``
    Timeout in seconds for API calls (default: 60.0)
    - Applies to both streaming and validation
    - Adjust for large responses

* ``--verbose``
    Enable detailed logging including:
    - Token usage statistics
    - API request/response details
    - Error context and stack traces

* ``--validate-schema``
    Validate both schema and response:
    - Schema validation: Checks JSON Schema Draft 7 compliance
    - Response validation: Ensures response matches schema
    - Type validation: Verifies data types and constraints

Streaming Behavior
----------------

All responses are streamed by default:

* Response Processing
    - Chunk-based processing with 8KB default chunk size
    - JSON validation of complete objects
    - Automatic buffer management
    - Debug logging of significant size changes
    - Resource cleanup on completion

* Error Handling
    - StreamBufferError for buffer overflow
    - StreamParseError after 5 failed parse attempts
    - StreamInterruptedError for network issues and connection problems
    - ValidationError for schema violations
    - Automatic resource cleanup on errors
    - Detailed error messages with context

* Resource Management
    - Automatic buffer cleanup
    - Connection closing in finally blocks
    - Buffer reset after successful parse
    - Proper error propagation
    - Debug logging support

Buffer Management
---------------

The CLI uses efficient buffer management for streaming responses:

* Buffer Size Control
    - Default maximum buffer size: 1MB
    - Default cleanup threshold: 512KB
    - Default chunk size: 8KB
    - Automatic cleanup when buffer exceeds threshold
    - StreamBufferError protection with clear error messages

* Cleanup Strategy
    - Uses ijson for efficient JSON parsing and finding complete objects
    - Fallback to regex pattern matching for partial JSON objects
    - Maximum 3 cleanup attempts before StreamBufferError
    - Tracks cleanup statistics for debugging
    - Preserves partial valid responses when possible

* Error Handling
    - StreamBufferError when size exceeds limit
    - StreamParseError after 5 failed parse attempts
    - StreamInterruptedError for network and connection issues
    - Automatic resource cleanup on errors

* Memory Efficiency
    - Chunk-based processing using write() method
    - Content cache invalidation on write
    - Automatic buffer reset after successful parse
    - Total bytes tracking for size management
    - Cleanup triggered at configurable threshold

Model Support
------------

The following models support OpenAI Structured Outputs:

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
    * Minimum version: 2024-12-17

Development Aliases
~~~~~~~~~~~~~~~~~

* ``gpt-4o``: Latest GPT-4 with OpenAI Structured Outputs
* ``gpt-4o-mini``: Latest mini variant with OpenAI Structured Outputs
* ``o1``: Latest model optimized for OpenAI Structured Outputs

Version Validation
~~~~~~~~~~~~~~~~

The CLI validates model versions to ensure compatibility with OpenAI Structured Outputs:

* Version Format: ``{base_model}-{YYYY}-{MM}-{DD}``
    * Example: ``gpt-4o-2024-08-06``
    * Validation regex: ``^[\w-]+?-\d{4}-\d{2}-\d{2}$``
    * Supports hyphens and underscores in base model name

* Alias Resolution
    * Aliases automatically use latest compatible version
    * Enforces minimum version requirements
    * Clear error messages for version mismatches

The ``--validate-schema`` option provides validation using JSON Schema Draft 7:

Schema File Validation
~~~~~~~~~~~~~~~~~~~~

* JSON Schema Draft 7 compliance check using ``jsonschema`` package
* Required properties validation
* Type definitions (string, integer, number, boolean, array, object)
* Basic constraints (minimum, maximum, pattern)
* Array validation (minItems, maxItems)
* Object property validation
* Validation errors include path and message

Response Validation
~~~~~~~~~~~~~~~~~

* JSON parsing validation
* Schema compliance verification
* Type checking against schema
* Required field validation
* Array and object validation
* Detailed error messages with context
* Real-time validation of each complete object in stream
* Immediate error reporting for validation failures

Error Types
~~~~~~~~~~

* APIResponseError (API response errors with response ID and content)
* ModelVersionError (invalid or unsupported model versions)
* Schema validation errors (invalid schema format)
* JSON parse errors (with position and context)
* Type mismatches (wrong data type)
* Missing required fields
* Invalid field values
* Token limit errors (input too long, output limit exceeded)
* Stream parse errors (after 5 attempts)
* StreamBufferError (buffer size exceeded)
* StreamInterruptedError (network issues)

Exit Codes
---------

The CLI uses these exit codes:

* ``0`` (SUCCESS)
    Command completed successfully

* ``1`` (VALIDATION_ERROR)
    - Schema validation failed
    - Response validation failed
    - Token limit exceeded (input too long or output limit exceeded)
    - Invalid template
    - Type mismatch
    - Format error

* ``2`` (USAGE_ERROR)
    - Missing required arguments
    - Invalid argument values
    - File not found
    - Permission denied
    - Invalid configuration
    - Schema error

* ``3`` (API_ERROR)
    - Authentication failed
    - Rate limit exceeded
    - Model not supported
    - Network error
    - Timeout
    - Version error

* ``4`` (IO_ERROR)
    - File read/write error
    - Directory creation failed
    - Permission issues
    - Disk space issues
    - Network I/O
    - Buffer overflow

* ``5`` (UNKNOWN_ERROR)
    - Unexpected exceptions
    - Internal errors
    - System errors
    - Resource errors
    - State errors

* ``6`` (INTERRUPTED)
    - User interrupted (Ctrl+C)
    - Signal received
    - Forced termination
    - Cleanup triggered
    - Resource release

Examples
--------

Basic Analysis
~~~~~~~~~~~~~

Analyze a text file with a custom schema::

    # schema.json
    {
      "type": "object",
      "properties": {
        "summary": { "type": "string" },
        "key_points": {
          "type": "array",
          "items": { "type": "string" }
        },
        "sentiment": {
          "type": "string",
          "enum": ["positive", "neutral", "negative"]
        }
      },
      "required": ["summary", "key_points", "sentiment"]
    }

    ostruct \
      --system-prompt "You are an expert analyst." \
      --template "Analyze this text: {input}" \
      --schema-file schema.json \
      --file input=document.txt \
      --output-file analysis.json \
      --verbose

Multiple Files
~~~~~~~~~~~~

Compare two documents::

    ostruct \
      --system-prompt "You are a legal AI." \
      --template "Compare these documents:\n1: {doc1}\n2: {doc2}" \
      --schema-file comparison_schema.json \
      --file doc1=contract1.txt \
      --file doc2=contract2.txt \
      --validate-schema

Using stdin
~~~~~~~~~~

Process data from stdin::

    cat data.txt | ostruct \
      --system-prompt "Analyze this data" \
      --template "Process this: {stdin}" \
      --schema-file schema.json \
      --model gpt-4o \
      --temperature 0.7

