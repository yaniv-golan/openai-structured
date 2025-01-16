.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

Command Line Interface
-------------------

The ``ostruct`` command provides a CLI for working with `OpenAI Structured Outputs <https://platform.openai.com/docs/guides/function-calling>`_ using OpenAI models.

Arguments
--------

Required:

* ``--system-prompt TEXT``: System prompt for the model.
* ``--template TEXT``: Template with {{ file }} placeholders for input substitution using Jinja2.
* ``--schema-file PATH``: JSON Schema file defining the expected response structure.

Optional:

* ``--file KEY=PATH``: File to read as input, can be specified multiple times.
* ``--output-file PATH``: Write JSON output to file instead of stdout.
* ``--model TEXT``: Model to use (default: gpt-4o-2024-08-06).
* ``--temperature FLOAT``: Temperature for sampling (default: 0.0).
* ``--max-tokens INTEGER``: Maximum tokens to generate (defaults to model-specific limit).
* ``--validate-schema``: Validate the JSON schema file and the response.
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
    Validate both the JSON schema file and response:
    - Schema validation: Checks JSON Schema Draft 7 compliance
    - Response validation: Ensures response matches schema
    - Type validation: Verifies data types and constraints
    - Raises validation error if schema or response is invalid
    - Exits with code 1 on validation failure

* ``--dry-run``
    Simulate API call and show parameters without making the actual call:
    - Shows system prompt and rendered user prompt
    - Displays estimated token count
    - Shows all model parameters (temperature, top_p, etc.)
    - Validates schema if --validate-schema is used
    - Shows output file path if specified
    - No API call is made, safe for testing
    - Useful for verifying template rendering and configuration

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

* Buffer Configuration
    - Default maximum buffer size: 1MB
    - Default cleanup threshold: 512KB
    - Default chunk size: 8KB
    - Configurable via StreamConfig

* Cleanup Strategies
    - ijson parsing for efficient JSON detection
    - Pattern matching for partial JSON
    - Maximum 3 cleanup attempts
    - Error context tracking
    - Cleanup statistics for debugging

* Error Handling
    - BufferOverflowError for size limits
    - ParseError for JSON parsing issues
    - StreamBufferError for general buffer issues
    - Automatic resource cleanup
    - Detailed error messages with context

* Schema Validation
    - Optional Pydantic model validation
    - JSON syntax validation
    - Error position tracking
    - Validation error context

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
      --template "Analyze this text: {{ input }}" \
      --schema-file schema.json \
      --file input=document.txt \
      --output-file analysis.json \
      --verbose

Multiple Files
~~~~~~~~~~~~

Compare two documents::

    ostruct \
      --system-prompt "You are a legal AI." \
      --template "Compare these documents:\n1: {{ doc1 }}\n2: {{ doc2 }}" \
      --schema-file comparison_schema.json \
      --file doc1=contract1.txt \
      --file doc2=contract2.txt \
      --validate-schema

Using stdin
~~~~~~~~~~

Process data from stdin::

    cat data.txt | ostruct \
      --system-prompt "Analyze this data" \
      --template "Process this: {{ stdin }}" \
      --schema-file schema.json \
      --model gpt-4o \
      --temperature 0.7

### Using Jinja2 Templates

The CLI supports comprehensive Jinja2 template features for advanced content processing.

Line Statements and Comments
-------------------------

Use line-based syntax for cleaner templates:

.. code-block:: jinja

    ## This is a line comment
    # for item in items
        {{ item }}
    # endfor

    ## Using line statements for control flow
    # if content is multiline
        {{ content|wrap(80)|indent(4) }}
    # endif

Whitespace Control
---------------

Fine-grained control over template whitespace:

.. code-block:: jinja

    {%- if header %}
    {{ header }}
    {% endif -%}
    
    {#- Remove whitespace around this comment -#}
    
    {{- content|normalize -}}

Block Scoping and Inheritance
--------------------------

Create reusable template hierarchies:

.. code-block:: jinja

    {# base_analysis.j2 #}
    {% block metadata %}
    Generated: {{ now() }}
    Analysis ID: {{ uuid() }}
    {% endblock %}

    {% block content %}{% endblock %}

    {% block footer %}
    Token count: {{ estimate_tokens(content) }}
    {% endblock %}

    {# code_analysis.j2 #}
    {% extends "base_analysis.j2" %}
    
    {% block content %}
        {% if content is contains_code %}
            {%- filter indent %}
            Language: Python
            {{ content|remove_comments|syntax_highlight('python') }}
            {%- endfilter %}
        {% endif %}
    {% endblock %}

Advanced Text Processing
---------------------

Comprehensive text manipulation:

.. code-block:: jinja

    {# Text wrapping and indentation #}
    {{ long_text|wrap(80)|indent(4) }}
    
    {# Clean up text #}
    {{ messy_text|dedent|normalize }}
    
    {# Format documentation #}
    {% if content is is_markdown %}
        Clean text: {{ content|strip_markdown }}
    {% endif %}

Environment and File Operations
---------------------------

Access system information and files:

.. code-block:: jinja

    API Key: {{ env('OPENAI_API_KEY', '[not set]') }}
    
    {% if file_exists('config.json') %}
        Config: {{ read_file('config.json')|from_json }}
    {% endif %}

Content Validation
---------------

Enhanced content testing:

.. code-block:: jinja

    {% if content is not is_empty %}
        {% if content is is_markdown %}
            Markdown content detected
        {% endif %}
        
        {% if content is has_urls %}
            URLs found in content
        {% endif %}
        
        {% if content is is_multiline %}
            Multi-line content detected
        {% endif %}
    {% endif %}

Example Usage
-----------

Complex template combining multiple features:

.. code-block:: jinja

    {# template.j2 #}
    {% extends "base_analysis.j2" %}
    
    {% block content %}
        ## Process each file with appropriate handling
        # for name, content in files.items()
            {%- if not content is is_empty %}
                File: {{ name }}
                {% if content is contains_code %}
                    {%- filter indent %}
                    {{ content|remove_comments|syntax_highlight('python') }}
                    {%- endfilter %}
                {% elif content is is_markdown %}
                    {%- filter indent %}
                    {{ content|strip_markdown|wrap(80) }}
                    {%- endfilter %}
                {% elif content is is_json %}
                    {%- filter indent %}
                    {{ content|from_json|to_json }}
                    {%- endfilter %}
                {% endif %}
                
                Stats:
                - Lines: {{ content|count('\n') + 1 }}
                - Words: {{ content|word_count }}
                - Chars: {{ content|char_count }}
                - Estimated tokens: {{ estimate_tokens(content) }}
            {% endif -%}
        # endfor
    {% endblock %}

Command line usage:

.. code-block:: bash

    ostruct \
      --system-prompt "Analyze multiple file types" \
      --template template.j2 \
      --file code=source.py \
      --file docs=README.md \
      --file config=settings.json \
      --schema-file analysis_schema.json

The template engine provides comprehensive features for content processing, validation, and formatting while maintaining clean and maintainable templates.

Markdown Processing
------------------

The template engine provides comprehensive markdown processing capabilities:

Raw Blocks
~~~~~~~~~

Escape Jinja2 syntax in markdown:

.. code-block:: jinja

    {% raw %}
    # Template Example
    Use {{ variable }} for substitution
    {% endraw %}

Markdown Formatting
~~~~~~~~~~~~~~~~

Generate markdown elements:

.. code-block:: jinja

    {# Headings #}
    {{ title|heading(1) }}
    {{ subtitle|heading(2) }}

    {# Text formatting #}
    {{ text|bold }}
    {{ text|italic }}
    {{ code|inline_code }}
    {{ code|code_block('python') }}

    {# Lists #}
    {{ items|unordered_list }}
    {{ items|ordered_list }}

    {# Blockquotes #}
    {{ quote|blockquote }}

Tables and Links
~~~~~~~~~~~~~

Create tables and process links:

.. code-block:: jinja

    {# Tables #}
    {{ table(headers=['Name', 'Value'], rows=data) }}

    {# Auto-link URLs #}
    {{ text|urlize }}

Footnotes
~~~~~~~~

Add and manage footnotes:

.. code-block:: jinja

    {{ text|footnote('ref1') }}
    {{ 'Additional information'|footnote_def('ref1') }}

Front Matter
~~~~~~~~~~

Process YAML front matter:

.. code-block:: jinja

    {% if content is has_frontmatter %}
        {% set meta = extract_frontmatter(content) %}
        Title: {{ meta.title }}
        Date: {{ meta.date }}
    {% endif %}

Table of Contents
~~~~~~~~~~~~~~~

Generate and manage TOC:

.. code-block:: jinja

    {% if content is has_toc %}
        ## Table of Contents
        {{ generate_toc(content, max_depth=3) }}
    {% endif %}

Code Blocks
~~~~~~~~~

Process code blocks:

.. code-block:: jinja

    {% set blocks = extract_code_blocks(content) %}
    {% for block in blocks %}
        Language: {{ block.lang }}
        {{ block.code|process_code(block.lang, 'plain') }}
    {% endfor %}

Complex Markdown Example
~~~~~~~~~~~~~~~~~~~~

Comprehensive markdown processing:

.. code-block:: jinja

    {# template.j2 #}
    {% extends "base.j2" %}
    
    {% block content %}
        {# Extract and process front matter #}
        {% if content is has_frontmatter %}
            {% set meta = extract_frontmatter(content) %}
            {{ meta.title|heading(1) }}
            Author: {{ meta.author|bold }}
            Date: {{ meta.date }}
        {% endif %}

        {# Generate TOC for long content #}
        {% if content is has_toc %}
            {{ "Table of Contents"|heading(2) }}
            {{ generate_toc(content) }}
        {% endif %}

        {# Process main content #}
        {% for section in sections %}
            {{ section.title|heading(2) }}
            
            {% if section.code is is_fenced_code %}
                {{ section.code|process_code(section.language, 'plain') }}
            {% else %}
                {{ section.text|urlize }}
            {% endif %}
            
            {% if section.notes %}
                {{ "Notes"|heading(3) }}
                {{ section.notes|blockquote }}
            {% endif %}
        {% endfor %}

        {# Add footnotes #}
        {% if footnotes %}
            {{ "Footnotes"|heading(2) }}
            {% for ref, text in footnotes.items() %}
                {{ text|footnote_def(ref) }}
            {% endfor %}
        {% endif %}
    {% endblock %}

Command line usage:

.. code-block:: bash

    ostruct \
      --system-prompt "Process markdown documentation" \
      --template template.j2 \
      --file content=document.md \
      --schema-file output_schema.json

The template engine provides comprehensive markdown processing capabilities while maintaining clean and maintainable templates.

Data Processing with Templates
=========================

The CLI supports advanced data processing through templates. Here are some examples:

Processing JSON Data
------------------

.. code-block:: bash

    # Analyze API response data
    $ cat response.json | ostruct process --template '
    {% set data = from_json(_input) %}
    
    Response Summary:
    {{ summarize(data)|dict_to_table }}
    
    Status Distribution:
    {{ data|pluck("status")|frequency|dict_to_table }}
    '

    # Generate pivot analysis
    $ cat metrics.json | ostruct process --template '
    {% set data = from_json(_input) %}
    
    Average Values by Category:
    {{ pivot_table(data, "category", "value", "mean")|dict_to_table }}
    '

Transforming Data
---------------

.. code-block:: bash

    # Sort and filter data
    $ cat users.json | ostruct process --template '
    {% set users = from_json(_input) %}
    
    Active Users by Age:
    {{ users|filter_by("status", "active")|sort_by("age")|list_to_table(["name", "age"]) }}
    '

    # Group and aggregate data
    $ cat transactions.json | ostruct process --template '
    {% set txns = from_json(_input) %}
    {% set by_type = txns|group_by("type") %}
    
    Transaction Summary by Type:
    {% for type, items in by_type.items() %}
    {{ type }}:
    {{ items|aggregate("amount")|dict_to_table }}
    {% endfor %}
    '

Generating Reports
---------------

.. code-block:: bash

    # Create detailed analysis report
    $ cat data.json | ostruct process --template '
    {% set data = from_json(_input) %}
    
    # Data Overview
    {{ summarize(data)|dict_to_table }}
    
    # Key Metrics
    {{ data|aggregate(["value", "count"])|dict_to_table }}
    
    # Distribution Analysis
    {% set dist = data|pluck("category")|frequency %}
    {{ dist|dict_to_table }}
    
    # Pivot Analysis
    {% set pivot = pivot_table(data, "category", "value", "mean") %}
    {{ pivot|dict_to_table }}
    '

Template Filters and Globals
-------------------------

The CLI provides several template filters and globals for advanced text processing and data manipulation.

Filters
~~~~~~~

* ``remove_comments(text)``
    Remove comments from code
    
* ``dedent(text)``
    Remove common leading whitespace
    
* ``normalize(text)``
    Normalize whitespace
    
* ``wrap(text, width=80)``
    Wrap text to specified width
    
* ``indent(text, width=4)``
    Indent text by specified width

Data Processing
~~~~~~~~~~~~~

* ``sort_by(items, key)``
    Sort items by key
    
* ``group_by(items, key)``
    Group items by key
    
* ``filter_by(items, key, value)``
    Filter items by key-value pair
    
* ``pluck(items, key)``
    Extract values for key
    
* ``unique(items)``
    Get unique values
    
* ``frequency(items)``
    Count value frequencies
    
* ``aggregate(items, key=None)``
    Calculate aggregate statistics

Table Formatting
~~~~~~~~~~~~~~

* ``table(headers, rows)``
    Create markdown table
    
* ``align_table(headers, rows, alignments)``
    Create aligned markdown table
    
* ``dict_to_table(data)``
    Convert dict to table
    
* ``list_to_table(items, headers=None)``
    Convert list to table
    
* ``auto_table(data)``
    Auto-format data as table

Globals
~~~~~~~

* ``estimate_tokens(text, model=None)``
    Estimate token count using tiktoken
    
* ``format_json(obj)``
    Format JSON with indentation
    
* ``now()``
    Current datetime
    
* ``validate_json(text)``
    Validate JSON string
    
* ``count_tokens(text, model=None)``
    Count tokens using tiktoken

Template Functions
----------------

The template engine provides several functions for file operations and code processing:

``read_file(path, encoding='utf-8', use_cache=True)``
    Read file contents safely with path validation and caching
    - Prevents directory traversal attacks
    - Optional content caching for performance
    - Example: ``{{ read_file('config.json') }}``

``process_code(text, lang='python', format='terminal')``
    Process code by removing comments and normalizing whitespace
    - Removes comments and normalizes whitespace
    - Example: ``{{ code|process_code('python', 'plain') }}``

Progress Indicators
----------------

The CLI provides progress feedback during template rendering:

- Visual progress bars when ``rich`` is installed
- Fallback to simple logging when ``rich`` is not available
- Configurable through environment variables:
    - ``OSTRUCT_PROGRESS=0``: Disable progress indicators
    - ``OSTRUCT_PROGRESS=1``: Enable progress indicators (default)

