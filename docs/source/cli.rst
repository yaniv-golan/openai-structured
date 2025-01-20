.. Copyright (c) 2025 Yaniv Golan. All rights reserved.

Command Line Interface
====================

The ``ostruct`` command line interface provides tools for making structured OpenAI API calls from the command line.

Basic Usage
----------

.. code-block:: bash

    ostruct--task template.j2 --file input=data.txt

Security and File Access
----------------------

By default, the CLI only allows access to files within the current working directory and temporary directories. To access files in other directories, use the ``--allowed-dir`` argument.

**--allowed-dir**
    Specify additional directories that the CLI can access. Can be used multiple times.
    
    - Direct specification:
      
      .. code-block:: bash
      
          ostruct --task template.j2 --dir data=/path/to/data --allowed-dir /path/to/data
    
    - Multiple directories:
      
      .. code-block:: bash
      
          ostruct --task template.j2 --allowed-dir /path/to/data --allowed-dir /another/dir
    
    - Using a file list (prefix with @):
      
      .. code-block:: bash
      
          ostruct --task template.j2 --allowed-dir @allowed_dirs.txt
      
      File format (allowed_dirs.txt):
      
      .. code-block:: text
      
          /path/to/data
          /another/dir
          # Lines starting with # are comments
          /yet/another/dir

Security Considerations
--------------------

When using ``--allowed-dir``:

- Only grant access to trusted directories necessary for your task
- Avoid including sensitive system directories
- Use absolute paths to prevent ambiguity
- Consider using a file list (@) for better maintainability in production
- Validate all paths before including them in allowed directories

The CLI implements several security measures:

- Path traversal prevention
- Base directory restrictions
- Explicit directory allowlisting
- File access validation

Arguments Reference
----------------

Required Arguments:
  --task TEMPLATE     Task template string or @file

File Access:
  --file NAME=PATH   Map file to variable
  --files NAME=PATTERN  Map glob pattern to variable
  --dir NAME=PATH    Map directory to variable
  --allowed-dir PATH Additional allowed directory or @file
  --recursive        Process directories recursively

Variable Arguments
~~~~~~~~~~~~~~~

--var NAME=VALUE
    Pass simple variables to task template. Can be specified multiple times.
    Example: ``--var language=python --var style=concise``

--json-var NAME=JSON
    Pass JSON-structured variables to task template. Can be specified multiple times.
    Example: ``--json-var settings={"indent": 2}``
    Example: ``--json-var data={"items": [1, 2, 3]}``
    Example: ``--json-var config={"any": {"nested": "structure"}}``

The JSON variables can have any structure you need in your templates. Access them using standard Jinja2 dot notation or dictionary syntax:

.. code-block:: jinja

    {{ settings.indent }}
    {{ data.items[0] }}
    {{ config.any.nested }}

File Arguments
~~~~~~~~~~~~

--file NAME=PATH
    Map a single file to a name in the template.
    Example: ``--file input=data.txt``

--files NAME=PATTERN
    Map multiple files using glob patterns.
    Example: ``--files sources=src/*.py``

--dir NAME=PATH
    Map an entire directory.
    Example: ``--dir docs=./documentation``

System Prompt Options
~~~~~~~~~~~~~~~~~~

--system-prompt TEXT
    Override the system prompt. Takes precedence over task template prompt.
    Example: ``--system-prompt "You are a helpful assistant."``
    Example with file: ``--system-prompt @system.txt``

--ignore-task-sysprompt
    Ignore system prompt from task template.

File Access Options
~~~~~~~~~~~~~~~~

--recursive
    Process directories recursively when using --dir.

--ext EXTENSIONS
    Comma-separated list of file extensions to include.
    Example: ``--ext .py,.js``

Output Options
~~~~~~~~~~~~

--output-file PATH
    Write JSON output to file instead of stdout.

--validate-schema
    Validate the JSON schema and response structure.

--dry-run
    Show what would be sent to the API without making the actual call.

--no-progress
    Disable progress indicators.

Model Options
~~~~~~~~~~~

--model TEXT
    OpenAI model to use (default: gpt-4o-2024-08-06).
    Supported models:
    - gpt-4o: 128K context, 16K output
    - gpt-4o-mini: 128K context, 16K output
    - o1: 200K context, 100K output

--temperature FLOAT
    Temperature for sampling (default: 0.0).

--max-tokens INTEGER
    Maximum tokens to generate.

--top-p FLOAT
    Top-p sampling parameter (default: 1.0).

--frequency-penalty FLOAT
    Frequency penalty parameter (default: 0.0).

--presence-penalty FLOAT
    Presence penalty parameter (default: 0.0).

Other Options
~~~~~~~~~~~

--timeout FLOAT
    Timeout in seconds for API calls (default: 60.0).

--verbose
    Enable verbose logging.

--api-key KEY
    OpenAI API key. Overrides OPENAI_API_KEY environment variable.

Template Features
--------------

Task templates use Jinja2 syntax with special features:

System Prompts
~~~~~~~~~~~~

Define a system prompt within the template:

.. code-block:: jinja

    {% system_prompt %}
    You are a helpful assistant.
    {% end_system_prompt %}

File Content Access
~~~~~~~~~~~~~~~~

Always use the .content attribute to access file contents:

.. code-block:: jinja

    # Correct
    {{ input.content }}
    {{ file.content }}
    
    # Incorrect
    {{ input }}
    {{ file }}

Template Functions
~~~~~~~~~~~~~~~

Text Processing:
    - ``format_code(text, lang='python')``
    - ``strip_comments(text, lang='python')``
    - ``wrap(text, width=80)``
    - ``indent(text, width=4)``

Data Analysis:
    - ``extract_field(items, key)``
    - ``pivot_table(data, index, value, aggfunc='sum')``
    - ``summarize(data, keys=None)``
    - ``frequency(items)``

Formatting:
    - ``dict_to_table(data)``
    - ``list_to_table(items, headers=None)``
    - ``format_json(obj)``

Examples
-------

Basic Analysis
~~~~~~~~~~~~

Analyze a text file with a custom schema:

.. code-block:: bash

    # schema.json
    {
        "type": "object",
        "properties": {
            "summary": { "type": "string" },
            "key_points": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["summary", "key_points"]
    }

    ostruct \
        --task "Analyze this text: {{ input.content }}" \
        --file input=@document.txt \
        --schema-file schema.json

Code Review
~~~~~~~~~

Review code using a task template file:

.. code-block:: bash

    # review.txt
    {% system_prompt %}
    You are an expert code reviewer.
    {% end_system_prompt %}

    Review this code:
    {{ code.content }}

    ostruct \
        --task @review.txt \
        --file code=app.py \
        --schema-file review_schema.json

Multiple Files
~~~~~~~~~~~

Process multiple files in a directory:

.. code-block:: bash

    ostruct \
        --task @analyze_code.txt \
        --dir src=./src \
        --recursive \
        --ext .py \
        --schema-file analysis_schema.json

Using Variables
~~~~~~~~~~~~

Pass configuration through variables:

.. code-block:: bash

    ostruct \
        --task @process.txt \
        --file input=data.txt \
        --var format=html \
        --json-var config={"mode": "strict", "flags": ["validate"]} \
        --schema-file output_schema.json

Exit Codes
---------

* ``0`` (SUCCESS): Command completed successfully
* ``1`` (VALIDATION_ERROR): Schema/response validation failed
* ``2`` (USAGE_ERROR): Invalid arguments or configuration
* ``3`` (API_ERROR): OpenAI API issues
* ``4`` (IO_ERROR): File system issues
* ``5`` (UNKNOWN_ERROR): Unexpected errors
* ``6`` (INTERRUPTED): User interruption

Troubleshooting
-------------

Common Issues
~~~~~~~~~~~

1. **Missing Variables**
   Error: ``TaskTemplateVariableError: Template uses undefined variable 'xyz'``
   Solution: Ensure all template variables are provided.

2. **File Access**
   Error: ``FileNotFoundError: File not found: 'missing.txt'``
   Solution: Verify file paths and permissions.

3. **JSON Parsing**
   Error: ``InvalidJSONError: Invalid JSON value``
   Solution: Check JSON syntax in --json-var and schema files.

4. **Schema Validation**
   Error: ``SchemaValidationError: Response does not match schema``
   Solution: Verify schema matches expected response structure.

5. **Path Security**
   Error: ``PathSecurityError: Path is outside the base directory``
   Solution: Keep all files within the working directory.

Best Practices
~~~~~~~~~~~~

1. Use ``--dry-run`` to verify template rendering before API calls
2. Store complex templates in files
3. Use ``--verbose`` for troubleshooting
4. Validate schemas during development
5. Use meaningful variable names

