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

YAML Frontmatter
~~~~~~~~~~~~~

Templates can include YAML frontmatter at the beginning of the file to configure template behavior:

.. code-block:: jinja

    ---
    system_prompt: |
      You are a helpful assistant.
      You will help analyze code.
    schema: schema.json
    ---
    
    Analyze this code: {{ code.content }}

The frontmatter section must:
- Start with ``---`` on the first line
- End with ``---``
- Contain valid YAML

Supported frontmatter fields:
- ``system_prompt``: Set the system prompt (can be overridden by --system-prompt)
- ``schema``: Specify the schema file (can be overridden by --schema)

System Prompts
~~~~~~~~~~~~

System prompts can be specified in two ways (in order of precedence):

1. Command line argument:
   .. code-block:: bash
   
       ostruct --system-prompt "You are a helpful assistant"
       ostruct --system-prompt @system_prompt.txt

2. YAML frontmatter in template:
   .. code-block:: yaml
   
       ---
       system_prompt: You are a helpful assistant
       ---

Use --ignore-task-sysprompt to ignore system prompts from the template's YAML frontmatter.

File Content Access
~~~~~~~~~~~~~~~~

Use the `.content` attribute to access file contents within your Jinja templates. However, be mindful of file sizes and their impact on the overall prompt length.

.. code-block:: jinja

    {{ input.content }}
    {{ file.content }}

**Important Considerations for File Sizes:**

*   **Small Files:** For small files, it's generally safe to include the entire content directly in the prompt using `{{ input.content }}`.
*   **Medium to Large Files:** For larger files, strategically place the content at the **end** of your prompt, clearly delimited by XML tags or other markers. This helps the model process the instructions and schema first, then focus on the content.
*   **Very Large Files:**  If a file approaches or exceeds the model's context window, you **must** reduce its size. Consider:
    *   **Pre-processing:** Extract the most relevant sections of the file before passing it to `ostruct`.
    *   **Chunking:** Divide the file into smaller chunks and process them in multiple calls to `ostruct`.
    *   **Summarization:** Use another tool or model to summarize the file content before analysis.
*   **Token Limits:** Always be aware of the model's token limit. Use the `--verbose` flag to see the total token count in your prompt and adjust accordingly.

**Example of strategic placement for medium to large files:**

.. code-block:: bash

    ostruct --task "Distill all claims from the document in the <doc> element into the JSON response. Place the claim itself in claim element, and the source (if available) in the source element. <doc>{{ input.content }}</doc>" --file input=input.txt --schema schema.json

**Note:** The effectiveness of this approach can vary depending on the model and the specific task. Experimentation is key.

Template Functions
~~~~~~~~~~~~~~~

The CLI provides a rich set of template functions for text processing, data manipulation, and formatting:

Text Processing
^^^^^^^^^^^^^

- ``word_count(text)``: Count words in text
    .. code-block:: jinja
    
        Words: {{ input.content | word_count }}

- ``char_count(text)``: Count characters in text
    .. code-block:: jinja
    
        Characters: {{ input.content | char_count }}

- ``wrap_text(text, width=80)``: Wrap text to specified width
    .. code-block:: jinja
    
        {{ long_text | wrap_text(width=60) }}

- ``indent_text(text, width=4)``: Indent text by specified width
    .. code-block:: jinja
    
        {{ code.content | indent_text(width=2) }}

- ``dedent_text(text)``: Remove common leading whitespace
    .. code-block:: jinja
    
        {{ indented_text | dedent_text }}

- ``normalize_text(text)``: Normalize whitespace
    .. code-block:: jinja
    
        {{ messy_text | normalize_text }}

- ``strip_markdown(text)``: Remove markdown formatting
    .. code-block:: jinja
    
        {{ markdown | strip_markdown }}

Code Processing
^^^^^^^^^^^^

- ``format_code(text, lang='python', output_format='terminal')``: Format and highlight code
    .. code-block:: jinja
    
        {{ code.content | format_code(lang='javascript') }}

- ``strip_comments(text, lang='python')``: Remove code comments
    .. code-block:: jinja
    
        {{ code.content | strip_comments(lang='python') }}

Data Analysis
^^^^^^^^^^

- ``extract_keywords(text)``: Extract words as keywords
    .. code-block:: jinja
    
        Keywords: {{ text | extract_keywords }}

- ``frequency(items)``: Count item frequencies
    .. code-block:: jinja
    
        {{ words | frequency | dict_to_table }}

- ``aggregate(items, key=None)``: Calculate statistics (count, sum, mean, etc.)
    .. code-block:: jinja
    
        {{ numbers | aggregate | dict_to_table }}

- ``unique(items)``: Get unique items
    .. code-block:: jinja
    
        {{ items | unique }}

- ``sort_by(items, key)``: Sort items by key
    .. code-block:: jinja
    
        {{ users | sort_by('name') }}

- ``group_by(items, key)``: Group items by key
    .. code-block:: jinja
    
        {% for group, items in data | group_by('category') %}
        Group {{ group }}:
        {{ items | list_to_table }}
        {% endfor %}

- ``filter_by(items, key, value)``: Filter items by key-value
    .. code-block:: jinja
    
        {{ users | filter_by('active', true) }}

- ``extract_field(items, key)``: Extract values of a field
    .. code-block:: jinja
    
        {{ users | extract_field('email') }}

- ``pivot_table(data, index, value, aggfunc='sum')``: Create pivot table
    .. code-block:: jinja
    
        {{ sales | pivot_table(index='category', value='amount') | dict_to_table }}

- ``summarize(data, keys=None)``: Generate statistical summary
    .. code-block:: jinja
    
        {{ dataset | summarize | dict_to_table }}

Formatting
^^^^^^^^

- ``to_json(obj)``: Convert to JSON string
    .. code-block:: jinja
    
        {{ data | to_json }}

- ``from_json(text)``: Parse JSON string
    .. code-block:: jinja
    
        {{ json_text | from_json | dict_to_table }}

- ``format_json(obj)``: Format JSON with indentation
    .. code-block:: jinja
    
        {{ data | format_json }}

- ``dict_to_table(data)``: Convert dictionary to markdown table
    .. code-block:: jinja
    
        {{ stats | dict_to_table }}

- ``list_to_table(items, headers=None)``: Convert list to markdown table
    .. code-block:: jinja
    
        {{ users | list_to_table(headers=['Name', 'Email']) }}

- ``auto_table(data)``: Auto-format data as table
    .. code-block:: jinja
    
        {{ data | auto_table }}

- ``format_table(headers, rows)``: Create markdown table
    .. code-block:: jinja
    
        {{ format_table(['Name', 'Age'], [['Alice', 25], ['Bob', 30]]) }}

- ``align_table(headers, rows, alignments=None)``: Create aligned markdown table
    .. code-block:: jinja
    
        {{ align_table(['Name', 'Age'], users, ['left', 'right']) }}

Utility Functions
^^^^^^^^^^^^^

- ``estimate_tokens(text)``: Estimate token count
    .. code-block:: jinja
    
        Tokens: {{ text | estimate_tokens }}

- ``validate_json(text)``: Check if text is valid JSON
    .. code-block:: jinja
    
        {% if json_text | validate_json %}Valid JSON{% endif %}

- ``type_of(x)``: Get type name
    .. code-block:: jinja
    
        Type: {{ value | type_of }}

- ``len_of(x)``: Get length if available
    .. code-block:: jinja
    
        Length: {{ value | len_of }}

- ``escape_special(text)``: Escape special characters
    .. code-block:: jinja
    
        {{ text | escape_special }}

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

File Access in Templates
----------------------

The CLI provides convenient access to file contents in templates:

Single File Access
~~~~~~~~~~~~~~~~

When using ``--file``, you can access content directly:

.. code-block:: jinja

    {{ doc.content }}  # Returns the file content
    {{ doc[0].content }}  # Traditional access (still works)
    {{ doc.path }}  # Access the file path

Multiple File Access
~~~~~~~~~~~~~~~~~~

When using ``--files`` or ``--dir``, content is returned as a list:

.. code-block:: jinja

    {% for content in doc.content %}
        {{ content }}
    {% endfor %}

    # Or access individual files:
    {{ doc[0].content }}
    {{ doc.path }}  # Returns list of paths

Available Properties
~~~~~~~~~~~~~~~~~~

The following properties are available for both single and multiple files:

- ``content``: File content(s)
- ``path``: File path(s)
- ``abs_path``: Absolute file path(s)
- ``size``: File size(s) in bytes

For single files (``--file``), these properties return single values.
For multiple files (``--files``, ``--dir``), they return lists of values.

Examples
~~~~~~~~

.. code-block:: bash

    # Single file
    ostruct --task "Content: {{ doc.content }}" --file doc=input.txt

    # Multiple files
    ostruct --task "Files: {% for c in docs.content %}{{ c }}{% endfor %}" --files docs=*.txt

    # Mixed usage
    ostruct --task "Single: {{ doc.content }}, Multiple: {{ files.content }}" \
        --file doc=input.txt --files files=*.txt

