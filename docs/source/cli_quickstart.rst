CLI Quickstart
============

This guide will help you get started with the OpenAI Structured CLI.

Installation
-----------

.. code-block:: bash

    pip install openai-structured

Basic Usage
----------

1. Create a schema file ``schema.json``:

   .. code-block:: json

       {
           "type": "object",
           "properties": {
               "summary": {
                   "type": "string",
                   "description": "Brief summary of the text"
               },
               "key_points": {
                   "type": "array",
                   "items": {"type": "string"}
               }
           },
           "required": ["summary", "key_points"]
       }

2. Run a simple analysis:

   .. code-block:: bash

       ostruct \
           --task "Summarize this text: {{ input.content }}" \
           --file input=document.txt \
           --schema-file schema.json

3. Use a task template file ``analyze.txt``:

   .. code-block:: jinja

       {% system_prompt %}
       You are an expert code reviewer.
       {% end_system_prompt %}

       Review this code:
       {{ code.content }}

   .. code-block:: bash

       ostruct \
           --task @analyze.txt \
           --file code=app.py \
           --schema-file review_schema.json

Next Steps
---------

- Read the full :doc:`CLI Reference <cli>` for all available options
- Check out :doc:`Examples <examples>` for more use cases
- Review :doc:`Best Practices <best_practices>` for tips 