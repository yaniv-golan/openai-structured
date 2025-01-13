Quick Start Guide
=================

Installation
------------

Install using pip:

.. code-block:: bash

   pip install openai-structured

Or install from source using Poetry:

.. code-block:: bash

   git clone https://github.com/yaniv-golan/openai-structured.git
   cd openai-structured
   poetry install

Basic Usage
-----------

Here's a basic example of extracting structured data with proper error handling:

.. code-block:: python

   from typing import List
   from openai import OpenAI
   from pydantic import BaseModel, Field
   from openai_structured import openai_structured_call, OpenAIClientError

   class TodoItem(BaseModel):
       task: str = Field(..., description="The task description")
       priority: str = Field(..., description="Priority level (high/medium/low)")

   class TodoList(BaseModel):
       items: List[TodoItem]
       total_count: int

   client = OpenAI()
   try:
       result = openai_structured_call(
           client=client,
           model="gpt-4o-2024-08-06",
           output_schema=TodoList,
           user_prompt="Create a todo list with 2 tasks",
           system_prompt="You are a task manager that creates todo lists"
       )
       print(f"\nTotal tasks: {result.total_count}")
       for item in result.items:
           print(f"- {item.task} (Priority: {item.priority})")
   except OpenAIClientError as e:
       print(f"Error: {e}")
   finally:
       client.close()

Supported Models and Token Limits
--------------------------------

The following models support structured output:

Aliases (convenient for development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``gpt-4o``: Latest GPT-4 model (128K context window, 16K output tokens)
* ``gpt-4o-mini``: Smaller, faster GPT-4 model (128K context window, 16K output tokens)
* ``o1``: Optimized model (200K context window, 100K output tokens)

Dated Versions (recommended for production)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``gpt-4o-2024-08-06``: Specific version of GPT-4 model
* ``gpt-4o-mini-2024-07-18``: Specific version of smaller GPT-4 model
* ``o1-2024-12-17``: Specific version of optimized model

Model Version Guidelines
~~~~~~~~~~~~~~~~~~~~~

* For development and testing, you can use the simpler alias format (e.g., ``gpt-4o``)
* For production applications, use dated versions (e.g., ``gpt-4o-2024-08-06``) for better stability
* Both formats are fully supported and will work with all library features
* Aliases are resolved by OpenAI to the latest compatible version
* We validate that dated versions meet minimum version requirements

You can check if a model supports structured output before making API calls:

.. code-block:: python

   from openai_structured import supports_structured_output

   # Check aliases
   print(supports_structured_output("gpt-4o"))  # True
   print(supports_structured_output("gpt-3.5-turbo"))  # False

   # Check dated versions
   print(supports_structured_output("gpt-4o-2024-08-06"))  # True
   print(supports_structured_output("gpt-4o-2024-09-01"))  # True (newer version)

Choose the appropriate model based on your context size and output requirements.

For more examples, see the :doc:`examples` section. 