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

The following models are supported for structured output:

* ``gpt-4o``: 128K context window, 16K output tokens
* ``gpt-4o-mini``: 128K context window, 16K output tokens
* ``O1``: 200K context window, 100K output tokens (including reasoning)

When using these models, specify them with a version date in the format ``<model>-YYYY-MM-DD``.
The minimum supported versions are:

* ``gpt-4o-2024-08-06``
* ``gpt-4o-mini-2024-07-18``
* ``o1-2024-12-17``

Choose the appropriate model based on your context size and output requirements.

For more examples, see the :doc:`examples` section. 