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

Here's a basic example of extracting structured data from text:

.. code-block:: python

   from openai import OpenAI
   from openai_structured import openai_structured_call
   from pydantic import BaseModel

   class UserInfo(BaseModel):
       name: str
       age: int

   client = OpenAI()
   result = openai_structured_call(
       client=client,
       model="gpt-4o-2024-08-06",
       output_schema=UserInfo,
       user_prompt="Tell me about John who is 30 years old",
       system_prompt="Extract user information"
   )
   print(f"Name: {result.name}, Age: {result.age}")

Supported Models and Token Limits
--------------------------------

The following models are supported for structured output:

* ``gpt-4o-2024-08-06``: 128K context window, 16K output tokens
* ``gpt-4o-mini-2024-07-18``: 128K context window, 16K output tokens
* ``o1-2024-12-17``: 200K context window, 100K output tokens (including reasoning)

Choose the appropriate model based on your context size and output requirements.

For more examples, see the :doc:`examples` section. 