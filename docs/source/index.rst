Welcome to openai-structured's documentation!
==============================================

A Python library for structured output from OpenAI's API.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Installation
-----------------

.. code-block:: bash

   pip install openai-structured

Quick Example
-----------------

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
       model="gpt-4",
       output_schema=UserInfo,
       user_prompt="Tell me about John who is 30 years old",
       system_prompt="Extract user information"
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
