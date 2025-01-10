# openai-structured

[![PyPI version](https://badge.fury.io/py/openai-structured.svg)](https://badge.fury.io/py/openai-structured)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/openai-structured)](https://pypi.org/project/openai-structured/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/openai-structured/badge/?version=latest)](https://openai-structured.readthedocs.io/en/latest/)
[![Build Status](https://github.com/yaniv-golan/openai-structured/actions/workflows/python-package.yml/badge.svg)](https://github.com/yaniv-golan/openai-structured/actions/workflows/python-package.yml)

A Python library for structured output from OpenAI's API using Pydantic models.

## Key Features

* **Structured Output:** Get structured data from OpenAI using Pydantic models
* **Async Streaming:** Process responses efficiently with async streaming support
* **Type Safety:** Full type hints and Pydantic validation
* **Simple API:** Clean and intuitive interface
* **Error Handling:** Well-defined exceptions for better error management
* **Modern:** Built for OpenAI's latest API and Python 3.9+

## Requirements

* Python 3.9 or higher
* OpenAI API key

### Dependencies

These will be installed automatically:

* `openai>=1.12.0`: OpenAI Python SDK
* `pydantic>=2.6.3`: Data validation

## Installation

```bash
pip install openai-structured
```

## Quick Start

```python
from openai import OpenAI
from openai_structured import openai_structured_call
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

client = OpenAI()  # Uses OPENAI_API_KEY environment variable
result = openai_structured_call(
    client=client,
    model="gpt-4",
    output_schema=UserInfo,
    user_prompt="Tell me about John who is 30 years old",
    system_prompt="Extract user information"
)
print(f"Name: {result.name}, Age: {result.age}")
```

## Streaming Example

```python
import asyncio
from openai import OpenAI
from openai_structured import openai_structured_stream
from pydantic import BaseModel

class TodoItem(BaseModel):
    task: str
    priority: str

async def main():
    client = OpenAI()
    async for item in openai_structured_stream(
        client=client,
        model="gpt-4",
        output_schema=TodoItem,
        user_prompt=(
            "Create a list of 3 tasks for a software developer "
            "with different priorities"
        ),
        system_prompt="Generate tasks with priorities (high/medium/low)"
    ):
        print(f"Task: {item.task}, Priority: {item.priority}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Supported Models

* `gpt-3.5-turbo`
* `gpt-4`
* `gpt-4-turbo-preview`

## Error Handling

The library provides several exception classes for better error handling:

* `OpenAIClientError`: Base class for all errors
* `APIResponseError`: When the API call fails
* `ModelNotSupportedError`: When using an unsupported model
* `EmptyResponseError`: When receiving an empty response
* `InvalidResponseFormatError`: When the response can't be parsed

Example:

```python
from openai_structured import openai_structured_call, OpenAIClientError

try:
    result = openai_structured_call(...)
except OpenAIClientError as error:
    print(f"Error occurred: {error}")
```

## Python Version Support

* Python 3.9+
* Tested on CPython implementations
* Compatible with Linux, macOS, and Windows

## Documentation

For full documentation, visit [openai-structured.readthedocs.io](https://openai-structured.readthedocs.io/).

## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.
