# openai-structured

[![PyPI version](https://badge.fury.io/py/openai-structured.svg)](https://badge.fury.io/py/openai-structured)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/openai-structured)](https://pypi.org/project/openai-structured/)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-informational)](https://flake8.pycqa.org/en/latest/)
[![Documentation Status](https://readthedocs.org/projects/your-project-slug/badge/?version=latest)](https://your-project-slug.readthedocs.io/en/latest/)
[![Build Status](https://github.com/your-username/openai-structured/actions/workflows/python-package.yml/badge.svg)](https://github.com/your-username/openai-structured/actions/workflows/python-package.yml)

A Python library for seamless interaction with the OpenAI API, focusing on structured output using Pydantic models.

## Key Features

* **Simplified Structured Calls:** Effortlessly fetch structured data from OpenAI using Pydantic models.
* **Asynchronous Streaming Support:** Process large responses efficiently with asynchronous streaming.
* **Clear Error Handling:** Well-defined exceptions for API and client-related issues.
* **Type Hinting:** Enhanced code readability and maintainability with comprehensive type hints.
* **Lightweight and Focused:** Designed specifically for structured output, minimizing unnecessary overhead.
* **Easy to Integrate:** Clean and intuitive API for seamless integration into your projects.

## Installation

```bash
pip install openai-structured
```

## Quick Start

```python
import os
from openai import OpenAI
from openai_structured import openai_structured_call
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

user_prompt = "Tell me about a user named Alice who is 25 years old."
system_prompt = "Extract user information."

try:
    user_info = openai_structured_call(
        client=client,
        model="gpt-4o-2024-08-06",
        output_schema=UserInfo,
        user_prompt=user_prompt,
        system_prompt=system_prompt
    )
    print(user_info)
except Exception as e:
    print(f"Error: {e}")
```

## Asynchronous Streaming Example

```python
import os
import asyncio
from openai import OpenAI
from openai_structured import openai_structured_stream
from pydantic import BaseModel

class TodoItem(BaseModel):
    task: str
    priority: str

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

user_prompt = "Create a list of two todo items."
system_prompt = "Generate a list of tasks with their priorities in JSON format."

async def main():
    try:
        async for item in openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=TodoItem,
            user_prompt=user_prompt,
            system_prompt=system_prompt
        ):
            print(f"Received item: {item}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation

The full documentation is available at [Read the Docs link (replace with your actual link)].

## Contributing

We welcome contributions! Please see the [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on how to get involved.

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
