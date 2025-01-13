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

### Using pip

```bash
pip install openai-structured
```

### Using Poetry (recommended for development)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/yaniv-golan/openai-structured.git
cd openai-structured

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Configuration

### Environment Variables

Create a `.env` file in your project root (see `.env.example`):

```bash
OPENAI_API_KEY=your-api-key-here

# Optional settings
OPENAI_API_BASE=https://api.openai.com/v1  # Custom API endpoint
OPENAI_API_VERSION=2024-02-01  # Specific API version
```

### Project Settings

The library uses these configurations (in `pyproject.toml`):

```toml
[tool.openai_structured]
max_buffer_size = 1048576  # 1MB default
buffer_cleanup_threshold = 524288  # 512KB default
chunk_size = 8192  # 8KB default
```

## Quick Start

### Basic Usage with Type Hints

```python
from typing import List
from openai import OpenAI
from openai_structured import openai_structured_call
from pydantic import BaseModel, Field

class TodoItem(BaseModel):
    """A single todo item with priority."""
    task: str = Field(..., description="The task description")
    priority: str = Field(..., pattern="^(high|medium|low)$")

class TodoList(BaseModel):
    """A list of todo items."""
    items: List[TodoItem]
    total_count: int = Field(..., ge=0)

# Initialize client
client = OpenAI()  # Uses OPENAI_API_KEY environment variable

# Get structured response
result = openai_structured_call(
    client=client,
    model="gpt-4o-2024-08-06",
    output_schema=TodoList,
    user_prompt="Create a todo list with 3 items",
    system_prompt="Generate a todo list with priorities"
)

# Type-safe access to response
for item in result.items:
    print(f"Task: {item.task}, Priority: {item.priority}")
print(f"Total items: {result.total_count}")
```

### Async Streaming Example

```python
import asyncio
from openai import AsyncOpenAI
from openai_structured import openai_structured_stream
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    """Progressive analysis result."""
    topic: str = Field(..., description="Current topic being analyzed")
    insight: str = Field(..., min_length=10)
    confidence: float = Field(..., ge=0.0, le=1.0)

async def main():
    client = AsyncOpenAI()  # Uses OPENAI_API_KEY environment variable
    async for result in openai_structured_stream(
        client=client,
        model="gpt-4o-2024-08-06",
        output_schema=AnalysisResult,
        system_prompt="Analyze the text progressively",
        user_prompt="Analyze the impact of AI on society",
    ):
        print(f"Topic: {result.topic}")
        print(f"Insight: {result.insight}")
        print(f"Confidence: {result.confidence:.2f}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

## Supported Models

Models can be specified using either aliases or dated versions:

### Aliases (convenient for development)

* `gpt-4o`: Latest GPT-4 model with structured output support (128K context window, 16K output tokens)
* `gpt-4o-mini`: Smaller, faster GPT-4 model (128K context window, 16K output tokens)
* `o1`: Optimized model for structured output (200K context window, 100K output tokens)

### Dated Versions (recommended for production)

* `gpt-4o-2024-08-06`: Specific version of GPT-4 model (128K context window, 16K output tokens)
* `gpt-4o-mini-2024-07-18`: Specific version of smaller GPT-4 model (128K context window, 16K output tokens)
* `o1-2024-12-17`: Specific version of optimized model (200K context window, 100K output tokens)

Note: Following OpenAI's best practices, it is recommended to use dated model versions in production applications for better version stability. When using aliases, OpenAI will automatically resolve them to the latest compatible version. Our library validates that dated versions meet minimum version requirements.

## Error Handling

The library provides several exception classes for better error handling:

```python
from openai_structured import (
    OpenAIClientError,  # Base class for all errors
    APIResponseError,   # API call failures
    ModelNotSupportedError,  # Unsupported model
    EmptyResponseError,  # Empty response
    InvalidResponseFormatError,  # Response parsing failures
    BufferOverflowError,  # Stream buffer exceeded
)

try:
    result = openai_structured_call(...)
except ModelNotSupportedError as e:
    print(f"Model not supported: {e}")
except InvalidResponseFormatError as e:
    print(f"Invalid response format: {e}")
except OpenAIClientError as e:
    print(f"General error: {e}")
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
