# openai-structured

[![PyPI version](https://badge.fury.io/py/openai-structured.svg)](https://badge.fury.io/py/openai-structured)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/openai-structured)](https://pypi.org/project/openai-structured/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/openai-structured/badge/?version=latest)](https://openai-structured.readthedocs.io/en/latest/)
[![Build Status](https://github.com/yaniv-golan/openai-structured/actions/workflows/python-package.yml/badge.svg)](https://github.com/yaniv-golan/openai-structured/actions/workflows/python-package.yml)

openai-structured is a Python library that provides a structured approach to interfacing with [OpenAI's Structured Outputs](https://platform.openai.com/docs/guides/function-calling) using Pydantic models. The library simplifies the processing and validation of structured outputs with support for both synchronous and asynchronous calls, robust error handling, and efficient streaming.

> **Note:** Looking for the CLI tool? It has been moved to a separate package called `ostruct-cli`.
> Visit <https://github.com/yaniv-golan/ostruct> for installation and usage instructions.

## Key Features

- **Structured Output**: Validate API responses directly against Pydantic models.
- **Streaming Support**: Process large or incremental responses efficiently.
- **Type Safety**: Full type annotations ensure reliable and predictable behavior.
- **Robust Error Handling**: Specific exception types for handling API and streaming errors.
- **Multi-Model Support**: Compatible with various models that offer structured outputs.
- **Enhanced Validation**: Comprehensive parameter validation with constraints.
- **Live Testing**: Built-in support for live testing capabilities.

## Requirements

- Python 3.9 or higher
- OpenAI API key

### Dependencies

These will be installed automatically:

- `openai>=1.12.0`: OpenAI Python SDK
- `pydantic>=2.6.3`: Data validation
- `jsonschema>=4.23.0`: JSON Schema validation
- `aiohttp>=3.11.11`: Async HTTP client
- `typing-extensions>=4.9.0`: Enhanced typing support

## Installation

### Using pip

```bash
pip install "openai-structured>=2.0.0"
```

### Using Poetry (recommended for development)

```bash
# Install Poetry (if needed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone, install, and activate
git clone https://github.com/yaniv-golan/openai-structured.git
cd openai-structured
poetry install
poetry shell
```

## Supported Models

The library supports models with [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/function-calling) capabilities:

### Production Models (Recommended)

- `gpt-4o-2024-08-06`: GPT-4 with OpenAI Structured Outputs
  - 128K context window
  - 16K output tokens
  - Supports streaming

- `gpt-4o-mini-2024-07-18`: GPT-4 variant
  - 128K context window
  - 16K output tokens
  - Supports streaming

- `o1-2024-12-17`
  - 200K context window
  - 100K output tokens
  - Does not support streaming (will return 400 error if attempted)
  - Limited parameter support (see note below)

- `o3-mini-2025-01-31`
  - 200K context window
  - 100K output tokens
  - Supports streaming
  - Limited parameter support (see note below)

### Development Aliases

- `gpt-4o`: Latest GPT-4 structured model (supports streaming)
- `gpt-4o-mini`: Latest mini variant (supports streaming)
- `o1`: Latest optimized model (limited parameters, no streaming)
- `o3`: Latest optimized model (limited parameters, no streaming)
- `o3-mini`: Latest mini optimized model (limited parameters, supports streaming)

Note: Use dated versions in production for stability. Aliases automatically use the latest compatible version.

> **Important Notes**:
>
> 1. o1 and o3 models only support the following parameters:
>    - max_completion_tokens
>    - reasoning_effort
>    Attempting to use other parameters (temperature, top_p, etc.) will raise an OpenAIClientError.
>
> 2. Streaming Support:
>    - o1-2024-12-17: Does not support streaming. Setting stream=True will result in a 400 error with message: "Unsupported value: 'stream' does not support true with this model. Supported values are: false"
>    - o3: Does not support streaming. Setting stream=True will result in a 400 error
>    - o3-mini and o3-mini-high: Support streaming
>
> Use other models if you need to adjust temperature, top_p, or other parameters not supported by o1/o3 models.

## Error Handling

The library provides comprehensive error handling with specific exception types:

```python
from openai_structured import (
    OpenAIClientError,        # Base class for all errors
    ModelNotSupportedError,   # Model doesn't support structured output
    ModelVersionError,        # Model version not supported
    APIResponseError,         # API call failures
    InvalidResponseFormatError,  # Response format issues
    EmptyResponseError,       # Empty response from API
    JSONParseError,          # JSON parsing failures
    StreamInterruptedError,   # Stream interruption
    StreamBufferError,        # Buffer management issues
    StreamParseError,        # Stream parsing failures
    BufferOverflowError,     # Buffer size exceeded
    TokenLimitError,         # ValueError: Token limit exceeded
)

try:
    result = openai_structured_call(...)
except ModelNotSupportedError as e:
    print(f"Model not supported: {e}")
except StreamInterruptedError as e:
    print(f"Stream interrupted: {e}")
except StreamParseError as e:
    print(f"Stream parse error after {e.attempts} attempts: {e}")
except OpenAIClientError as e:
    print(f"General error: {e}")
```

## Python API

### Basic Usage

```python
from typing import List
from openai import OpenAI
from openai_structured import async_openai_structured_stream
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

# Process streamed response using OpenAI Structured Outputs
async for result in async_openai_structured_stream(
    client=client,
    model="gpt-4o-2024-08-06",
    output_schema=TodoList,
    user_prompt="Create a todo list with 3 items",
    system_prompt="Generate a todo list with priorities"
):
    # Type-safe access to response
    for item in result.items:
        print(f"Task: {item.task}, Priority: {item.priority}")
    print(f"Total items: {result.total_count}")
```

### Stream Configuration

Control streaming behavior with `StreamConfig`:

```python
from openai_structured import StreamConfig, async_openai_structured_stream

stream_config = StreamConfig(
    max_buffer_size=1024 * 1024,  # 1MB max buffer
    cleanup_threshold=512 * 1024,  # Clean up at 512KB
    chunk_size=8192,  # 8KB chunks
)

async for result in async_openai_structured_stream(
    client=client,
    model="gpt-4o",
    output_schema=MySchema,
    system_prompt="...",
    user_prompt="...",
    stream_config=stream_config,
):
    process_result(result)
```

### Production Best Practices

1. Use dated model versions for stability:

```python
model = "gpt-4o-2024-08-06"  # Instead of "gpt-4o"
```

2. Implement retries for resilience:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
async def resilient_call():
    async for result in async_openai_structured_stream(...):
        yield result
```

3. Add rate limiting:

```python
from asyncio_throttle import Throttler

async with Throttler(rate_limit=100, period=60):
    async for result in async_openai_structured_stream(...):
        process_result(result)
```

4. Handle all error types:

```python
try:
    async for result in async_openai_structured_stream(...):
        process_result(result)
except StreamInterruptedError:
    handle_interruption()
except StreamBufferError:
    handle_buffer_issue()
except OpenAIClientError as e:
    handle_general_error(e)
finally:
    cleanup_resources()
```

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1  # Optional custom endpoint
OPENAI_API_VERSION=2024-02-01  # Optional API version
```

### Python Version Support

- Python 3.9+
- Tested on CPython implementations
- Compatible with Linux, macOS, and Windows

## Documentation

For full documentation, visit [openai-structured.readthedocs.io](https://openai-structured.readthedocs.io/).

## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.

## Testing

### Basic Tests

```python
def test_simple_call():
    client = OpenAI()  # Configure with test credentials
    result = openai_structured_call(
        client=client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        user_prompt="test"
    )
    assert isinstance(result, SimpleMessage)
    assert result.message

def test_stream():
    client = OpenAI()  # Configure with test credentials
    results = []
    for result in openai_structured_stream(
        client=client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        user_prompt="test"
    ):
        results.append(result)

    assert len(results) > 0
    for result in results:
        assert isinstance(result, SimpleMessage)

### Error Handling

```python
def test_invalid_credentials():
    client = OpenAI(api_key="invalid-key")
    with pytest.raises(StreamInterruptedError):
        list(openai_structured_stream(
            client=client,
            model="gpt-4o",
            output_schema=SimpleMessage,
            user_prompt="test"
        ))

### Async Testing

```python
async def test_async_stream():
    client = AsyncOpenAI()  # Configure with test credentials
    results = []
    async for result in async_openai_structured_stream(
        client=client,
        model="gpt-4o",
        output_schema=SimpleMessage,
        user_prompt="test"
    ):
        results.append(result)

    assert len(results) > 0
    for result in results:
        assert isinstance(result, SimpleMessage)
```

## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.
