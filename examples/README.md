# Examples

This directory contains example scripts demonstrating how to use the `openai-structured` library.

## Setup

1. Install the library:

```bash
pip install openai-structured
```

2. Set up your environment:

```bash
# Create a .env file in the examples directory
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Examples

### Basic Usage (`basic_usage.py`)

Demonstrates basic synchronous API calls with structured output:

- Creating Pydantic models
- Making API calls
- Processing structured responses

```bash
python basic_usage.py
```

### Streaming (`streaming.py`)

Shows how to use async streaming for real-time structured output:

- Async client setup
- Streaming responses
- Proper resource cleanup

```bash
python streaming.py
```

### Advanced Usage (`advanced_usage.py`)

Illustrates advanced features and best practices:

- Comprehensive error handling
- Logging configuration
- Complex Pydantic models
- Resource management
- Configuration options

```bash
python advanced_usage.py
```

## Notes

- Each example includes detailed comments explaining the code
- Error handling follows best practices
- Examples demonstrate proper resource cleanup
- Configuration options are documented inline

## Common Issues

1. **API Key Not Found**
   - Ensure you've set the `OPENAI_API_KEY` environment variable
   - Check that your `.env` file is in the correct location

2. **Model Availability**
   - The examples use specific model versions
   - Update the model names if newer versions are available

3. **Rate Limits**
   - Be aware of OpenAI's rate limits
   - Add appropriate delays if running examples multiple times
