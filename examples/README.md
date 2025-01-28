# openai-structured Examples

This directory contains examples demonstrating various features of the `openai-structured` library for working with [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/function-calling).

## Basic Examples

### streaming.py

Demonstrates streaming with OpenAI Structured Outputs:

- Processing large structured responses piece by piece
- Buffer management and cleanup
- Progress tracking
- Error handling

### error_handling.py

Comprehensive error handling examples:

- Stream interruption handling
- Buffer overflow management
- Parse error recovery
- Rate limiting and retries
- Network error handling
- Resource cleanup

## Production Examples

### production_setup.py

Production-ready setup for OpenAI Structured Outputs with:

- Proper error handling
- Retries with backoff
- Rate limiting
- Resource cleanup
- Logging configuration
- Monitoring hooks

### logging_setup.py

Advanced logging configuration:

- Structured logging
- Multiple handlers
- Log rotation
- Error tracking
- Performance monitoring

## Running the Examples

1. Install the package:

```bash
pip install openai-structured
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-api-key
```

3. Run any example:

```bash
python examples/streaming.py
python examples/cli_basic.py
# etc.
```

## Notes

- All examples use streaming by default
- Examples use the latest `gpt-4o-2024-08-06` model with OpenAI Structured Outputs
- Some examples create temporary files
- Error examples demonstrate failure handling
- Logging examples create log files
