"""Command-line interface for making structured OpenAI API calls."""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Type

import tiktoken
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, create_model

from .client import async_openai_structured_stream, supports_structured_output
from .errors import (
    APIResponseError,
    EmptyResponseError,
    InvalidResponseFormatError,
    JSONParseError,
    ModelNotSupportedError,
    ModelVersionError,
    OpenAIClientError,
    StreamBufferError,
    StreamInterruptedError,
    StreamParseError,
)

# Check Python version
if sys.version_info < (3, 9):
    sys.exit("Python 3.9 or higher is required")


# Make jsonschema optional
try:
    from jsonschema import Draft7Validator, SchemaError

    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False


def validate_json_schema(schema: Dict[str, Any]) -> None:
    """Validate that the provided schema is a valid JSON Schema."""
    if not HAVE_JSONSCHEMA:
        logging.warning(
            "jsonschema package not installed. Schema validation disabled."
        )
        return

    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as e:
        raise ValueError(f"Invalid JSON Schema: {e}")


def validate_response(
    response: Dict[str, Any], schema: Dict[str, Any]
) -> None:
    """Validate that the response matches the provided JSON Schema."""
    if not HAVE_JSONSCHEMA:
        logging.warning(
            "jsonschema package not installed. Response validation disabled."
        )
        return

    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(response))
    if errors:
        error_messages = []
        for error in errors:
            path = (
                " -> ".join(str(p) for p in error.path)
                if error.path
                else "root"
            )
            error_messages.append(f"At {path}: {error.message}")
        raise ValueError(
            "Response validation errors:\n" + "\n".join(error_messages)
        )


def create_dynamic_model(schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from a JSON schema."""
    # Extract properties and their types
    properties = schema.get("properties", {})
    field_definitions: Dict[str, Any] = {}

    for name, prop in properties.items():
        field_type: Any  # Allow any type to be assigned
        if prop.get("type") == "string":
            field_type = str
        elif prop.get("type") == "integer":
            field_type = int
        elif prop.get("type") == "number":
            field_type = float
        elif prop.get("type") == "boolean":
            field_type = bool
        elif prop.get("type") == "array":
            field_type = List[Any]
        elif prop.get("type") == "object":
            field_type = Dict[str, Any]
        else:
            field_type = Any

        field_definitions[name] = (field_type, ...)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Type[BaseModel] = create_model(
        "DynamicModel", __config__=model_config, **field_definitions
    )
    return model


def validate_template_placeholders(
    template: str, available_files: Set[str]
) -> None:
    """Validate that all placeholders in the template have corresponding files."""
    placeholders = {m.group(1) for m in re.finditer(r"\{([^}]+)\}", template)}
    missing = placeholders - available_files
    if missing:
        raise ValueError(
            f"Template placeholders missing files: {', '.join(missing)}"
        )


def estimate_tokens_for_chat(
    messages: List[Dict[str, str]], model: str
) -> int:
    """Estimate the number of tokens in a chat completion."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        # Add message overhead
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def get_default_token_limit(model: str) -> int:
    """Get the default token limit for a given model.

    Note: These limits are based on current OpenAI model specifications as of 2024 and may
    need to be updated if OpenAI changes the models' capabilities.

    Args:
        model: The model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'o1')

    Returns:
        The default token limit for the model
    """
    if "o1" in model:
        return 100_000  # o1 supports up to 100K output tokens
    elif "gpt-4o" in model:
        return 16_384  # gpt-4o and gpt-4o-mini support up to 16K output tokens
    else:
        return 4_096  # default fallback


def get_context_window_limit(model: str) -> int:
    """Get the total context window limit for a given model.

    Note: These limits are based on current OpenAI model specifications as of 2024 and may
    need to be updated if OpenAI changes the models' capabilities.

    Args:
        model: The model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'o1')

    Returns:
        The context window limit for the model
    """
    if "o1" in model:
        return 200_000  # o1 supports 200K total context window
    elif "gpt-4o" in model:
        return 128_000  # gpt-4o and gpt-4o-mini support 128K context window
    else:
        return 8_192  # default fallback


def validate_token_limits(
    model: str, total_tokens: int, max_token_limit: Optional[int] = None
) -> None:
    """Validate token counts against model limits.

    Args:
        model: The model name
        total_tokens: Total number of tokens in the prompt
        max_token_limit: Optional user-specified token limit

    Raises:
        ValueError: If token limits are exceeded
    """
    context_limit = get_context_window_limit(model)
    output_limit = (
        max_token_limit
        if max_token_limit is not None
        else get_default_token_limit(model)
    )

    # Check if total tokens exceed context window
    if total_tokens >= context_limit:
        raise ValueError(
            f"Total tokens ({total_tokens:,}) exceed model's context window limit "
            f"of {context_limit:,} tokens"
        )

    # Check if there's enough room for output tokens
    remaining_tokens = context_limit - total_tokens
    if remaining_tokens < output_limit:
        raise ValueError(
            f"Only {remaining_tokens:,} tokens remaining in context window, but "
            f"output may require up to {output_limit:,} tokens"
        )


class ExitCode(IntEnum):
    """Exit codes for the CLI."""

    SUCCESS = 0
    VALIDATION_ERROR = 1  # Schema/response validation errors
    USAGE_ERROR = 2  # Command line usage errors
    API_ERROR = 3  # OpenAI API errors
    IO_ERROR = 4  # File/network IO errors
    UNKNOWN_ERROR = 5  # Unexpected errors
    INTERRUPTED = 6  # Operation cancelled by user


async def _main() -> ExitCode:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Make structured OpenAI API calls from the command line."
    )
    parser.add_argument(
        "--system-prompt", required=True, help="System prompt for the model"
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Template string with {file} placeholders",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="File mapping in name=path format. Can be specified multiple times.",
    )
    parser.add_argument(
        "--schema-file",
        required=True,
        help="Path to JSON schema file defining the response structure",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-08-06",
        help=(
            "OpenAI model to use. Supported models:\n"
            "- gpt-4o: 128K context, 16K output\n"
            "- gpt-4o-mini: 128K context, 16K output\n"
            "- o1: 200K context, 100K output"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help=(
            "Maximum number of tokens to generate. Set to 0 or negative to disable "
            "token limit checks. Defaults to model-specific limit."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Frequency penalty parameter (default: 0.0)",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty parameter (default: 0.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for API calls (default: 60.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output-file",
        help="Write JSON output to this file instead of stdout",
    )
    parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate the JSON schema file and the response",
    )
    parser.add_argument(
        "--api-key",
        help=(
            "OpenAI API key. Overrides OPENAI_API_KEY environment variable. "
            "Warning: Key might be visible in process list or shell history."
        ),
    )

    try:
        args = parser.parse_args()
    except Exception:
        return ExitCode.USAGE_ERROR

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger("ostruct")

    # Load and validate schema
    try:
        with open(args.schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
        if args.validate_schema:
            try:
                validate_json_schema(schema)
                logger.debug("JSON Schema validation passed")
            except ValueError as e:
                logger.error(str(e))
                return ExitCode.VALIDATION_ERROR
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in schema file: {e}")
        return ExitCode.VALIDATION_ERROR
    except OSError as e:
        logger.error(f"Cannot read schema file '{args.schema_file}': {e}")
        return ExitCode.IO_ERROR

    # Parse file mappings and handle stdin
    file_mappings = {}
    for mapping in args.file:
        try:
            name, path = mapping.split("=", 1)
            with open(path, "r", encoding="utf-8") as f:
                file_mappings[name] = f.read()
        except ValueError:
            logger.error(f"Invalid file mapping: {mapping}")
            return ExitCode.USAGE_ERROR
        except OSError as e:
            logger.error(f"Cannot read file '{path}': {e}")
            return ExitCode.IO_ERROR

    # Handle stdin if referenced in template
    if "{stdin}" in args.template:
        if not sys.stdin.isatty():
            file_mappings["stdin"] = sys.stdin.read()
        else:
            logger.error(
                "Template references {stdin} but no input provided on stdin"
            )
            return ExitCode.USAGE_ERROR

    # Validate template and build user prompt
    try:
        validate_template_placeholders(
            args.template, set(file_mappings.keys())
        )
        user_prompt = args.template.format(**file_mappings)
    except KeyError as e:
        logger.error(f"Template placeholder not found: {e}")
        return ExitCode.USAGE_ERROR
    except ValueError as e:
        logger.error(str(e))
        return ExitCode.VALIDATION_ERROR

    # Estimate tokens and validate limits
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        total_tokens = estimate_tokens_for_chat(messages, args.model)
        validate_token_limits(args.model, total_tokens, args.max_tokens)
        logger.debug(f"Total tokens in prompt: {total_tokens:,}")
    except (ValueError, OpenAIClientError) as e:
        logger.error(str(e))
        return ExitCode.VALIDATION_ERROR

    # Create dynamic model for validation
    try:
        model_class = create_dynamic_model(schema)
    except Exception as e:
        logger.error(f"Failed to create model from schema: {e}")
        return ExitCode.VALIDATION_ERROR

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "No OpenAI API key provided (--api-key or OPENAI_API_KEY env var)"
        )
        return ExitCode.USAGE_ERROR

    client = AsyncOpenAI(api_key=api_key)

    try:
        if not supports_structured_output(args.model):
            logger.error(
                f"Model '{args.model}' does not support structured output"
            )
            return ExitCode.API_ERROR

        # Make API call
        first_result = True
        try:
            async for result in async_openai_structured_stream(
                client=client,
                model=args.model,
                output_schema=model_class,
                system_prompt=args.system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                timeout=args.timeout,
                on_log=logger.debug if args.verbose else None,
            ):
                result_dict = result.model_dump()
                if args.validate_schema:
                    validate_response(result_dict, schema)

                json_str = json.dumps(result_dict)
                if args.output_file:
                    mode = "w" if first_result else "a"
                    with open(args.output_file, mode, encoding="utf-8") as f:
                        if not first_result:
                            f.write("\n")
                        f.write(json_str)
                else:
                    if not first_result:
                        print()
                    print(json_str, flush=True)
                first_result = False

        except StreamInterruptedError as e:
            logger.error(f"Stream interrupted: {e}")
            return ExitCode.API_ERROR
        except (StreamParseError, StreamBufferError) as e:
            logger.error(f"Stream processing error: {e}")
            return ExitCode.API_ERROR

    except (
        AuthenticationError,
        BadRequestError,
        RateLimitError,
        APIConnectionError,
        InternalServerError,
    ) as e:
        logger.error(f"API error: {e}")
        return ExitCode.API_ERROR
    except (
        ModelNotSupportedError,
        ModelVersionError,
        APIResponseError,
        InvalidResponseFormatError,
        EmptyResponseError,
        JSONParseError,
        OpenAIClientError,
    ) as e:
        logger.error(str(e))
        return ExitCode.VALIDATION_ERROR
    except Exception as e:
        if args.verbose:
            logging.exception("Unexpected error")
        logger.error(f"Unexpected error: {e}")
        return ExitCode.API_ERROR

    return ExitCode.SUCCESS


async def main() -> ExitCode:
    """Async main entry point for the CLI."""
    try:
        return await _main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return ExitCode.INTERRUPTED
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return ExitCode.UNKNOWN_ERROR


def cli_main() -> int:
    """Synchronous entry point for command line usage."""
    try:
        return int(asyncio.run(main()))
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return int(ExitCode.UNKNOWN_ERROR)


if __name__ == "__main__":
    sys.exit(cli_main())

# Export public API
__all__ = [
    "create_dynamic_model",
    "validate_template_placeholders",
    "estimate_tokens_for_chat",
    "get_context_window_limit",
    "get_default_token_limit",
    "validate_token_limits",
    "supports_structured_output",
]
