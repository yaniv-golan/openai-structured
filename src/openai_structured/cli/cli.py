"""Command-line interface for making structured OpenAI API calls."""

import argparse
import asyncio
import json
import logging
import os
import sys
from enum import IntEnum
from typing import Any, Dict, List, Optional, Type, TypeVar

import jinja2
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

from ..client import async_openai_structured_stream, supports_structured_output
from ..errors import (
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
from .progress import ProgressContext
from .template_utils import (
    get_template_variables,
    render_template,
    validate_json_schema,
    validate_response,
    validate_template_placeholders,
)


class ExitCode(IntEnum):
    """Exit codes for the CLI."""

    SUCCESS = 0
    UNKNOWN_ERROR = 1
    INTERRUPTED = 2
    USAGE_ERROR = 64
    IO_ERROR = 74
    API_ERROR = 75
    VALIDATION_ERROR = 76


def create_dynamic_model(schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from a JSON schema."""
    properties = schema.get("properties", {})
    field_definitions: Dict[str, Any] = {}

    for name, prop in properties.items():
        field_type: Any
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


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


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
        "--dry-run",
        action="store_true",
        help="Simulate API call and show parameters without making the actual call",
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

    # Read stdin if available
    if not sys.stdin.isatty():
        try:
            file_mappings["stdin"] = sys.stdin.read()
        except OSError as e:
            logger.error(f"Cannot read from stdin: {e}")
            return ExitCode.IO_ERROR

    # Create Jinja2 environment for template parsing
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("."),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        line_statement_prefix="#",
        line_comment_prefix="##",
        undefined=jinja2.StrictUndefined,
    )

    # Create template from string
    template = env.from_string(args.template)
    template_vars = get_template_variables(template)

    # Read stdin if referenced in template
    if "stdin" in template_vars and not sys.stdin.isatty():
        try:
            file_mappings["stdin"] = sys.stdin.read()
        except OSError as e:
            logger.error(f"Cannot read from stdin: {e}")
            return ExitCode.IO_ERROR

    # Validate template placeholders
    missing_files = [var for var in template_vars if var not in file_mappings]
    if missing_files:
        logger.error(
            f"Template placeholders missing files: {', '.join(missing_files)}"
        )
        return ExitCode.VALIDATION_ERROR

    # Handle stdin if referenced in template
    if "stdin" in template_vars:
        if not sys.stdin.isatty() and "stdin" not in file_mappings:
            try:
                file_mappings["stdin"] = sys.stdin.read()
            except OSError as e:
                logger.error(f"Cannot read from stdin: {e}")
                return ExitCode.IO_ERROR
        elif "stdin" not in file_mappings:
            logger.error(
                "Template references {{ stdin }} but no input provided on stdin"
            )
            return ExitCode.USAGE_ERROR

    # Validate template and build user prompt
    try:
        validate_template_placeholders(
            args.template, set(file_mappings.keys())
        )
        user_prompt = render_template(
            args.template, file_mappings, jinja_env=env
        )
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

    # Handle dry run mode
    if args.dry_run:
        logger.info("*** DRY RUN MODE - No API call will be made ***\n")
        logger.info("System Prompt:\n%s\n", args.system_prompt)
        logger.info("User Prompt:\n%s\n", user_prompt)
        logger.info("Estimated Tokens: %s", total_tokens)
        logger.info("Model: %s", args.model)
        logger.info("Temperature: %s", args.temperature)

        if args.max_tokens is not None:
            logger.info("Max Tokens: %s", args.max_tokens)

        logger.info("Top P: %s", args.top_p)
        logger.info("Frequency Penalty: %s", args.frequency_penalty)
        logger.info("Presence Penalty: %s", args.presence_penalty)

        if args.validate_schema:
            logger.info("Schema: Valid")

        if args.output_file:
            logger.info("Output would be written to: %s", args.output_file)

        return ExitCode.SUCCESS

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

        except (
            StreamInterruptedError,
            StreamParseError,
            StreamBufferError,
        ) as e:
            logger.error(
                f"Stream {'interrupted' if isinstance(e, StreamInterruptedError) else 'processing'} error: {e}"
            )
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
    "ProgressContext",
]
