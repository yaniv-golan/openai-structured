"""Command-line interface for making structured OpenAI API calls."""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Any, Type, List, Union, Tuple

import tiktoken
from openai import AsyncOpenAI, APIConnectionError, AuthenticationError, BadRequestError, InternalServerError, RateLimitError
from pydantic import BaseModel, create_model, ConfigDict

from .client import openai_structured_call
from .errors import ModelNotSupportedError, OpenAIClientError


# Check Python version
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required")


# Make jsonschema optional
try:
    from jsonschema import Draft7Validator, ValidationError, SchemaError
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False


def validate_json_schema(schema: Dict[str, Any]) -> None:
    """Validate that the provided schema is a valid JSON Schema."""
    if not HAVE_JSONSCHEMA:
        logging.warning("jsonschema package not installed. Schema validation disabled.")
        return
        
    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as e:
        raise ValueError(f"Invalid JSON Schema: {e}")


def validate_response(response: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate that the response matches the provided JSON Schema."""
    if not HAVE_JSONSCHEMA:
        logging.warning("jsonschema package not installed. Response validation disabled.")
        return
        
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(response))
    if errors:
        error_messages = []
        for error in errors:
            path = " -> ".join(str(p) for p in error.path) if error.path else "root"
            error_messages.append(f"At {path}: {error.message}")
        raise ValueError("Response validation errors:\n" + "\n".join(error_messages))


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
        "DynamicModel",
        __config__=model_config,
        **field_definitions
    )
    return model


def validate_template(template: str, available_files: Set[str]) -> None:
    """Validate that all placeholders in the template have corresponding files."""
    placeholders = {m.group(1) for m in re.finditer(r"\{([^}]+)\}", template)}
    missing = placeholders - available_files
    if missing:
        raise ValueError(f"Template placeholders missing files: {', '.join(missing)}")


def estimate_tokens_for_chat(messages: List[Dict[str, str]], model: str) -> int:
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


def validate_token_limits(model: str, total_tokens: int, max_token_limit: Optional[int] = None) -> None:
    """Validate token counts against model limits.
    
    Args:
        model: The model name
        total_tokens: Total number of tokens in the prompt
        max_token_limit: Optional user-specified token limit
        
    Raises:
        ValueError: If token limits are exceeded
    """
    context_limit = get_context_window_limit(model)
    output_limit = max_token_limit if max_token_limit is not None else get_default_token_limit(model)
    
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


async def _main() -> None:
    """Main async function."""
    parser = argparse.ArgumentParser(
        description="Make structured OpenAI API calls from the command line."
    )
    parser.add_argument("--system-prompt", required=True,
                      help="System prompt for the model")
    parser.add_argument("--template", required=True,
                      help="Template string with {file} placeholders")
    parser.add_argument("--file", action="append", default=[],
                      help="File mapping in name=path format. Can be specified multiple times.")
    parser.add_argument("--schema-file", required=True,
                      help="Path to JSON schema file defining the response structure")
    parser.add_argument("--model", default="gpt-4o-2024-08-06",
                      help=(
                          "OpenAI model to use. Supported models:\n"
                          "- gpt-4o: 128K context, 16K output\n"
                          "- gpt-4o-mini: 128K context, 16K output\n"
                          "- o1: 200K context, 100K output"
                      ),
    )
    parser.add_argument("--max-token-limit", type=int,
                      help="Maximum tokens allowed. Set to 0 or negative to disable check.")
    parser.add_argument("--output-file",
                      help="Write JSON output to this file instead of stdout")
    parser.add_argument("--log-level", default="INFO",
                      help="Logging level: DEBUG, INFO, WARNING, or ERROR")
    parser.add_argument("--api-key",
                      help="OpenAI API key. Overrides OPENAI_API_KEY environment variable. "
                           "Warning: Key might be visible in process list or shell history.")
    parser.add_argument("--validate-schema", action="store_true",
                      help="Validate the JSON schema file and the response")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())
    logger = logging.getLogger("oai-structured-cli")

    # 1) Load JSON schema
    try:
        with open(args.schema_file, "r", encoding="utf-8") as sf:
            schema_data = json.load(sf)
            
        # Validate the schema if requested
        if args.validate_schema:
            try:
                validate_json_schema(schema_data)
                logger.debug("JSON Schema validation passed")
            except ValueError as e:
                parser.error(str(e))
                
    except json.JSONDecodeError as e:
        parser.error(f"Invalid JSON in schema file: {e}")
    except Exception as e:
        parser.error(f"Cannot read schema file '{args.schema_file}': {e}")

    # Create Pydantic model from schema
    try:
        output_schema = create_dynamic_model(schema_data)
    except Exception as e:
        parser.error(f"Failed to create model from schema: {e}")

    # 2) Gather file contents
    files_map: Dict[str, str] = {}
    for file_arg in args.file:
        if '=' not in file_arg:
            parser.error("--file must be 'name=path'")
        name, fpath = file_arg.split('=', 1)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                files_map[name] = f.read()
        except Exception as e:
            parser.error(f"Cannot read {fpath}: {e}")

    # 3) Read from stdin if needed
    if "{stdin}" in args.template:
        if not sys.stdin.isatty():
            files_map["stdin"] = sys.stdin.read()
        else:
            parser.error("Template references {stdin} but no input provided on stdin")

    # 4) Validate template and build user prompt
    try:
        validate_template(args.template, set(files_map.keys()))
    except ValueError as e:
        parser.error(str(e))

    user_prompt = args.template
    for name, content in files_map.items():
        user_prompt = user_prompt.replace(f"{{{name}}}", content)

    # 5) Estimate tokens if needed
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    max_tokens = args.max_token_limit if args.max_token_limit is not None else get_default_token_limit(args.model)
    if max_tokens > 0:  # Explicitly check > 0 to match spec
        total_tokens = estimate_tokens_for_chat(messages, args.model)
        if total_tokens > max_tokens:
            logger.error(
                f"Prompt requires {total_tokens} tokens, exceeds limit of {max_tokens}"
            )
            sys.exit(1)

    # 6) Get API key
    openai_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        parser.error("No OpenAI API key provided (--api-key or OPENAI_API_KEY env var)")

    # 7) Make the API call
    openai_client = AsyncOpenAI(api_key=openai_key)
    try:
        result = await openai_structured_call(
            client=openai_client,
            model=args.model,
            output_schema=output_schema,  # Use dynamic model
            user_prompt=user_prompt,
            system_prompt=args.system_prompt,
            logger=logger,
        )
    except ModelNotSupportedError as e:
        logger.error(f"Model not supported: {e}")
        sys.exit(1)
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        sys.exit(1)
    except (APIConnectionError, AuthenticationError, BadRequestError) as e:
        logger.error(f"API error: {e}")
        sys.exit(1)
    except InternalServerError as e:
        logger.error(f"OpenAI server error: {e}")
        sys.exit(1)
    except OpenAIClientError as e:
        logger.error(f"Client error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

    # 8) Output the result
    try:
        # Convert result to dict
        result_dict = result.model_dump()
        
        # Validate response if requested (and jsonschema is available)
        if args.validate_schema:
            try:
                validate_response(result_dict, schema_data)
                logger.debug("Response validation passed")
            except ValueError as e:
                logger.error(str(e))
                sys.exit(1)
        
        # Convert to string for output
        json_str = json.dumps(result_dict, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to process result: {e}")
        sys.exit(1)
    
    if args.output_file:
        # Ensure output directory exists
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(json_str)
    else:
        print(json_str)


def main() -> None:
    """Main CLI entrypoint."""
    asyncio.run(_main())


if __name__ == "__main__":
    main() 