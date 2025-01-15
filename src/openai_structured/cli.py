"""Command-line interface for making structured OpenAI API calls."""

import argparse
import asyncio
import datetime
import itertools
import json
import logging
import os
import re
import sys
import textwrap
from enum import IntEnum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)

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


def sort_by(items: Sequence[T], key: str) -> List[T]:
    """Sort items by a key, handling both dict and object attributes."""

    def get_key(x: T) -> Any:
        return x.get(key) if isinstance(x, dict) else getattr(x, key)

    return sorted(items, key=get_key)


def group_by(items: Sequence[T], key: str) -> Dict[Any, List[T]]:
    """Group items by a key, handling both dict and object attributes."""
    sorted_items = sort_by(items, key)

    def get_key(x: T) -> Any:
        return x.get(key) if isinstance(x, dict) else getattr(x, key)

    return {
        k: list(g) for k, g in itertools.groupby(sorted_items, key=get_key)
    }


def filter_by(items: Sequence[T], key: str, value: Any) -> List[T]:
    """Filter items by a key-value pair."""
    return [
        x
        for x in items
        if (x.get(key) if isinstance(x, dict) else getattr(x, key)) == value
    ]


def pluck(items: Sequence[T], key: str) -> List[Any]:
    """Extract a specific field from each item."""
    return [
        x.get(key) if isinstance(x, dict) else getattr(x, key) for x in items
    ]


def unique(items: Sequence[T]) -> List[T]:
    """Get unique values while preserving order."""
    return list(dict.fromkeys(items))


def frequency(items: Sequence[T]) -> Dict[T, int]:
    """Count frequency of each item."""
    return {item: items.count(item) for item in set(items)}


def aggregate(
    items: Sequence[Any], key: Optional[str] = None
) -> Dict[str, Union[int, float]]:
    """Calculate aggregate statistics."""
    if not items:
        return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}

    def get_value(x: Any) -> float:
        if key is None:
            if isinstance(x, (int, float)):
                return float(x)
            raise ValueError(f"Cannot convert {type(x)} to float")
        val = x.get(key) if isinstance(x, dict) else getattr(x, key, 0)
        if val is None:
            return 0.0
        return float(val)

    values = [get_value(x) for x in items]
    return {
        "count": len(values),
        "sum": sum(values),
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def render_template(template_str: str, context: Dict[str, Any]) -> str:
    """Render a Jinja2 template with the given context."""
    # Create environment with comprehensive features
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("."),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        line_statement_prefix="#",
        line_comment_prefix="##",
        finalize=lambda x: x if x is not None else "",
    )

    def syntax_highlight(text: str, lang: str = "python") -> str:
        return text

    def extract_keywords(text: str) -> List[str]:
        return text.split()

    def word_count(text: str) -> int:
        return len(text.split())

    def char_count(text: str) -> int:
        return len(text)

    def to_json(obj: Any) -> str:
        return json.dumps(obj, indent=2)

    def from_json(text: str) -> Any:
        return json.loads(text)

    def remove_comments(text: str) -> str:
        return re.sub(
            r"#.*$|//.*$|/\*[\s\S]*?\*/", "", text, flags=re.MULTILINE
        )

    def wrap_text(text: str, width: int = 80) -> str:
        return textwrap.fill(text, width)

    def indent_text(text: str, width: int = 4) -> str:
        return textwrap.indent(text, " " * width)

    def dedent_text(text: str) -> str:
        return textwrap.dedent(text)

    def normalize_text(text: str) -> str:
        return " ".join(text.split())

    def strip_markdown(text: str) -> str:
        return re.sub(r"[#*`_~]", "", text)

    def format_table(
        headers: Sequence[Any], rows: Sequence[Sequence[Any]]
    ) -> str:
        return (
            f"| {' | '.join(str(h) for h in headers)} |\n"
            f"| {' | '.join('-' * max(len(str(h)), 3) for h in headers)} |\n"
            + "\n".join(
                f"| {' | '.join(str(cell) for cell in row)} |" for row in rows
            )
        )

    def align_table(
        headers: Sequence[Any],
        rows: Sequence[Sequence[Any]],
        alignments: Optional[Sequence[str]] = None,
    ) -> str:
        alignments_list = alignments or ["left"] * len(headers)
        alignment_markers = []
        for a in alignments_list:
            if a == "center":
                alignment_markers.append(":---:")
            elif a == "left":
                alignment_markers.append(":---")
            elif a == "right":
                alignment_markers.append("---:")
            else:
                alignment_markers.append("---")

        return (
            f"| {' | '.join(str(h) for h in headers)} |\n"
            f"| {' | '.join(alignment_markers)} |\n"
            + "\n".join(
                f"| {' | '.join(str(cell) for cell in row)} |" for row in rows
            )
        )

    def dict_to_table(data: Dict[Any, Any]) -> str:
        return "| Key | Value |\n| --- | --- |\n" + "\n".join(
            f"| {k} | {v} |" for k, v in data.items()
        )

    def list_to_table(
        items: Sequence[Any], headers: Optional[Sequence[str]] = None
    ) -> str:
        if not headers:
            return "| # | Value |\n| --- | --- |\n" + "\n".join(
                f"| {i+1} | {item} |" for i, item in enumerate(items)
            )
        return (
            f"| {' | '.join(headers)} |\n| {' | '.join('-' * len(h) for h in headers)} |\n"
            + "\n".join(
                f"| {' | '.join(str(cell) for cell in row)} |" for row in items
            )
        )

    def process_code(text: str, lang: str = "python") -> str:
        """Process code by removing comments, dedenting, and highlighting."""
        return str(
            env.filters["remove_comments"](text)
            | env.filters["dedent"](text)
            | env.filters["syntax_highlight"](text, lang)
        )

    def format_prompt(text: str) -> str:
        """Format prompt text with normalization, wrapping, and indentation."""
        return str(
            env.filters["normalize"](text)
            | env.filters["wrap"](text, 80)
            | env.filters["indent"](text, 4)
        )

    def optimize_tokens(text: str, max_length: Optional[int] = None) -> str:
        """Optimize text for token usage."""
        if max_length is None:
            return str(env.filters["normalize"](text))
        return str(env.filters["normalize"](text[:max_length]))

    def escape_special(text: str) -> str:
        return re.sub(r'([{}\[\]"\'\\])', r"\\\1", text)

    def debug_format(obj: Any) -> str:
        return (
            f"Type: {type(obj).__name__}\n"
            f"Length: {len(str(obj))}\n"
            f"Content: {str(obj)[:200]}..."
        )

    def format_table_cell(x: Any) -> str:
        return str(x).replace("|", "\\|").replace("\n", "<br>")

    def auto_table(data: Any) -> str:
        """Convert data to a markdown table format."""
        if isinstance(data, dict):
            return str(env.filters["dict_to_table"](data))
        if isinstance(data, (list, tuple)):
            return str(env.filters["list_to_table"](data))
        return str(data)

    # Add custom filters
    env.filters.update(
        {
            "syntax_highlight": syntax_highlight,
            "extract_keywords": extract_keywords,
            "word_count": word_count,
            "char_count": char_count,
            "to_json": to_json,
            "from_json": from_json,
            "remove_comments": remove_comments,
            "wrap": wrap_text,
            "indent": indent_text,
            "dedent": dedent_text,
            "normalize": normalize_text,
            "strip_markdown": strip_markdown,
            # Data processing filters
            "sort_by": sort_by,
            "group_by": group_by,
            "filter_by": filter_by,
            "pluck": pluck,
            "unique": unique,
            "frequency": frequency,
            "aggregate": aggregate,
            # Table formatting filters
            "table": format_table,
            "align_table": align_table,
            "dict_to_table": dict_to_table,
            "list_to_table": list_to_table,
            # Advanced content processing
            "process_code": process_code,
            "format_prompt": format_prompt,
            "optimize_tokens": optimize_tokens,
            "escape_special": escape_special,
            "debug_format": debug_format,
        }
    )

    def estimate_tokens(text: str) -> float:
        return len(text.split()) * 1.3

    def format_json(obj: Any) -> str:
        return json.dumps(obj, indent=2)

    def debug_print(x: Any) -> None:
        print(f"DEBUG: {x}")

    def type_of(x: Any) -> str:
        return type(x).__name__

    def dir_of(x: Any) -> List[str]:
        return dir(x)

    def len_of(x: Any) -> Optional[int]:
        return len(x) if hasattr(x, "__len__") else None

    def create_prompt(template: str, **kwargs: Any) -> str:
        return env.from_string(template).render(**kwargs)

    def validate_json(text: str) -> bool:
        if not text:
            return False
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def count_tokens(text: str) -> int:
        return len(text.split())

    def format_error(e: Exception) -> str:
        return f"{type(e).__name__}: {str(e)}"

    # Add template globals
    env.globals.update(
        {
            "estimate_tokens": estimate_tokens,
            "format_json": format_json,
            "now": datetime.datetime.now,
            "debug": debug_print,
            "type_of": type_of,
            "dir_of": dir_of,
            "len_of": len_of,
            "create_prompt": create_prompt,
            "validate_json": validate_json,
            "count_tokens": count_tokens,
            "format_error": format_error,
            # Data analysis globals
            "summarize": summarize,
            "pivot_table": pivot_table,
            # Table utilities
            "format_table_cell": format_table_cell,
            "auto_table": auto_table,
        }
    )

    # Create template from string or file
    if template_str.endswith((".j2", ".jinja2", ".md")) and os.path.isfile(
        template_str
    ):
        template = env.get_template(template_str)
    else:
        template = env.from_string(template_str)

    # Add debug context
    template.globals["template_name"] = getattr(template, "name", "<string>")
    template.globals["template_path"] = getattr(template, "filename", None)

    return template.render(**context)


def validate_template_placeholders(
    template: str, available_files: Set[str]
) -> None:
    """Validate that all placeholders in the template have corresponding files.

    Supports Jinja2 syntax including:
    - Variable references: {{ var }}
    - Control structures: {% if/for/etc %}
    - Comments: {# comment #}
    """
    try:
        env = jinja2.Environment()
        ast = env.parse(template)
        variables = {
            node.name
            for node in ast.find_all(jinja2.nodes.Name)
            if isinstance(node.name, str)
        }

        # Remove built-in Jinja2 variables and functions
        builtin_vars = {
            "loop",
            "self",
            "range",
            "dict",
            "lipsum",
            "cycler",
            "namespace",
            "super",
            "varargs",
            "kwargs",
            "undefined",
            # Template functions
            "estimate_tokens",
            "format_json",
            "now",
            "debug",
            "type_of",
            "dir_of",
            "len_of",
            "create_prompt",
            "validate_json",
            "count_tokens",
            "format_error",
            "summarize",
            "pivot_table",
            "format_table_cell",
            "auto_table",
        }
        variables = variables - builtin_vars

        # Check for undefined variables
        missing = variables - available_files
        if missing:
            raise ValueError(
                f"Template placeholders missing files: {', '.join(sorted(missing))}"
            )
    except jinja2.TemplateSyntaxError as e:
        raise ValueError(f"Invalid template syntax: {str(e)}")


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
        user_prompt = render_template(args.template, file_mappings)
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


def summarize(
    data: Sequence[Any], keys: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Generate summary statistics for data fields."""
    if not data:
        return {"total_records": 0, "fields": {}}

    def get_field_value(item: Any, field: str) -> Any:
        if isinstance(item, dict):
            return item.get(field)
        try:
            return getattr(item, field)
        except AttributeError:
            return None

    def get_field_type(value: Any) -> str:
        return type(value).__name__ if value is not None else "NoneType"

    def analyze_field(field: str) -> Dict[str, Any]:
        values = [get_field_value(x, field) for x in data]
        non_null_values = [v for v in values if v is not None]
        return {
            "type": get_field_type(
                next((v for v in values if v is not None), None)
            ),
            "unique_values": len(set(non_null_values)),
            "null_count": len(values) - len(non_null_values),
        }

    available_keys = keys or (
        list(data[0].keys())
        if isinstance(data[0], dict)
        else list(vars(data[0]).keys())
    )

    return {
        "total_records": len(data),
        "fields": {key: analyze_field(key) for key in available_keys},
    }


def pivot_table(
    data: Sequence[Any], index: str, values: str, aggfunc: str = "sum"
) -> Dict[Any, float]:
    """Create a pivot table from data."""

    def get_field_value(item: Any, field: str) -> Any:
        if isinstance(item, dict):
            return item.get(field)
        try:
            return getattr(item, field)
        except AttributeError:
            return None

    def aggregate_values(vals: List[Any]) -> float:
        if not vals:
            return 0.0
        nums = [float(v) for v in vals if v is not None]
        if not nums:
            return 0.0
        if aggfunc == "sum":
            return sum(nums)
        elif aggfunc == "mean":
            return sum(nums) / len(nums)
        elif aggfunc == "min":
            return min(nums) if nums else 0.0
        elif aggfunc == "max":
            return max(nums) if nums else 0.0
        elif aggfunc == "count":
            return float(len(nums))
        return sum(nums)  # default to sum

    # Group data by index
    groups: Dict[Any, List[Any]] = {}
    for item in data:
        idx_val = get_field_value(item, index)
        if idx_val not in groups:
            groups[idx_val] = []
        groups[idx_val].append(get_field_value(item, values))

    # Aggregate each group
    return {idx: aggregate_values(group) for idx, group in groups.items()}
