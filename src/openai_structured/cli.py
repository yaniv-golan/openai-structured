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
import types
from enum import IntEnum
from functools import lru_cache
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
from jinja2 import Environment, Template, meta
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, create_model
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

# Make jsonschema optional
try:
    from jsonschema import Draft7Validator, SchemaError

    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

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


class ProgressContext:
    """Context manager for progress tracking."""

    def __init__(
        self, description: str, total: int = 1, enabled: bool = True
    ) -> None:
        self.description = description
        self.total = total
        self.enabled = enabled
        self.task: Optional[TaskID] = None
        self.progress: Optional[Progress] = None

    def __enter__(self) -> "ProgressContext":
        if not self.enabled:
            return self

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=Console(file=sys.stderr),
        )
        self.progress.start()
        self.task = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        if self.progress is not None:
            self.progress.stop()

    def update(self, advance: int = 1) -> None:
        """Update progress by the specified amount."""
        if not self.enabled:
            return

        if self.progress is not None and self.task is not None:
            try:
                self.progress.update(self.task, advance=advance)
            except Exception as e:
                logging.warning(f"Failed to update progress bar: {e}")
                self.enabled = False


_file_mtimes: Dict[str, float] = {}


@lru_cache(maxsize=128)
def _cached_read_file(full_path: str, encoding: str) -> str:
    """Cached file reading implementation.

    Note: Only successful reads are cached. Errors are always propagated.
    """
    try:
        with open(full_path, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise OSError(f"Failed to read file: {str(e)}")


def read_file(
    path: str, encoding: str = "utf-8", use_cache: bool = True
) -> str:
    """Read file contents safely with path validation and optional caching.

    Args:
        path: Path to file, relative to current directory
        encoding: File encoding (default: utf-8)
        use_cache: Whether to cache file contents (default: True)

    Returns:
        File contents as string

    Raises:
        ValueError: If path is invalid or outside base directory
        OSError: If file cannot be read or has encoding issues
    """
    # Resolve path relative to current directory
    base_dir = os.path.abspath(os.getcwd())
    full_path = os.path.abspath(os.path.join(base_dir, path))

    # Validate path is within base directory
    if not full_path.startswith(base_dir):
        raise ValueError(
            f"Access denied: Path {path} is outside base directory"
        )

    try:
        if use_cache:
            # Check if file was modified since last cache
            try:
                current_mtime = os.path.getmtime(full_path)
                last_mtime = _file_mtimes.get(full_path, 0.0)

                if current_mtime > last_mtime:
                    _cached_read_file.cache_clear()
                _file_mtimes[full_path] = current_mtime

                return _cached_read_file(full_path, encoding)
            except OSError:
                _cached_read_file.cache_clear()
                if full_path in _file_mtimes:
                    del _file_mtimes[full_path]
                raise
        else:
            with open(full_path, "r", encoding=encoding) as f:
                return f.read()
    except UnicodeDecodeError as e:
        raise OSError(f"Failed to read file: {str(e)}")
    except OSError:
        raise


def render_template(template_str: str, context: Dict[str, Any]) -> str:
    """Render a Jinja2 template with the given context.

    Args:
        template_str: Template string or path to template file
        context: Template variables

    Returns:
        Rendered template string

    Raises:
        ValueError: For template syntax errors or undefined variables
        OSError: For file reading errors
    """
    with ProgressContext("Rendering template", enabled=True) as progress:
        try:
            # Create environment with comprehensive features
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
            progress.update()

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
                        f"| {' | '.join(str(cell) for cell in row)} |"
                        for row in rows
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
                        f"| {' | '.join(str(cell) for cell in row)} |"
                        for row in rows
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
                        f"| {' | '.join(str(cell) for cell in row)} |"
                        for row in items
                    )
                )

            def process_code(
                text: str, lang: str = "python", format: str = "terminal"
            ) -> str:
                """Process code by removing comments and dedenting.

                Args:
                    text: Code to process
                    lang: Language for syntax highlighting (unused)
                    format: Output format (unused)

                Returns:
                    Processed code
                """
                processed = text
                if processed := env.filters["remove_comments"](processed):
                    processed = env.filters["dedent"](processed)
                    return processed
                return text

            def format_prompt(text: str) -> str:
                """Format prompt text with normalization, wrapping, and indentation."""
                return str(
                    env.filters["normalize"](text)
                    | env.filters["wrap"](text, 80)
                    | env.filters["indent"](text, 4)
                )

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
                    "escape_special": escape_special,
                    "debug_format": debug_format,
                }
            )

            def estimate_tokens(text: str, model: Optional[str] = None) -> int:
                """Estimate tokens using tiktoken."""
                try:
                    if model:
                        encoding = tiktoken.encoding_for_model(model)
                    else:
                        encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
                except Exception:
                    # Fallback to basic estimation only if tiktoken fails
                    return int(len(text.split()) * 1.3)

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

            def count_tokens(text: str, model: Optional[str] = None) -> int:
                """Count tokens using tiktoken."""
                return estimate_tokens(text, model)  # Use same implementation

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
                    # File utilities
                    "read_file": read_file,
                    "process_code": process_code,
                }
            )

            # Create template from string or file
            if template_str.endswith((".j2", ".jinja2", ".md")):
                if not os.path.isfile(template_str):
                    raise OSError(f"Template file not found: {template_str}")
                try:
                    template = env.get_template(template_str)
                except jinja2.TemplateNotFound as e:
                    raise OSError(f"Template file not found: {e.name}")
            else:
                try:
                    template = env.from_string(template_str)
                except jinja2.TemplateSyntaxError as e:
                    raise OSError(f"Template syntax error: {str(e)}")

            # Add debug context
            template.globals["template_name"] = getattr(
                template, "name", "<string>"
            )
            template.globals["template_path"] = getattr(
                template, "filename", None
            )

            # Update progress before final render
            progress.update()

            # Render template with error handling
            try:
                return template.render(**context)
            except jinja2.UndefinedError as e:
                raise OSError(f"Undefined template variable: {str(e)}")
            except jinja2.TemplateRuntimeError as e:
                raise OSError(f"Template rendering error: {str(e)}")
            except jinja2.TemplateError as e:
                raise OSError(f"Template error: {str(e)}")

        except jinja2.TemplateError as e:
            # Catch any other Jinja2 errors
            raise OSError(f"Template error: {str(e)}")


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
            # Jinja2 builtins
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
            "read_file",
            "process_code",
            # Data processing functions
            "sort_by",
            "group_by",
            "filter_by",
            "pluck",
            "unique",
            "frequency",
            "aggregate",
            # Table formatting functions
            "table",
            "align_table",
            "dict_to_table",
            "list_to_table",
            # Content processing functions
            "extract_keywords",
            "word_count",
            "char_count",
            "to_json",
            "from_json",
            "remove_comments",
            "wrap",
            "indent",
            "dedent",
            "normalize",
            "strip_markdown",
            "format_prompt",
            "escape_special",
            "debug_format",
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

    # Read stdin if available
    if not sys.stdin.isatty():
        try:
            file_mappings["stdin"] = sys.stdin.read()
        except OSError as e:
            logger.error(f"Cannot read from stdin: {e}")
            return ExitCode.IO_ERROR

    # Create Jinja2 environment for template parsing
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)

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
    "read_file",
    "ProgressContext",
    "summarize",
    "pivot_table",
]


def summarize(
    data: Sequence[Any], keys: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Generate summary statistics for data fields.

    Args:
        data: Sequence of dictionaries or objects to analyze
        keys: Optional sequence of keys to analyze (default: all keys)

    Returns:
        Dictionary containing:
        - total_records: Total number of records
        - fields: Per-field statistics including type, unique values, null count,
                 min/max (for numeric fields), and most common values

    Raises:
        ValueError: If data is empty or keys are invalid
    """
    if not data:
        return {"total_records": 0, "fields": {}}

    def get_field_value(item: Any, field: str) -> Any:
        """Extract field value from dict or object safely."""
        try:
            if isinstance(item, dict):
                return item.get(field)
            return getattr(item, field, None)
        except Exception:
            return None

    def get_field_type(values: List[Any]) -> str:
        """Determine field type from non-null values."""
        non_null = [v for v in values if v is not None]
        if not non_null:
            return "NoneType"

        # Check if all values are of the same type
        types = {type(v) for v in non_null}
        if len(types) == 1:
            return next(iter(types)).__name__

        # Handle mixed numeric types
        if all(isinstance(v, (int, float)) for v in non_null):
            return "number"

        # Default to most specific common ancestor type
        return "mixed"

    def analyze_field(field: str) -> Dict[str, Any]:
        """Generate comprehensive field statistics."""
        values = [get_field_value(x, field) for x in data]
        non_null_values = [v for v in values if v is not None]

        stats = {
            "type": get_field_type(values),
            "unique_values": len(set(non_null_values)),
            "null_count": len(values) - len(non_null_values),
            "total_count": len(values),
        }

        # Add numeric statistics if applicable
        if stats["type"] in ("int", "float", "number"):
            try:
                numeric_values = [float(v) for v in non_null_values]
                stats.update(
                    {
                        "min": min(numeric_values) if numeric_values else None,
                        "max": max(numeric_values) if numeric_values else None,
                        "mean": (
                            sum(numeric_values) / len(numeric_values)
                            if numeric_values
                            else None
                        ),
                    }
                )
            except (ValueError, TypeError):
                pass

        # Add most common values (up to 5)
        if non_null_values:
            from collections import Counter

            most_common = Counter(non_null_values).most_common(5)
            stats["most_common"] = [
                {"value": v, "count": c} for v, c in most_common
            ]

        return stats

    try:
        # Determine available keys if not provided
        available_keys = keys or (
            list(data[0].keys())
            if isinstance(data[0], dict)
            else [k for k in dir(data[0]) if not k.startswith("_")]
        )

        if not available_keys:
            raise ValueError("No valid keys found in data")

        return {
            "total_records": len(data),
            "fields": {key: analyze_field(key) for key in available_keys},
        }
    except Exception as e:
        raise ValueError(f"Failed to analyze data: {str(e)}")


def pivot_table(
    data: Sequence[Dict[str, Any]],
    index: str,
    value: str,
    aggfunc: str = "sum",
) -> Dict[str, Dict[str, Any]]:
    """Create a pivot table from data with specified index and value columns.

    Args:
        data: List of dictionaries containing the data
        index: Column to use as index
        value: Column to aggregate
        aggfunc: Aggregation function (sum, mean, count)

    Returns:
        Dictionary containing aggregated results and metadata
    """
    if not data:
        return {
            "aggregates": {},
            "metadata": {"total_records": 0, "null_index_count": 0},
        }

    # Count records with null index
    null_index_count = sum(1 for row in data if row.get(index) is None)

    # Group by index
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in data:
        idx = row.get(index)
        if idx is not None:
            idx_str = str(idx)
            if idx_str not in groups:
                groups[idx_str] = []
            groups[idx_str].append(row)

    # Aggregate values
    result: Dict[str, Dict[str, Any]] = {"aggregates": {}, "metadata": {}}

    for idx, group in groups.items():
        values = [
            float(row[value]) for row in group if row.get(value) is not None
        ]
        if not values:
            continue

        if aggfunc == "sum":
            result["aggregates"][idx] = {"value": sum(values)}
        elif aggfunc == "mean":
            result["aggregates"][idx] = {"value": sum(values) / len(values)}
        elif aggfunc == "count":
            result["aggregates"][idx] = {"count": len(values)}
        else:
            raise ValueError(f"Invalid aggfunc: {aggfunc}")

    result["metadata"] = {
        "total_records": len(data),
        "null_index_count": null_index_count,
    }

    return result


def get_template_variables(template: Union[str, Template]) -> Set[str]:
    """Extract all variable names from a Jinja2 template.

    Args:
        template: Either a string template or a Jinja2 Template object

    Returns:
        Set of variable names used in the template
    """
    # Always parse the template string to avoid issues with Template objects
    if isinstance(template, str):
        template_str = template
    else:
        # Fallback if ".source" is not available, converting the Template to a string
        template_str = getattr(template, "source", str(template))

    env = Environment()
    parsed_content = env.parse(template_str)
    variables = meta.find_undeclared_variables(parsed_content)
    return variables
