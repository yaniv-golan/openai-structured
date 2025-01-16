"""Template utilities for the CLI."""

import datetime
import itertools
import json
import logging
import os
import re
import textwrap
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set, TypeVar, Union

import jinja2
import tiktoken
from jinja2 import Environment, Template, meta

# Make jsonschema optional
try:
    from jsonschema import Draft7Validator, SchemaError

    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

from .progress import ProgressContext

logger = logging.getLogger(__name__)


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


_file_mtimes: Dict[str, float] = {}

T = TypeVar("T")


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
                if (
                    full_path not in _file_mtimes
                    or current_mtime > _file_mtimes[full_path]
                ):
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


def frequency(items: Sequence[T]) -> Dict[T, int]:
    """Count frequency of items in a sequence."""
    from collections import Counter

    return dict(Counter(items))


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


def sort_by(items: Sequence[T], key: str) -> List[T]:
    """Sort items by a key, handling both dict and object attributes."""

    def get_key(x: T) -> Any:
        val = x.get(key) if isinstance(x, dict) else getattr(x, key)
        return 0 if val is None else val

    return sorted(items, key=get_key)


def unique(items: Sequence[Any]) -> List[Any]:
    """Get unique values while preserving order."""
    return list(dict.fromkeys(items))


def pluck(items: Sequence[Any], key: str) -> List[Any]:
    """Extract a specific field from each item."""
    return [
        x.get(key) if isinstance(x, dict) else getattr(x, key) for x in items
    ]


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


def pivot_table(
    data: Sequence[Dict[str, Any]],
    index: str,
    value: str,
    aggfunc: str = "sum",
) -> Dict[str, Dict[str, Any]]:
    """Create a pivot table from data with specified index and value columns."""
    if not data:
        return {
            "aggregates": {},
            "metadata": {"total_records": 0, "null_index_count": 0},
        }

    # Count records with null index
    null_index_count = sum(1 for row in data if row.get(index) is None)

    # Group by index
    groups: Dict[str, List[float]] = {}
    for row in data:
        idx = str(row.get(index, ""))
        val = float(row.get(value, 0))
        if idx not in groups:
            groups[idx] = []
        groups[idx].append(val)

    result: Dict[str, Dict[str, Any]] = {"aggregates": {}, "metadata": {}}
    for idx, values in groups.items():
        if aggfunc == "sum":
            result["aggregates"][idx] = {"value": sum(values)}
        elif aggfunc == "mean":
            result["aggregates"][idx] = {"value": sum(values) / len(values)}
        elif aggfunc == "count":
            result["aggregates"][idx] = {"value": len(values)}
        else:
            raise ValueError(f"Invalid aggfunc: {aggfunc}")

    result["metadata"] = {
        "total_records": len(data),
        "null_index_count": null_index_count,
    }
    return result


def summarize(
    data: Sequence[Any], keys: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Generate summary statistics for data fields."""
    if not data:
        return {"total_records": 0, "fields": {}}

    def get_field_value(item: Any, field: str) -> Any:
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
        values = [get_field_value(x, field) for x in data]
        non_null = [v for v in values if v is not None]
        stats = {
            "type": get_field_type(values),
            "total": len(values),
            "null_count": len(values) - len(non_null),
            "unique": len(set(non_null)),
        }

        # Add numeric statistics if applicable
        if stats["type"] in ("int", "float", "number"):
            try:
                nums = [float(x) for x in non_null]
                stats.update(
                    {
                        "min": min(nums) if nums else None,
                        "max": max(nums) if nums else None,
                        "avg": sum(nums) / len(nums) if nums else None,
                    }
                )
            except (ValueError, TypeError):
                pass

        # Add most common values
        if non_null:
            from collections import Counter

            most_common = Counter(non_null).most_common(5)
            stats["most_common"] = [
                {"value": v, "count": c} for v, c in most_common
            ]

        return stats

    try:
        available_keys = keys or (
            list(data[0].keys())
            if isinstance(data[0], dict)
            else [k for k in dir(data[0]) if not k.startswith("_")]
        )

        if not available_keys:
            raise ValueError("No valid keys found in data")

        return {
            "total_records": len(data),
            "fields": {k: analyze_field(k) for k in available_keys},
        }
    except Exception as e:
        raise ValueError(f"Failed to analyze data: {str(e)}")


def render_template(
    template_str: str,
    context: Dict[str, Any],
    jinja_env: jinja2.Environment,
) -> str:
    """Render a Jinja2 template with the given context.

    Args:
        template_str: Template string or path to template file
        context: Template variables
        jinja_env: Jinja2 environment to use for rendering

    Returns:
        Rendered template string
    """
    with ProgressContext("Rendering template", enabled=True) as progress:
        try:
            if progress:
                task = progress.add_task("Setting up environment")
                progress.update(task, advance=1)

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
                if processed := jinja_env.filters["remove_comments"](
                    processed
                ):
                    processed = jinja_env.filters["dedent"](processed)
                    return processed
                return text

            def format_prompt(text: str) -> str:
                """Format prompt text with normalization, wrapping, and indentation."""
                return str(
                    jinja_env.filters["normalize"](text)
                    | jinja_env.filters["wrap"](text, 80)
                    | jinja_env.filters["indent"](text, 4)
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
                    return str(jinja_env.filters["dict_to_table"](data))
                if isinstance(data, (list, tuple)):
                    return str(jinja_env.filters["list_to_table"](data))
                return str(data)

            # Add custom filters
            jinja_env.filters.update(
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
                return jinja_env.from_string(template).render(**kwargs)

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
            jinja_env.globals.update(
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
                    template = jinja_env.get_template(template_str)
                except jinja2.TemplateNotFound as e:
                    raise OSError(f"Template file not found: {e.name}")
            else:
                try:
                    template = jinja_env.from_string(template_str)
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
            if progress:
                task = progress.add_task("Rendering template")
                progress.update(task, advance=1)

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
