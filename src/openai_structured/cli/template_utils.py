"""Template utilities for the CLI."""

# Standard library imports
import datetime
import itertools
import json
import logging
import os
import re
import textwrap
from collections import Counter
from functools import lru_cache
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Third-party imports
import jinja2
import tiktoken
import yaml
from jinja2 import Environment, Template, meta
from typing_extensions import TypedDict

# Optional dependencies
try:
    from jsonschema import Draft7Validator, SchemaError

    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

try:
    import pygments
    from pygments.formatters import (
        HtmlFormatter,
        NullFormatter,
        TerminalFormatter,
    )
    from pygments.lexers import TextLexer, get_lexer_by_name

    HAVE_PYGMENTS = True
except ImportError:
    HAVE_PYGMENTS = False

# Local imports
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


def _format_markdown_table_row(cells: Sequence[Any]) -> str:
    """Format a row of cells as a markdown table row."""
    return f"| {' | '.join(str(cell) for cell in cells)} |"


def _format_markdown_table_separator(
    widths: Sequence[int], alignments: Optional[Sequence[str]] = None
) -> str:
    """Format markdown table separator row with optional alignments."""
    if not alignments:
        return f"| {' | '.join('-' * max(w, 3) for w in widths)} |"

    markers = []
    for w, a in zip(widths, alignments):
        if a == "center":
            markers.append(":---:")
        elif a == "right":
            markers.append("---:")
        else:  # left or default
            markers.append(":---")
    return f"| {' | '.join(markers)} |"


def strip_comments(text: str, lang: str = "python") -> str:
    """Remove comments from code text based on language.

    Args:
        text: Code text to process
        lang: Programming language

    Returns:
        Text with comments removed if language is supported,
        otherwise returns original text with a warning
    """
    # Define comment patterns for different languages
    single_line_comments = {
        "python": "#",
        "javascript": "//",
        "typescript": "//",
        "java": "//",
        "c": "//",
        "cpp": "//",
        "go": "//",
        "rust": "//",
        "swift": "//",
        "ruby": "#",
        "perl": "#",
        "shell": "#",
        "bash": "#",
        "php": "//",
    }

    multi_line_comments = {
        "javascript": ("/*", "*/"),
        "typescript": ("/*", "*/"),
        "java": ("/*", "*/"),
        "c": ("/*", "*/"),
        "cpp": ("/*", "*/"),
        "go": ("/*", "*/"),
        "rust": ("/*", "*/"),
        "swift": ("/*", "*/"),
        "php": ("/*", "*/"),
    }

    # Return original text if language is not supported
    if lang not in single_line_comments and lang not in multi_line_comments:
        logger.debug(
            f"Language '{lang}' is not supported for comment removal. "
            f"Comments will be preserved in the output."
        )
        return text

    lines = text.splitlines()
    cleaned_lines = []

    # Handle single-line comments
    if lang in single_line_comments:
        comment_char = single_line_comments[lang]
        for line in lines:
            # Remove inline comments
            line = re.sub(f"\\s*{re.escape(comment_char)}.*$", "", line)
            # Keep non-empty lines
            if line.strip():
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

    # Handle multi-line comments
    if lang in multi_line_comments:
        start, end = multi_line_comments[lang]
        # Remove multi-line comments
        text = re.sub(
            f"{re.escape(start)}.*?{re.escape(end)}", "", text, flags=re.DOTALL
        )

    return text


def format_json(obj: Any, indent: int = 2) -> str:
    """Format JSON with indentation.

    Args:
        obj: Object to format as JSON
        indent: Number of spaces for indentation

    Returns:
        Formatted JSON string
    """
    return json.dumps(obj, indent=indent, default=str)


def format_table(
    headers: Sequence[Any],
    rows: Sequence[Sequence[Any]],
    alignments: Optional[Sequence[str]] = None,
) -> str:
    """Format data as a markdown table with optional column alignments."""
    header_widths = [max(3, len(str(h))) for h in headers]
    header_row = _format_markdown_table_row(headers)
    separator_row = _format_markdown_table_separator(header_widths, alignments)
    data_rows = [_format_markdown_table_row(row) for row in rows]
    return "\n".join([header_row, separator_row] + data_rows)


def dict_to_table(data: Dict[Any, Any]) -> str:
    """Convert a dictionary to a markdown table."""
    headers = ["Key", "Value"]
    rows = [[k, v] for k, v in data.items()]
    return format_table(headers, rows)


def list_to_table(
    items: Sequence[Any], headers: Optional[Sequence[str]] = None
) -> str:
    """Convert a list to a markdown table."""
    if not headers:
        headers = ["#", "Value"]
        rows: list[list[Any]] = [[i + 1, item] for i, item in enumerate(items)]
    else:
        rows = [[*row] for row in items]  # Convert each row to a list
    return format_table(headers, rows)


def format_code(
    text: str,
    lang: str = "python",
    format: str = "terminal",
) -> str:
    """Format and syntax highlight code.

    Args:
        text: Code text to format
        lang: Programming language for syntax highlighting
        format: Output format ('terminal', 'html', or 'plain')

    Returns:
        Formatted code string

    Raises:
        ValueError: If format is not one of 'terminal', 'html', or 'plain'
    """
    if not text.strip():
        logger.debug("Empty text provided to format_code")
        return ""

    if format not in ("terminal", "html", "plain"):
        raise ValueError(
            f"Invalid format: {format}. Must be 'terminal', 'html', or 'plain'"
        )

    if not HAVE_PYGMENTS:
        logger.debug("Pygments not available, returning plain text")
        return text

    # Get lexer
    try:
        lexer = get_lexer_by_name(lang)
    except Exception as e:
        logger.debug(f"Using generic lexer for language '{lang}': {e}")
        lexer = TextLexer()

    # Get formatter
    formatter: Union[
        HtmlFormatter[str], TerminalFormatter[str], NullFormatter[str]
    ]
    if format == "html":
        formatter = HtmlFormatter[str]()
    elif format == "terminal":
        formatter = TerminalFormatter[str]()
    else:
        formatter = NullFormatter[str]()

    # Format code with error handling
    try:
        formatted = pygments.highlight(text, lexer, formatter)
    except Exception as e:
        logger.error(f"Failed to format code: {e}")
        return text

    return formatted


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


def extract_field(items: Sequence[Any], key: str) -> List[Any]:
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
    """Create a pivot table from data with specified index and value columns.

    Args:
        data: Sequence of dictionaries containing the data
        index: Column to use as index
        value: Column to aggregate
        aggfunc: Aggregation function ('sum', 'mean', or 'count')

    Returns:
        Dictionary containing aggregated data and metadata

    Raises:
        ValueError: If aggfunc is invalid or if index/value columns don't exist
    """
    if not data:
        logger.debug("Empty data provided to pivot_table")
        return {
            "aggregates": {},
            "metadata": {"total_records": 0, "null_index_count": 0},
        }

    # Validate aggfunc
    valid_aggfuncs = {"sum", "mean", "count"}
    if aggfunc not in valid_aggfuncs:
        raise ValueError(
            f"Invalid aggfunc: {aggfunc}. Must be one of {valid_aggfuncs}"
        )

    # Validate columns exist in first row
    if data and (index not in data[0] or value not in data[0]):
        missing = []
        if index not in data[0]:
            missing.append(f"index column '{index}'")
        if value not in data[0]:
            missing.append(f"value column '{value}'")
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Count records with null index
    null_index_count = sum(1 for row in data if row.get(index) is None)
    if null_index_count:
        logger.warning(f"Found {null_index_count} rows with null index values")

    # Group by index
    groups: Dict[str, List[float]] = {}
    invalid_values = 0
    for row in data:
        idx = str(row.get(index, ""))
        try:
            val = float(row.get(value, 0))
        except (TypeError, ValueError):
            invalid_values += 1
            logger.warning(
                f"Invalid value for {value} in row with index {idx}, using 0"
            )
            val = 0.0

        if idx not in groups:
            groups[idx] = []
        groups[idx].append(val)

    if invalid_values:
        logger.warning(
            f"Found {invalid_values} invalid values in column {value}"
        )

    result: Dict[str, Dict[str, Any]] = {"aggregates": {}, "metadata": {}}
    for idx, values in groups.items():
        if aggfunc == "sum":
            result["aggregates"][idx] = {"value": sum(values)}
        elif aggfunc == "mean":
            result["aggregates"][idx] = {"value": sum(values) / len(values)}
        else:  # count
            result["aggregates"][idx] = {"value": len(values)}

    result["metadata"] = {
        "total_records": len(data),
        "null_index_count": null_index_count,
        "invalid_values": invalid_values,
    }
    return result


def summarize(
    data: Sequence[Any], keys: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Generate summary statistics for data fields.

    Args:
        data: Sequence of dictionaries or objects to analyze
        keys: Optional sequence of keys/attributes to analyze

    Returns:
        Dictionary containing summary statistics for each field

    Raises:
        ValueError: If data is empty or no valid keys are found
        TypeError: If data items are not dictionaries or objects
    """
    if not data:
        logger.debug("Empty data provided to summarize")
        return {"total_records": 0, "fields": {}}

    # Validate data type
    if not isinstance(data[0], dict) and not hasattr(data[0], "__dict__"):
        raise TypeError("Data items must be dictionaries or objects")

    def get_field_value(item: Any, field: str) -> Any:
        try:
            if isinstance(item, dict):
                return item.get(field)
            return getattr(item, field, None)
        except Exception as e:
            logger.warning(f"Error accessing field {field}: {e}")
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
        logger.debug(f"Analyzing field: {field}")
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
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Error calculating numeric stats for {field}: {e}"
                )

        # Add most common values
        if non_null:
            try:
                most_common = Counter(non_null).most_common(5)
                stats["most_common"] = [
                    {"value": str(v), "count": c} for v, c in most_common
                ]
            except TypeError as e:
                logger.warning(
                    f"Error calculating most common values for {field}: {e}"
                )

        return stats

    try:
        available_keys = keys or (
            list(data[0].keys())
            if isinstance(data[0], dict)
            else [k for k in dir(data[0]) if not k.startswith("_")]
        )

        if not available_keys:
            raise ValueError("No valid keys found in data")

        logger.debug(
            f"Analyzing {len(data)} records with {len(available_keys)} fields"
        )
        result = {
            "total_records": len(data),
            "fields": {k: analyze_field(k) for k in available_keys},
        }
        logger.debug("Analysis complete")
        return result

    except Exception as e:
        logger.error(f"Failed to analyze data: {e}", exc_info=True)
        raise ValueError(f"Failed to analyze data: {str(e)}")


def render_template(
    template_str: str,
    context: Dict[str, Any],
    jinja_env: jinja2.Environment,
    progress_enabled: bool = True,
) -> str:
    """Render a Jinja2 template with the given context.

    Args:
        template_str: Template string or path to template file
        context: Template variables
        jinja_env: Jinja2 environment to use for rendering
        progress_enabled: Whether to show progress indicators

    Returns:
        Rendered template string

    Raises:
        OSError: If template cannot be loaded or rendered
    """
    with ProgressContext(
        "Rendering template", enabled=progress_enabled
    ) as progress:
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

            def escape_special(text: str) -> str:
                return re.sub(r'([{}\[\]"\'\\])', r"\\\1", text)

            def debug_print(x: Any) -> None:
                print(f"DEBUG: {x}")

            def type_of(x: Any) -> str:
                return type(x).__name__

            def dir_of(x: Any) -> List[str]:
                return dir(x)

            def len_of(x: Any) -> Optional[int]:
                return len(x) if hasattr(x, "__len__") else None

            def validate_json(text: str) -> bool:
                if not text:
                    return False
                try:
                    json.loads(text)
                    return True
                except json.JSONDecodeError:
                    return False

            def format_error(e: Exception) -> str:
                return f"{type(e).__name__}: {str(e)}"

            def estimate_tokens(text: str) -> int:
                """Estimate the number of tokens in a text string."""
                try:
                    encoding = tiktoken.encoding_for_model("gpt-4")
                    return len(encoding.encode(str(text)))
                except Exception as e:
                    logger.warning(f"Failed to estimate tokens: {e}")
                    return len(str(text).split())

            def format_json(obj: Any) -> str:
                """Format JSON with indentation."""
                return json.dumps(obj, indent=2, default=str)

            def auto_table(data: Any) -> str:
                """Automatically format data as a table based on its type."""
                if isinstance(data, dict):
                    return dict_to_table(data)
                if isinstance(data, (list, tuple)):
                    return list_to_table(data)
                return str(data)

            # Add custom filters
            jinja_env.filters.update(
                {
                    "extract_keywords": extract_keywords,
                    "word_count": word_count,
                    "char_count": char_count,
                    "to_json": format_json,
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
                    "extract_field": extract_field,
                    "unique": unique,
                    "frequency": frequency,
                    "aggregate": aggregate,
                    # Table formatting filters
                    "table": format_table,
                    "align_table": align_table,
                    "dict_to_table": dict_to_table,
                    "list_to_table": list_to_table,
                    # Code processing filters
                    "format_code": format_code,
                    "strip_comments": strip_comments,
                    # Special character handling
                    "escape_special": escape_special,
                    # Table utilities
                    "auto_table": auto_table,
                    # File utilities
                    "read_file": read_file,
                    # Code utilities
                    "format_code": format_code,
                    "strip_comments": strip_comments,
                }
            )

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
                    "validate_json": validate_json,
                    "format_error": format_error,
                    # Data analysis globals
                    "summarize": summarize,
                    "pivot_table": pivot_table,
                    # Table utilities
                    "auto_table": auto_table,
                    # File utilities
                    "read_file": read_file,
                    # Code utilities
                    "format_code": format_code,
                    "strip_comments": strip_comments,
                }
            )

            # Create template from string or file
            template: Optional[jinja2.Template] = None
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

            if template is None:
                raise OSError("Failed to create template")

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
                # Attempt to render the template
                return template.render(**context)
            except (
                jinja2.UndefinedError,
                jinja2.TemplateRuntimeError,
                jinja2.TemplateError,
            ) as e:
                # Convert all Jinja2 errors to OSError with specific messages
                error_type = type(e).__name__.replace("Template", "")
                raise OSError(f"Template {error_type.lower()}: {str(e)}")
            except Exception as e:
                # Convert any other exceptions to OSError
                if isinstance(e, OSError):
                    raise
                raise OSError(f"Template processing error: {str(e)}")

        except OSError:
            # Re-raise OSError exceptions as is
            raise
        except Exception as e:
            # Convert any other exceptions to OSError
            raise OSError(f"Template processing error: {str(e)}")


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
            "validate_json",
            "format_error",
            # Data analysis functions
            "summarize",
            "pivot_table",
            # Table utilities
            "auto_table",
            # File utilities
            "read_file",
            "format_code",
            # Data processing functions
            "sort_by",
            "group_by",
            "filter_by",
            "extract_field",
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
            "strip_comments",
            "wrap",
            "indent",
            "dedent",
            "normalize",
            "strip_markdown",
            "escape_special",
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

    # Create environment that allows undefined filters
    env = Environment(undefined=jinja2.StrictUndefined)
    env.filters["filter"] = lambda x: x  # Add dummy filter

    parsed_content = env.parse(template_str)
    variables = meta.find_undeclared_variables(parsed_content)
    return variables


class TemplateMetadataError(ValueError):
    """Base class for template metadata errors."""

    pass


class SystemPromptError(TemplateMetadataError):
    """Error processing system prompt template."""

    pass


class TemplateMetadata(TypedDict, total=False):
    """Type for template metadata from frontmatter."""

    system_prompt: Optional[str]


def extract_template_metadata(
    template_content: str, context: Dict[str, Any], env: jinja2.Environment
) -> Tuple[TemplateMetadata, str]:
    """Extract and process template metadata from frontmatter.

    Args:
        template_content: The full template content including potential frontmatter
        context: Template context for variable rendering
        env: Jinja2 environment for rendering

    Returns:
        Tuple of (metadata dict, remaining template content)

    Raises:
        TemplateMetadataError: If frontmatter is invalid
    """
    frontmatter_match = re.match(
        r"^---\s*\n(.*?)\n---\s*\n(.*)", template_content, re.DOTALL
    )
    if not frontmatter_match:
        return TemplateMetadata(system_prompt=None), template_content

    try:
        metadata = yaml.safe_load(frontmatter_match.group(1))
        if not isinstance(metadata, dict):
            raise TemplateMetadataError(
                "Frontmatter must be a YAML dictionary"
            )

        return TemplateMetadata(
            system_prompt=metadata.get("system_prompt")
        ), frontmatter_match.group(2)
    except yaml.YAMLError as e:
        raise TemplateMetadataError(f"Invalid YAML frontmatter: {e}")


def extract_metadata(template_content: str) -> Optional[Dict[str, Any]]:
    """Extract YAML frontmatter metadata from template content.

    Args:
        template_content: The template content to parse

    Returns:
        Dictionary containing metadata if found, None otherwise

    Raises:
        ValueError: If YAML frontmatter is invalid
    """
    # Match YAML frontmatter between --- delimiters
    frontmatter_match = re.match(
        r"^---\s*\n(.*?)\n---\s*\n", template_content, re.DOTALL
    )
    if not frontmatter_match:
        return None

    try:
        # Parse YAML content
        metadata = yaml.safe_load(frontmatter_match.group(1))
        if not isinstance(metadata, dict):
            raise ValueError("YAML frontmatter must be a dictionary")
        return metadata
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML frontmatter: {e}")
