"""Template validation using Jinja2.

This module provides functionality for validating Jinja2 templates, ensuring that:
1. All variables used in templates exist in the context
2. Nested attribute/key access is valid according to schema
3. Loop variables and filters are used correctly

Key Components:
    - SafeUndefined: Custom undefined type for validation
    - validate_template_placeholders: Main validation function

Examples:
    Basic template validation:
    >>> template = "Hello {{ name }}, your score is {{ results.score }}"
    >>> template_context = {'name': 'value', 'results': {'score': 100}}
    >>> validate_template_placeholders(template, template_context)  # OK
    >>>
    >>> # Missing variable raises ValueError:
    >>> template = "Hello {{ missing }}"
    >>> validate_template_placeholders(template, {'name'})  # Raises ValueError

    Nested attribute validation:
    >>> template = '''
    ... Debug mode: {{ config.debug }}
    ... Settings: {{ config.settings.mode }}
    ... Invalid: {{ config.invalid }}
    ... '''
    >>> validate_template_placeholders(template, {'config'})  # Raises ValueError

    Loop variable validation:
    >>> template = '''
    ... {% for item in items %}
    ...   - {{ item.name }}: {{ item.value }}
    ...   Invalid: {{ item.invalid }}
    ... {% endfor %}
    ... '''
    >>> validate_template_placeholders(template, {'items'})  # Raises ValueError

    Filter validation:
    >>> template = '''
    ... {{ code | format_code }}  {# OK - valid filter #}
    ... {{ data | invalid_filter }}  {# Raises ValueError #}
    ... '''
    >>> validate_template_placeholders(template, {'code', 'data'})

Notes:
    - Uses Jinja2's meta API to find undeclared variables
    - Supports custom filters through safe wrappers
    - Provides detailed error messages for validation failures
"""

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
    cast,
)

import jinja2
from jinja2 import Environment, meta
from jinja2.nodes import For, Name, Node

from . import template_filters
from .template_extensions import CommentExtension
from .template_schema import (
    DictProxy,
    FileInfoProxy,
    ValidationProxy,
    create_validation_context,
)

T = TypeVar("T")
FilterFunc = Callable[..., Any]
FilterWrapper = Callable[[Any, Any, Any], Optional[Union[Any, str, List[Any]]]]


class SafeUndefined(jinja2.StrictUndefined):
    """A strict Undefined class that allows any attribute access during validation."""

    def __getattr__(self, name: str) -> Any:
        # Allow any attribute access during validation
        return self

    def __getitem__(self, key: Any) -> Any:
        # Allow any key access during validation
        return self


def safe_filter(func: FilterFunc) -> FilterWrapper:
    """Wrap a filter function to handle None and proxy values safely."""

    def wrapper(
        value: Any, *args: Any, **kwargs: Any
    ) -> Optional[Union[Any, str, List[Any]]]:
        if value is None:
            return None
        if isinstance(value, (ValidationProxy, FileInfoProxy, DictProxy)):
            # For validation, just return an empty result of the appropriate type
            if func.__name__ in (
                "extract_field",
                "frequency",
                "aggregate",
                "pivot_table",
                "summarize",
            ):
                return []
            elif func.__name__ in ("dict_to_table", "list_to_table"):
                return ""
            return value
        return func(value, *args, **kwargs)

    return wrapper


def find_loop_vars(nodes: List[Node]) -> Set[str]:
    """Find variables used in loop constructs."""
    loop_vars = set()
    for node in nodes:
        if isinstance(node, For):
            target = node.target
            if isinstance(target, Name):
                loop_vars.add(target.name)
            elif hasattr(target, "items"):
                items = cast(List[Name], target.items)
                for item in items:
                    loop_vars.add(item.name)
        if hasattr(node, "body"):
            loop_vars.update(find_loop_vars(cast(List[Node], node.body)))
        if hasattr(node, "else_"):
            loop_vars.update(find_loop_vars(cast(List[Node], node.else_)))
    return loop_vars


def validate_template_placeholders(
    task_template: str,
    template_context: Dict[str, Any],
    env: Optional[Environment] = None,
) -> None:
    """Validate that a task template only uses available variables.

    This function checks that all variables used in the template exist in the
    template context and that any nested attribute access is valid.

    Args:
        task_template: The template string to validate
        template_context: Dictionary mapping variable names to their values
        env: Optional Jinja2 Environment to use for validation

    Raises:
        ValueError: If the template uses undefined variables or has invalid syntax
    """
    logger = logging.getLogger(__name__)

    logger.debug("=== validate_template_placeholders called ===")
    logger.debug(
        "Args: task_template=%s, template_context=%s",
        task_template,
        {k: type(v).__name__ for k, v in template_context.items()},
    )

    try:
        # 1) Create Jinja2 environment with meta extension and safe undefined
        if env is None:
            env = jinja2.Environment(
                extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", CommentExtension],
                undefined=SafeUndefined,
            )

        # Register custom filters with None-safe wrappers
        env.filters.update(
            {
                "format_code": safe_filter(template_filters.format_code),
                "strip_comments": safe_filter(template_filters.strip_comments),
                "extract_field": safe_filter(template_filters.extract_field),
                "frequency": safe_filter(template_filters.frequency),
                "aggregate": safe_filter(template_filters.aggregate),
                "pivot_table": safe_filter(template_filters.pivot_table),
                "summarize": safe_filter(template_filters.summarize),
                "dict_to_table": safe_filter(template_filters.dict_to_table),
                "list_to_table": safe_filter(template_filters.list_to_table),
            }
        )

        # Add built-in Jinja2 functions and filters
        builtin_vars = {
            # Core functions
            "range",
            "dict",
            "lipsum",
            "cycler",
            "joiner",
            "namespace",
            "loop",
            "super",
            "self",
            "varargs",
            "kwargs",
            "items",
            # String filters
            "upper",
            "lower",
            "title",
            "capitalize",
            "trim",
            "strip",
            "lstrip",
            "rstrip",
            "center",
            "ljust",
            "rjust",
            "wordcount",
            "truncate",
            "striptags",
            # List filters
            "first",
            "last",
            "length",
            "max",
            "min",
            "sum",
            "sort",
            "unique",
            "reverse",
            "reject",
            "select",
            "map",
            "join",
            "count",
            # Type conversion
            "abs",
            "round",
            "int",
            "float",
            "string",
            "list",
            "bool",
            "batch",
            "slice",
            "attr",
            # Common filters
            "default",
            "replace",
            "safe",
            "urlencode",
            "indent",
            "format",
            "escape",
            "e",
            "nl2br",
            "urlize",
            # Dictionary operations
            "items",
            "keys",
            "values",
            "dictsort",
            # Math operations
            "round",
            "ceil",
            "floor",
            "abs",
            "max",
            "min",
            # Date/time
            "now",
            "strftime",
            "strptime",
            "datetimeformat",
        }
        builtin_vars.update(env.filters.keys())
        builtin_vars.update(env.globals.keys())

        # 2) Parse template and find variables
        ast = env.parse(task_template)
        available_vars = set(template_context.keys())
        logger.debug("Available variables: %s", available_vars)

        # Find loop variables
        loop_vars = find_loop_vars(ast.body)
        logger.debug("Found loop variables: %s", loop_vars)

        # Find all undeclared variables using jinja2.meta
        found_vars = meta.find_undeclared_variables(ast)
        logger.debug("Found undeclared variables: %s", found_vars)

        # Check for missing variables
        missing = {
            name
            for name in found_vars
            if name not in available_vars
            and name not in builtin_vars
            and name not in loop_vars
            and name
            != "stdin"  # Special case for stdin which may be added dynamically
        }

        if missing:
            raise ValueError(
                f"Task template uses undefined variables: {', '.join(sorted(missing))}. "
                f"Available variables: {', '.join(sorted(available_vars))}"
            )

        logger.debug(
            "Before create_validation_context - available_vars type: %s, value: %s",
            type(available_vars),
            available_vars,
        )
        logger.debug(
            "Before create_validation_context - template_context type: %s, value: %s",
            type(template_context),
            template_context,
        )

        # 3) Create validation context with proxy objects
        logger.debug(
            "Creating validation context with template_context: %s",
            template_context,
        )
        validation_context = create_validation_context(template_context)

        # 4) Try to render with validation context
        try:
            env.from_string(task_template).render(validation_context)
        except jinja2.UndefinedError as e:
            # Convert Jinja2 undefined errors to our own ValueError
            raise ValueError(str(e))
        except Exception as e:
            logger.error(
                "Unexpected error during template validation: %s", str(e)
            )
            raise

    except jinja2.TemplateSyntaxError as e:
        # Convert Jinja2 syntax errors to our own ValueError
        raise ValueError(f"Invalid task template syntax: {str(e)}")
    except Exception as e:
        logger.error("Unexpected error during template validation: %s", str(e))
        raise
