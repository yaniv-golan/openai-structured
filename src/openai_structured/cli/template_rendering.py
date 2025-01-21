"""Template rendering with Jinja2.

This module provides functionality for rendering Jinja2 templates with support for:
1. Custom filters and functions
2. Dot notation access for dictionaries
3. Error handling and reporting

Key Components:
    - render_template: Main rendering function
    - DotDict: Dictionary wrapper for dot notation access
    - Custom filters for code formatting and data manipulation

Examples:
    Basic template rendering:
    >>> template = "Hello {{ name }}!"
    >>> context = {'name': 'World'}
    >>> result = render_template(template, context)
    >>> print(result)
    Hello World!

    Dictionary access with dot notation:
    >>> template = '''
    ... Debug: {{ config.debug }}
    ... Mode: {{ config.settings.mode }}
    ... '''
    >>> config = {
    ...     'debug': True,
    ...     'settings': {'mode': 'test'}
    ... }
    >>> result = render_template(template, {'config': config})
    >>> print(result)
    Debug: True
    Mode: test

    Using custom filters:
    >>> template = '''
    ... {{ code | format_code('python') }}
    ... {{ data | dict_to_table }}
    ... '''
    >>> context = {
    ...     'code': 'def hello(): print("Hello")',
    ...     'data': {'name': 'test', 'value': 42}
    ... }
    >>> result = render_template(template, context)

    File content rendering:
    >>> template = "Content: {{ file.content }}"
    >>> context = {'file': FileInfo('test.txt')}
    >>> result = render_template(template, context)

Notes:
    - All dictionaries are wrapped in DotDict for dot notation access
    - Custom filters are registered automatically
    - Provides detailed error messages for rendering failures
"""

import datetime
import logging
import os
import sys
from typing import Any, Dict, Optional, Union, List, cast, Tuple, TypeVar

import jinja2
from jinja2 import Environment

from .file_utils import FileInfo
from . import template_filters
from .template_schema import StdinProxy, DotDict

__all__ = ['render_template', 'DotDict', 'create_jinja_env']

logger = logging.getLogger(__name__)

# Type alias for values that can appear in the template context
TemplateContextValue = Union[
    DotDict,
    StdinProxy, 
    FileInfo,
    List[Union[FileInfo, Any]],  # For file lists
    str,
    int,
    float,
    bool,
    None
]

def create_jinja_env(env: Optional[Environment] = None) -> Environment:
    """Create and configure a Jinja2 environment with custom filters and globals."""
    if env is None:
        env = Environment(
            undefined=jinja2.StrictUndefined,
            autoescape=True
        )

    # Add template filters
    env.filters.update({
        'extract_keywords': template_filters.extract_keywords,
        'word_count': template_filters.word_count,
        'char_count': template_filters.char_count,
        'to_json': template_filters.format_json,
        'from_json': template_filters.from_json,
        'remove_comments': template_filters.remove_comments,
        'wrap': template_filters.wrap_text,
        'indent': template_filters.indent_text,
        'dedent': template_filters.dedent_text,
        'normalize': template_filters.normalize_text,
        'strip_markdown': template_filters.strip_markdown,
        # Data processing filters
        'sort_by': template_filters.sort_by,
        'group_by': template_filters.group_by,
        'filter_by': template_filters.filter_by,
        'extract_field': template_filters.extract_field,
        'unique': template_filters.unique,
        'frequency': template_filters.frequency,
        'aggregate': template_filters.aggregate,
        # Table formatting filters
        'table': template_filters.format_table,
        'align_table': template_filters.align_table,
        'dict_to_table': template_filters.dict_to_table,
        'list_to_table': template_filters.list_to_table,
        # Code processing filters
        'format_code': template_filters.format_code,
        'strip_comments': template_filters.strip_comments,
        # Special character handling
        'escape_special': template_filters.escape_special,
        # Table utilities
        'auto_table': template_filters.auto_table,
    })

    # Add template globals
    env.globals.update({
        'estimate_tokens': template_filters.estimate_tokens,
        'format_json': template_filters.format_json,
        'now': datetime.datetime.now,
        'debug': template_filters.debug_print,
        'type_of': template_filters.type_of,
        'dir_of': template_filters.dir_of,
        'len_of': template_filters.len_of,
        'validate_json': template_filters.validate_json,
        'format_error': template_filters.format_error,
        # Data analysis globals
        'summarize': template_filters.summarize,
        'pivot_table': template_filters.pivot_table,
        # Table utilities
        'auto_table': template_filters.auto_table,
    })

    return env

def render_template(
    template_str: str,
    context: Dict[str, Any],
    jinja_env: Optional[Environment] = None,
    progress_enabled: bool = True
) -> str:
    """Render a task template with the given context.

    Args:
        template_str: Task template string or path to task template file
        context: Task template variables
        jinja_env: Optional Jinja2 environment to use
        progress_enabled: Whether to show progress indicators

    Returns:
        Rendered task template string

    Raises:
        ValueError: If task template cannot be loaded or rendered. The original error
                  will be chained using `from` for proper error context.
    """
    from .progress import ProgressContext  # Import here to avoid circular dependency

    with ProgressContext(
        "Rendering task template", show_progress=progress_enabled
    ) as progress:
        try:
            if progress:
                progress.update(1)  # Update progress for setup

            if jinja_env is None:
                jinja_env = create_jinja_env()

            # Wrap JSON variables in DotDict and handle special cases
            wrapped_context: Dict[str, TemplateContextValue] = {}
            for key, value in context.items():
                if isinstance(value, dict):
                    wrapped_context[key] = DotDict(value)
                else:
                    wrapped_context[key] = value
            
            # Add stdin only if not already in context
            if 'stdin' not in wrapped_context:
                wrapped_context['stdin'] = StdinProxy()

            # Load file content for FileInfo objects
            for key, value in context.items():
                if isinstance(value, FileInfo):
                    value.load_content()
                elif isinstance(value, list) and value and isinstance(value[0], FileInfo):
                    for file_info in value:
                        file_info.load_content()

            if progress:
                progress.update(1)  # Update progress for template creation

            # Create template from string or file
            template: Optional[jinja2.Template] = None
            if template_str.endswith((".j2", ".jinja2", ".md")):
                if not os.path.isfile(template_str):
                    raise ValueError(f"Task template file not found: {template_str}")
                try:
                    template = jinja_env.get_template(template_str)
                except jinja2.TemplateNotFound as e:
                    raise ValueError(f"Task template file not found: {e.name}") from e
            else:
                try:
                    template = jinja_env.from_string(template_str)
                except jinja2.TemplateSyntaxError as e:
                    raise ValueError(f"Task template syntax error: {str(e)}") from e

            if template is None:
                raise ValueError("Failed to create task template")
            assert template is not None  # Help mypy understand control flow

            # Add template globals
            template.globals["template_name"] = getattr(
                template, "name", "<string>"
            )
            template.globals["template_path"] = getattr(
                template, "filename", None
            )

            try:
                # Attempt to render the template
                result = template.render(**wrapped_context)
                if progress:
                    progress.update(1)  # Update progress for successful render
                return result
            except (jinja2.TemplateError, Exception) as e:
                # Convert all errors to ValueError with proper context
                raise ValueError(f"Template rendering failed: {str(e)}") from e

        except ValueError as e:
            # Re-raise with original context
            raise e