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
from typing import Any, Dict, List, Optional, Union

import jinja2
from jinja2 import Environment

from . import template_filters
from .file_utils import FileInfo
from .template_extensions import CommentExtension
from .template_schema import DotDict, StdinProxy
from .template_env import create_jinja_env

__all__ = ["render_template", "DotDict"]

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
    None,
]

def render_template(
    template_str: str,
    context: Dict[str, Any],
    jinja_env: Optional[Environment] = None,
    progress_enabled: bool = True,
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
    from .progress import (  # Import here to avoid circular dependency
        ProgressContext,
    )

    with ProgressContext(
        "Rendering task template", show_progress=progress_enabled
    ) as progress:
        try:
            if progress:
                progress.update(1)  # Update progress for setup

            if jinja_env is None:
                jinja_env = create_jinja_env(loader=jinja2.FileSystemLoader("."))

            # Wrap JSON variables in DotDict and handle special cases
            wrapped_context: Dict[str, TemplateContextValue] = {}
            for key, value in context.items():
                if isinstance(value, dict):
                    wrapped_context[key] = DotDict(value)
                else:
                    wrapped_context[key] = value

            # Add stdin only if not already in context
            if "stdin" not in wrapped_context:
                wrapped_context["stdin"] = StdinProxy()

            # Load file content for FileInfo objects
            for key, value in context.items():
                if isinstance(value, FileInfo):
                    # Access content property to trigger loading
                    _ = value.content
                elif (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], FileInfo)
                ):
                    for file_info in value:
                        # Access content property to trigger loading
                        _ = file_info.content

            if progress:
                progress.update(1)  # Update progress for template creation

            # Create template from string or file
            template: Optional[jinja2.Template] = None
            if template_str.endswith((".j2", ".jinja2", ".md")):
                if not os.path.isfile(template_str):
                    raise ValueError(
                        f"Task template file not found: {template_str}"
                    )
                try:
                    template = jinja_env.get_template(template_str)
                except jinja2.TemplateNotFound as e:
                    raise ValueError(
                        f"Task template file not found: {e.name}"
                    ) from e
            else:
                try:
                    template = jinja_env.from_string(template_str)
                except jinja2.TemplateSyntaxError as e:
                    raise ValueError(
                        f"Task template syntax error: {str(e)}"
                    ) from e

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
