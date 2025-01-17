"""CLI package for structured OpenAI API calls."""

from openai import AsyncOpenAI

from .cli import (
    ExitCode,
    _main,
    cli_main,
    create_dynamic_model,
    estimate_tokens_for_chat,
    get_context_window_limit,
    get_default_token_limit,
    main,
)
from .file_utils import FileInfo, collect_files
from .template_utils import (
    read_file,
    render_template,
    validate_template_placeholders,
)

__all__ = [
    "cli_main",
    "render_template",
    "AsyncOpenAI",
    "ExitCode",
    "main",
    "_main",
    "create_dynamic_model",
    "estimate_tokens_for_chat",
    "get_context_window_limit",
    "get_default_token_limit",
    "read_file",
    "validate_template_placeholders",
    "FileInfo",
    "collect_files",
]
