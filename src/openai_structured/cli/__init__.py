"""CLI package for structured OpenAI API calls."""

from openai import AsyncOpenAI

from .cli import (
    ExitCode,
    ProgressContext,
    create_dynamic_model,
    estimate_tokens_for_chat,
    get_context_window_limit,
    get_default_token_limit,
    main,
    supports_structured_output,
    validate_template_placeholders,
    validate_token_limits,
)
from .template_utils import read_file, render_template

__all__ = [
    "ExitCode",
    "main",
    "create_dynamic_model",
    "validate_template_placeholders",
    "estimate_tokens_for_chat",
    "get_context_window_limit",
    "get_default_token_limit",
    "validate_token_limits",
    "supports_structured_output",
    "ProgressContext",
    "render_template",
    "AsyncOpenAI",
    "read_file",
]
