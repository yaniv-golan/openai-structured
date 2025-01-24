"""Command-line interface for making structured OpenAI API calls."""

from .cli import (
    ExitCode,
    create_dynamic_model,
    estimate_tokens_for_chat,
    get_context_window_limit,
    get_default_token_limit,
    main,
    parse_json_var,
)

__all__ = [
    "ExitCode",
    "estimate_tokens_for_chat",
    "get_context_window_limit",
    "get_default_token_limit",
    "parse_json_var",
    "create_dynamic_model",
    "main",
]
