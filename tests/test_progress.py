"""Tests for output handling utilities."""

import io
from unittest.mock import patch

import pytest

from openai_structured.cli.progress import ProgressContext


def test_stdout_output() -> None:
    """Test direct output to stdout."""
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        with ProgressContext() as progress:
            progress.print_output("test message")
        assert fake_out.getvalue() == "test message\n"


def test_file_output() -> None:
    """Test output to file."""
    with patch("builtins.open", mock_open := io.StringIO()) as fake_file:
        with ProgressContext(output_file="test.txt") as progress:
            progress.print_output("test message")
        assert mock_open.getvalue() == "test message\n"


def test_compatibility_methods() -> None:
    """Test that compatibility methods don't raise errors."""
    with ProgressContext() as progress:
        # These should be no-ops but shouldn't raise errors
        progress.update()
        with progress.step("test"):
            pass
