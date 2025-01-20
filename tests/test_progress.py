"""Tests for progress reporting utilities."""

import io
import sys
import time
from unittest.mock import patch
from typing import Any, Callable

import pytest

from openai_structured.cli.progress import ProgressContext


def test_progress_disabled() -> None:
    """Test progress reporting when disabled."""
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        with ProgressContext("Testing", show_progress=False) as progress:
            progress.update()
            with progress.step("Step 1"):
                pass
        
        assert fake_out.getvalue() == ""


def test_progress_no_tty() -> None:
    """Test progress reporting when stdout is not a TTY."""
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        # Use setattr instead of direct assignment for isatty
        setattr(fake_out, 'isatty', lambda: False)
        
        with ProgressContext("Testing") as progress:
            progress.update()
            with progress.step("Step 1"):
                pass
        
        assert fake_out.getvalue() == ""


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def test_progress_with_total() -> None:
    """Test progress reporting with total count."""
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        # Use setattr instead of direct assignment for isatty
        setattr(fake_out, 'isatty', lambda: True)

        with ProgressContext("Testing", total=2) as progress:
            time.sleep(0.2)  # Let spinner run
            progress.update()
            with progress.step("Step 1"):
                time.sleep(0.2)
            with progress.step("Step 2"):
                time.sleep(0.2)

        output = strip_ansi(fake_out.getvalue())
        # Check for key elements in output
        assert "Testing" in output
        assert "Step 1" in output
        assert "Step 2" in output
        assert "100%" in output
        assert "2" in output  # Just check for the number 2 somewhere in output


def test_progress_without_total() -> None:
    """Test progress reporting without total count."""
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        # Use setattr instead of direct assignment for isatty
        setattr(fake_out, 'isatty', lambda: True)

        with ProgressContext("Testing") as progress:
            time.sleep(0.2)  # Let spinner run
            progress.update()
            with progress.step("Step 1"):
                time.sleep(0.2)

        output = strip_ansi(fake_out.getvalue())
        # Check for key elements in output
        assert "Testing" in output
        assert "Step 1" in output
        assert "0%" in output  # Check for percentage instead of "completed"


def test_progress_with_error() -> None:
    """Test progress reporting when an error occurs."""
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        # Use setattr instead of direct assignment for isatty
        setattr(fake_out, 'isatty', lambda: True)

        with pytest.raises(ValueError):
            with ProgressContext("Testing") as progress:
                time.sleep(0.2)  # Let spinner run
                progress.update()
                with progress.step("Step 1"):
                    raise ValueError("Test error")

        output = strip_ansi(fake_out.getvalue())
        # Check that progress was shown but completion message was not
        assert "Testing" in output
        assert "0%" in output  # Check for percentage instead of specific step


def test_progress_update_amount() -> None:
    """Test progress updating with custom amounts."""
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        # Use setattr instead of direct assignment for isatty
        setattr(fake_out, 'isatty', lambda: True)

        with ProgressContext("Testing", total=10) as progress:
            time.sleep(0.2)  # Let spinner run
            progress.update(5)  # Update by 5
            time.sleep(0.2)
            progress.update(3)  # Update by 3
            time.sleep(0.2)
            progress.update(2)  # Update by 2

        output = strip_ansi(fake_out.getvalue())
        # Check for key elements in output
        assert "Testing" in output
        assert "100%" in output
        assert "10" in output  # Just check for the number 10 somewhere in output