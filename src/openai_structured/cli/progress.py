"""Progress reporting utilities for CLI operations."""

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Type


class ProgressContext:
    """Simple context manager for output handling.

    This is a minimal implementation that just handles direct output to stdout.
    No progress reporting is implemented - it simply prints output directly.
    """

    def __init__(
        self,
        description: str = "Processing",
        total: Optional[int] = None,
        level: str = "basic",
        output_file: Optional[str] = None,
    ):
        self._output_file = output_file
        self._level = level
        self.enabled = level != "none"
        self.current: int = 0

    def __enter__(self) -> "ProgressContext":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        pass

    def update(self, amount: int = 1, force: bool = False) -> None:
        """No-op update method kept for compatibility."""
        pass

    def print_output(self, text: str) -> None:
        """Print output to stdout or file.

        Args:
            text: Text to print
        """
        if self._output_file:
            with open(self._output_file, "a", encoding="utf-8") as f:
                f.write(text)
                f.write("\n")
        else:
            print(text, flush=True)

    @contextmanager
    def step(
        self,
        description: Optional[str] = None,
        show_spinner: bool = True,
    ) -> Iterator[None]:
        """No-op step method kept for compatibility."""
        yield
