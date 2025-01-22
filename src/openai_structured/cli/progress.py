"""Progress reporting utilities for CLI operations."""

import sys
from contextlib import contextmanager
from types import TracebackType
from typing import Iterator, Optional, Type

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

try:
    # Just check if rich is available
    import rich  # noqa: F401

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ProgressContext:
    """Context manager for showing progress indicators.

    Attributes:
        description: Description of the operation
        total: Total number of items (optional)
        enabled: Whether progress reporting is enabled
        current: Current progress count
    """

    def __init__(
        self,
        description: str = "Processing",
        total: Optional[int] = None,
        show_progress: bool = True,
        output_file: Optional[str] = None,
    ):
        self.description = description
        self.total = total
        self.enabled = show_progress and sys.stdout.isatty() and RICH_AVAILABLE
        self.current = 0
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None
        self._output_file = output_file

    def __enter__(self) -> "ProgressContext":
        """Start progress reporting."""
        if self.enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=Console(force_terminal=True),
                transient=True,
                expand=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.description,
                total=self.total if self.total is not None else None,
            )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stop progress reporting."""
        if self.enabled and self._progress:
            if exc_type is None:
                # Complete the progress bar
                if self._task_id is not None:
                    self._progress.update(
                        self._task_id, completed=self.total or self.current
                    )
            self._progress.stop()

    def update(self, amount: int = 1) -> None:
        """Update progress by the specified amount."""
        if self.enabled and self._progress and self._task_id is not None:
            self.current += amount
            self._progress.update(self._task_id, advance=amount)

    @contextmanager
    def step(self, description: Optional[str] = None) -> Iterator[None]:
        """Context manager for a single step in the progress.

        Args:
            description: Optional description for this step
        """
        if (
            description
            and self.enabled
            and self._progress
            and self._task_id is not None
        ):
            old_description = self._progress.tasks[self._task_id].description
            self._progress.update(
                self._task_id, description=f"{old_description} • {description}"
            )
        try:
            yield
        finally:
            if self.enabled and self._progress and self._task_id is not None:
                self.update()
                if description:
                    self._progress.update(
                        self._task_id, description=self.description
                    )
