"""Progress indicator utilities."""

from contextlib import contextmanager
from typing import Generator, Optional

try:
    from rich.progress import Progress, SpinnerColumn, TextColumn

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@contextmanager
def ProgressContext(
    description: str, enabled: bool = True
) -> Generator[Optional[Progress], None, None]:
    """Context manager for progress indicators.

    Args:
        description: Progress description
        enabled: Whether to show progress (default: True)
    """
    if not enabled or not RICH_AVAILABLE:
        yield None
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=None)
        yield progress
        progress.update(task, completed=True)
