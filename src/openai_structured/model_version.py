"""Model version information."""

import re
from datetime import date
from functools import total_ordering
from typing import NamedTuple


@total_ordering
class ModelVersion(NamedTuple):
    """Model version information."""

    year: int
    month: int
    day: int

    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        """Create a ModelVersion from a string in YYYY-MM-DD format."""
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")

        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        try:
            date(year, month, day)
        except ValueError as e:
            raise ValueError(f"Invalid date: {version_str} - {str(e)}")

        return cls(year, month, day)

    def __str__(self) -> str:
        """Return string representation in YYYY-MM-DD format."""
        return f"{self.year}-{self.month:02d}-{self.day:02d}"

    def __lt__(self, other: object) -> bool:
        """Compare versions for less than."""
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return (self.year, self.month, self.day) < (
            other.year,
            other.month,
            other.day,
        )

    def __eq__(self, other: object) -> bool:
        """Compare versions for equality."""
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return (self.year, self.month, self.day) == (
            other.year,
            other.month,
            other.day,
        )
