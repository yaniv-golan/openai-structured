"""Model version information."""

import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import total_ordering
from typing import Optional


@total_ordering
@dataclass
class ModelVersion:
    """Represents a model version with year, month, and day components."""

    year: int
    month: int
    day: int

    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        """Create a ModelVersion from a string in YYYY-MM-DD format.

        Args:
            version_str: Version string in YYYY-MM-DD format

        Returns:
            ModelVersion instance

        Raises:
            ValueError: If the version string is invalid or represents an invalid date

        Examples:
            >>> ModelVersion.from_string("2024-01-15")
            ModelVersion(year=2024, month=1, day=15)
        """
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

        return cls(year=year, month=month, day=day)

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


def parse_model_version(model: str) -> Optional[ModelVersion]:
    """
    Parse a model name to extract its version information.

    Args:
        model: The model name to parse (e.g., "gpt-4-0314", "gpt-4-0613", "gpt-4-1106-preview")

    Returns:
        ModelVersion if the model name contains a valid version, None otherwise.

    Examples:
        >>> parse_model_version("gpt-4-0314")
        ModelVersion(year=2023, month=3, day=14)
        >>> parse_model_version("gpt-4-1106-preview")
        ModelVersion(year=2023, month=11, day=6)
        >>> parse_model_version("gpt-4")  # Alias without version
        None
    """
    # Match MMDD pattern in model name
    match = re.search(r"(\d{2})(\d{2})(?:-preview)?$", model)
    if not match:
        return None

    month, day = map(int, match.groups())
    # Assume current year for now - could be made configurable if needed
    year = datetime.now().year
    return ModelVersion(year=year, month=month, day=day)
