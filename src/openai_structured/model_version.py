"""Model version handling with validation.

This module provides functionality for parsing and validating OpenAI model version strings.
Model versions follow a strict YYYY-MM-DD format to ensure consistent version comparison
and validation across the codebase.

Example:
    >>> from openai_structured.model_version import ModelVersion
    >>> # Parse a model name into base model and version
    >>> base_model, version_str = ModelVersion.parse_version_string("gpt-4o-2024-08-06")
    >>> print(base_model)  # "gpt-4o"
    >>> print(version_str)  # "2024-08-06"
    >>>
    >>> # Create a version object from a string
    >>> version = ModelVersion.from_string(version_str)
    >>> print(version)  # "2024-08-06"
    >>>
    >>> # Compare versions
    >>> newer = ModelVersion.from_string("2024-09-01")
    >>> print(version < newer)  # True
"""

import re
from dataclasses import dataclass
from datetime import date
from functools import total_ordering
from typing import Match, Optional, Tuple

from .errors import InvalidDateError, InvalidVersionFormatError

VERSION_PATTERN = r"^([a-zA-Z0-9-]+)-(\d{4})-(\d{2})-(\d{2})$"


@total_ordering
@dataclass
class ModelVersion:
    """Represents a model version with year, month, and day components.

    This class provides functionality for parsing, validating, and comparing
    OpenAI model versions. All versions must follow the YYYY-MM-DD format,
    where:
    - YYYY is a four-digit year (2000 or later)
    - MM is a two-digit month (01-12)
    - DD is a two-digit day (01-31)

    The class supports:
    1. Parsing model names with versions (e.g., "gpt-4o-2024-08-06")
    2. Creating version objects from date strings (e.g., "2024-08-06")
    3. Comparing versions for ordering (e.g., v1 < v2)
    4. String representation in YYYY-MM-DD format

    Examples:
        >>> # Create a version
        >>> v1 = ModelVersion(year=2024, month=8, day=6)
        >>> print(v1)  # "2024-08-06"
        >>>
        >>> # Parse from string
        >>> v2 = ModelVersion.from_string("2024-09-01")
        >>> print(v2)  # "2024-09-01"
        >>>
        >>> # Compare versions
        >>> print(v1 < v2)  # True
        >>> print(v1 == ModelVersion(2024, 8, 6))  # True
    """

    year: int
    month: int
    day: int

    @classmethod
    def parse_version_string(cls, model_name: str) -> Tuple[str, str]:
        """Parse a model name into base model and version string.

        This method extracts the base model name and version string from a full
        model name. The version must be in YYYY-MM-DD format.

        Args:
            model_name: Full model name with version (e.g., "gpt-4o-2024-08-06")

        Returns:
            Tuple of (base_model, version_str), where:
            - base_model is the name without version (e.g., "gpt-4o")
            - version_str is the version part (e.g., "2024-08-06")

        Raises:
            InvalidVersionFormatError: If the version format is invalid

        Examples:
            >>> base, version = ModelVersion.parse_version_string("gpt-4o-2024-08-06")
            >>> print(base)  # "gpt-4o"
            >>> print(version)  # "2024-08-06"
            >>>
            >>> # Invalid format raises error
            >>> ModelVersion.parse_version_string("gpt-4o-2024")  # Raises InvalidVersionFormatError
        """
        match: Optional[Match] = re.match(VERSION_PATTERN, model_name)
        if not match:
            raise InvalidVersionFormatError(
                model_name, "Version must be in format: <model>-YYYY-MM-DD"
            )

        base_model, year, month, day = match.groups()
        version_str = f"{year}-{month}-{day}"

        return base_model, version_str

    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        """Create a ModelVersion from a string in YYYY-MM-DD format.

        This method parses a version string and validates its components to
        ensure they represent a valid date.

        Args:
            version_str: Version string in YYYY-MM-DD format

        Returns:
            ModelVersion instance

        Raises:
            InvalidVersionFormatError: If the version string format is invalid
            InvalidDateError: If the date components are invalid

        Examples:
            >>> v1 = ModelVersion.from_string("2024-08-06")
            >>> print(v1)  # "2024-08-06"
            >>>
            >>> # Invalid format raises error
            >>> ModelVersion.from_string("2024-13-01")  # Raises InvalidDateError (invalid month)
            >>> ModelVersion.from_string("1999-12-31")  # Raises InvalidDateError (year too old)
        """
        match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", version_str)
        if not match:
            raise InvalidVersionFormatError(
                version_str, "Version must be in YYYY-MM-DD format"
            )

        try:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))

            # Validate date components
            if year < 2000:
                raise ValueError("Year must be 2000 or later")
            if month < 1 or month > 12:
                raise ValueError("Month must be between 1 and 12")

            # This will raise ValueError for invalid dates (e.g., Feb 30)
            date(year, month, day)

        except ValueError as e:
            raise InvalidDateError(
                "version",  # Will be replaced by model name in registry
                version_str,
                f"Invalid date format in model version: {str(e)}\n"
                f"Use format: YYYY-MM-DD (e.g. 2024-08-06)",
            )

        return cls(year=year, month=month, day=day)

    def __str__(self) -> str:
        """Return string representation in YYYY-MM-DD format.

        Returns:
            str: The version in YYYY-MM-DD format with zero-padded month and day

        Examples:
            >>> v = ModelVersion(2024, 8, 6)
            >>> print(str(v))  # "2024-08-06"
        """
        return f"{self.year}-{self.month:02d}-{self.day:02d}"

    def __lt__(self, other: object) -> bool:
        """Compare versions for less than.

        Args:
            other: Object to compare with. Can be another ModelVersion or None.
                  When comparing with None, this version is considered greater.

        Returns:
            bool: True if this version is less than other, False otherwise.
                 When comparing with None, always returns False.

        Examples:
            >>> v1 = ModelVersion(2024, 8, 6)
            >>> v2 = ModelVersion(2024, 9, 1)
            >>> print(v1 < v2)  # True
            >>> print(v1 < None)  # False (version is never less than None)
        """
        if other is None:
            return False  # Any version is greater than None
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return (self.year, self.month, self.day) < (
            other.year,
            other.month,
            other.day,
        )

    def __eq__(self, other: object) -> bool:
        """Compare versions for equality.

        Args:
            other: Object to compare with. Can be another ModelVersion or None.
                  When comparing with None, this version is considered not equal.

        Returns:
            bool: True if versions are equal, False otherwise.
                 When comparing with None, always returns False.

        Examples:
            >>> v1 = ModelVersion(2024, 8, 6)
            >>> v2 = ModelVersion(2024, 8, 6)
            >>> print(v1 == v2)  # True
            >>> print(v1 == None)  # False (version is never equal to None)
        """
        if other is None:
            return False  # A version is never equal to None
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return (self.year, self.month, self.day) == (
            other.year,
            other.month,
            other.day,
        )
