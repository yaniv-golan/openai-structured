import pytest

from openai_structured.client import supports_structured_output


@pytest.mark.parametrize(
    "model_name,expected",
    [  # type: ignore[misc]
        # Alias cases
        ("gpt-4o", True),  # Basic alias
        ("gpt-3.5-turbo", False),  # Unsupported model
        ("o3-mini", True),  # New o3-mini alias
        # Dated versions - valid
        ("gpt-4o-2024-08-06", True),  # Minimum version
        ("gpt-4o-2024-09-01", True),  # Newer version
        ("o3-mini-2025-01-31", True),  # Minimum o3-mini version
        ("o3-mini-2025-02-01", True),  # Newer o3-mini version
        # Dated versions - invalid
        ("gpt-4o-2024-07-01", False),  # Too old
        ("gpt-4o-2023-12-01", False),  # Much too old
        ("o3-mini-2025-01-30", False),  # Too old for o3-mini
        # Edge cases
        ("", False),  # Empty string
        ("gpt-4o-", False),  # Incomplete version
        ("gpt-4o-invalid-date", False),  # Invalid date format
        ("gpt-4o-9999-99-99", False),  # Invalid date values
        ("not-a-model", False),  # Random string
        ("gpt-4o-2024-08-06-extra", False),  # Extra version components
    ],
)
def test_supports_structured_output(model_name: str, expected: bool) -> None:
    """Test model support validation with various model names.

    This test covers:
    1. Basic alias validation
    2. Dated version validation (minimum and newer versions)
    3. Invalid version handling
    4. Edge cases and error conditions
    """
    assert supports_structured_output(model_name) == expected
