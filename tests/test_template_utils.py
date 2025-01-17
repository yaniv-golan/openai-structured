"""Tests for template utilities."""

from typing import Set

import pytest
from jinja2 import Environment

from openai_structured.cli.template_utils import (
    extract_field,
    format_code,
    get_template_variables,
    pivot_table,
    summarize,
    validate_template_placeholders,
)


@pytest.fixture
def jinja_env() -> Environment:
    """Create a basic Jinja2 environment for testing."""
    return Environment()


class TestExtractField:
    """Tests for extract_field function."""

    def test_extract_from_dicts(self) -> None:
        """Test extracting field from dictionaries."""
        data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
        ]
        assert extract_field(data, "name") == ["Alice", "Bob"]

    def test_extract_from_objects(self) -> None:
        """Test extracting field from objects."""

        class Person:
            def __init__(self, name: str):
                self.name = name

        data = [Person("Alice"), Person("Bob")]
        assert extract_field(data, "name") == ["Alice", "Bob"]

    def test_extract_missing_field(self) -> None:
        """Test extracting non-existent field."""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        assert extract_field(data, "age") == [None, None]


class TestFormatCode:
    """Tests for the format_code function."""

    def test_basic_formatting(self) -> None:
        """Test basic code formatting without syntax highlighting."""
        code = """
        def hello():
            print("Hello")
        """
        result = format_code(code, format="plain")
        assert "def hello():" in result
        assert 'print("Hello")' in result

    def test_syntax_highlighting_terminal(self) -> None:
        """Test syntax highlighting in terminal format."""
        code = 'print("Hello")'
        result = format_code(code, lang="python", format="terminal")
        # Terminal format should include ANSI color codes if pygments is available
        assert result != code

    def test_syntax_highlighting_html(self) -> None:
        """Test syntax highlighting in HTML format."""
        code = 'print("Hello")'
        result = format_code(code, lang="python", format="html")
        # HTML format should include HTML tags if pygments is available
        assert "<div" in result or code in result

    def test_invalid_language(self) -> None:
        """Test handling of invalid language specification."""
        code = 'print("Hello")'
        result = format_code(code, lang="nonexistent", format="terminal")
        assert "Hello" in result

    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        assert format_code("") == ""

    def test_invalid_format(self) -> None:
        """Test handling of invalid format specification."""
        with pytest.raises(ValueError, match="Invalid format"):
            format_code("test", format="invalid")


class TestPivotTable:
    """Tests for pivot_table function."""

    def test_basic_pivot(self) -> None:
        """Test basic pivot table creation."""
        data = [
            {"category": "A", "value": 1},
            {"category": "B", "value": 2},
            {"category": "A", "value": 3},
        ]
        result = pivot_table(data, "category", "value", "sum")
        assert result["aggregates"]["A"]["value"] == 4
        assert result["aggregates"]["B"]["value"] == 2

    def test_empty_data(self) -> None:
        """Test pivot table with empty data."""
        result = pivot_table([], "category", "value")
        assert result["aggregates"] == {}
        assert result["metadata"]["total_records"] == 0

    def test_invalid_aggfunc(self) -> None:
        """Test pivot table with invalid aggregation function."""
        data = [{"category": "A", "value": 1}]
        with pytest.raises(ValueError, match="Invalid aggfunc"):
            pivot_table(data, "category", "value", "invalid")


class TestSummarize:
    """Tests for summarize function."""

    def test_basic_summary(self) -> None:
        """Test basic data summarization."""
        data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
        ]
        result = summarize(data)
        assert result["total_records"] == 2
        assert "name" in result["fields"]
        assert "age" in result["fields"]

    def test_empty_data(self) -> None:
        """Test summarization with empty data."""
        result = summarize([])
        assert result["total_records"] == 0
        assert result["fields"] == {}

    def test_invalid_data(self) -> None:
        """Test summarization with invalid data type."""
        with pytest.raises(TypeError):
            summarize([1, 2, 3])


class TestTemplateValidation:
    """Tests for template validation functions."""

    available_files: Set[str] = {"input.txt", "config.json"}

    def test_get_template_variables(self) -> None:
        """Test extracting variables from template."""
        template = "{{ var1 }} {% set var2 = 123 %} {{ var3|filter }}"
        variables = get_template_variables(template)
        assert variables == {"var1", "var3"}

    def test_validate_template_basic(self) -> None:
        """Test basic template validation."""
        template = "{{ input }} {{ config }}"
        validate_template_placeholders(template, {"input", "config"})

    def test_validate_template_missing_files(self) -> None:
        """Test validation with missing files."""
        template = "{{ missing }}"
        with pytest.raises(ValueError, match="missing files"):
            validate_template_placeholders(template, set())

    def test_validate_template_invalid_syntax(self) -> None:
        """Test validation with invalid syntax."""
        template = "{% if x %}"  # Missing endif
        with pytest.raises(ValueError, match="Invalid template syntax"):
            validate_template_placeholders(template, set())
