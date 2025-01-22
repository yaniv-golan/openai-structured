"""Tests for template rendering functionality."""

import logging
import os
import tempfile
from typing import Any, Dict

import pytest
from jinja2 import Environment, StrictUndefined
from pyfakefs.fake_filesystem import FakeFilesystem

from openai_structured.cli.file_utils import FileInfo
from openai_structured.cli.template_rendering import (
    DotDict,
    create_jinja_env,
    render_template,
)
from openai_structured.cli.template_utils import (
    read_file,
    validate_template_placeholders,
)


def test_create_jinja_env() -> None:
    """Test creation of Jinja2 environment with custom filters and globals."""
    env = create_jinja_env()

    # Test that environment is properly configured
    assert isinstance(env, Environment)
    assert env.undefined == StrictUndefined
    assert env.autoescape

    # Test that custom filters are registered
    assert "format_code" in env.filters
    assert "strip_comments" in env.filters
    assert "dict_to_table" in env.filters
    assert "list_to_table" in env.filters

    # Test that custom globals are registered
    assert "now" in env.globals
    assert "debug" in env.globals
    assert "type_of" in env.globals
    assert "dir_of" in env.globals


def test_render_template_basic() -> None:
    """Test basic template rendering with simple context."""
    template: str = "Hello {{ name }}!"
    context: Dict[str, str] = {"name": "World"}
    result = render_template(template, context)
    assert result == "Hello World!"


def test_render_template_with_filters() -> None:
    """Test template rendering with custom filters."""
    template: str = "{{ data | dict_to_table }}"
    context: Dict[str, Dict[str, str]] = {"data": {"key": "value"}}
    result = render_template(template, context)
    expected_table = "| Key | Value |\n| --- | --- |\n| key | value |"
    assert expected_table in result


def test_render_template_with_file_info() -> None:
    """Test template rendering with FileInfo objects."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name

    try:
        file_info = FileInfo.from_path(path=file_path)
        file_info.load_content()

        template: str = (
            "Content: {{ file.content }}, Path: {{ file.abs_path }}"
        )
        env = Environment()
        result = env.from_string(template).render(file=file_info)

        assert "test content" in result
        assert file_path in result
    finally:
        os.unlink(file_path)


def test_render_template_with_immediate_loading() -> None:
    """Test template rendering with immediate loading of file content."""
    logger = logging.getLogger(__name__)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
        logger.debug(
            "Created test file %s with content 'test content'", file_path
        )

    try:
        file_info = FileInfo.from_path(path=file_path)
        logger.debug("Created FileInfo instance")

        # Content should be loaded immediately
        assert file_info.content == "test content"

        # Render template with file info
        result = render_template("{{ file.content }}", {"file": file_info})
        assert result == "test content"
    finally:
        os.unlink(file_path)


def test_render_template_with_dot_dict() -> None:
    """Test template rendering with nested dictionary access."""
    template: str = "{{ config.settings.mode }}"
    context: Dict[str, Dict[str, Dict[str, str]]] = {
        "config": {"settings": {"mode": "test"}}
    }
    result = render_template(template, context)
    assert result == "test"


def test_render_template_error_handling() -> None:
    """Test error handling in template rendering."""
    # Test undefined variable
    with pytest.raises(ValueError) as exc:
        render_template("{{ undefined }}", {})
    assert "undefined" in str(exc.value)

    # Test syntax error
    with pytest.raises(ValueError) as exc:
        render_template("{% if %}", {})
    assert "syntax error" in str(exc.value)

    # Test runtime error
    with pytest.raises(ValueError) as exc:
        render_template("{{ x + y }}", {"x": "string", "y": 1})
    assert "can only concatenate str" in str(exc.value)


def test_dot_dict() -> None:
    """Test DotDict functionality."""
    data: Dict[str, Any] = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    dot_dict = DotDict(data)

    # Test attribute access
    assert dot_dict.a == 1
    assert dot_dict.b.c == 2
    assert dot_dict.b.d.e == 3

    # Test dictionary access
    assert dot_dict["a"] == 1
    assert dot_dict["b"]["c"] == 2
    assert dot_dict["b"]["d"]["e"] == 3

    # Test contains
    assert "a" in dot_dict
    assert "c" in dot_dict.b

    # Test get with default
    assert dot_dict.get("missing", "default") == "default"

    # Test iteration methods
    assert list(dot_dict.keys()) == ["a", "b"]
    assert list(dot_dict.values())[0] == 1  # Test first value
    assert isinstance(dot_dict.b, DotDict)  # Test nested dict is DotDict

    # Test items returns DotDict for nested dicts
    items = list(dot_dict.items())
    assert items[0] == ("a", 1)
    assert items[1][0] == "b"
    assert isinstance(items[1][1], DotDict)
    assert items[1][1].c == 2
    assert items[1][1].d.e == 3

    # Test error handling
    with pytest.raises(AttributeError):
        dot_dict.missing
    with pytest.raises(KeyError):
        dot_dict["missing"]


def test_read_file(fs: FakeFilesystem) -> None:
    """Test reading file contents."""
    # Create test file
    fs.create_file("test.txt", contents="test content")

    # Read file
    file_info = read_file("test.txt")
    assert file_info.content == "test content"


def test_render_template_with_file(fs: FakeFilesystem) -> None:
    """Test template rendering with file content."""
    # Create test file
    fs.create_file("test.txt", contents="test content")

    # Create template using file content
    template = "Content: {{ file.content }}"
    context = {"file": read_file("test.txt")}
    env = Environment()
    result = render_template(template, context, env)
    assert result == "Content: test content"


def test_validate_template_placeholders_basic() -> None:
    """Test basic template placeholder validation."""
    template = "Hello {{ name }}!"
    context = {"name": "World"}
    env = Environment()
    # Should not raise any exceptions
    validate_template_placeholders(template, context, env)


def test_validate_template_placeholders_missing() -> None:
    """Test template validation with missing variables."""
    template = "Hello {{ name }}!"
    context: Dict[str, str] = {}  # Empty context
    env = Environment()
    with pytest.raises(ValueError):
        validate_template_placeholders(template, context, env)


def test_validate_template_placeholders_undefined() -> None:
    """Test template validation with undefined variables."""
    template = "Hello {{ name }}!"
    context = {"wrong_name": "World"}
    env = Environment()
    with pytest.raises(ValueError):
        validate_template_placeholders(template, context, env)


def test_validate_template_placeholders_nested() -> None:
    """Test template validation with nested variables."""
    template = "{{ user.name }} is {{ user.age }} years old"
    context = {"user": {"name": "Alice", "age": 30}}
    env = Environment()
    # Should not raise any exceptions
    validate_template_placeholders(template, context, env)


def test_validate_template_placeholders_complex() -> None:
    """Test template validation with complex expressions."""
    template = """
    {% for item in items %}
        {{ item.name }}: {{ item.value }}
    {% endfor %}
    """
    context = {
        "items": [
            {"name": "A", "value": 1},
            {"name": "B", "value": 2},
        ]
    }
    env = Environment()
    # Should not raise any exceptions
    validate_template_placeholders(template, context, env)
