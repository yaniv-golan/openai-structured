"""Tests for template validation."""

from typing import Any, Dict, List, Set, TypedDict, Union, cast

import jinja2
import pytest
from jinja2 import Environment
from pyfakefs.fake_filesystem import FakeFilesystem

from openai_structured.cli.file_utils import FileInfo
from openai_structured.cli.template_validation import (
    SafeUndefined,
    validate_template_placeholders,
)


class FileMapping(TypedDict, total=False):
    name: str
    dict: Dict[str, str]
    deep: Dict[str, Dict[str, Dict[str, str]]]
    content: str
    items: List[Dict[str, str]]
    condition: bool
    config: Dict[str, Union[Dict[str, str], Dict[str, Dict[str, str]]]]
    source_files: List[FileInfo]


@pytest.mark.parametrize(
    "template,available_vars",
    [
        ("{{ any_dict.any_key }}", {"any_dict"}),  # Basic undefined access
        ("{{ nested.deep.key }}", {"nested"}),  # Nested key access
    ],
)
def test_safe_undefined(template: str, available_vars: Set[str]) -> None:
    """Test SafeUndefined behavior for various access patterns."""
    env = Environment(undefined=SafeUndefined)
    temp = env.from_string(template)
    with pytest.raises(jinja2.UndefinedError):
        temp.render()


@pytest.mark.parametrize(
    "template,file_mappings,should_pass",
    [
        (
            "Hello {{ name }}!",
            cast(Dict[str, Any], {"name": "test"}),
            True,
        ),  # Basic variable
        (
            "{{ dict.key }}",
            cast(Dict[str, Any], {"dict": {"key": "value"}}),
            True,
        ),  # Nested access
        (
            "{{ deep.nested.key }}",
            cast(Dict[str, Any], {"deep": {"nested": {"key": "value"}}}),
            True,
        ),  # Deep nested access
        (
            "{{ content | trim | upper }}",
            cast(Dict[str, Any], {"content": "test"}),
            True,
        ),  # Multiple filters
        (
            "{% for item in items %}{{ item.name }}{% endfor %}",
            cast(Dict[str, Any], {"items": [{"name": "test"}]}),
            True,
        ),  # Loop variable
    ],
)
def test_validate_template_success(
    template: str, file_mappings: Dict[str, Any], should_pass: bool
) -> None:
    """Test successful template validation cases."""
    validate_template_placeholders(template, file_mappings)


@pytest.mark.parametrize(
    "template,file_mappings,error_phrase",
    [
        (
            "Hello {{ name }}!",
            cast(Dict[str, Any], {}),
            "undefined variables",
        ),  # Missing variable
        (
            "Hello {{ name!",
            cast(Dict[str, Any], {"name": "test"}),
            "syntax",
        ),  # Invalid syntax
        (
            "{% for item in items %}{{ item.undefined }}{% endfor %}",
            cast(Dict[str, Any], {"items": [{"name": "test"}]}),
            "items",
        ),  # Invalid loop variable
        (
            "{% if condition %}{{ undefined_var }}{% endif %}",
            cast(Dict[str, Any], {"condition": True}),
            "undefined variable",
        ),  # Undefined in conditional
    ],
)
def test_validate_template_errors(
    template: str, file_mappings: Dict[str, Any], error_phrase: str
) -> None:
    """Test template validation error cases."""
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)
    assert error_phrase in str(exc.value)


def test_validate_with_filters() -> None:
    """Test validation with template filters."""
    template = """
    {{ content | trim | upper }}
    {{ data | extract_field("status") | frequency | dict_to_table }}
    """
    file_mappings: Dict[str, Any] = {
        "content": "test",
        "data": [{"status": "active"}],
    }
    validate_template_placeholders(template, file_mappings)


def test_validate_fileinfo_attributes(fs: FakeFilesystem) -> None:
    """Test validation of FileInfo attribute access."""
    # Create test file in fake filesystem
    fs.create_file("/path/to/file.txt", contents="test content")

    template = "Content: {{ file.content }}, Path: {{ file.abs_path }}"
    file_info = FileInfo.from_path(path="/path/to/file.txt")
    file_mappings: Dict[str, Any] = {"file": file_info}
    validate_template_placeholders(template, file_mappings)


@pytest.mark.parametrize(
    "template,context_setup",
    [
        (
            "{{ file.invalid_attr }}",
            lambda fs: {"file": create_test_file(fs, "test.txt")},
        ),  # Invalid FileInfo attribute
        (
            "{{ config['invalid'] }}",
            lambda _: {"config": {}},
        ),  # Invalid dict key
        (
            "{{ config.settings.invalid }}",
            lambda _: {"config": {"settings": {}}},
        ),  # Invalid nested dict key
    ],
)
def test_validate_invalid_access(
    template: str, context_setup: Any, fs: FakeFilesystem
) -> None:
    """Test validation with invalid attribute/key access."""
    file_mappings = context_setup(fs)
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)
    assert "undefined" in str(exc.value)


def create_test_file(fs: FakeFilesystem, filename: str) -> FileInfo:
    """Create a test file and return FileInfo instance."""
    fs.create_file(filename, contents="test content")
    return FileInfo.from_path(path=filename)


def test_validate_complex_template(fs: FakeFilesystem) -> None:
    """Test validation of complex template with multiple features."""
    # Set up fake filesystem
    fs.create_file("/test/file1.txt", contents="content1")
    fs.create_file("/test/file2.txt", contents="content2")

    template = """
    {% for file in source_files %}
        File: {{ file.abs_path }}
        Content: {{ file.content }}
        {% if file.name in config['exclude'] %}
            Excluded: {{ config['exclude'][file.name] }}
        {% endif %}
    {% endfor %}

    Settings:
    {% for key, value in config['settings'].items() %}
        {{ key }}: {{ value }}
    {% endfor %}
    """
    file_mappings: Dict[str, Any] = {
        "source_files": [
            FileInfo.from_path(path="/test/file1.txt"),
            FileInfo.from_path(path="/test/file2.txt"),
        ],
        "config": {
            "exclude": {"file1.txt": "reason1"},
            "settings": {"mode": "test"},
        },
    }
    validate_template_placeholders(template, file_mappings)


def test_validate_template_placeholders_invalid_json() -> None:
    """Test validation with invalid JSON value."""
    template = "{{ invalid_json }}"
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, {})
    assert "undefined variable" in str(exc.value)
