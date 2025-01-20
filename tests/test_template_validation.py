"""Tests for template validation."""

import pytest
import jinja2
from jinja2 import Environment
from openai_structured.cli.template_validation import (
    SafeUndefined,
    validate_template_placeholders
)
from openai_structured.cli.file_utils import FileInfo

@pytest.mark.parametrize("template,available_vars", [
    ("{{ any_dict.any_key }}", {"any_dict"}),  # Basic undefined access
    ("{{ nested.deep.key }}", {"nested"}),  # Nested key access
])
def test_safe_undefined(template, available_vars):
    """Test SafeUndefined behavior for various access patterns."""
    env = Environment(undefined=SafeUndefined)
    temp = env.from_string(template)
    with pytest.raises(jinja2.UndefinedError):
        temp.render()

@pytest.mark.parametrize("template,file_mappings,should_pass", [
    ("Hello {{ name }}!", {"name": "test"}, True),  # Basic variable
    ("{{ dict.key }}", {"dict": {"key": "value"}}, True),  # Nested access
    ("{{ deep.nested.key }}", {"deep": {"nested": {"key": "value"}}}, True),  # Deep nested access
    ("{{ content | trim | upper }}", {"content": "test"}, True),  # Multiple filters
    ("{% for item in items %}{{ item.name }}{% endfor %}", {"items": [{"name": "test"}]}, True),  # Loop variable
])
def test_validate_template_success(template, file_mappings, should_pass):
    """Test successful template validation cases."""
    validate_template_placeholders(template, file_mappings)

@pytest.mark.parametrize("template,file_mappings,error_phrase", [
    ("Hello {{ name }}!", {}, "undefined variables"),  # Missing variable
    ("Hello {{ name!", {"name": "test"}, "syntax"),  # Invalid syntax
    ("{% for item in items %}{{ item.undefined }}{% endfor %}", {"items": [{"name": "test"}]}, "items"),  # Invalid loop variable
    ("{% if condition %}{{ undefined_var }}{% endif %}", {"condition": True}, "undefined variable"),  # Undefined in conditional
])
def test_validate_template_errors(template, file_mappings, error_phrase):
    """Test template validation error cases."""
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)
    assert error_phrase in str(exc.value)

def test_validate_with_filters():
    """Test validation with template filters."""
    template = """
    {{ content | trim | upper }}
    {{ data | extract_field("status") | frequency | dict_to_table }}
    """
    file_mappings = {
        "content": "test",
        "data": [{"status": "active"}]
    }
    validate_template_placeholders(template, file_mappings)

def test_validate_fileinfo_attributes(fs):
    """Test validation of FileInfo attribute access."""
    template = "Content: {{ file.content }}, Path: {{ file.abs_path }}"
    file_info = FileInfo(
        name="file",
        path="/path/to/file.txt"
    )
    file_mappings = {"file": file_info}
    validate_template_placeholders(template, file_mappings)

@pytest.mark.parametrize("template,file_mappings", [
    ("{{ file.invalid_attr }}", {"file": FileInfo("file", "/test/file.txt")}),  # Invalid FileInfo attribute
    ("{{ config['invalid'] }}", {"config": {}}),  # Invalid dict key
    ("{{ config.settings.invalid }}", {"config": {"settings": {}}}),  # Invalid nested dict key
])
def test_validate_invalid_access(template, file_mappings):
    """Test validation with invalid attribute/key access."""
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)

def test_validate_complex_template(fs):
    """Test validation of complex template with multiple features."""
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
    file_mappings = {
        "source_files": [
            FileInfo("file1", "/test/file1.txt"),
            FileInfo("file2", "/test/file2.txt")
        ],
        "config": {
            "exclude": {"file1": "reason1"},
            "settings": {"mode": "test"}
        }
    }
    validate_template_placeholders(template, file_mappings) 