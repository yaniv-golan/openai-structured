"""Tests for task template utilities."""

import pytest
from jinja2 import Environment
from pyfakefs.fake_filesystem_unittest import Patcher
from openai_structured.cli.file_utils import FileInfo
from openai_structured.cli.template_utils import (
    render_template,
    validate_template_placeholders,
)
from openai_structured.cli.template_validation import validate_template_placeholders

def test_render_task_template_basic():
    """Test basic task template rendering."""
    template = "Hello {{ name }}!"
    context = {"name": "World"}
    result = render_template(template, context)
    assert result == "Hello World!"

def test_render_task_template_missing_var():
    """Test task template rendering with missing variable."""
    template = "Hello {{ name }}!"
    context = {}
    with pytest.raises(ValueError) as exc:
        render_template(template, context)
    assert "'name' is undefined" in str(exc.value)

def test_validate_task_template_basic():
    """Test basic task template validation."""
    template = "Hello {{ name }}!"
    file_mappings = {"name": "test"}  # Simple value
    validate_template_placeholders(template, file_mappings)

def test_validate_task_template_missing_var():
    """Test task template validation with missing variable."""
    template = "Hello {{ name }}!"
    file_mappings = {}  # Empty dict instead of set()
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)
    assert "undefined variable" in str(exc.value)

def test_validate_task_template_invalid_syntax():
    """Test task template validation with invalid syntax."""
    template = "Hello {{ name!"  # Missing closing brace
    file_mappings = {"name": "test"}
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)
    assert "Invalid task template syntax" in str(exc.value)

@pytest.fixture
def fs():
    """Fixture to set up fake filesystem."""
    with Patcher() as patcher:
        # Create test files and directories
        patcher.fs.create_file('/path/to/file.txt', contents='Test file content')
        patcher.fs.create_file('/absolute/path/to/file.txt', contents='Test file content')
        yield patcher.fs

def test_validate_fileinfo_attributes(fs):
    """Test validation of FileInfo attribute access."""
    template = "Content: {{ file.content }}, Path: {{ file.abs_path }}"
    file_info = FileInfo(
        name="file",
        path="/path/to/file.txt"
    )
    file_mappings = {"file": file_info}
    validate_template_placeholders(template, file_mappings)

def test_validate_fileinfo_invalid_attribute(fs):
    """Test validation with invalid FileInfo attribute."""
    template = "{{ file.invalid_attr }}"
    file_info = FileInfo(
        name="file",
        path="/path/to/file.txt"
    )
    file_mappings = {"file": file_info}
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)

def test_validate_nested_json_access():
    """Test validation of nested JSON dictionary access."""
    template = "{{ config['debug'] }}, {{ config['settings']['mode'] }}"
    file_mappings = {"config": {"debug": True, "settings": {"mode": "test"}}}
    validate_template_placeholders(template, file_mappings)

def test_validate_nested_json_invalid_key():
    """Test validation with invalid nested JSON key."""
    template = "{{ config['invalid_key'] }}"
    file_mappings = {"config": {"debug": True}}
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)

def test_validate_complex_template(fs):
    """Test validation of complex template with multiple features."""
    # Set up test files
    fs.create_file('/test/file1.txt', contents='File 1 content')
    fs.create_file('/test/file2.txt', contents='File 2 content')
    
    template = """
    {% for file in source_files %}
        File: {{ file.abs_path }}
        Content: {{ file.content }}
        {% if file.name in config.exclude %}
            Excluded: {{ config.exclude[file.name] }}
        {% endif %}
    {% endfor %}

    Settings:
    {% for key, value in config.settings.items() %}
        {{ key }}: {{ value }}
    {% endfor %}
    """
    file_mappings = {
        "source_files": [
            FileInfo("file1.txt", "/test/file1.txt"),
            FileInfo("file2.txt", "/test/file2.txt")
        ],
        "config": {
            "exclude": {"file1.txt": "reason1"},
            "settings": {"mode": "test"}
        }
    }
    validate_template_placeholders(template, file_mappings)

def test_validate_template_with_filters(fs):
    """Test validation of template using built-in filters and functions."""
    fs.create_file('/test/data.txt', contents='Test data content')

    template = """
    {% set content = file.content|trim %}
    {{ content|wordcount }}
    {{ content|extract_field("status")|frequency|dict_to_table }}
    """
    file_mappings = {
        "file": FileInfo("file", "/test/data.txt")
    }
    validate_template_placeholders(template, file_mappings)

def test_validate_template_undefined_in_loop():
    """Test validation catches undefined variables in loops."""
    template = """
    {% for item in items %}
        {{ item.undefined_var }}
    {% endfor %}
    """
    file_mappings = {
        "items": [
            {"name": "item1"},
            {"name": "item2"}
        ]
    }
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)

def test_validate_template_conditional_vars():
    """Test validation with variables in conditional blocks."""
    template = """
    {% if condition %}
        {{ defined_var }}
    {% else %}
        {{ undefined_var }}
    {% endif %}
    """
    file_mappings = {
        "condition": True,
        "defined_var": "test"
    }
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)

def test_validate_template_builtin_functions():
    """Test validation allows built-in Jinja2 functions and filters."""
    template = """
    {% set items = range(5) %}
    {% for i in items %}
        {{ loop.index }}: {{ i }}
    {% endfor %}
    {{ "text"|upper }}
    {{ lipsum(2) }}
    """
    file_mappings = {}  # No variables needed
    validate_template_placeholders(template, file_mappings)

def test_validate_template_custom_functions(fs):
    """Test validation allows custom template functions."""
    fs.create_file('/test/file.txt', contents='Test file content')
    fs.create_file('/test/data.json', contents='{"status": "active"}')

    template = """
    {{ file.content|extract_field("status") }}
    {{ data|pivot_table("category", "value", "mean") }}
    {{ text|format_code("python") }}
    """
    file_mappings = {
        "file": FileInfo("file", "/test/file.txt"),
        "data": {"category": "test", "value": 1},
        "text": "print('hello')"
    }
    validate_template_placeholders(template, file_mappings)

def test_render_template_with_file_content(fs):
    """Test rendering template with actual file content."""
    fs.create_file('/test/input.txt', contents='Hello from file!')

    template = "Content: {{ file.content }}"
    file_info = FileInfo(
        name="file",
        path="/test/input.txt"
    )
    file_info.load_content()
    context = {"file": file_info}
    result = render_template(template, context)
    assert result == "Content: Hello from file!"

def test_validate_json_variable_access():
    """Test validation of JSON variable access using both dot notation and dictionary syntax."""
    template = """
    Dot notation: {{ config.debug }}, {{ config.settings.mode }}
    Dict access: {{ config['debug'] }}, {{ config['settings']['mode'] }}
    Mixed: {{ config.settings['mode'] }}, {{ config['settings'].mode }}
    """
    file_mappings = {
        "config": {
            "debug": True,
            "settings": {"mode": "test"}
        }
    }
    validate_template_placeholders(template, file_mappings)

def test_render_json_variable_access():
    """Test rendering with JSON variables using both access methods."""
    config = {
        "debug": True,
        "settings": {
            "mode": "test"
        }
    }
    
    template = """
    Dot notation: {{ config.debug }}, {{ config.settings.mode }}
    Dict access: {{ config['debug'] }}, {{ config['settings']['mode'] }}
    Mixed: {{ config.settings['mode'] }}, {{ config['settings'].mode }}
    """
    
    result = render_template(template, {"config": config})
    assert "Dot notation: True, test" in result
    assert "Dict access: True, test" in result
    assert "Mixed: test, test" in result

def test_invalid_json_variable_access():
    """Test validation catches invalid JSON variable access."""
    template = "{{ config.invalid }}"
    file_mappings = {
        "config": {
            "debug": True,
            "settings": {"mode": "test"}
        }
    }
    with pytest.raises(ValueError) as exc:
        validate_template_placeholders(template, file_mappings)
