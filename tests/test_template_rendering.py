"""Tests for template rendering functionality."""

import os
import tempfile
from typing import Dict, Any
import logging

import pytest
from jinja2 import Environment, StrictUndefined

from openai_structured.cli.template_rendering import (
    create_jinja_env,
    render_template,
    DotDict
)
from openai_structured.cli.file_utils import FileInfo

def test_create_jinja_env():
    """Test creation of Jinja2 environment with custom filters and globals."""
    env = create_jinja_env()
    
    # Test that environment is properly configured
    assert isinstance(env, Environment)
    assert env.undefined == StrictUndefined
    assert env.autoescape
    
    # Test that custom filters are registered
    assert 'format_code' in env.filters
    assert 'strip_comments' in env.filters
    assert 'dict_to_table' in env.filters
    assert 'list_to_table' in env.filters
    
    # Test that custom globals are registered
    assert 'now' in env.globals
    assert 'debug' in env.globals
    assert 'type_of' in env.globals
    assert 'dir_of' in env.globals

def test_render_template_basic():
    """Test basic template rendering with simple context."""
    template = "Hello {{ name }}!"
    context = {"name": "World"}
    result = render_template(template, context)
    assert result == "Hello World!"

def test_render_template_with_filters():
    """Test template rendering with custom filters."""
    template = "{{ data | dict_to_table }}"
    context = {"data": {"key": "value"}}
    result = render_template(template, context)
    expected_table = "| Key | Value |\n| --- | --- |\n| key | value |"
    assert expected_table in result

def test_render_template_with_file_info():
    """Test template rendering with FileInfo objects."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
        file_name = os.path.basename(file_path)

    try:
        file_info = FileInfo(name=file_name, path=file_path)
        file_info.load_content()

        template = "Content: {{ file.content }}, Path: {{ file.abs_path }}"
        env = Environment()
        result = env.from_string(template).render(file=file_info)

        assert "test content" in result
        assert file_path in result
    finally:
        os.unlink(file_path)

def test_render_template_with_lazy_loading():
    """Test template rendering with lazy loading of file content."""
    logger = logging.getLogger(__name__)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
        file_name = os.path.basename(file_path)
        logger.debug("Created test file %s with content 'test content'", file_path)

    try:
        file_info = FileInfo(name=file_name, path=file_path)
        logger.debug("Created FileInfo instance (lazy=%s)", file_info.lazy)
        
        # Content should not be loaded initially
        assert file_info._content is None
        logger.debug("Verified content is not loaded initially")

        # Content should load automatically when rendered
        template = "Content: {{ file.content }}"
        env = Environment()
        logger.debug("Created Jinja environment with template %r", template)
        
        result = env.from_string(template).render(file=file_info)
        logger.debug("Rendered template result: %r", result)

        # Content should now be loaded
        assert "test content" in result
        logger.debug("Verified content is in rendered result")
    finally:
        os.unlink(file_path)

def test_render_template_with_dot_dict():
    """Test template rendering with nested dictionary access."""
    template = "{{ config.settings.mode }}"
    context = {"config": {"settings": {"mode": "test"}}}
    result = render_template(template, context)
    assert result == "test"

def test_render_template_error_handling():
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
    assert "error" in str(exc.value)

def test_dot_dict():
    """Test DotDict functionality."""
    data = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3
            }
        }
    }
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