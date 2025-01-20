"""Tests for template schema validation."""

from typing import Any, Dict, List, Iterator, Tuple, Union, cast, Type

from openai_structured.cli.template_schema import (
    ValidationProxy,
    FileInfoProxy,
    DictProxy,
    ListProxy,
    create_validation_context
)
from openai_structured.cli.file_utils import FileInfo
import tempfile
import os
import pytest

ProxyType = Union[Type[ValidationProxy], Type[DictProxy]]
ProxyInstance = Union[ValidationProxy, DictProxy]

def test_dict_proxy_access() -> None:
    """Test DictProxy allows any attribute/key access."""
    test_dict: Dict[str, Any] = {
        'debug': True,
        'settings': {
            'mode': 'test'
        },
        'key': 'value'
    }
    proxy = DictProxy("config", test_dict)
    
    # Test attribute access
    debug = getattr(proxy, 'debug')
    assert isinstance(debug, ValidationProxy)
    
    # Test nested access
    settings = getattr(proxy, 'settings')
    pytest.fail(f"settings type = {type(settings)}")  # This will show
    mode = getattr(settings, 'mode')
    pytest.fail(f"mode type = {type(mode)}")  # This won't show since we failed above
    
    # Test dictionary methods
    items = proxy.items()
    print(f"DEBUG: items type = {type(items)}")
    items_list = list(items)
    print(f"DEBUG: First item type = {type(items_list[0][1])}")
    
    values = proxy.values()
    print(f"DEBUG: values type = {type(values)}")
    values_list = list(values)
    print(f"DEBUG: First value type = {type(values_list[0])}")

    # Test invalid access
    with pytest.raises(ValueError, match="undefined attribute 'config.invalid'"):
        _ = getattr(proxy, 'invalid')

def test_list_proxy_access() -> None:
    """Test ListProxy allows iteration and indexing."""
    test_list: List[Dict[str, str]] = [
        {'name': 'item1'},
        {'name': 'item2'},
        {'name': 'item3'}
    ]
    proxy = ListProxy("items", test_list)
    
    # Test iteration
    items = list(proxy)
    assert len(items) == 3
    assert all(isinstance(item, ValidationProxy) for item in items)
    
    # Test indexing
    item = proxy[0]
    assert isinstance(item, ValidationProxy)
    assert item._name == "items[0]"
    
    # Test invalid index
    with pytest.raises(ValueError, match="List index"):
        _ = proxy[len(test_list)]

def test_file_info_proxy() -> None:
    """Test FileInfoProxy provides standard file attributes."""
    # Create a test FileInfo instance
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
        file_info = FileInfo(name=os.path.basename(file_path), path=file_path)

    try:
        # Create proxy with the test FileInfo
        proxy = FileInfoProxy("file", file_info)

        # Test basic attribute access
        assert str(getattr(proxy, 'name')) == os.path.basename(file_path)
        assert str(getattr(proxy, 'path')) == file_path
        assert str(getattr(proxy, 'abs_path')) == os.path.realpath(file_path)
        # Content should be empty string to support filtering
        assert str(getattr(proxy, 'content')) == ""
        assert str(getattr(proxy, 'size')) == str(len("test content"))

        # Test invalid attribute access
        with pytest.raises(ValueError):
            _ = getattr(proxy, 'invalid_attr')

    finally:
        os.unlink(file_path) 