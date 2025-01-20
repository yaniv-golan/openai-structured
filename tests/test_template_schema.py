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
    assert isinstance(settings, DictProxy)  # Settings should be a DictProxy since it's a nested dict
    mode = getattr(settings, 'mode')
    assert isinstance(mode, ValidationProxy)
    
    # Test dictionary methods
    items = proxy.items()
    items_list = list(items)
    assert len(items_list) == 3
    assert isinstance(items_list[0][1], ValidationProxy)  # Value should be a ValidationProxy
    
    values = proxy.values()
    values_list = list(values)
    assert len(values_list) == 3
    assert isinstance(values_list[0], ValidationProxy)  # Value should be a ValidationProxy

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
    # Each item should be a DictProxy since they are dictionaries
    assert all(isinstance(item, DictProxy) for item in items)
    
    # Test indexing
    item = proxy[0]
    assert isinstance(item, DictProxy)  # Should be DictProxy since it's a dictionary
    assert item._name == "items[0]"
    
    # Test accessing dictionary values through the proxy
    assert isinstance(getattr(item, 'name'), ValidationProxy)
    
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