"""Tests for template schema validation."""

from typing import Any, Dict, List, cast, Iterator, Tuple

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
    debug = proxy.debug
    assert isinstance(debug, ValidationProxy)
    assert debug._var_name == "config.debug"
    
    # Test nested access
    nested = proxy.settings.mode
    assert isinstance(nested, ValidationProxy)
    assert nested._var_name == "config.settings.mode"
    
    # Test dictionary methods
    items: Iterator[Tuple[str, Any]] = proxy.items()
    items_list = list(items)
    assert len(items_list) == 3
    assert 'key' in dict(items_list)
    
    keys: Iterator[str] = proxy.keys()
    keys_list = list(keys)
    assert len(keys_list) == 3
    assert 'debug' in keys_list
    
    values: Iterator[Any] = proxy.values()
    values_list = list(values)
    assert len(values_list) == 3
    assert any(isinstance(v, DictProxy) for v in values_list)

    # Test invalid access
    with pytest.raises(ValueError, match="undefined attribute 'config.invalid'"):
        _ = proxy.invalid

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
    assert all(isinstance(item, DictProxy) for item in items)
    
    # Test indexing
    item = proxy[0]
    assert isinstance(item, DictProxy)
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
        assert str(proxy.name) == os.path.basename(file_path)
        assert str(proxy.path) == file_path
        assert str(proxy.abs_path) == os.path.realpath(file_path)
        # Content should be empty string to support filtering
        assert str(proxy.content) == ""
        assert str(proxy.size) == str(len("test content"))

        # Test invalid attribute access
        with pytest.raises(ValueError):
            _ = getattr(proxy, 'invalid_attr')

    finally:
        os.unlink(file_path) 