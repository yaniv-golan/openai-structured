"""Tests for template I/O operations."""

import os
import tempfile
from typing import Dict, Any, cast, Union, List

import pytest
from pytest import ExceptionInfo

from openai_structured.cli.template_io import (
    read_file,
    extract_metadata,
    extract_template_metadata
)
from openai_structured.cli.file_utils import FileInfo

def test_read_file_basic() -> None:
    """Test basic file reading."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
        
    try:
        # Read file
        file_info = read_file(file_path)
        
        # Verify content
        assert file_info.content == "test content"
        assert file_info.encoding is not None
        assert file_info.hash is not None
    finally:
        os.unlink(file_path)

def test_read_file_with_encoding() -> None:
    """Test file reading with specific encoding."""
    content = "test content with unicode: 🚀"
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write(content)
        f.flush()
        file_path = f.name
        
    try:
        file_info = read_file(file_path, encoding='utf-8')
        assert file_info.content == content
        assert file_info.encoding == 'utf-8'
    finally:
        os.unlink(file_path)

def test_read_file_content_loading() -> None:
    """Test immediate content loading behavior."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name

    try:
        # Create FileInfo - content should be loaded immediately
        file_info = read_file(file_path)
        
        # Content should be available immediately
        assert file_info.content == "test content"
        
        # Internal state should show content is loaded
        assert file_info._content == "test content"
        
        # Second access should return same content
        assert file_info.content == "test content"
    finally:
        os.unlink(file_path)

def test_read_file_not_found() -> None:
    """Test error handling for non-existent files."""
    with pytest.raises(ValueError) as exc:
        read_file("nonexistent_file.txt")
    assert "File not found" in str(exc.value)

def test_read_file_caching() -> None:
    """Test file content caching."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
        
    try:
        # First read should cache the file
        file_info1 = read_file(file_path)
        mtime1 = os.path.getmtime(file_path)
        
        # Second read should use cached content
        file_info2 = read_file(file_path)
        assert file_info2.content == file_info1.content
        
        # Modify file
        with open(file_path, 'w') as f:
            f.write("new content")
        
        # Third read should detect file change and update cache
        file_info3 = read_file(file_path)
        assert file_info3.content == "new content"
    finally:
        os.unlink(file_path)

def test_extract_metadata() -> None:
    """Test metadata extraction from FileInfo."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
        
    try:
        file_info = read_file(file_path)
        metadata = extract_metadata(file_info)
        
        # Test basic metadata
        assert metadata["name"] == os.path.basename(file_path)
        assert metadata["path"] == file_path
        assert metadata["abs_path"] == os.path.realpath(file_path)
        assert metadata["content"] == "test content"
        assert metadata["size"] == len("test content")
        assert isinstance(metadata["mtime"], float)
        
        # Test optional metadata
        assert "encoding" not in metadata
        assert "mime_type" not in metadata
    finally:
        os.unlink(file_path)

def test_extract_template_metadata() -> None:
    """Test metadata extraction from template and context."""
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
        
    try:
        template_str = "template.j2"
        context: Dict[str, Union[FileInfo, Dict[str, str], List[str], str]] = {
            "file": read_file(file_path),
            "config": {"key": "value"},
            "items": ["item1", "item2"],
            "name": "test"
        }
        
        metadata = extract_template_metadata(template_str, context)
        
        # Test template metadata
        assert metadata["template"]["is_file"] is True
        assert metadata["template"]["path"] == "template.j2"
        
        # Test context metadata
        assert set(metadata["context"]["variables"]) == {"file", "config", "items", "name"}
        assert metadata["context"]["file_info_vars"] == ["file"]
        assert metadata["context"]["dict_vars"] == ["config"]
        assert metadata["context"]["list_vars"] == ["items"]
    finally:
        os.unlink(file_path) 