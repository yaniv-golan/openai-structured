"""Tests for template I/O operations."""

import os
import tempfile
from typing import Dict, List, Union

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

from openai_structured.cli.file_info import FileInfo
from openai_structured.cli.security import SecurityManager
from openai_structured.cli.template_io import (
    extract_metadata,
    extract_template_metadata,
    read_file,
)


@pytest.fixture  # type: ignore[misc]
def security_manager() -> SecurityManager:
    """Create a security manager for testing."""
    manager = SecurityManager(base_dir=os.getcwd())
    manager.add_allowed_dir(tempfile.gettempdir())
    return manager


def test_read_file_basic(
    fs: FakeFilesystem, security_manager: SecurityManager
) -> None:
    """Test basic file reading."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
    try:
        # Read file
        file_info = read_file(file_path, security_manager=security_manager)
        # Verify content
        assert file_info.content == "test content"
        assert file_info.encoding is not None
        assert file_info.hash is not None
    finally:
        os.unlink(file_path)


def test_read_file_with_encoding(security_manager: SecurityManager) -> None:
    """Test file reading with specific encoding."""
    content = "test content with unicode: 🚀"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(content)
        f.flush()
        file_path = f.name
    try:
        file_info = read_file(
            file_path, encoding="utf-8", security_manager=security_manager
        )
        assert file_info.content == content
        assert file_info.encoding == "utf-8"
    finally:
        os.unlink(file_path)


def test_read_file_content_loading(security_manager: SecurityManager) -> None:
    """Test immediate content loading behavior."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name
    try:
        # Create FileInfo - content should be loaded immediately
        file_info = read_file(file_path, security_manager=security_manager)
        # Content should be available immediately
        assert file_info.content == "test content"
        # Internal state should show content is loaded
        assert getattr(file_info, "_FileInfo__content") == "test content"
    finally:
        os.unlink(file_path)


def test_read_file_not_found(security_manager: SecurityManager) -> None:
    """Test error handling for non-existent files."""
    with pytest.raises(ValueError) as exc:
        read_file("nonexistent_file.txt", security_manager=security_manager)
    assert "File not found" in str(exc.value)


def test_read_file_caching(security_manager: SecurityManager) -> None:
    """Test file content caching."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name

    try:
        # First read should cache the file
        file_info1 = read_file(file_path, security_manager=security_manager)
        initial_content = file_info1.content

        # Second read should use cached content
        file_info2 = read_file(file_path, security_manager=security_manager)
        assert file_info2.content == initial_content

        # Modify file
        with open(file_path, "w") as f:
            f.write("new content")

        # Third read should detect file change and update cache
        file_info3 = read_file(file_path, security_manager=security_manager)
        assert file_info3.content == "new content"
    finally:
        os.unlink(file_path)


def test_extract_metadata(security_manager: SecurityManager) -> None:
    """Test metadata extraction from FileInfo."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name

    try:
        file_info = read_file(file_path, security_manager=security_manager)
        metadata = extract_metadata(file_info)

        # Test basic metadata
        assert metadata["name"] == os.path.basename(file_path)
        assert metadata["path"] == file_path
        assert metadata["abs_path"] == os.path.realpath(file_path)
        assert isinstance(metadata["mtime"], float)

        # Test optional metadata
        assert "encoding" not in metadata
        assert "mime_type" not in metadata
    finally:
        os.unlink(file_path)


def test_extract_template_metadata(security_manager: SecurityManager) -> None:
    """Test metadata extraction from template and context."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        f.flush()
        file_path = f.name

    try:
        template_str = "template.j2"
        context: Dict[str, Union[FileInfo, Dict[str, str], List[str], str]] = {
            "file": read_file(file_path, security_manager=security_manager),
            "config": {"key": "value"},
            "items": ["item1", "item2"],
            "name": "test",
        }

        metadata = extract_template_metadata(template_str, context)

        # Test template metadata
        assert metadata["template"]["path"] == "template.j2"

        # Test context metadata
        assert metadata["context"]["file_info_vars"] == ["file"]
        assert metadata["context"]["dict_vars"] == ["config"]
        assert metadata["context"]["list_vars"] == ["items"]
    finally:
        os.unlink(file_path)
