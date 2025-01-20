"""Tests for file utilities."""

import os
from pathlib import Path
import pytest
from typing import Any, Dict, List, Union
from pyfakefs.fake_filesystem import FakeFilesystem

from openai_structured.cli.file_utils import (
    FileInfo,
    collect_files,
    collect_files_from_pattern,
    collect_files_from_directory,
)
from openai_structured.cli.errors import (
    FileNotFoundError,
    DirectoryNotFoundError,
    PathSecurityError,
)


def test_file_info_creation(fs: FakeFilesystem) -> None:
    """Test FileInfo creation and properties."""
    # Create test file
    fs.create_file("test.txt", contents="test content")
    
    # Create FileInfo instance
    file_info = FileInfo(name="test", path="test.txt")
    
    # Check basic properties
    assert file_info.name == "test"
    assert file_info.path == "test.txt"
    assert file_info.abs_path == os.path.abspath("test.txt")
    assert file_info.extension == "txt"
    
    # Check initial state - content should be loaded
    assert file_info._content == "test content"
    assert isinstance(file_info._size, int)
    assert isinstance(file_info._mtime, float)
    assert file_info._encoding is not None
    assert file_info._hash is not None
    assert file_info._stats_loaded is True
    
    # Content should be available immediately
    assert file_info.content == "test content"


def test_file_info_cache_update(fs: FakeFilesystem) -> None:
    """Test cache update mechanism."""
    fs.create_file("test.txt", contents="test content")
    
    file_info = FileInfo(name="test", path="test.txt")
    
    # Update from cache
    cached_content = "cached content"
    file_info.update_cache(
        content=cached_content,
        encoding="utf-8",
        hash_value="test_hash"
    )
    
    assert file_info.content == cached_content
    assert file_info.encoding == "utf-8"
    assert file_info.hash == "test_hash"


def test_file_info_property_protection(fs: FakeFilesystem) -> None:
    """Test that private fields cannot be set directly."""
    fs.create_file("test.txt", contents="test content")
    
    file_info = FileInfo(name="test", path="test.txt")
    
    # Attempt to set private fields should raise AttributeError
    with pytest.raises(AttributeError):
        file_info._content = "new content"  # pyright: ignore[reportPrivateUsage]
    
    with pytest.raises(AttributeError):
        file_info._hash = "new hash"  # pyright: ignore[reportPrivateUsage]


def test_file_info_directory_traversal(fs: FakeFilesystem) -> None:
    """Test FileInfo protection against directory traversal."""
    # Set up directory structure
    fs.create_dir("/base")
    fs.create_file("/base/test.txt", contents="test")
    fs.create_file("/outside/test.txt", contents="test")
    
    # Change to base directory
    os.chdir("/base")
    
    # Try to access file outside base directory
    with pytest.raises(PathSecurityError, match="Access denied: .* is outside base directory and not in allowed directories"):
        FileInfo.from_path("test", "../outside/test.txt")
    
    with pytest.raises(PathSecurityError, match="Directory mapping .* error: Directory .* is outside the current working directory .*"):
        collect_files(dir_args=["test=../outside"])


def test_file_info_missing_file(fs: FakeFilesystem) -> None:
    """Test FileInfo handling of missing files."""
    with pytest.raises(FileNotFoundError, match="File not found: nonexistent.txt"):
        FileInfo.from_path("test", "nonexistent.txt")


def test_collect_files_from_pattern(fs: FakeFilesystem) -> None:
    """Test collecting files using glob patterns."""
    # Create test files
    fs.create_file("test1.py", contents="test1")
    fs.create_file("test2.py", contents="test2")
    fs.create_file("test.txt", contents="test")
    fs.create_file("subdir/test3.py", contents="test3")
    
    # Test basic pattern
    files = collect_files_from_pattern("test", "*.py")
    assert len(files) == 2
    assert {f.path for f in files} == {"test1.py", "test2.py"}
    
    # Test with extension filter
    files = collect_files_from_pattern("test", "*.*", allowed_extensions={".py"})
    assert len(files) == 2
    assert {f.path for f in files} == {"test1.py", "test2.py"}
    
    # Test recursive pattern
    files = collect_files_from_pattern("test", "**/*.py", recursive=True)
    assert len(files) == 3
    assert {f.path for f in files} == {"test1.py", "test2.py", "subdir/test3.py"}


def test_collect_files_from_directory(fs: FakeFilesystem) -> None:
    """Test collecting files from directory."""
    # Create test files
    fs.create_file("dir/test1.py", contents="test1")
    fs.create_file("dir/test2.py", contents="test2")
    fs.create_file("dir/test.txt", contents="test")
    fs.create_file("dir/subdir/test3.py", contents="test3")
    
    # Test non-recursive collection
    files = collect_files_from_directory("test", "dir")
    assert len(files) == 3
    assert {f.path for f in files} == {"dir/test1.py", "dir/test2.py", "dir/test.txt"}
    
    # Test recursive collection
    files = collect_files_from_directory("test", "dir", recursive=True)
    assert len(files) == 4
    assert {f.path for f in files} == {
        "dir/test1.py",
        "dir/test2.py",
        "dir/test.txt",
        "dir/subdir/test3.py"
    }
    
    # Test with extension filter
    files = collect_files_from_directory(
        "test", "dir", recursive=True, allowed_extensions={".py"}
    )
    assert len(files) == 3
    assert {f.path for f in files} == {
        "dir/test1.py",
        "dir/test2.py",
        "dir/subdir/test3.py"
    }


def test_collect_files(fs: FakeFilesystem) -> None:
    """Test collecting files from multiple sources."""
    # Create test files
    fs.create_file("single.txt", contents="single")
    fs.create_file("test1.py", contents="test1")
    fs.create_file("test2.py", contents="test2")
    fs.create_file("dir/test3.py", contents="test3")
    fs.create_file("dir/test4.py", contents="test4")
    
    # Test collecting single file
    result = collect_files(file_args=["single=single.txt"])
    assert len(result) == 1
    assert isinstance(result["single"], FileInfo)
    assert result["single"].path == "single.txt"
    
    # Test collecting multiple files
    result = collect_files(files_args=["tests=*.py"])
    assert len(result) == 1
    assert isinstance(result["tests"], list)
    file_list = result["tests"]
    assert len(file_list) == 2
    assert {f.path for f in file_list} == {"test1.py", "test2.py"}
    
    # Test collecting from directory
    result = collect_files(dir_args=["dir=dir"], recursive=True)
    assert len(result) == 1
    assert isinstance(result["dir"], list)
    dir_list = result["dir"]
    assert len(dir_list) == 2
    assert {f.path for f in dir_list} == {"dir/test3.py", "dir/test4.py"}
    
    # Test collecting from all sources
    result = collect_files(
        file_args=["single=single.txt"],
        files_args=["tests=*.py"],
        dir_args=["dir=dir"],
        recursive=True,
    )
    assert len(result) == 3
    assert isinstance(result["single"], FileInfo)
    assert isinstance(result["tests"], list)
    assert isinstance(result["dir"], list)


def test_collect_files_errors(fs: FakeFilesystem) -> None:
    """Test error handling in collect_files."""
    # Set up directory structure
    fs.create_dir("/base")
    fs.create_dir("/outside")
    fs.create_file("/outside/test.txt", contents="test")
    os.chdir("/base")  # Change to base directory
    
    # Test invalid file mapping
    with pytest.raises(ValueError, match="Invalid file mapping"):
        collect_files(file_args=["invalid_mapping"])
    
    # Test no files found for pattern
    with pytest.raises(ValueError, match="No files found matching pattern"):
        collect_files(files_args=["test=*.nonexistent"])
    
    # Test no files found in directory
    fs.create_dir("empty_dir")
    with pytest.raises(ValueError, match="No files found in directory"):
        collect_files(dir_args=["test=empty_dir"])
    
    # Test directory outside base
    with pytest.raises(PathSecurityError, match="Directory mapping .* error: Directory .* is outside the current working directory .*"):
        collect_files(dir_args=["test=../outside"])


def test_file_info_stats_loading(fs: FakeFilesystem) -> None:
    """Test that file stats are loaded with content."""
    fs.create_file("test.txt", contents="test content")
    
    # Create FileInfo instance
    file_info = FileInfo(name="test", path="test.txt")
    
    # Check that stats are loaded
    assert file_info.size == len("test content")
    assert file_info.mtime is not None
    assert file_info._content == "test content"
    
    # Content should be available
    assert file_info.content == "test content"


def test_file_info_stats_security(fs: FakeFilesystem) -> None:
    """Test security checks when loading stats."""
    # Set up directory structure
    fs.create_dir("/base")
    fs.create_file("/base/test.txt", contents="test")
    fs.create_file("/outside/test.txt", contents="test")
    
    # Change to base directory
    os.chdir("/base")
    
    # Create FileInfo for file outside base directory
    with pytest.raises(PathSecurityError, match="Access denied: .* is outside base directory and not in allowed directories"):
        FileInfo(name="test", path="../outside/test.txt")


def test_file_info_missing_file_stats(fs: FakeFilesystem) -> None:
    """Test stats loading for missing file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        FileInfo(name="test", path="nonexistent.txt")


def test_file_info_content_errors(fs: FakeFilesystem) -> None:
    """Test error handling in content loading."""
    # Test with missing file
    with pytest.raises(FileNotFoundError, match="File not found"):
        FileInfo(name="test", path="nonexistent.txt")
    
    # Test security error
    fs.create_dir("/base")
    fs.create_file("/outside/test.txt", contents="test")
    os.chdir("/base")
    
    # Security error should be raised immediately
    with pytest.raises(PathSecurityError, match="Access denied: .* is outside base directory and not in allowed directories"):
        FileInfo(name="test", path="../outside/test.txt")