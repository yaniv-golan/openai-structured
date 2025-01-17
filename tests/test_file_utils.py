"""Tests for file utilities."""

import os
from pathlib import Path
from typing import List, Set
from unittest.mock import mock_open, patch

import pytest

from openai_structured.cli.file_utils import (
    FileInfo,
    collect_files,
    collect_files_from_directory,
    collect_files_from_pattern,
)


def test_file_info_from_path() -> None:
    """Test creating FileInfo from path."""
    with patch("os.path.isfile", return_value=True):
        # Test valid file
        info = FileInfo.from_path("test", "test.txt")
        assert info.name == "test"
        assert info.path == "test.txt"
        assert info.abs_path.endswith("test.txt")
        
        # Test directory property
        info = FileInfo.from_path("test", "dir/test.txt")
        assert info.dir == "dir"
        
        # Test path traversal
        with pytest.raises(ValueError, match="Access denied"):
            FileInfo.from_path("test", "../outside.txt")
            
        # Test missing file
        with patch("os.path.isfile", return_value=False):
            with pytest.raises(OSError, match="File not found"):
                FileInfo.from_path("test", "missing.txt")


def test_collect_files_from_pattern() -> None:
    """Test collecting files from glob pattern."""
    mock_glob = ["file1.py", "file2.py", "file3.js"]
    
    with (
        patch("glob.glob", return_value=mock_glob),
        patch("os.path.isfile", return_value=True),
    ):
        # Test basic collection
        files = collect_files_from_pattern("test", "*.py")
        assert len(files) == 3
        assert all(isinstance(f, FileInfo) for f in files)
        assert [f.name for f in files] == ["test_1", "test_2", "test_3"]
        
        # Test with extension filter
        files = collect_files_from_pattern(
            "test", "*.py", allowed_extensions={".py"}
        )
        assert len(files) == 2
        assert all(f.path.endswith(".py") for f in files)
        
        # Test invalid pattern
        with patch("glob.glob", side_effect=Exception("Invalid pattern")):
            with pytest.raises(ValueError, match="Invalid glob pattern"):
                collect_files_from_pattern("test", "[invalid")


def test_collect_files_from_directory() -> None:
    """Test collecting files from directory."""
    with (
        patch("os.path.isdir", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch(
            "glob.glob",
            return_value=["dir/file1.py", "dir/file2.py", "dir/sub/file3.py"],
        ),
    ):
        # Test non-recursive collection
        files = collect_files_from_directory("test", "dir")
        assert len(files) == 3
        assert all(isinstance(f, FileInfo) for f in files)
        
        # Test with extension filter
        files = collect_files_from_directory(
            "test", "dir", allowed_extensions={".py"}
        )
        assert len(files) == 3
        assert all(f.path.endswith(".py") for f in files)
        
        # Test directory traversal
        with pytest.raises(ValueError, match="Access denied"):
            collect_files_from_directory("test", "../outside")
            
        # Test missing directory
        with patch("os.path.isdir", return_value=False):
            with pytest.raises(OSError, match="Directory not found"):
                collect_files_from_directory("test", "missing")


def test_collect_files() -> None:
    """Test collecting files from CLI arguments."""
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch(
            "glob.glob",
            return_value=["test1.py", "test2.py", "dir/test3.py"],
        ),
    ):
        # Test single file
        result = collect_files(
            file_args=["input=test.py"],
            files_args=[],
            dir_args=[],
        )
        assert len(result) == 1
        assert isinstance(result["input"], FileInfo)
        assert result["input"].path == "test.py"
        
        # Test multiple files
        result = collect_files(
            file_args=[],
            files_args=["inputs=*.py"],
            dir_args=[],
        )
        assert len(result) == 1
        assert isinstance(result["inputs"], list)
        files = result["inputs"]
        assert isinstance(files, list)
        assert len(files) == 3
        assert all(isinstance(f, FileInfo) for f in files)
        
        # Test directory
        result = collect_files(
            file_args=[],
            files_args=[],
            dir_args=["dir=testdir"],
        )
        assert len(result) == 1
        assert isinstance(result["dir"], list)
        files = result["dir"]
        assert isinstance(files, list)
        assert len(files) == 3
        assert all(isinstance(f, FileInfo) for f in files)
        
        # Test invalid mappings
        with pytest.raises(ValueError, match="Invalid file mapping"):
            collect_files(
                file_args=["invalid"],
                files_args=[],
                dir_args=[],
            )
            
        with pytest.raises(ValueError, match="Invalid files mapping"):
            collect_files(
                file_args=[],
                files_args=["invalid"],
                dir_args=[],
            )
            
        with pytest.raises(ValueError, match="Invalid directory mapping"):
            collect_files(
                file_args=[],
                files_args=[],
                dir_args=["invalid"],
            ) 