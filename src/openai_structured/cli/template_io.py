"""File I/O operations for template processing.

This module provides functionality for file operations related to template processing:
1. Reading files with encoding detection and caching
2. Extracting metadata from files and templates
3. Managing file content caching and eviction
4. Progress tracking for file operations

Key Components:
    - read_file: Main function for reading files
    - extract_metadata: Extract metadata from files
    - extract_template_metadata: Extract metadata from templates
    - Cache management for file content

Examples:
    Basic file reading:
    >>> file_info = read_file('example.txt')
    >>> print(file_info.name)  # 'example.txt'
    >>> print(file_info.content)  # File contents
    >>> print(file_info.encoding)  # Detected encoding

    Lazy loading:
    >>> file_info = read_file('large_file.txt', lazy=True)
    >>> # Content not loaded yet
    >>> print(file_info.content)  # Now content is loaded
    >>> print(file_info.size)  # File size in bytes

    Metadata extraction:
    >>> metadata = extract_metadata(file_info)
    >>> print(metadata['size'])  # File size
    >>> print(metadata['encoding'])  # File encoding
    >>> print(metadata['mtime'])  # Last modified time

    Template metadata:
    >>> template = "Hello {{ name }}, files: {% for f in files %}{{ f.name }}{% endfor %}"
    >>> metadata = extract_template_metadata(template)
    >>> print(metadata['variables'])  # ['name', 'files']
    >>> print(metadata['has_loops'])  # True
    >>> print(metadata['filters'])  # []

    Cache management:
    >>> # Files are automatically cached
    >>> file_info1 = read_file('example.txt')
    >>> file_info2 = read_file('example.txt')  # Uses cached content
    >>> # Cache is invalidated if file changes
    >>> # Large files evicted from cache based on size

Notes:
    - Automatically detects file encoding
    - Caches file content for performance
    - Tracks file modifications
    - Provides progress updates for large files
    - Handles various error conditions gracefully
"""

import logging
import os
import threading
from typing import Any, Dict, Optional

from jinja2 import Environment

from .file_utils import FileInfo
from .progress import ProgressContext

logger = logging.getLogger(__name__)

# Cache settings
MAX_CACHE_SIZE = 50 * 1024 * 1024  # 50MB total cache size
_cache_lock = threading.Lock()
_file_mtimes: Dict[str, float] = {}
_file_cache: Dict[str, str] = {}
_file_encodings: Dict[str, str] = {}
_file_hashes: Dict[str, str] = {}
_cache_size: int = 0


def read_file(
    file_path: str,
    encoding: Optional[str] = None,
    progress_enabled: bool = True,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
) -> FileInfo:
    """Read a file and return its contents."""
    logger = logging.getLogger(__name__)
    logger.debug("\n=== read_file called ===")
    logger.debug("Args: file_path=%s, encoding=%s", file_path, encoding)

    with ProgressContext(
        "Reading file", show_progress=progress_enabled
    ) as progress:
        try:
            if progress:
                progress.update(1)  # Update progress for setup

            # Get absolute path and check file exists
            abs_path = os.path.abspath(file_path)
            logger.debug("Absolute path: %s", abs_path)
            if not os.path.isfile(abs_path):
                raise ValueError(f"File not found: {file_path}")

            # Check if file is in cache and up to date
            mtime = os.path.getmtime(abs_path)
            with _cache_lock:
                logger.debug(
                    "Cache state - mtimes: %s, cache: %s",
                    _file_mtimes,
                    _file_cache,
                )
                if (
                    abs_path in _file_mtimes
                    and _file_mtimes[abs_path] == mtime
                ):
                    logger.debug("Cache hit for %s", abs_path)
                    if progress:
                        progress.update(1)  # Update progress for cache hit
                    # Create FileInfo and update from cache
                    file_info = FileInfo.from_path(path=file_path)
                    file_info.update_cache(
                        content=_file_cache[abs_path],
                        encoding=_file_encodings.get(abs_path),
                        hash_value=_file_hashes.get(abs_path),
                    )
                    return file_info

            # Create new FileInfo - content will be loaded immediately
            file_info = FileInfo.from_path(path=file_path)

            # Update cache with loaded content
            with _cache_lock:
                logger.debug("Updating cache for %s", abs_path)
                _file_mtimes[abs_path] = mtime
                _file_cache[abs_path] = file_info.content
                if file_info.encoding is not None:
                    _file_encodings[abs_path] = file_info.encoding
                if file_info.hash is not None:
                    _file_hashes[abs_path] = file_info.hash

                global _cache_size
                _cache_size = sum(
                    len(content) for content in _file_cache.values()
                )
                logger.debug("Cache updated - size: %d", _cache_size)

                # Remove old entries if cache is too large
                while _cache_size > MAX_CACHE_SIZE:
                    oldest = min(_file_mtimes.items(), key=lambda x: x[1])
                    old_path = oldest[0]
                    if old_path in _file_cache:
                        _cache_size -= len(_file_cache[old_path])
                        del _file_cache[old_path]
                        del _file_encodings[old_path]
                        del _file_hashes[old_path]
                    del _file_mtimes[old_path]
                    logger.debug("Removed old cache entry: %s", old_path)

            if progress:
                progress.update(1)  # Update progress for successful read

            return file_info

        except Exception as e:
            logger.error("Error reading file: %s", str(e))
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to read file: {str(e)}")


def extract_metadata(file_info: FileInfo) -> Dict[str, Any]:
    """Extract metadata from a FileInfo object.

    This function respects lazy loading - it will not force content loading
    if the content hasn't been loaded yet.
    """
    metadata: Dict[str, Any] = {
        "name": os.path.basename(file_info.path),
        "path": file_info.path,
        "abs_path": os.path.realpath(file_info.path),
        "mtime": file_info.mtime,
    }

    # Only include content-related fields if content is already loaded
    if not file_info.lazy or file_info._content is not None:
        metadata["content"] = file_info.content
        metadata["size"] = len(file_info.content) if file_info.content else 0

    return metadata


def extract_template_metadata(
    template_str: str,
    context: Dict[str, Any],
    jinja_env: Optional[Environment] = None,
    progress_enabled: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Extract metadata about a template string."""
    metadata: Dict[str, Dict[str, Any]] = {
        "template": {"is_file": True, "path": template_str},
        "context": {
            "variables": sorted(context.keys()),
            "dict_vars": [],
            "list_vars": [],
            "file_info_vars": [],
            "other_vars": [],
        },
    }

    with ProgressContext(
        description="Analyzing template", show_progress=progress_enabled
    ) as progress:
        # Categorize variables by type
        for key, value in context.items():
            if isinstance(value, dict):
                metadata["context"]["dict_vars"].append(key)
            elif isinstance(value, list):
                metadata["context"]["list_vars"].append(key)
            elif isinstance(value, FileInfo):
                metadata["context"]["file_info_vars"].append(key)
            else:
                metadata["context"]["other_vars"].append(key)

        # Sort lists for consistent output
        for key in ["dict_vars", "list_vars", "file_info_vars", "other_vars"]:
            metadata["context"][key].sort()

        if progress.enabled:
            progress.current = 1

        return metadata
