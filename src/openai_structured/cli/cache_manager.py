"""Cache management for file content.

This module provides a thread-safe cache manager for file content
with LRU eviction and automatic invalidation on file changes.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from cachetools import LRUCache
from cachetools.keys import hashkey

logger = logging.getLogger(__name__)

# Type alias for cache keys
CacheKey = Tuple[Any, ...]


@dataclass(frozen=True)
class CacheEntry:
    """Represents a cached file entry.

    Note: This class is immutable (frozen) to ensure thread safety
    when used as a cache value.
    """

    content: str
    encoding: Optional[str]
    hash_value: Optional[str]
    mtime: float
    size: int


class FileCache:
    """Thread-safe LRU cache for file content with size limit."""

    def __init__(self, max_size_bytes: int = 50 * 1024 * 1024):  # 50MB default
        """Initialize cache with maximum size in bytes.

        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        self._max_size = max_size_bytes
        self._current_size = 0
        self._cache: LRUCache[CacheKey, CacheEntry] = LRUCache(maxsize=1024)

    def _remove_entry(self, key: CacheKey) -> None:
        """Remove entry from cache and update size.

        Args:
            key: Cache key to remove
        """
        entry = self._cache.get(key)
        if entry is not None:
            self._current_size -= entry.size
        self._cache.pop(key, None)

    def get(self, path: str, current_mtime: float) -> Optional[CacheEntry]:
        """Get cache entry if it exists and is valid.

        Args:
            path: Absolute path to the file
            current_mtime: Current modification time of the file

        Returns:
            CacheEntry if valid cache exists, None otherwise
        """
        key = hashkey(path)
        entry = self._cache.get(key)

        if entry is None:
            return None

        # Check if file has been modified
        if entry.mtime != current_mtime:
            self._remove_entry(key)
            return None

        return entry

    def put(
        self,
        path: str,
        content: str,
        encoding: Optional[str],
        hash_value: Optional[str],
        mtime: float,
    ) -> None:
        """Add or update cache entry.

        Args:
            path: Absolute path to the file
            content: File content
            encoding: File encoding
            hash_value: Content hash
            mtime: File modification time
        """
        size = len(content.encode("utf-8"))

        if size > self._max_size:
            logger.warning(
                "File %s size (%d bytes) exceeds cache max size (%d bytes)",
                path,
                size,
                self._max_size,
            )
            return

        key = hashkey(path)
        self._remove_entry(key)

        entry = CacheEntry(content, encoding, hash_value, mtime, size)

        while self._current_size + size > self._max_size and self._cache:
            _, evicted = self._cache.popitem()
            self._current_size -= evicted.size

        self._cache[key] = entry
        self._current_size += size
