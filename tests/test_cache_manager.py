"""Tests for cache manager implementation."""

import dataclasses
import os
import time
from typing import Generator

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem
from pyfakefs.fake_filesystem_unittest import Patcher
from typing_extensions import TypeAlias

from openai_structured.cli.cache_manager import CacheEntry, FileCache

# Type alias for the fixture
FSFixture: TypeAlias = Generator[FakeFilesystem, None, None]


@pytest.fixture  # type: ignore[misc] # Decorator is typed in pytest's stub
def fs() -> FSFixture:
    """Fixture to set up fake filesystem."""
    with Patcher() as patcher:
        fs = patcher.fs
        assert fs is not None  # Type assertion for mypy
        yield fs


def test_cache_entry_immutability() -> None:
    """Test that CacheEntry is immutable."""
    entry = CacheEntry(
        content="test",
        encoding="utf-8",
        hash_value="hash",
        mtime=123.0,
        size=4,
    )

    # Verify we can't modify attributes
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.content = "new"  # type: ignore
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.encoding = "ascii"  # type: ignore
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.hash_value = "new_hash"  # type: ignore


def test_cache_basic_operations(fs: FakeFilesystem) -> None:
    """Test basic cache operations."""
    cache = FileCache(max_size_bytes=1024)  # Small cache for testing

    # Create test file
    fs.create_file("/test.txt", contents="test content")
    mtime = os.path.getmtime("/test.txt")

    # Test cache miss
    assert cache.get("/test.txt", mtime) is None

    # Test cache put and hit
    cache.put("/test.txt", "test content", "utf-8", "hash", mtime)
    entry = cache.get("/test.txt", mtime)
    assert entry is not None
    assert entry.content == "test content"
    assert entry.encoding == "utf-8"
    assert entry.hash_value == "hash"
    assert entry.mtime == mtime


def test_cache_size_limit(fs: FakeFilesystem) -> None:
    """Test cache size limiting."""
    # Create cache with 100 byte limit
    cache = FileCache(max_size_bytes=100)

    # Create test files
    fs.create_file("/small.txt", contents="small")
    fs.create_file("/large.txt", contents="x" * 200)  # Exceeds cache size

    small_mtime = os.path.getmtime("/small.txt")
    large_mtime = os.path.getmtime("/large.txt")

    # Small file should be cached
    cache.put("/small.txt", "small", "utf-8", "hash", small_mtime)
    assert cache.get("/small.txt", small_mtime) is not None

    # Large file should be rejected
    cache.put("/large.txt", "x" * 200, "utf-8", "hash", large_mtime)
    assert cache.get("/large.txt", large_mtime) is None


def test_cache_lru_eviction(fs: FakeFilesystem) -> None:
    """Test LRU eviction behavior."""
    # Create cache with size limit that can hold 2 small files
    # Each file is ~8 bytes (content1, content2, content3)
    cache = FileCache(max_size_bytes=16)  # Only fits 2 files

    # Create test files
    files = ["/file1.txt", "/file2.txt", "/file3.txt"]
    mtimes = []

    for i, path in enumerate(files, 1):
        fs.create_file(path, contents=f"content{i}")
        mtimes.append(os.path.getmtime(path))
        cache.put(path, f"content{i}", "utf-8", f"hash{i}", mtimes[-1])
        time.sleep(0.01)  # Ensure different mtimes

    # First file should be evicted due to size limit
    assert cache.get(files[0], mtimes[0]) is None
    # Later files should still be cached
    assert cache.get(files[1], mtimes[1]) is not None
    assert cache.get(files[2], mtimes[2]) is not None


def test_cache_modification_detection(fs: FakeFilesystem) -> None:
    """Test detection of file modifications."""
    cache = FileCache()

    # Create and cache file
    fs.create_file("/test.txt", contents="original")
    original_mtime = os.path.getmtime("/test.txt")
    cache.put("/test.txt", "original", "utf-8", "hash1", original_mtime)

    # Modify file
    time.sleep(0.01)  # Ensure different mtime
    fs.remove("/test.txt")  # Remove first
    fs.create_file("/test.txt", contents="modified")  # Then create new
    new_mtime = os.path.getmtime("/test.txt")

    # Cache should detect modification
    assert original_mtime != new_mtime
    assert cache.get("/test.txt", new_mtime) is None


def test_cache_concurrent_access() -> None:
    """Test concurrent cache access."""
    import random
    import threading

    cache = FileCache(max_size_bytes=1000)
    errors: list[Exception] = []

    def cache_operation(thread_id: int) -> None:
        """Perform random cache operations."""
        try:
            for _ in range(100):
                path = f"/file{random.randint(1, 5)}.txt"
                mtime = float(random.randint(1, 1000))

                if random.random() < 0.5:
                    # Read operation
                    cache.get(path, mtime)
                else:
                    # Write operation
                    cache.put(
                        path,
                        f"content{thread_id}",
                        "utf-8",
                        f"hash{thread_id}",
                        mtime,
                    )
        except Exception as e:
            errors.append(e)

    # Create and start threads
    threads = [
        threading.Thread(target=cache_operation, args=(i,)) for i in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Check for errors
    assert not errors, f"Encountered errors during concurrent access: {errors}"


def test_cache_key_collisions() -> None:
    """Test handling of potential key collisions."""
    cache = FileCache()
    mtime = time.time()

    # Test paths that might hash similarly
    paths = [
        "/test/path/file.txt",
        "/test/path/file.txt/",  # Trailing slash
        "//test/path/file.txt",  # Double slash
        "/test//path/file.txt",
        "/test/./path/file.txt",
        "/test/path/../path/file.txt",
    ]

    # Add each path to cache
    for i, path in enumerate(paths):
        cache.put(path, f"content{i}", "utf-8", f"hash{i}", mtime)

    # Verify each path has correct content
    for i, path in enumerate(paths):
        entry = cache.get(path, mtime)
        assert entry is not None
        assert entry.content == f"content{i}"
