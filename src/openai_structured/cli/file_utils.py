"""File utilities for the CLI.

This module provides utilities for file operations with security controls:

1. File Information:
   - FileInfo class for safe file access and metadata
   - Support for file content caching
   - Automatic encoding detection

2. Path Handling:
   - Supports ~ expansion for home directory
   - Supports environment variable expansion
   - Security checks for file access
   - Requires explicit allowed directories for access outside CWD

3. Security Features:
   - Directory traversal prevention
   - Explicit allowed directory configuration
   - Temporary file access controls
   - Path validation and normalization

Usage Examples:
    Basic file access (from current directory):
    >>> info = FileInfo.from_path("var_name", "local_file.txt")
    >>> content = info.content

    Access home directory files (requires --allowed-dir):
    >>> info = FileInfo.from_path("var_name", "~/file.txt", allowed_dirs=["~/"])
    >>> content = info.content

    Multiple file collection:
    >>> files = collect_files(
    ...     file_args=["var=path.txt"],
    ...     allowed_dirs=["/allowed/path"],
    ...     recursive=True
    ... )

Security Notes:
    - Files must be in current directory or explicitly allowed directories
    - Use --allowed-dir to access files outside current directory
    - Home directory (~) is not automatically allowed
    - Environment variables are expanded in paths
"""

import glob
import hashlib
import inspect
import logging
import os
from typing import Any, Dict, List, Optional, Type, Union

import chardet

from .errors import (
    DirectoryNotFoundError,
    FileNotFoundError,
    PathSecurityError,
)
from .path_utils import validate_path_mapping
from .security import SecurityManager, is_temp_file
from .security_types import SecurityManagerProtocol
from .utils import parse_mapping

# Type for values in template context
TemplateValue = Union[str, "FileInfo", List["FileInfo"]]


def _get_security_manager() -> Type[SecurityManagerProtocol]:
    """Get the SecurityManager class.

    Returns:
        The SecurityManager class type
    """
    return SecurityManager


def expand_path(path: str, force_absolute: bool = False) -> str:
    """Expand user home directory and environment variables in path.

    Args:
        path: Path that may contain ~ or environment variables
        force_absolute: Whether to force conversion to absolute path

    Returns:
        Expanded path, maintaining relative paths unless force_absolute=True
        or the path contains ~ or environment variables
    """
    # First expand user and environment variables
    expanded = os.path.expanduser(os.path.expandvars(path))

    # If the path hasn't changed and we're not forcing absolute, keep it relative
    if expanded == path and not force_absolute:
        return path

    # Otherwise return absolute path
    return os.path.abspath(expanded)


class FileInfo:
    """File information class that includes file path and content."""

    def __init__(
        self, path: str, security_manager: Optional[SecurityManager] = None
    ) -> None:
        """Initialize FileInfo instance.

        Args:
            path: Path to the file
            security_manager: Optional security manager for path validation

        Raises:
            FileNotFoundError: If file does not exist
            PathSecurityError: If file is outside allowed directories
            OSError: If file cannot be read
        """
        self._initialized = False  # Add initialization flag
        self.logger = logging.getLogger("ostruct")
        self.logger.debug("Creating FileInfo instance for %s", path)
        self._path = path
        self._abs_path = os.path.abspath(path)
        self._content = None
        self._encoding = None
        self._hash = None
        self._security_checked = False
        self._security_manager = security_manager
        self._stats_loaded = False
        self._size: Optional[int] = None
        self._mtime: Optional[float] = None

        # Load stats immediately
        self._load_stats()

        # Detect encoding immediately
        try:
            self._encoding = detect_encoding(self._abs_path)
        except Exception:
            self.logger.warning("Error detecting encoding")
            self._encoding = "utf-8"  # Fallback to UTF-8

        # Load content and compute hash
        try:
            with open(self._abs_path, "r", encoding=self._encoding) as f:
                content = f.read()
                self._content = content
                self._hash = hashlib.sha256(content.encode()).hexdigest()
        except Exception:
            self.logger.warning("Error loading content")
            raise RuntimeError("Failed to load content")

        self._initialized = True  # Mark initialization as complete

    @property
    def path(self) -> str:
        """Get the file path.

        Returns:
            The file path as a string
        """
        return self._path

    @classmethod
    def from_path(
        cls, path: str, security_manager: Optional[SecurityManager] = None
    ) -> "FileInfo":
        """Create a FileInfo instance from a file path.

        Args:
            path: The path to the file
            security_manager: Optional security manager for path validation

        Returns:
            A new FileInfo instance

        Raises:
            FileNotFoundError: If file does not exist
            PathSecurityError: If file is outside allowed directories
            OSError: If file cannot be read
        """
        logger = logging.getLogger("ostruct")
        logger.debug("Creating FileInfo from path: %s", path)

        expanded_path = expand_path(path)
        logger.debug("Expanded path: %s", expanded_path)

        # For security checks, we need the absolute path
        abs_path = os.path.abspath(expanded_path)

        # Check file exists before creating instance
        if not os.path.exists(abs_path):
            msg = (
                f"File not found: {path}\n"
                f"Expanded path: {expanded_path}\n"
                "Note: Use --allowed-dir to allow access to directories outside the current path"
            )
            raise FileNotFoundError(msg)

        if security_manager:
            logger.debug("Checking security for path: %s", expanded_path)
            security_manager.validate_path(expanded_path)
            logger.debug("Security check passed for path: %s", expanded_path)

        return cls(path=expanded_path, security_manager=security_manager)

    @property
    def name(self) -> str:
        """Get variable name for the file."""
        return self.path.split("/")[-1]

    @property
    def abs_path(self) -> str:
        """Get absolute path to file with symlinks resolved."""
        return os.path.realpath(self.path)

    @property
    def content(self) -> str:
        """Get file content.

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file does not exist
            OSError: If file cannot be read
            PathSecurityError: If file access is denied
        """
        if self._content is None:
            try:
                self.load_content()
            except (
                OSError,
                IOError,
                FileNotFoundError,
                PathSecurityError,
            ) as e:
                self.logger.error(
                    "Failed to load content for %s: %s", self.path, e
                )
                raise
            except Exception:
                self.logger.error(
                    "Unexpected error loading content for %s",
                    self.path,
                )
                raise RuntimeError("Failed to load content")
        return self._content if self._content is not None else ""

    @property
    def size(self) -> Optional[int]:
        """Get file size in bytes."""
        if not self._stats_loaded:
            try:
                self._load_stats()
            except (FileNotFoundError, PathSecurityError) as e:
                self.logger.error(str(e))
                raise
            except Exception:
                self.logger.warning("Failed to load stats for %s", self.path)
                return None
        return self._size

    @property
    def mtime(self) -> Optional[float]:
        """Get file modification time."""
        if not self._stats_loaded:
            try:
                self._load_stats()
            except (FileNotFoundError, PathSecurityError) as e:
                self.logger.error(str(e))
                raise
            except Exception:
                self.logger.warning("Failed to load stats for %s", self.path)
                return None
        return self._mtime

    @property
    def encoding(self) -> Optional[str]:
        """Get file encoding."""
        return self._encoding

    @property
    def hash(self) -> Optional[str]:
        """Get file content hash."""
        return self._hash

    @property
    def extension(self) -> str:
        """Get file extension without the leading dot."""
        return os.path.splitext(self.path)[1].lstrip(".")

    @property
    def basename(self) -> str:
        """Get file basename."""
        return os.path.basename(self.path)

    @property
    def dirname(self) -> str:
        """Get directory name."""
        return os.path.dirname(self.path)

    @property
    def parent(self) -> str:
        """Get parent directory."""
        return os.path.dirname(self.path)

    @property
    def stem(self) -> str:
        """Get file stem (name without extension)."""
        return os.path.splitext(self.basename)[0]

    @property
    def suffix(self) -> str:
        """Get file extension (alias for extension)."""
        return self.extension

    @property
    def exists(self) -> bool:
        """Check if file exists."""
        return os.path.exists(self.path)

    @property
    def is_file(self) -> bool:
        """Check if path is a file."""
        return os.path.isfile(self.path)

    @property
    def is_dir(self) -> bool:
        """Check if path is a directory."""
        return os.path.isdir(self.path)

    @property
    def lazy(self) -> bool:
        """Get whether this instance uses lazy loading."""
        return False  # Always return False since lazy loading is removed

    def __setattr__(self, name: str, value: Any) -> None:
        """Override attribute setting to prevent modification of private fields after initialization.

        Args:
            name: Attribute name
            value: Value to set

        Raises:
            AttributeError: If attempting to modify a private field after initialization
        """
        # Allow all modifications during initialization
        if not hasattr(self, "_initialized") or not self._initialized:
            super().__setattr__(name, value)
            return

        # Allow updates from specific methods
        frame = inspect.currentframe()
        if frame is not None:
            caller = frame.f_back
            if caller is not None and caller.f_code.co_name in [
                "_check_security",
                "_load_stats",
                "load_content",
                "update_cache",
            ]:
                super().__setattr__(name, value)
                return

        # Prevent modification of private attributes after initialization
        if name.startswith("_") and hasattr(self, name):
            raise AttributeError(f"Cannot modify private attribute {name}")
        super().__setattr__(name, value)

    def _check_security(
        self, allowed_dirs: Optional[List[str]] = None
    ) -> None:
        """Perform security checks without loading stats.

        Args:
            allowed_dirs: Optional list of allowed directories

        Raises:
            PathSecurityError: If file access is denied
        """
        if self._security_checked:
            return

        # If we have a security manager, use it for validation
        if self._security_manager:
            try:
                self._security_manager.validate_path(self.path)
                self._security_checked = True
                return
            except PathSecurityError as e:
                self.logger.error(str(e))
                raise

        # Otherwise fall back to basic security checks
        abs_path = os.path.realpath(os.path.abspath(self.path))
        base_dir = os.path.realpath(os.getcwd())

        # Expand any allowed directories (force absolute for security checks)
        expanded_allowed_dirs = [
            expand_path(d, force_absolute=True) for d in (allowed_dirs or [])
        ]

        # Check if path is in base directory, temp directory, or allowed directory
        is_allowed = (
            os.path.commonpath([abs_path, base_dir]) == base_dir
            or is_temp_file(abs_path)
            or any(
                os.path.commonpath([abs_path, os.path.realpath(allowed_dir)])
                == os.path.realpath(allowed_dir)
                for allowed_dir in expanded_allowed_dirs
                if os.path.exists(allowed_dir)
            )
        )

        if not is_allowed:
            # Use enhanced PathSecurityError with expanded paths
            error = PathSecurityError.from_expanded_paths(
                original_path=str(self.path),
                expanded_path=abs_path,
                base_dir=base_dir,
                allowed_dirs=expanded_allowed_dirs if allowed_dirs else None,
                error_logged=False,  # Set to False to ensure error is logged
            )
            self.logger.error(str(error))
            error.error_logged = True  # Mark as logged after logging
            raise error

        self._security_checked = True

    def _load_stats(self) -> None:
        """Load file statistics and perform security checks.

        This method:
        1. Performs security checks (always)
        2. Loads basic file stats (size, mtime)

        Stats are loaded when first accessed via properties.

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If path is not a file
            OSError: If file cannot be read
            PathSecurityError: If file access is denied
        """
        # Always perform security check
        try:
            self._check_security()
        except PathSecurityError:
            raise  # Don't log again, already logged in _check_security

        if self._stats_loaded:
            self.logger.debug("Stats already loaded for %s", self.path)
            return

        self.logger.debug("Loading stats for %s", self.path)

        # Check file exists and is readable
        abs_path = os.path.realpath(self.path)
        if not os.path.exists(abs_path):
            msg = f"File not found: {self.path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)  # Using our custom FileNotFoundError
        if not os.path.isfile(abs_path):
            msg = f"Path is not a file: {self.path}"
            self.logger.error(msg)
            raise ValueError(msg)

        try:
            stats = os.stat(abs_path)
            self._stats = stats
            self._size = stats.st_size
            self._mtime = stats.st_mtime
            self._stats_loaded = True
            self.logger.debug("Successfully loaded stats for %s", self.path)
        except (OSError, IOError) as e:
            self.logger.error("Error reading file stats: %s", e)
            raise

    def load_content(self, encoding: Optional[str] = None) -> None:
        """Load file content if not already loaded.

        This method performs security checks and loads the file content.
        It is called automatically by the content property in lazy mode,
        or during initialization in non-lazy mode.

        The method uses a multi-step approach:
        1. Check if content is already loaded
        2. Perform security checks via _load_stats
        3. Detect or use provided encoding
        4. Load and decode file content
        5. Update instance state with loaded content

        Args:
            encoding: Optional encoding to use for reading the file.
                     If not provided, encoding will be detected.

        Raises:
            OSError: If file cannot be read or accessed
            UnicodeDecodeError: If file content cannot be decoded with detected/specified encoding
            PathSecurityError: If file access is not allowed
            RuntimeError: For unexpected errors during content loading
        """
        if self._content is not None:
            self.logger.debug("Content already loaded for %s", self.path)
            return

        self.logger.debug(
            "Loading content for %s (encoding=%s)", self.path, encoding
        )

        # Load stats first to perform security checks
        try:
            self._load_stats()
        except (FileNotFoundError, PathSecurityError) as e:
            self.logger.error("Security check failed for %s: %s", self.path, e)
            raise
        except Exception:
            self.logger.error(
                "Unexpected error in security check for %s",
                self.path,
            )
            raise RuntimeError("Security check failed")

        try:
            # Detect or use provided encoding
            file_encoding = encoding
            if file_encoding is None:
                try:
                    file_encoding = detect_encoding(self.path)
                    self.logger.debug(
                        "Using detected encoding %s for %s",
                        file_encoding,
                        self.path,
                    )
                except (OSError, ValueError):
                    self.logger.error(
                        "Error detecting encoding for %s", self.path
                    )
                    raise RuntimeError("Failed to detect encoding")

            # Attempt to read and decode file content
            try:
                with open(self.path, "r", encoding=file_encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                self.logger.error(
                    "Failed to decode %s with encoding %s",
                    self.path,
                    file_encoding,
                )
                # Try fallback to UTF-8 if a different encoding was attempted
                if file_encoding != "utf-8":
                    self.logger.debug(
                        "Attempting UTF-8 fallback for %s", self.path
                    )
                    try:
                        with open(self.path, "r", encoding="utf-8") as f:
                            content = f.read()
                        file_encoding = "utf-8"
                        self.logger.debug(
                            "UTF-8 fallback successful for %s", self.path
                        )
                    except UnicodeDecodeError:
                        self.logger.error(
                            "UTF-8 fallback failed for %s",
                            self.path,
                        )
                        raise
                else:
                    raise
            except OSError:
                self.logger.error("Failed to read %s", self.path)
                raise

            # Update instance state
            self._content = content
            self._encoding = file_encoding
            self._hash = hashlib.sha256(content.encode()).hexdigest()

            self.logger.debug(
                "Successfully loaded %d bytes from %s using encoding %s",
                len(content),
                self.path,
                file_encoding,
            )

        except (OSError, UnicodeDecodeError):
            # Let these exceptions propagate with their original type
            raise
        except Exception:
            self.logger.error(
                "Unexpected error loading content for %s",
                self.path,
            )
            raise RuntimeError("Failed to load content")

    def update_cache(
        self,
        content: str,
        encoding: Optional[str],
        hash_value: Optional[str] = None,
    ) -> None:
        """Update private fields from external cache logic."""
        self._content = content
        self._encoding = encoding
        self._hash = hash_value or hashlib.sha256(content.encode()).hexdigest()
        self.logger.debug("Updated cache for %s", self.path)

    def dir(self) -> str:
        """Get directory containing the file."""
        return os.path.dirname(self.abs_path)


def collect_files_from_pattern(
    pattern: str,
    security_manager: Optional[SecurityManager] = None,
) -> List[FileInfo]:
    """Collect files matching a glob pattern.

    Args:
        pattern: Glob pattern to match files
        security_manager: Optional security manager for path validation

    Returns:
        List of FileInfo objects for matching files

    Raises:
        PathSecurityError: If any matched file is outside base directory and not in allowed directories
        ValueError: If no files match pattern
    """
    # Expand glob pattern
    try:
        matches = glob.glob(pattern, recursive=True)
        matches.sort()  # Sort for consistent ordering
    except Exception as e:
        raise ValueError(f"Invalid glob pattern '{pattern}': {e}")

    # Filter and convert matches to FileInfo objects
    result: List[FileInfo] = []
    seen_paths = set()  # Track unique paths

    for path in matches:
        # Skip if already processed (handles case-insensitive duplicates)
        norm_path = os.path.normcase(path)
        if norm_path in seen_paths:
            continue
        seen_paths.add(norm_path)

        # Skip if not a file
        if not os.path.isfile(path):
            continue

        try:
            file_info = FileInfo.from_path(
                path=path, security_manager=security_manager
            )
            result.append(file_info)
        except PathSecurityError:
            # Propagate security errors
            raise
        except (OSError, ValueError):
            # Skip other errors but continue processing
            continue

    if not result:
        raise ValueError(f"No files found matching pattern: {pattern}")

    return result


def collect_files_from_directory(
    directory: str,
    base_dir: str,
    recursive: bool = False,
    allowed_extensions: Optional[List[str]] = None,
    allowed_dirs: Optional[List[str]] = None,
    security_manager: Optional[SecurityManager] = None,
) -> List[FileInfo]:
    """Collect files from a directory.

    Args:
        directory: Directory path relative to current directory
        recursive: Whether to traverse subdirectories
        allowed_extensions: Optional set of allowed file extensions (e.g. {'.py', '.js'})
        allowed_dirs: Optional list of allowed directories
        security_manager: Optional security manager for path validation

    Returns:
        List of FileInfo objects for files in directory

    Raises:
        DirectoryNotFoundError: If directory does not exist
        PathSecurityError: If directory is outside base directory and not in allowed directories
        ValueError: If no files are found in directory
    """
    # Resolve paths and check security first
    abs_dir = os.path.abspath(os.path.join(base_dir, directory))

    # Security check - prevent directory traversal
    if not abs_dir.startswith(base_dir) and not any(
        abs_dir.startswith(allowed_dir) for allowed_dir in (allowed_dirs or [])
    ):
        raise PathSecurityError.from_expanded_paths(
            original_path=directory,
            expanded_path=abs_dir,
            base_dir=base_dir,
            allowed_dirs=allowed_dirs,
            error_logged=True,
        )

    # Verify directory exists
    if not os.path.isdir(abs_dir):
        raise DirectoryNotFoundError(f"Directory not found: {directory}")

    # Collect files
    files = []

    for root, _, filenames in os.walk(abs_dir):
        # Skip subdirectories if not recursive
        if not recursive and root != abs_dir:
            continue

        for filename in filenames:
            abs_path = os.path.join(root, filename)

            # Check file extension if specified
            if (
                allowed_extensions
                and os.path.splitext(filename)[1] not in allowed_extensions
            ):
                continue

            # Create relative path from current directory
            rel_path = os.path.relpath(abs_path, base_dir)

            try:
                # Use the base name for all files in the collection
                files.append(
                    FileInfo.from_path(
                        path=rel_path, security_manager=security_manager
                    )
                )
            except PathSecurityError:
                # Propagate security errors
                raise
            except (OSError, ValueError):
                # Skip other errors but continue processing
                continue

    if not files:
        raise ValueError(f"No files found in directory: {directory}")

    return files


def collect_files(
    file_mappings: Optional[List[str]] = None,
    file_pattern_mappings: Optional[List[str]] = None,
    dir_mappings: Optional[List[str]] = None,
    recursive: bool = False,
    extensions: Optional[List[str]] = None,
    security_manager: Optional[SecurityManager] = None,
) -> Dict[str, List[FileInfo]]:
    """Collect files from various mapping types.

    Args:
        file_mappings: List of name=path mappings for single files
        file_pattern_mappings: List of name=pattern mappings for file patterns
        dir_mappings: List of name=path mappings for directories
        recursive: Whether to process directories recursively
        extensions: Optional list of file extensions to include
        security_manager: Optional security manager for path validation

    Returns:
        Dictionary mapping names to lists of FileInfo objects

    Raises:
        PathSecurityError: If any paths violate security constraints
        ValueError: If mappings are invalid or no files are found
    """
    result: Dict[str, List[FileInfo]] = {}

    # Process single file mappings
    if file_mappings:
        for mapping in file_mappings:
            try:
                name, path = validate_path_mapping(
                    mapping, security_manager=security_manager
                )
                result[name] = [
                    FileInfo.from_path(path, security_manager=security_manager)
                ]
            except (OSError, ValueError) as e:
                raise ValueError(f"Invalid file mapping {mapping!r}: {e}")

    # Process file pattern mappings
    if file_pattern_mappings:
        for mapping in file_pattern_mappings:
            try:
                name, pattern = parse_mapping(mapping)
                result[name] = collect_files_from_pattern(
                    pattern, security_manager=security_manager
                )
            except (OSError, ValueError) as e:
                raise ValueError(f"Invalid pattern mapping {mapping!r}: {e}")

    # Process directory mappings
    if dir_mappings:
        for mapping in dir_mappings:
            try:
                name, directory = validate_path_mapping(
                    mapping, is_dir=True, security_manager=security_manager
                )
                result[name] = collect_files_from_directory(
                    directory,
                    os.path.abspath(os.getcwd()),
                    recursive=recursive,
                    allowed_extensions=(
                        list(extensions) if extensions else None
                    ),
                    security_manager=security_manager,
                )
            except PathSecurityError as e:
                # Create a new error with the directory mapping prefix
                new_error = PathSecurityError(
                    f"Directory mapping {mapping} error: {str(e)}",
                    error_logged=True,
                )
                raise new_error
            except (OSError, ValueError) as e:
                raise ValueError(f"Invalid directory mapping {mapping!r}: {e}")

    return result


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet, with a fallback to UTF-8.

    This function uses a multi-step approach to detect file encoding:
    1. Check for BOM markers to identify UTF encodings
    2. Try chardet for content-based detection
    3. Attempt UTF-8 decode as a validation step
    4. Fall back to UTF-8 if all else fails

    Args:
        file_path: Path to file to check

    Returns:
        Detected encoding or 'utf-8' if detection fails. Note that ASCII detection
        is automatically converted to UTF-8 since UTF-8 is a superset of ASCII.

    Raises:
        OSError: If file cannot be read or accessed
        ValueError: If file path is invalid
    """
    logger = logging.getLogger(__name__)
    logger.debug("Detecting encoding for file: %s", file_path)

    try:
        # Read a sample of the file
        with open(file_path, "rb") as f:
            # First check for BOM markers (4 bytes)
            raw_data = f.read(4)
            if not raw_data:  # Empty file
                logger.debug("Empty file detected, using UTF-8")
                return "utf-8"

            # Check for BOM markers
            if raw_data.startswith(b"\xef\xbb\xbf"):
                logger.debug("UTF-8 BOM detected")
                return "utf-8-sig"
            elif raw_data.startswith(b"\xff\xfe") or raw_data.startswith(
                b"\xfe\xff"
            ):
                logger.debug("UTF-16 BOM detected")
                return "utf-16"
            elif raw_data.startswith(
                b"\xff\xfe\x00\x00"
            ) or raw_data.startswith(b"\x00\x00\xfe\xff"):
                logger.debug("UTF-32 BOM detected")
                return "utf-32"

            # Read more data for chardet (up to 1MB)
            f.seek(0)
            raw_data = f.read(
                1024 * 1024
            )  # Read up to 1MB for better detection

            # Try chardet detection
            result = chardet.detect(raw_data)
            logger.debug("Chardet detection result: %s", result)

            if result["encoding"]:
                detected = result["encoding"].lower()
                confidence = result["confidence"]

                # Handle ASCII detection
                if detected == "ascii":
                    logger.debug(
                        "ASCII detected, converting to UTF-8 (confidence: %f)",
                        confidence,
                    )
                    return "utf-8"

                # High confidence detection
                if confidence > 0.9:
                    logger.debug(
                        "High confidence encoding detected: %s (confidence: %f)",
                        detected,
                        confidence,
                    )
                    return detected

                # Medium confidence - validate with UTF-8 attempt
                if confidence > 0.6:
                    logger.debug(
                        "Medium confidence for %s (confidence: %f), validating",
                        detected,
                        confidence,
                    )
                    try:
                        raw_data.decode("utf-8")
                        logger.debug("Successfully validated as UTF-8")
                        return "utf-8"
                    except UnicodeDecodeError:
                        logger.debug(
                            "UTF-8 validation failed, using detected encoding: %s",
                            detected,
                        )
                        return detected

            # Low confidence or no detection - try UTF-8
            try:
                raw_data.decode("utf-8")
                logger.debug(
                    "No confident detection, but UTF-8 decode successful"
                )
                return "utf-8"
            except UnicodeDecodeError:
                if result["encoding"]:
                    logger.debug(
                        "Falling back to detected encoding with low confidence: %s",
                        result["encoding"].lower(),
                    )
                    return result["encoding"].lower()

                logger.warning(
                    "Could not confidently detect encoding for %s, defaulting to UTF-8",
                    file_path,
                )
                return "utf-8"

    except OSError:
        logger.error("Error reading file %s", file_path)
        raise
    except Exception:
        logger.error(
            "Unexpected error detecting encoding for %s",
            file_path,
        )
        raise ValueError("Failed to detect encoding")


def read_allowed_dirs_from_file(filepath: str) -> List[str]:
    """Reads a list of allowed directories from a file.

    Args:
        filepath: The path to the file.

    Returns:
        A list of allowed directories as absolute paths.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid data.
    """
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except OSError as e:
        raise FileNotFoundError(
            f"Error reading allowed directories from file: {filepath}: {e}"
        )

    allowed_dirs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(
            "#"
        ):  # Ignore empty lines and comments
            abs_path = os.path.abspath(line)
            if not os.path.isdir(abs_path):
                raise ValueError(
                    f"Invalid directory in allowed directories file '{filepath}': "
                    f"'{line}' is not a directory or does not exist."
                )
            allowed_dirs.append(abs_path)
    return allowed_dirs
