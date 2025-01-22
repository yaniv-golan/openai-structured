"""Security management for file access.

This module provides security checks for file access, including:
- Base directory restrictions
- Allowed directory validation
- Path traversal prevention
- Temporary directory handling
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Set

from .errors import DirectoryNotFoundError, PathSecurityError
from .security_types import SecurityManagerProtocol


def is_temp_file(path: str) -> bool:
    """Check if a file is in a temporary directory.

    Args:
        path: Path to check (will be converted to absolute path)

    Returns:
        True if the path is in a temporary directory, False otherwise

    Note:
        This function handles platform-specific path normalization, including symlinks
        (e.g., on macOS where /var is symlinked to /private/var).
    """
    # Normalize the input path (resolve symlinks)
    abs_path = os.path.realpath(path)

    # Get all potential temp directories and normalize them
    temp_dirs = set()
    # System temp dir (platform independent)
    temp_dirs.add(os.path.realpath(tempfile.gettempdir()))

    # Common Unix/Linux/macOS temp locations
    unix_temp_dirs = ["/tmp", "/var/tmp", "/var/folders"]
    for temp_dir in unix_temp_dirs:
        if os.path.exists(temp_dir):
            temp_dirs.add(os.path.realpath(temp_dir))

    # Windows temp locations (if on Windows)
    if os.name == "nt":
        if "TEMP" in os.environ:
            temp_dirs.add(os.path.realpath(os.environ["TEMP"]))
        if "TMP" in os.environ:
            temp_dirs.add(os.path.realpath(os.environ["TMP"]))

    # Check if file is in any temp directory using normalized paths
    abs_path_parts = os.path.normpath(abs_path).split(os.sep)
    for temp_dir in temp_dirs:
        temp_dir_parts = os.path.normpath(temp_dir).split(os.sep)
        # Check if the path starts with the temp directory components
        if len(abs_path_parts) >= len(temp_dir_parts) and all(
            a == b
            for a, b in zip(
                abs_path_parts[: len(temp_dir_parts)], temp_dir_parts
            )
        ):
            return True

    return False


class SecurityManager(SecurityManagerProtocol):
    """Manages security for file access.

    Validates all file access against a base directory and optional
    allowed directories. Prevents unauthorized access and directory
    traversal attacks.

    The security model is based on:
    1. A base directory that serves as the root for all file operations
    2. A set of explicitly allowed directories that can be accessed outside the base directory
    3. Special handling for temporary directories that are always allowed

    All paths are normalized using realpath() to handle symlinks consistently across platforms.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        allowed_dirs: Optional[List[str]] = None,
    ):
        """Initialize security manager.

        Args:
            base_dir: Base directory for file access. Defaults to current working directory.
            allowed_dirs: Optional list of additional allowed directories

        All paths are normalized using realpath to handle symlinks
        and relative paths consistently across platforms.
        """
        self._base_dir = Path(os.path.realpath(base_dir or os.getcwd()))
        self._allowed_dirs: Set[Path] = set()
        if allowed_dirs:
            for directory in allowed_dirs:
                self.add_allowed_dir(directory)

    @property
    def base_dir(self) -> Path:
        """Get the base directory."""
        return self._base_dir

    @property
    def allowed_dirs(self) -> List[Path]:
        """Get the list of allowed directories."""
        return sorted(self._allowed_dirs)  # Sort for consistent ordering

    def add_allowed_dir(self, directory: str) -> None:
        """Add a directory to the set of allowed directories.

        Args:
            directory: Directory to allow access to

        Raises:
            DirectoryNotFoundError: If directory does not exist
        """
        real_path = Path(os.path.realpath(directory))
        if not real_path.exists():
            raise DirectoryNotFoundError(f"Directory not found: {directory}")
        if not real_path.is_dir():
            raise DirectoryNotFoundError(
                f"Path is not a directory: {directory}"
            )
        self._allowed_dirs.add(real_path)

    def add_allowed_dirs_from_file(self, file_path: str) -> None:
        """Add allowed directories from a file.

        Args:
            file_path: Path to file containing allowed directories (one per line)

        Raises:
            PathSecurityError: If file_path is outside allowed directories
            FileNotFoundError: If file does not exist
            ValueError: If file contains invalid directories
        """
        real_path = Path(os.path.realpath(file_path))

        # First validate the file path itself
        try:
            self.validate_path(
                str(real_path), purpose="read allowed directories"
            )
        except PathSecurityError:
            raise PathSecurityError(
                f"Cannot read allowed directories file outside allowed paths: {file_path}"
            )

        if not real_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(real_path) as f:
            for line in f:
                directory = line.strip()
                if directory and not directory.startswith("#"):
                    self.add_allowed_dir(directory)

    def is_path_allowed(self, path: str) -> bool:
        """Check if a path is allowed.

        A path is allowed if it is:
        1. Under the normalized base directory
        2. Under any normalized allowed directory

        The path must also exist.

        Args:
            path: Path to check

        Returns:
            bool: True if path exists and is allowed, False otherwise
        """
        try:
            real_path = Path(os.path.realpath(path))
            if not real_path.exists():
                return False
        except (ValueError, OSError):
            return False

        # Check if path is in base directory
        try:
            if real_path.is_relative_to(self._base_dir):
                return True
        except ValueError:
            pass

        # Check if path is in allowed directories
        for allowed_dir in self._allowed_dirs:
            try:
                if real_path.is_relative_to(allowed_dir):
                    return True
            except ValueError:
                continue

        return False

    def validate_path(self, path: str, purpose: str = "access") -> Path:
        """Validate and normalize a path.

        Args:
            path: Path to validate
            purpose: Description of intended access (for error messages)

        Returns:
            Path: Normalized path if valid

        Raises:
            PathSecurityError: If path is not allowed
        """
        try:
            real_path = Path(os.path.realpath(path))
        except (ValueError, OSError) as e:
            raise PathSecurityError(f"Invalid path format: {e}")

        if not self.is_path_allowed(str(real_path)):
            raise PathSecurityError(
                f"Path {purpose} denied: {path} is outside allowed directories"
            )

        return real_path

    def is_allowed_file(self, path: str) -> bool:
        """Check if file access is allowed.

        Args:
            path: Path to check

        Returns:
            bool: True if file exists and is allowed
        """
        try:
            real_path = Path(os.path.realpath(path))
            return self.is_path_allowed(str(real_path)) and real_path.is_file()
        except (ValueError, OSError):
            return False

    def is_allowed_path(self, path_str: str) -> bool:
        """Check if string path is allowed.

        Args:
            path_str: Path string to check

        Returns:
            bool: True if path is allowed
        """
        try:
            return self.is_path_allowed(path_str)
        except (ValueError, OSError):
            return False
