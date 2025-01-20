"""File utilities for the CLI."""

import os
import glob
import hashlib
import logging
import chardet
import inspect
from pathlib import Path
from typing import List, Dict, Union, Optional, Set, Any

from .errors import (
    DirectoryNotFoundError,
    PathSecurityError,
    FileNotFoundError,
)
from .security import is_temp_file

# Type for values in template context
TemplateValue = Union[str, "FileInfo", List["FileInfo"]]

# Import SecurityManager at runtime to avoid circular import
SecurityManager = None

def _get_security_manager():
    global SecurityManager
    if SecurityManager is None:
        from .security import SecurityManager
    return SecurityManager

class FileInfo:
    """Information about a file.
    
    This class provides a safe interface for accessing file information and content,
    with support for lazy loading and caching. It includes security checks to prevent
    directory traversal attacks and ensures files are accessed only within allowed directories.
    """
    
    def __init__(self, name: str, path: str, lazy: bool = True) -> None:
        """Initialize FileInfo instance.
        
        Args:
            name: Variable name for this file
            path: Path to file
            lazy: Whether to load content lazily (default: True)
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Initialize basic attributes
        self._name = name
        self._path = path
        self._lazy = lazy
        
        # Stats and content attributes - will be loaded when needed
        self._content: Optional[str] = None
        self._size: Optional[int] = None
        self._mtime: Optional[float] = None
        self._stats: Optional[os.stat_result] = None
        self._stats_loaded = False
        self._security_checked = False
        self._encoding: Optional[str] = None
        self._hash: Optional[str] = None
        
        # Mark as initialized
        self._initialized = True
        
        # In non-lazy mode, load stats and content immediately
        if not lazy:
            self._load_stats()
            self.load_content()

    @property
    def name(self) -> str:
        """Get variable name for the file."""
        return self._name
        
    @property
    def path(self) -> str:
        """Get path to the file."""
        return str(self._path)
        
    @property
    def abs_path(self) -> str:
        """Get absolute path to file with symlinks resolved."""
        return os.path.realpath(self.path)
        
    @property
    def content(self) -> str:
        """Get file content, loading it if necessary.
        
        The content loading behavior depends on the lazy mode:
        - In lazy mode: Content is loaded when this property is first accessed
        - In non-lazy mode: Content should already be loaded during initialization
        
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file does not exist
            OSError: If file cannot be read
            PathSecurityError: If file access is denied
        """
        self.logger.debug("Accessing content for %s (lazy=%s)", self._path, self._lazy)
        if self._content is None:
            if not self._lazy:
                # In non-lazy mode, content should have been loaded during initialization
                self.logger.error("Content not loaded for non-lazy FileInfo: %s", self._path)
                raise RuntimeError(f"Content not loaded for non-lazy FileInfo: {self._path}")
            try:
                self.load_content()
            except (OSError, IOError, FileNotFoundError, PathSecurityError) as e:
                self.logger.error("Failed to load content for %s: %s", self._path, e)
                raise
            except Exception as e:
                self.logger.error("Unexpected error loading content for %s: %s", self._path, e)
                raise RuntimeError(f"Failed to load content: {str(e)}")
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
            except Exception as e:
                self.logger.warning("Failed to load stats for %s: %s", self._path, e)
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
            except Exception as e:
                self.logger.warning("Failed to load stats for %s: %s", self._path, e)
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
        return os.path.splitext(self._path)[1].lstrip('.')
        
    @property
    def basename(self) -> str:
        """Get file basename."""
        return os.path.basename(self._path)
        
    @property
    def dirname(self) -> str:
        """Get directory name."""
        return os.path.dirname(self._path)
        
    @property
    def parent(self) -> str:
        """Get parent directory."""
        return str(self._path.parent)
        
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
        return os.path.exists(self._path)
        
    @property
    def is_file(self) -> bool:
        """Check if path is a file."""
        return os.path.isfile(self._path)
        
    @property
    def is_dir(self) -> bool:
        """Check if path is a directory."""
        return os.path.isdir(self._path)

    @property
    def lazy(self) -> bool:
        """Whether this FileInfo uses lazy loading."""
        return self._lazy

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification of private attributes after initialization."""
        # Allow all modifications during initialization
        if not hasattr(self, '_initialized'):
            super().__setattr__(name, value)
            if name == '_initialized':
                return
            return
            
        # Allow internal updates from specific methods
        frame = inspect.currentframe()
        if frame is not None:
            caller = frame.f_back
            if caller is not None and caller.f_code.co_name in ['_check_security', '_load_stats', 'load_content', 'update_cache']:
                super().__setattr__(name, value)
                return
                
        # Prevent external modifications of private attributes
        if name.startswith('_') and hasattr(self, name):
            raise AttributeError(f"Can't modify {name} after initialization")
            
        # Allow setting public attributes
        super().__setattr__(name, value)

    @classmethod
    def from_path(cls, name: str, path: str, lazy: bool = True, allowed_dirs: Optional[List[str]] = None) -> "FileInfo":
        """Create FileInfo instance from file path.
        
        Args:
            name: Variable name for this file
            path: Path to file
            lazy: Whether to load content lazily (default: True)
            allowed_dirs: Optional list of allowed directories
            
        Returns:
            FileInfo instance
            
        Raises:
            PathSecurityError: If file access is denied
            FileNotFoundError: If file does not exist
            OSError: If file cannot be read
        """
        # Check file exists before creating instance
        abs_path = os.path.realpath(path)
        if not os.path.exists(abs_path):
            msg = f"File not found: {path}"
            logging.getLogger(__name__).error(msg)
            raise FileNotFoundError(msg)  # Using our custom FileNotFoundError

        info = cls(name=name, path=path, lazy=lazy)
        
        # Always check security first
        info._check_security(allowed_dirs)
        
        # In non-lazy mode, load stats and content immediately
        if not lazy:
            info._load_stats()
            info.load_content()
            
        return info

    def _check_security(self, allowed_dirs: Optional[List[str]] = None) -> None:
        """Perform security checks without loading stats.
        
        Args:
            allowed_dirs: Optional list of allowed directories
            
        Raises:
            PathSecurityError: If file access is denied
        """
        if self._security_checked:
            return

        abs_path = os.path.realpath(self._path)
        base_dir = os.path.realpath(os.getcwd())

        # Check if path is in base directory, temp directory, or allowed directory
        is_allowed = (
            os.path.commonpath([abs_path, base_dir]) == base_dir or
            is_temp_file(abs_path) or
            any(
                os.path.commonpath([abs_path, os.path.realpath(allowed_dir)]) == os.path.realpath(allowed_dir)
                for allowed_dir in (allowed_dirs or [])
                if os.path.exists(allowed_dir)
            )
        )

        if not is_allowed:
            msg = f"Access denied: {self._path} is outside base directory and not in allowed directories"
            self.logger.error(msg)
            raise PathSecurityError(msg)

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
        except PathSecurityError as e:
            self.logger.error(str(e))
            raise

        if self._stats_loaded:
            self.logger.debug("Stats already loaded for %s", self._path)
            return

        self.logger.debug("Loading stats for %s", self._path)

        # Check file exists and is readable
        abs_path = os.path.realpath(self._path)
        if not os.path.exists(abs_path):
            msg = f"File not found: {self._path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)  # Using our custom FileNotFoundError
        if not os.path.isfile(abs_path):
            msg = f"Path is not a file: {self._path}"
            self.logger.error(msg)
            raise ValueError(msg)

        try:
            stats = os.stat(abs_path)
            self._stats = stats
            self._size = stats.st_size
            self._mtime = stats.st_mtime
            self._stats_loaded = True
            self.logger.debug("Successfully loaded stats for %s", self._path)
        except (OSError, IOError) as e:
            self.logger.error("Error reading file stats: %s", e)
            raise

    def load_content(self, encoding: Optional[str] = None) -> None:
        """Load file content if not already loaded.
        
        This method performs security checks and loads the file content.
        It is called automatically by the content property in lazy mode,
        or during initialization in non-lazy mode.
        
        Args:
            encoding: Optional encoding to use for reading the file.
                     If not provided, encoding will be detected.
        
        Raises:
            OSError: If file cannot be read
            IOError: If file cannot be read
            PathSecurityError: If file access is not allowed
        """
        if self._content is not None:
            self.logger.debug("Content already loaded for %s", self._path)
            return

        self.logger.debug("Loading content for %s (encoding=%s)", self._path, encoding)
        
        # Load stats first to perform security checks
        self._load_stats()

        try:
            if encoding is None:
                encoding = detect_encoding(self._path)
                self.logger.debug("Using detected encoding %s for %s", encoding, self._path)

            with open(self._path, 'r', encoding=encoding) as f:
                content = f.read()

            self._content = content
            self._encoding = encoding
            self._hash = hashlib.sha256(content.encode()).hexdigest()
            self.logger.debug("Successfully loaded %d bytes from %s", len(content), self._path)
        except (OSError, IOError) as e:
            self.logger.error("Error reading file: %s", e)
            raise

    def update_cache(self, content: str, encoding: Optional[str], hash_value: Optional[str] = None) -> None:
        """Update private fields from external cache logic."""
        self._content = content
        self._encoding = encoding
        self._hash = hash_value or hashlib.sha256(content.encode()).hexdigest()
        self.logger.debug("Updated cache for %s", self._path)

    def dir(self) -> str:
        """Get directory containing the file."""
        return os.path.dirname(self.abs_path)


def collect_files_from_pattern(
    name: str,
    pattern: str,
    allowed_extensions: Optional[Set[str]] = None,
    recursive: bool = False,
    allowed_dirs: Optional[List[str]] = None,
) -> List[FileInfo]:
    """Collect files matching a glob pattern.
    
    Args:
        name: Base name for the files in template
        pattern: Glob pattern to match files
        allowed_extensions: Optional set of allowed file extensions (e.g. {'.py', '.js'})
        recursive: Whether to allow recursive glob patterns
        allowed_dirs: Optional list of allowed directories
        
    Returns:
        List of FileInfo objects for matching files
        
    Raises:
        ValueError: If pattern is invalid or matches files outside base directory
        OSError: If files cannot be read
    """
    # Resolve base directory
    base_dir = os.path.abspath(os.getcwd())
    
    # Expand glob pattern
    try:
        matches = glob.glob(pattern, recursive=recursive)
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
            
        # Skip if extension not allowed
        if allowed_extensions:
            ext = os.path.splitext(path)[1].lower()
            if ext not in allowed_extensions:
                continue
                
        try:
            # Use the base name for all files in the collection
            file_info = FileInfo.from_path(
                name=name,
                path=path,
                allowed_dirs=allowed_dirs,
            )
            result.append(file_info)
        except (ValueError, OSError):
            # Skip invalid files but continue processing
            continue
            
    return result


def collect_files_from_directory(
    name: str,
    directory: str,
    recursive: bool = False,
    allowed_extensions: Optional[Set[str]] = None,
    allowed_dirs: Optional[List[str]] = None,
) -> List[FileInfo]:
    """Collect files from a directory.

    Args:
        name: Base name for the files in template
        directory: Directory path relative to current directory
        recursive: Whether to traverse subdirectories
        allowed_extensions: Optional set of allowed file extensions (e.g. {'.py', '.js'})
        allowed_dirs: Optional list of allowed directories

    Returns:
        List of FileInfo objects for files in directory

    Raises:
        DirectoryNotFoundError: If directory does not exist
        PathSecurityError: If directory is outside base directory and not in allowed directories
        ValueError: If no files are found in directory
    """
    # Resolve paths and check security first
    base_dir = os.path.abspath(os.getcwd())
    abs_dir = os.path.abspath(os.path.join(base_dir, directory))

    # Security check - prevent directory traversal
    if not abs_dir.startswith(base_dir) and not any(
        abs_dir.startswith(allowed_dir) for allowed_dir in (allowed_dirs or [])
    ):
        raise PathSecurityError(
            f"Access denied: Directory {directory} is outside base directory and not in allowed directories"
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
            if allowed_extensions and os.path.splitext(filename)[1] not in allowed_extensions:
                continue

            # Create relative path from current directory
            rel_path = os.path.relpath(abs_path, base_dir)

            try:
                # Use the base name for all files in the collection
                files.append(FileInfo.from_path(name=name, path=rel_path, allowed_dirs=allowed_dirs))
            except (OSError, ValueError):
                continue

    if not files:
        raise ValueError(f"No files found in directory: {directory}")

    return files


def collect_files(
    file_args: Optional[List[str]] = None,
    files_args: Optional[List[str]] = None,
    dir_args: Optional[List[str]] = None,
    recursive: bool = False,
    allowed_extensions: Optional[Set[str]] = None,
    load_content: bool = False,
    allowed_dirs: Optional[List[str]] = None,
) -> Dict[str, Union[FileInfo, List[FileInfo]]]:
    """Collect files from various sources.
    
    Args:
        file_args: List of name=path mappings for single files
        files_args: List of name=pattern mappings for multiple files
        dir_args: List of name=path mappings for directories
        recursive: Whether to process directories recursively
        allowed_extensions: Optional set of allowed file extensions (e.g. {'.py', '.js'})
        load_content: Whether to load file content immediately
        allowed_dirs: Optional list of allowed directories
        
    Returns:
        Dictionary mapping variable names to FileInfo objects or lists of FileInfo objects
        
    Raises:
        ValueError: If arguments are invalid
        OSError: If files cannot be read
    """
    result: Dict[str, Union[FileInfo, List[FileInfo]]] = {}
    
    # Expand allowed directories first
    expanded_allowed_dirs: List[str] = []
    if allowed_dirs:
        for allowed_dir in allowed_dirs:
            if allowed_dir.startswith("@"):
                try:
                    expanded_allowed_dirs.extend(
                        read_allowed_dirs_from_file(allowed_dir[1:])
                    )
                except (FileNotFoundError, ValueError) as e:
                    raise ValueError(f"Error processing allowed directories: {e}")
            else:
                if not os.path.isdir(allowed_dir):
                    raise ValueError(
                        f"Invalid allowed directory: '{allowed_dir}' is not a directory or does not exist."
                    )
                expanded_allowed_dirs.append(os.path.abspath(allowed_dir))
    
    # Process single files
    if file_args:
        for mapping in file_args:
            try:
                if "=" not in mapping:
                    raise ValueError(f"Invalid file mapping format: {mapping}. Expected name=path")
                name, path = mapping.split("=", 1)
                if not name:
                    raise ValueError(f"Empty variable name in mapping: {mapping}")
                if not path:
                    raise ValueError(f"Empty file path in mapping: {mapping}")
                result[name] = FileInfo.from_path(name=name, path=path, lazy=not load_content, allowed_dirs=expanded_allowed_dirs)
            except ValueError as e:
                raise ValueError(f"Invalid file mapping: {str(e)}")
    
    # Process multiple files
    if files_args:
        for mapping in files_args:
            try:
                if "=" not in mapping:
                    raise ValueError(f"Invalid files mapping format: {mapping}. Expected name=pattern")
                name, pattern = mapping.split("=", 1)
                if not name:
                    raise ValueError(f"Empty variable name in mapping: {mapping}")
                if not pattern:
                    raise ValueError(f"Empty glob pattern in mapping: {mapping}")
                files = collect_files_from_pattern(
                    name=name,
                    pattern=pattern,
                    allowed_extensions=allowed_extensions,
                    recursive=recursive,
                    allowed_dirs=expanded_allowed_dirs,
                )
                if not files:
                    raise ValueError(f"No files found matching pattern: {pattern}")
                result[name] = files
            except ValueError as e:
                raise ValueError(f"Invalid files mapping: {str(e)}")
    
    # Process directories
    if dir_args:
        for mapping in dir_args:
            try:
                # Format validation with specific error
                if "=" not in mapping:
                    raise ValueError(f"Directory mapping must contain '='. Got: '{mapping}'")
                name, path = mapping.split("=", 1)
                
                # Empty checks with specific errors
                if not name:
                    raise ValueError(f"Directory mapping contains empty variable name: '{mapping}'")
                if not path:
                    raise ValueError(f"Directory mapping contains empty path: '{mapping}'")

                # Directory existence check with clear error
                if not os.path.exists(path):
                    raise DirectoryNotFoundError(f"Directory does not exist: '{path}'")
                if not os.path.isdir(path):
                    raise DirectoryNotFoundError(f"Path exists but is not a directory: '{path}'")

                # Security check with clear error
                base_dir = os.path.abspath(os.getcwd())
                abs_dir = os.path.abspath(path)
                if not abs_dir.startswith(base_dir):
                    raise PathSecurityError(f"Directory '{path}' is outside the current working directory '{base_dir}'")

                # Collect files with clear error if empty
                files = collect_files_from_directory(
                    name=name,
                    directory=path,
                    recursive=recursive,
                    allowed_extensions=allowed_extensions,
                    allowed_dirs=expanded_allowed_dirs,
                )
                if not files:
                    raise ValueError(f"No files found in directory '{path}'" + 
                                  (f" with extensions {allowed_extensions}" if allowed_extensions else ""))
                
                result[name] = files

            except ValueError as e:
                raise ValueError(f"Directory mapping '{mapping}' error: {str(e)}")
            except DirectoryNotFoundError as e:
                raise DirectoryNotFoundError(f"Directory mapping '{mapping}' error: {str(e)}")
            except PathSecurityError as e:
                raise PathSecurityError(f"Directory mapping '{mapping}' error: {str(e)}")
    
    return result


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet.
    
    Args:
        file_path: Path to file to check
        
    Returns:
        Detected encoding or 'utf-8' if detection fails
    """
    try:
        # Read a sample of the file
        with open(file_path, 'rb') as f:
            raw_data = f.read(4)  # Read first 4 bytes to check for BOM
            if not raw_data:  # Empty file
                return 'utf-8'
            
            # Check for BOM markers
            if raw_data.startswith(b'\xef\xbb\xbf'):
                return 'utf-8-sig'
            elif raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
                return 'utf-16'
            elif raw_data.startswith(b'\xff\xfe\x00\x00') or raw_data.startswith(b'\x00\x00\xfe\xff'):
                return 'utf-32'
            
            # Read more data for chardet
            f.seek(0)
            raw_data = f.read(1024)  # Read first 1KB
            
        # Detect encoding
        result = chardet.detect(raw_data)
        if result['encoding'] and result['confidence'] > 0.9:
            return result['encoding'].lower()
            
        # Default to utf-8 for high confidence unicode detection
        try:
            raw_data.decode('utf-8')
            return 'utf-8'
        except UnicodeDecodeError:
            pass
            
        # If no high confidence detection, use chardet's best guess
        return result['encoding'].lower() if result['encoding'] else 'utf-8'
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning("Error detecting encoding for %s: %s", file_path, e)
        return 'utf-8'


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
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except OSError as e:
        raise FileNotFoundError(f"Error reading allowed directories from file: {filepath}: {e}")

    allowed_dirs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):  # Ignore empty lines and comments
            abs_path = os.path.abspath(line)
            if not os.path.isdir(abs_path):
                raise ValueError(f"Invalid directory in allowed directories file '{filepath}': '{line}' is not a directory or does not exist.")
            allowed_dirs.append(abs_path)
    return allowed_dirs 