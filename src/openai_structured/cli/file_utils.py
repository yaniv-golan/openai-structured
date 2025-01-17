"""File utilities for the CLI."""

import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

# Type for values in template context
TemplateValue = Union[str, "FileInfo", List["FileInfo"]]


@dataclass
class FileInfo:
    """Information about a file."""

    name: str  # Variable name in template
    path: str  # Relative path from CWD
    abs_path: str  # Absolute path
    content: Optional[str] = None  # File content, loaded on demand

    @property
    def dir(self) -> str:
        """Get the directory containing this file, relative to CWD."""
        return str(Path(self.path).parent)

    def load_content(self) -> None:
        """Load the file content if not already loaded."""
        if self.content is None:
            with open(self.abs_path, "r", encoding="utf-8") as f:
                self.content = f.read()

    @classmethod
    def from_path(cls, name: str, path: str, load_content: bool = False) -> "FileInfo":
        """Create a FileInfo from a file path.
        
        Args:
            name: Variable name to use in template
            path: Path to file, relative to current directory
            load_content: Whether to load file content immediately
            
        Returns:
            FileInfo object
            
        Raises:
            ValueError: If path is invalid or outside base directory
            OSError: If file cannot be read
        """
        # Resolve paths
        base_dir = os.path.abspath(os.getcwd())
        abs_path = os.path.abspath(os.path.join(base_dir, path))
        
        # Security check - prevent directory traversal
        if not abs_path.startswith(base_dir):
            raise ValueError(f"Access denied: Path {path} is outside base directory")
            
        # Verify file exists
        if not os.path.isfile(abs_path):
            raise OSError(f"File not found: {path}")
            
        file_info = cls(
            name=name,
            path=path,
            abs_path=abs_path,
        )
        
        if load_content:
            file_info.load_content()
            
        return file_info


def collect_files_from_pattern(
    name: str,
    pattern: str,
    allowed_extensions: Optional[Set[str]] = None,
) -> List[FileInfo]:
    """Collect files matching a glob pattern.
    
    Args:
        name: Base name for the files in template
        pattern: Glob pattern to match files
        allowed_extensions: Optional set of allowed file extensions (e.g. {'.py', '.js'})
        
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
        matches = glob.glob(pattern, recursive=True)
    except Exception as e:
        raise ValueError(f"Invalid glob pattern '{pattern}': {e}")
        
    # Filter and convert matches to FileInfo objects
    result: List[FileInfo] = []
    for path in matches:
        # Skip if extension not allowed
        if allowed_extensions:
            ext = os.path.splitext(path)[1].lower()
            if ext not in allowed_extensions:
                continue
                
        try:
            # Use index as suffix for name to ensure uniqueness
            file_info = FileInfo.from_path(
                name=f"{name}_{len(result) + 1}",
                path=path,
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
) -> List[FileInfo]:
    """Collect files from a directory.
    
    Args:
        name: Base name for the files in template
        directory: Directory path relative to current directory
        recursive: Whether to traverse subdirectories
        allowed_extensions: Optional set of allowed file extensions (e.g. {'.py', '.js'})
        
    Returns:
        List of FileInfo objects for files in directory
        
    Raises:
        ValueError: If directory is invalid or outside base directory
        OSError: If directory cannot be read
    """
    # Resolve directory path
    base_dir = os.path.abspath(os.getcwd())
    abs_dir = os.path.abspath(os.path.join(base_dir, directory))
    
    # Security check - prevent directory traversal
    if not abs_dir.startswith(base_dir):
        raise ValueError(f"Access denied: Directory {directory} is outside base directory")
        
    # Verify directory exists
    if not os.path.isdir(abs_dir):
        raise OSError(f"Directory not found: {directory}")
        
    # Build glob pattern based on recursion and extensions
    if recursive:
        pattern = os.path.join(directory, "**", "*")
    else:
        pattern = os.path.join(directory, "*")
        
    return collect_files_from_pattern(name, pattern, allowed_extensions)


def collect_files(
    file_args: Optional[List[str]] = None,
    files_args: Optional[List[str]] = None,
    dir_args: Optional[List[str]] = None,
    recursive: bool = False,
    allowed_extensions: Optional[Set[str]] = None,
    load_content: bool = False,
) -> Dict[str, TemplateValue]:
    """Collect files from various sources.
    
    Args:
        file_args: List of name=path mappings for single files
        files_args: List of name=pattern mappings for multiple files
        dir_args: List of name=path mappings for directories
        recursive: Whether to process directories recursively
        allowed_extensions: Optional set of allowed file extensions (e.g. {'.py', '.js'})
        load_content: Whether to load file content immediately
        
    Returns:
        Dictionary mapping variable names to FileInfo objects or lists of FileInfo objects
        
    Raises:
        ValueError: If arguments are invalid
        OSError: If files cannot be read
    """
    result: Dict[str, TemplateValue] = {}
    
    # Process single files
    if file_args:
        for mapping in file_args:
            try:
                name, path = mapping.split("=", 1)
                result[name] = FileInfo.from_path(name, path, load_content)
            except ValueError:
                raise ValueError(f"Invalid file mapping: {mapping}")
    
    # Process multiple files
    if files_args:
        for mapping in files_args:
            try:
                name, pattern = mapping.split("=", 1)
                files = collect_files_from_pattern(
                    name, pattern, allowed_extensions
                )
                if not files:
                    raise ValueError(
                        f"No files found matching pattern: {pattern}"
                    )
                result[name] = files
            except ValueError as e:
                raise ValueError(f"Invalid files mapping: {e}")
    
    # Process directories
    if dir_args:
        for mapping in dir_args:
            try:
                name, path = mapping.split("=", 1)
                files = collect_files_from_directory(
                    name, path, recursive, allowed_extensions
                )
                if not files:
                    raise ValueError(
                        f"No files found in directory: {path}"
                    )
                result[name] = files
            except ValueError as e:
                raise ValueError(f"Invalid directory mapping: {e}")
    
    return result 