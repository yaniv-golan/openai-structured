"""Command-line interface for making structured OpenAI API calls."""

import argparse
import asyncio
import json
import logging
import os
import sys
from enum import IntEnum
from importlib.metadata import version
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
    overload,
)

import jinja2
import tiktoken
import yaml
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, create_model

from ..client import async_openai_structured_stream, supports_structured_output
from ..errors import (
    APIResponseError,
    EmptyResponseError,
    InvalidResponseFormatError,
    JSONParseError,
    ModelNotSupportedError,
    ModelVersionError,
    OpenAIClientError,
    SchemaFileError,
    SchemaValidationError,
    StreamBufferError,
    StreamInterruptedError,
    StreamParseError,
)
from .errors import (
    DirectoryNotFoundError,
    FileNotFoundError,
    InvalidJSONError,
    PathSecurityError,
    TaskTemplateSyntaxError,
    TaskTemplateVariableError,
    VariableError,
    VariableNameError,
    VariableValueError,
)
from .file_utils import TemplateValue, collect_files
from .path_utils import validate_path_mapping
from .progress import ProgressContext
from .security import SecurityManager
from .template_env import create_jinja_env
from .template_utils import (
    SystemPromptError,
    TemplateMetadataError,
    render_template,
    validate_json_schema,
    validate_template_placeholders,
)

# Set up logging
logger = logging.getLogger("ostruct")

# Constants
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Get package version
try:
    __version__ = version("openai-structured")
except Exception:
    __version__ = "unknown"


class ExitCode(IntEnum):
    """Exit codes for the CLI following standard Unix conventions.

    Categories:
    - Success (0-1)
    - User Interruption (2-3)
    - Input/Validation (64-69)
    - I/O and File Access (70-79)
    - API and External Services (80-89)
    - Internal Errors (90-99)
    """

    # Success codes
    SUCCESS = 0

    # User interruption
    INTERRUPTED = 2

    # Input/Validation errors (64-69)
    USAGE_ERROR = 64
    DATA_ERROR = 65
    SCHEMA_ERROR = 66
    VALIDATION_ERROR = 67

    # I/O and File Access errors (70-79)
    IO_ERROR = 70
    FILE_NOT_FOUND = 71
    PERMISSION_ERROR = 72
    SECURITY_ERROR = 73

    # API and External Service errors (80-89)
    API_ERROR = 80
    API_TIMEOUT = 81

    # Internal errors (90-99)
    INTERNAL_ERROR = 90
    UNKNOWN_ERROR = 91


def create_dynamic_model(schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from a JSON schema."""
    properties = schema.get("properties", {})
    field_definitions: Dict[str, Any] = {}

    for name, prop in properties.items():
        field_type: Any
        if prop.get("type") == "string":
            field_type = str
        elif prop.get("type") == "integer":
            field_type = int
        elif prop.get("type") == "number":
            field_type = float
        elif prop.get("type") == "boolean":
            field_type = bool
        elif prop.get("type") == "array":
            field_type = List[Any]
        elif prop.get("type") == "object":
            field_type = Dict[str, Any]
        else:
            field_type = Any

        field_definitions[name] = (field_type, ...)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Type[BaseModel] = create_model(
        "DynamicModel", __config__=model_config, **field_definitions
    )
    return model


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def estimate_tokens_for_chat(
    messages: List[Dict[str, str]], model: str
) -> int:
    """Estimate the number of tokens in a chat completion."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        # Add message overhead
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def get_default_token_limit(model: str) -> int:
    """Get the default token limit for a given model.

    Note: These limits are based on current OpenAI model specifications as of 2024 and may
    need to be updated if OpenAI changes the models' capabilities.

    Args:
        model: The model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'o1')

    Returns:
        The default token limit for the model
    """
    if "o1" in model:
        return 100_000  # o1 supports up to 100K output tokens
    elif "gpt-4o" in model:
        return 16_384  # gpt-4o and gpt-4o-mini support up to 16K output tokens
    else:
        return 4_096  # default fallback


def get_context_window_limit(model: str) -> int:
    """Get the total context window limit for a given model.

    Note: These limits are based on current OpenAI model specifications as of 2024 and may
    need to be updated if OpenAI changes the models' capabilities.

    Args:
        model: The model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'o1')

    Returns:
        The context window limit for the model
    """
    if "o1" in model:
        return 200_000  # o1 supports 200K total context window
    elif "gpt-4o" in model:
        return 128_000  # gpt-4o and gpt-4o-mini support 128K context window
    else:
        return 8_192  # default fallback


def validate_token_limits(
    model: str, total_tokens: int, max_token_limit: Optional[int] = None
) -> None:
    """Validate token counts against model limits.

    Args:
        model: The model name
        total_tokens: Total number of tokens in the prompt
        max_token_limit: Optional user-specified token limit

    Raises:
        ValueError: If token limits are exceeded
    """
    context_limit = get_context_window_limit(model)
    output_limit = (
        max_token_limit
        if max_token_limit is not None
        else get_default_token_limit(model)
    )

    # Check if total tokens exceed context window
    if total_tokens >= context_limit:
        raise ValueError(
            f"Total tokens ({total_tokens:,}) exceed model's context window limit "
            f"of {context_limit:,} tokens"
        )

    # Check if there's enough room for output tokens
    remaining_tokens = context_limit - total_tokens
    if remaining_tokens < output_limit:
        raise ValueError(
            f"Only {remaining_tokens:,} tokens remaining in context window, but "
            f"output may require up to {output_limit:,} tokens"
        )


def process_system_prompt(
    task_template: str,
    system_prompt: Optional[str],
    template_context: Dict[str, Any],
    env: jinja2.Environment,
    ignore_task_sysprompt: bool = False,
) -> str:
    """Process system prompt from various sources.

    Args:
        task_template: The task template string
        system_prompt: Optional system prompt string or file path (with @ prefix)
        template_context: Template context for rendering
        env: Jinja2 environment
        ignore_task_sysprompt: Whether to ignore system prompt in task template

    Returns:
        The final system prompt string

    Raises:
        SystemPromptError: If the system prompt cannot be loaded or rendered
        FileNotFoundError: If a prompt file does not exist
        PathSecurityError: If a prompt file path violates security constraints
    """
    # Default system prompt
    default_prompt = "You are a helpful assistant."

    # Try to get system prompt from CLI argument first
    if system_prompt:
        if system_prompt.startswith("@"):
            # Load from file
            path = system_prompt[1:]
            try:
                name, path = validate_path_mapping(f"system_prompt={path}")
                with open(path, "r", encoding="utf-8") as f:
                    system_prompt = f.read().strip()
            except (FileNotFoundError, PathSecurityError) as e:
                raise SystemPromptError(f"Invalid system prompt file: {e}")

        # Render system prompt with template context
        try:
            template = env.from_string(system_prompt)
            return template.render(**template_context).strip()
        except jinja2.TemplateError as e:
            raise SystemPromptError(f"Error rendering system prompt: {e}")

    # If not ignoring task template system prompt, try to extract it
    if not ignore_task_sysprompt:
        try:
            # Extract YAML frontmatter
            if task_template.startswith("---\n"):
                end = task_template.find("\n---\n", 4)
                if end != -1:
                    frontmatter = task_template[4:end]
                    try:
                        metadata = yaml.safe_load(frontmatter)
                        if (
                            isinstance(metadata, dict)
                            and "system_prompt" in metadata
                        ):
                            system_prompt = str(metadata["system_prompt"])
                            # Render system prompt with template context
                            try:
                                template = env.from_string(system_prompt)
                                return template.render(
                                    **template_context
                                ).strip()
                            except jinja2.TemplateError as e:
                                raise SystemPromptError(
                                    f"Error rendering system prompt: {e}"
                                )
                    except yaml.YAMLError as e:
                        raise SystemPromptError(
                            f"Invalid YAML frontmatter: {e}"
                        )

        except Exception as e:
            raise SystemPromptError(
                f"Error extracting system prompt from template: {e}"
            )

    # Fall back to default
    return default_prompt


def validate_variable_mapping(
    mapping: str, is_json: bool = False
) -> tuple[str, Any]:
    """Validate a variable mapping in name=value format."""
    try:
        name, value = mapping.split("=", 1)
        if not name:
            raise VariableNameError(
                f"Empty name in {'JSON ' if is_json else ''}variable mapping"
            )

        if is_json:
            try:
                value = json.loads(value)
            except json.JSONDecodeError as e:
                raise InvalidJSONError(
                    f"Invalid JSON value for variable {name!r}: {value!r}"
                ) from e

        return name, value

    except ValueError as e:
        if "not enough values to unpack" in str(e):
            raise VariableValueError(
                f"Invalid {'JSON ' if is_json else ''}variable mapping "
                f"(expected name=value format): {mapping!r}"
            )
        raise


@overload
def _validate_path_mapping_internal(
    mapping: str,
    is_dir: Literal[True],
    base_dir: Optional[str] = None,
    security_manager: Optional[SecurityManager] = None,
) -> Tuple[str, str]: ...


@overload
def _validate_path_mapping_internal(
    mapping: str,
    is_dir: Literal[False] = False,
    base_dir: Optional[str] = None,
    security_manager: Optional[SecurityManager] = None,
) -> Tuple[str, str]: ...


def _validate_path_mapping_internal(
    mapping: str,
    is_dir: bool = False,
    base_dir: Optional[str] = None,
    security_manager: Optional[SecurityManager] = None,
) -> Tuple[str, str]:
    """Validate a path mapping in the format "name=path".

    Args:
        mapping: The path mapping string (e.g., "myvar=/path/to/file").
        is_dir: Whether the path is expected to be a directory (True) or file (False).
        base_dir: Optional base directory to resolve relative paths against.
        security_manager: Optional security manager to validate paths.

    Returns:
        A (name, path) tuple.

    Raises:
        VariableNameError: If the variable name portion is empty or invalid.
        DirectoryNotFoundError: If is_dir=True and the path is not a directory or doesn't exist.
        FileNotFoundError: If is_dir=False and the path is not a file or doesn't exist.
        PathSecurityError: If the path is inaccessible or outside the allowed directory.
        ValueError: If the format is invalid (missing "=").
        OSError: If there is an underlying OS error (permissions, etc.).
    """
    try:
        if not mapping or "=" not in mapping:
            raise ValueError(
                "Invalid path mapping format. Expected format: name=path"
            )

        name, path = mapping.split("=", 1)
        if not name:
            raise VariableNameError(
                f"Empty name in {'directory' if is_dir else 'file'} mapping"
            )

        if not path:
            raise VariableValueError("Path cannot be empty")

        # Convert to Path object and resolve against base_dir if provided
        path_obj = Path(path)
        if base_dir:
            path_obj = Path(base_dir) / path_obj

        # Resolve the path to catch directory traversal attempts
        try:
            resolved_path = path_obj.resolve()
        except OSError as e:
            raise OSError(f"Failed to resolve path: {e}")

        # Check for directory traversal
        try:
            base_path = (
                Path.cwd() if base_dir is None else Path(base_dir).resolve()
            )
            if not str(resolved_path).startswith(str(base_path)):
                raise PathSecurityError(
                    f"Path {str(path)!r} resolves to {str(resolved_path)!r} which is outside "
                    f"base directory {str(base_path)!r}"
                )
        except OSError as e:
            raise OSError(f"Failed to resolve base path: {e}")

        # Check if path exists
        if not resolved_path.exists():
            if is_dir:
                raise DirectoryNotFoundError(f"Directory not found: {path!r}")
            else:
                raise FileNotFoundError(f"File not found: {path!r}")

        # Check if path is correct type
        if is_dir and not resolved_path.is_dir():
            raise DirectoryNotFoundError(f"Path is not a directory: {path!r}")
        elif not is_dir and not resolved_path.is_file():
            raise FileNotFoundError(f"Path is not a file: {path!r}")

        # Check if path is accessible
        try:
            if is_dir:
                os.listdir(str(resolved_path))
            else:
                with open(str(resolved_path), "r", encoding="utf-8") as f:
                    f.read(1)
        except OSError as e:
            if e.errno == 13:  # Permission denied
                raise PathSecurityError(
                    f"Permission denied accessing path: {path!r}",
                    error_logged=True,
                )
            raise

        if security_manager:
            if not security_manager.is_allowed_file(str(resolved_path)):
                raise PathSecurityError.from_expanded_paths(
                    original_path=str(path),
                    expanded_path=str(resolved_path),
                    base_dir=str(security_manager.base_dir),
                    allowed_dirs=[
                        str(d) for d in security_manager.allowed_dirs
                    ],
                    error_logged=True,
                )

        # Return the original path to maintain relative paths in the output
        return name, path

    except ValueError as e:
        if "not enough values to unpack" in str(e):
            raise VariableValueError(
                f"Invalid {'directory' if is_dir else 'file'} mapping "
                f"(expected name=path format): {mapping!r}"
            )
        raise


def validate_task_template(task: str) -> str:
    """Validate and load a task template.

    Args:
        task: The task template string or path to task template file (with @ prefix)

    Returns:
        The task template string

    Raises:
        TaskTemplateVariableError: If the template file cannot be read or is invalid
        TaskTemplateSyntaxError: If the template has invalid syntax
        FileNotFoundError: If the template file does not exist
        PathSecurityError: If the template file path violates security constraints
    """
    template_content = task

    # Check if task is a file path
    if task.startswith("@"):
        path = task[1:]
        try:
            name, path = validate_path_mapping(f"task={path}")
            with open(path, "r", encoding="utf-8") as f:
                template_content = f.read()
        except (FileNotFoundError, PathSecurityError) as e:
            raise TaskTemplateVariableError(f"Invalid task template file: {e}")

    # Validate template syntax
    try:
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        env.parse(template_content)
        return template_content
    except jinja2.TemplateSyntaxError as e:
        raise TaskTemplateSyntaxError(
            f"Invalid task template syntax at line {e.lineno}: {e.message}"
        )


def validate_schema_file(path: str) -> Dict[str, Any]:
    """Validate and load a JSON schema file.

    Args:
        path: Path to the JSON schema file

    Returns:
        The loaded and validated schema

    Raises:
        InvalidJSONError: If the schema file contains invalid JSON
        SchemaValidationError: If the schema is invalid
        FileNotFoundError: If the schema file does not exist
        PathSecurityError: If the schema file is outside base directory
    """
    try:
        # Validate file exists and is readable
        name, path = validate_path_mapping(f"schema={path}")

        # Load and parse JSON
        with open(path, "r", encoding="utf-8") as f:
            try:
                schema = json.load(f)
            except json.JSONDecodeError as e:
                raise InvalidJSONError(f"Invalid JSON: {e.msg}")

        # Validate schema
        validate_json_schema(schema)
        return cast(Dict[str, Any], schema)
    except json.JSONDecodeError as e:
        raise InvalidJSONError(f"Invalid JSON: {e.msg}")


def collect_template_files(
    args: argparse.Namespace,
    security_manager: SecurityManager,
) -> Dict[str, TemplateValue]:
    """Collect files from command line arguments.

    Args:
        args: Parsed command line arguments
        security_manager: Security manager for path validation

    Returns:
        Dictionary mapping variable names to file info objects

    Raises:
        PathSecurityError: If any file paths violate security constraints
        ValueError: If file mappings are invalid or files cannot be accessed
    """
    try:
        result = collect_files(
            file_mappings=args.file,
            pattern_mappings=args.files,
            dir_mappings=args.dir,
            recursive=args.recursive,
            extensions=args.ext.split(",") if args.ext else None,
            security_manager=security_manager,
        )
        return cast(Dict[str, TemplateValue], result)
    except PathSecurityError:
        # Let PathSecurityError propagate without wrapping
        raise
    except (FileNotFoundError, DirectoryNotFoundError) as e:
        # Wrap file-related errors
        raise ValueError(f"File access error: {e}")
    except Exception as e:
        # Check if this is a wrapped security error
        if isinstance(e.__cause__, PathSecurityError):
            raise e.__cause__
        # Wrap unexpected errors
        raise ValueError(f"Error collecting files: {e}")


def collect_simple_variables(args: argparse.Namespace) -> Dict[str, str]:
    """Collect simple string variables from --var arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary mapping variable names to string values

    Raises:
        VariableNameError: If a variable name is invalid or duplicate
    """
    variables: Dict[str, str] = {}
    all_names: Set[str] = set()

    if args.var:
        for mapping in args.var:
            try:
                name, value = mapping.split("=", 1)
                if not name.isidentifier():
                    raise VariableNameError(f"Invalid variable name: {name}")
                if name in all_names:
                    raise VariableNameError(f"Duplicate variable name: {name}")
                variables[name] = value
                all_names.add(name)
            except ValueError:
                raise VariableNameError(
                    f"Invalid variable mapping (expected name=value format): {mapping!r}"
                )

    return variables


def collect_json_variables(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect JSON variables from --json-var arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary mapping variable names to parsed JSON values

    Raises:
        VariableNameError: If a variable name is invalid or duplicate
        InvalidJSONError: If a JSON value is invalid
    """
    variables: Dict[str, Any] = {}
    all_names: Set[str] = set()

    if args.json_var:
        for mapping in args.json_var:
            try:
                name, json_str = mapping.split("=", 1)
                if not name.isidentifier():
                    raise VariableNameError(f"Invalid variable name: {name}")
                if name in all_names:
                    raise VariableNameError(f"Duplicate variable name: {name}")
                try:
                    value = json.loads(json_str)
                    variables[name] = value
                    all_names.add(name)
                except json.JSONDecodeError as e:
                    raise InvalidJSONError(
                        f"Invalid JSON value for {name}: {str(e)}"
                    )
            except ValueError:
                raise VariableNameError(
                    f"Invalid JSON variable mapping format: {mapping}. Expected name=json"
                )

    return variables


def create_template_context(
    args: argparse.Namespace,
    security_manager: SecurityManager,
) -> Dict[str, Any]:
    """Create template context from command line arguments.

    Args:
        args: Parsed command line arguments
        security_manager: Security manager for path validation

    Returns:
        Template context dictionary with files accessible as:
            doc.content  # For single files
            doc[0].content  # Traditional access (still works)
            doc.content  # Returns list for multiple files

    Raises:
        PathSecurityError: If any file paths violate security constraints
        VariableError: If variable mappings are invalid
        ValueError: If file mappings are invalid or files cannot be accessed
    """
    try:
        context: Dict[str, Any] = {}

        # Only collect files if there are file mappings
        if any([args.file, args.files, args.dir]):
            files = collect_files(
                file_mappings=args.file,
                pattern_mappings=args.files,
                dir_mappings=args.dir,
                recursive=args.recursive,
                extensions=args.ext.split(",") if args.ext else None,
                security_manager=security_manager,
            )
            context.update(files)

        # Add simple variables
        try:
            variables = collect_simple_variables(args)
            context.update(variables)
        except VariableNameError as e:
            raise VariableError(str(e))

        # Add JSON variables
        if args.json_var:
            for mapping in args.json_var:
                try:
                    name, value = mapping.split("=", 1)
                    if not name.isidentifier():
                        raise VariableNameError(
                            f"Invalid variable name: {name}"
                        )
                    try:
                        json_value = json.loads(value)
                    except json.JSONDecodeError as e:
                        raise InvalidJSONError(
                            f"Invalid JSON value for {name} ({value!r}): {str(e)}"
                        )
                    if name in context:
                        raise VariableNameError(
                            f"Duplicate variable name: {name}"
                        )
                    context[name] = json_value
                except ValueError:
                    raise VariableNameError(
                        f"Invalid JSON variable mapping format: {mapping}. Expected name=json"
                    )

        # Add stdin if available and readable
        try:
            if not sys.stdin.isatty():
                context["stdin"] = sys.stdin.read()
        except (OSError, IOError):
            # Skip stdin if it can't be read (e.g. in pytest environment)
            pass

        return context

    except PathSecurityError:
        # Let PathSecurityError propagate without wrapping
        raise
    except (FileNotFoundError, DirectoryNotFoundError) as e:
        # Wrap file-related errors
        raise ValueError(f"File access error: {e}")
    except Exception as e:
        # Check if this is a wrapped security error
        if isinstance(e.__cause__, PathSecurityError):
            raise e.__cause__
        # Wrap unexpected errors
        raise ValueError(f"Error collecting files: {e}")


def validate_security_manager(
    base_dir: Optional[str] = None,
    allowed_dirs: Optional[List[str]] = None,
    allowed_dirs_file: Optional[str] = None,
) -> SecurityManager:
    """Create and validate a security manager.

    Args:
        base_dir: Optional base directory to resolve paths against
        allowed_dirs: Optional list of allowed directory paths
        allowed_dirs_file: Optional path to file containing allowed directories

    Returns:
        Configured SecurityManager instance

    Raises:
        FileNotFoundError: If allowed_dirs_file does not exist
        PathSecurityError: If any paths are outside base directory
    """
    # Convert base_dir to string if it's a Path
    base_dir_str = str(base_dir) if base_dir else None
    security_manager = SecurityManager(base_dir_str)

    if allowed_dirs_file:
        security_manager.add_allowed_dirs_from_file(str(allowed_dirs_file))

    if allowed_dirs:
        for allowed_dir in allowed_dirs:
            security_manager.add_allowed_dir(str(allowed_dir))

    return security_manager


def parse_var(var_str: str) -> Tuple[str, str]:
    """Parse a variable string in the format 'name=value'.

    Args:
        var_str: Variable string in format 'name=value'

    Returns:
        Tuple of (name, value)

    Raises:
        VariableNameError: If variable name is empty or invalid
        VariableValueError: If variable format is invalid
    """
    try:
        name, value = var_str.split("=", 1)
        if not name:
            raise VariableNameError("Empty name in variable mapping")
        if not name.isidentifier():
            raise VariableNameError(
                f"Invalid variable name: {name}. Must be a valid Python identifier"
            )
        return name, value

    except ValueError as e:
        if "not enough values to unpack" in str(e):
            raise VariableValueError(
                f"Invalid variable mapping (expected name=value format): {var_str!r}"
            )
        raise


def parse_json_var(var_str: str) -> Tuple[str, Any]:
    """Parse a JSON variable string in the format 'name=json_value'.

    Args:
        var_str: Variable string in format 'name=json_value'

    Returns:
        Tuple of (name, parsed_value)

    Raises:
        VariableNameError: If variable name is empty or invalid
        VariableValueError: If variable format is invalid
        InvalidJSONError: If JSON value is invalid
    """
    try:
        name, json_str = var_str.split("=", 1)
        if not name:
            raise VariableNameError("Empty name in JSON variable mapping")
        if not name.isidentifier():
            raise VariableNameError(
                f"Invalid variable name: {name}. Must be a valid Python identifier"
            )

        try:
            value = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise InvalidJSONError(
                f"Invalid JSON value for variable {name!r}: {json_str!r}"
            ) from e

        return name, value

    except ValueError as e:
        if "not enough values to unpack" in str(e):
            raise VariableValueError(
                f"Invalid JSON variable mapping (expected name=json format): {var_str!r}"
            )
        raise


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the CLI.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Make structured OpenAI API calls from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--task",
        required=True,
        help="Task template string or @file",
    )

    # File access arguments
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Map file to variable (name=path)",
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--files",
        action="append",
        default=[],
        help="Map file pattern to variable (name=pattern)",
        metavar="NAME=PATTERN",
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        help="Map directory to variable (name=path)",
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--allowed-dir",
        action="append",
        default=[],
        help="Additional allowed directory or @file",
        metavar="PATH",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively",
    )
    parser.add_argument(
        "--ext",
        help="Comma-separated list of file extensions to include",
    )

    # Variable arguments
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        help="Pass simple variables (name=value)",
        metavar="NAME=VALUE",
    )
    parser.add_argument(
        "--json-var",
        action="append",
        default=[],
        help="Pass JSON variables (name=json)",
        metavar="NAME=JSON",
    )

    # System prompt options
    parser.add_argument(
        "--system-prompt",
        help=(
            "System prompt for the model (use @file to load from file, "
            "can also be specified in task template YAML frontmatter)"
        ),
        default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--ignore-task-sysprompt",
        action="store_true",
        help="Ignore system prompt from task template YAML frontmatter",
    )

    # Schema validation
    parser.add_argument(
        "--schema",
        dest="schema_file",
        required=True,
        help="JSON schema file for response validation",
    )
    parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate schema and response",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-08-06",
        help="Model to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature (0.0-2.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling (0.0-1.0)",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Frequency penalty (-2.0-2.0)",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty (-2.0-2.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="API timeout in seconds",
    )

    # Output options
    parser.add_argument(
        "--output-file",
        help="Write JSON output to file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate API call without making request",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress indicators",
    )

    # Other options
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (overrides env var)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


async def _main() -> ExitCode:
    """Main entry point for the CLI.

    Returns:
        Exit code indicating success or type of failure
    """
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    log_dir = os.path.expanduser("~/.ostruct/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "ostruct.log")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    root_logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s:%(name)s:%(funcName)s: %(message)s")
    )
    root_logger.addHandler(console_handler)

    logger = logging.getLogger("ostruct")
    logger.debug("Starting ostruct CLI with log file: %s", log_file)

    # Initialize security manager with current directory as base
    security_manager = SecurityManager(str(Path.cwd()))
    logger.debug("Initialized security manager with base dir: %s", Path.cwd())

    # Process allowed directories
    if args.allowed_dir:
        for allowed_dir in args.allowed_dir:
            if allowed_dir.startswith("@"):
                # Read allowed directories from file
                allowed_file = allowed_dir[1:]
                try:
                    security_manager.add_allowed_dirs_from_file(allowed_file)
                except PathSecurityError as e:
                    if not e.has_been_logged:
                        logger.error(str(e))
                    return ExitCode.SECURITY_ERROR
                except OSError as e:
                    logger.error(
                        f"Could not read allowed directories file: {e}"
                    )
                    return ExitCode.IO_ERROR
            else:
                # Add single allowed directory
                try:
                    security_manager.add_allowed_dir(allowed_dir)
                except OSError as e:
                    logger.error(f"Invalid allowed directory path: {e}")
                    return ExitCode.IO_ERROR

    # Create template context from arguments with security checks
    try:
        logger.debug("[_main] Creating template context from arguments")
        template_context = create_template_context(args, security_manager)
    except PathSecurityError as e:
        logger.debug(
            "[_main] Caught PathSecurityError: %s (logged=%s)",
            str(e),
            getattr(e, "has_been_logged", False),
        )
        if not getattr(e, "has_been_logged", False):
            logger.error(str(e))
        return ExitCode.SECURITY_ERROR
    except VariableError as e:
        logger.debug("[_main] Caught VariableError: %s", str(e))
        logger.error(str(e))
        return ExitCode.DATA_ERROR
    except OSError as e:
        logger.debug("[_main] Caught OSError: %s", str(e))
        logger.error(f"File access error: {e}")
        return ExitCode.IO_ERROR
    except ValueError as e:
        # Check if this is a wrapped security error
        if isinstance(e.__cause__, PathSecurityError):
            logger.debug(
                "[_main] Caught wrapped PathSecurityError in ValueError: %s (logged=%s)",
                str(e.__cause__),
                getattr(e.__cause__, "has_been_logged", False),
            )
            if not getattr(e.__cause__, "has_been_logged", False):
                logger.error(str(e.__cause__))
            return ExitCode.SECURITY_ERROR
        # Check if this is a wrapped security error in the error message
        if "Access denied:" in str(e):
            logger.debug(
                "[_main] Detected security error in ValueError message: %s",
                str(e),
            )
            logger.error(f"Invalid input: {e}")
            return ExitCode.SECURITY_ERROR
        logger.debug("[_main] Caught ValueError: %s", str(e))
        logger.error(f"Invalid input: {e}")
        return ExitCode.DATA_ERROR

    # Create Jinja2 environment
    env = create_jinja_env(
        loader=jinja2.FileSystemLoader("."),
    )

    # Load and validate task template
    try:
        task_template = validate_task_template(args.task)
    except TaskTemplateSyntaxError as e:
        logger.error(f"Template syntax error: {e}")
        return ExitCode.VALIDATION_ERROR
    except PathSecurityError as e:
        if not e.has_been_logged:
            logger.error(str(e))
        return ExitCode.SECURITY_ERROR
    except OSError as e:
        logger.error(f"Could not read template file: {e}")
        return ExitCode.IO_ERROR

    # Validate template placeholders
    try:
        validate_template_placeholders(task_template, template_context, env)
    except ValueError as e:
        logger.error(f"Template validation error: {e}")
        return ExitCode.VALIDATION_ERROR

    # Load and validate schema
    try:
        schema = validate_schema_file(args.schema_file)
    except SchemaValidationError as e:
        logger.error(f"Schema validation error: {e}")
        return ExitCode.SCHEMA_ERROR
    except PathSecurityError as e:
        if not e.has_been_logged:
            logger.error(str(e))
        return ExitCode.SECURITY_ERROR
    except OSError as e:
        logger.error(f"Could not read schema file: {e}")
        return ExitCode.IO_ERROR

    # Process system prompt
    try:
        system_prompt = process_system_prompt(
            task_template,
            args.system_prompt,
            template_context,
            env,
            args.ignore_task_sysprompt,
        )
    except (TemplateMetadataError, SystemPromptError) as e:
        logger.error(str(e))
        return ExitCode.VALIDATION_ERROR

    # Render task template
    try:
        user_message = render_template(task_template, template_context, env)
    except jinja2.TemplateSyntaxError as e:
        logger.error(f"Template syntax error: {e}")
        return ExitCode.VALIDATION_ERROR
    except jinja2.UndefinedError as e:
        logger.error(f"Template variable error: {e}")
        return ExitCode.VALIDATION_ERROR
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return ExitCode.VALIDATION_ERROR

    # Create messages for chat completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Estimate tokens
    try:
        total_tokens = estimate_tokens_for_chat(messages, args.model)
        validate_token_limits(args.model, total_tokens, args.max_tokens)
        logger.debug(f"Total tokens in prompt: {total_tokens:,}")
    except (ValueError, OpenAIClientError) as e:
        logger.error(str(e))
        return ExitCode.VALIDATION_ERROR

    # Handle dry run mode
    if args.dry_run:
        logger.info("*** DRY RUN MODE - No API call will be made ***\n")
        logger.info("System Prompt:\n%s\n", system_prompt)
        logger.info("User Prompt:\n%s\n", user_message)
        logger.info("Estimated Tokens: %s", total_tokens)
        logger.info("Model: %s", args.model)
        logger.info("Temperature: %s", args.temperature)
        if args.max_tokens is not None:
            logger.info("Max Tokens: %s", args.max_tokens)
        logger.info("Top P: %s", args.top_p)
        logger.info("Frequency Penalty: %s", args.frequency_penalty)
        logger.info("Presence Penalty: %s", args.presence_penalty)
        if args.validate_schema:
            logger.info("Schema: Valid")
        if args.output_file:
            logger.info("Output would be written to: %s", args.output_file)
        return ExitCode.SUCCESS

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "No OpenAI API key provided (--api-key or OPENAI_API_KEY env var)"
        )
        return ExitCode.USAGE_ERROR

    # Create OpenAI client
    client = AsyncOpenAI(api_key=api_key, timeout=args.timeout)

    # Make API request
    try:
        if not supports_structured_output(args.model):
            logger.error(
                f"Model '{args.model}' does not support structured output"
            )
            return ExitCode.API_ERROR

        # Create output schema model
        output_model = (
            create_dynamic_model(schema)
            if args.validate_schema
            else create_model(
                "DynamicModel",
                __config__=ConfigDict(arbitrary_types_allowed=True),
                result=(Any, ...),
            )
        )

        # Initialize response
        response = None
        async for chunk in async_openai_structured_stream(
            client=client,
            model=args.model,
            output_schema=output_model,
            system_prompt=system_prompt,
            user_prompt=user_message,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
        ):
            # Each chunk is already a structured response
            logger.debug("Received chunk: %s", chunk)
            response = chunk
            # Print each chunk as it arrives
            if not args.output_file:
                # Convert Pydantic model to dict before JSON serialization
                chunk_dict = (
                    chunk.model_dump()
                    if hasattr(chunk, "model_dump")
                    else chunk
                )
                print(json.dumps(chunk_dict, indent=2))

        if response is None:
            raise EmptyResponseError("No response received from API")

        # Write response to file if requested
        if args.output_file:
            try:
                # Convert Pydantic model to dict before JSON serialization
                response_dict = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else response
                )
                with open(args.output_file, "w") as f:
                    json.dump(response_dict, f, indent=2)
            except OSError:
                raise ValueError("Could not write to output file")

        return ExitCode.SUCCESS

    except KeyboardInterrupt:
        return ExitCode.INTERRUPTED

    except (
        APIConnectionError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
        InternalServerError,
    ) as e:
        logger.error(f"API error: {e}")
        return ExitCode.API_ERROR

    except (
        APIResponseError,
        EmptyResponseError,
        InvalidResponseFormatError,
        JSONParseError,
        ModelNotSupportedError,
        ModelVersionError,
        StreamBufferError,
        StreamInterruptedError,
        StreamParseError,
    ) as e:
        logger.error(f"Stream error: {e}")
        return ExitCode.API_ERROR

    except KeyboardInterrupt:
        return ExitCode.INTERRUPTED

    except (
        FileNotFoundError,
        DirectoryNotFoundError,
        PathSecurityError,
        SchemaFileError,
    ) as e:
        logger.error(str(e))
        return ExitCode.IO_ERROR

    except (
        InvalidJSONError,
        SchemaValidationError,
        TaskTemplateSyntaxError,
        TaskTemplateVariableError,
        VariableNameError,
        VariableValueError,
    ) as e:
        logger.error(str(e))
        return ExitCode.VALIDATION_ERROR

    except OpenAIClientError as e:
        logger.error(str(e))
        return ExitCode.API_ERROR

    except Exception:
        logger.error("Unexpected error occurred")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return ExitCode.INTERNAL_ERROR


def main() -> None:
    """CLI entry point that handles all errors."""
    try:
        logger.debug("[main] Starting main execution")
        exit_code = asyncio.run(_main())
        sys.exit(exit_code.value)
    except KeyboardInterrupt:
        logger.error("Operation cancelled by user")
        sys.exit(ExitCode.INTERRUPTED.value)
    except PathSecurityError as e:
        # Only log security errors if they haven't been logged already
        logger.debug(
            "[main] Caught PathSecurityError: %s (logged=%s)",
            str(e),
            getattr(e, "has_been_logged", False),
        )
        if not getattr(e, "has_been_logged", False):
            logger.error(str(e))
        sys.exit(ExitCode.SECURITY_ERROR.value)
    except ValueError as e:
        # Get the original cause of the error
        cause = e.__cause__ or e.__context__
        if isinstance(cause, PathSecurityError):
            logger.debug(
                "[main] Caught wrapped PathSecurityError in ValueError: %s (logged=%s)",
                str(cause),
                getattr(cause, "has_been_logged", False),
            )
            # Only log security errors if they haven't been logged already
            if not getattr(cause, "has_been_logged", False):
                logger.error(str(cause))
            sys.exit(ExitCode.SECURITY_ERROR.value)
        else:
            logger.debug("[main] Caught ValueError: %s", str(e))
            logger.error(f"Invalid input: {e}")
            sys.exit(ExitCode.DATA_ERROR.value)
    except Exception as e:
        # Check if this is a wrapped security error
        if isinstance(e.__cause__, PathSecurityError):
            logger.debug(
                "[main] Caught wrapped PathSecurityError in Exception: %s (logged=%s)",
                str(e.__cause__),
                getattr(e.__cause__, "has_been_logged", False),
            )
            # Only log security errors if they haven't been logged already
            if not getattr(e.__cause__, "has_been_logged", False):
                logger.error(str(e.__cause__))
            sys.exit(ExitCode.SECURITY_ERROR.value)
        logger.debug("[main] Caught unexpected error: %s", str(e))
        logger.error(f"Invalid input: {e}")
        sys.exit(ExitCode.DATA_ERROR.value)


if __name__ == "__main__":
    main()

# Export public API
__all__ = [
    "create_dynamic_model",
    "validate_template_placeholders",
    "estimate_tokens_for_chat",
    "get_context_window_limit",
    "get_default_token_limit",
    "validate_token_limits",
    "supports_structured_output",
    "ProgressContext",
    "validate_path_mapping",
]
