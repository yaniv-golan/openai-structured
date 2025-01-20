"""Command-line interface for making structured OpenAI API calls."""

import argparse
import asyncio
import json
import logging
import os
import sys
from enum import IntEnum
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Type, TypeVar, Set, Union
from pathlib import Path

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
from jinja2 import meta

from ..client import async_openai_structured_stream, supports_structured_output
from ..errors import (
    APIResponseError,
    EmptyResponseError,
    InvalidResponseFormatError,
    JSONParseError,
    ModelNotSupportedError,
    ModelVersionError,
    OpenAIClientError,
    StreamBufferError,
    StreamInterruptedError,
    StreamParseError,
    DirectoryNotFoundError,
    FileNotFoundError,
    InvalidJSONError,
    PathSecurityError,
    SchemaFileError,
    SchemaValidationError,
    TaskTemplateSyntaxError,
    TaskTemplateVariableError,
    VariableNameError,
    VariableValueError,
    VariableError,
)
from .file_utils import TemplateValue, collect_files, FileInfo
from .progress import ProgressContext
from .template_utils import (
    SystemPromptError,
    TemplateMetadataError,
    extract_metadata,
    find_all_template_variables,
    render_template,
    validate_json_schema,
    validate_response,
    validate_template_placeholders,
    extract_template_metadata,
)
from .security import SecurityManager

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
    ignore_task_sysprompt: bool = False
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
                        if isinstance(metadata, dict) and "system_prompt" in metadata:
                            system_prompt = str(metadata["system_prompt"])
                            # Render system prompt with template context
                            try:
                                template = env.from_string(system_prompt)
                                return template.render(**template_context).strip()
                            except jinja2.TemplateError as e:
                                raise SystemPromptError(f"Error rendering system prompt: {e}")
                    except yaml.YAMLError as e:
                        raise SystemPromptError(f"Invalid YAML frontmatter: {e}")
                        
        except Exception as e:
            raise SystemPromptError(f"Error extracting system prompt from template: {e}")
            
    # Fall back to default
    return default_prompt


def validate_variable_mapping(mapping: str, is_json: bool = False) -> tuple[str, Any]:
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


def validate_path_mapping(mapping: str, is_dir: bool = False) -> tuple[str, str]:
    """Validate a path mapping in name=path format.
    
    Args:
        mapping: The path mapping string in name=path format
        is_dir: Whether the path should be a directory
        
    Returns:
        Tuple of (name, path)
        
    Raises:
        ValueError: If the mapping is invalid
        OSError: If the path does not exist or is inaccessible
        VariableNameError: If the name part is empty
        FileNotFoundError: If the specified file does not exist
        DirectoryNotFoundError: If the specified directory does not exist
        PathSecurityError: If the path violates security constraints
    """
    try:
        name, path = mapping.split("=", 1)
        if not name:
            raise VariableNameError(
                f"Empty name in {'directory' if is_dir else 'file'} mapping"
            )
            
        # Check if path exists
        if not os.path.exists(path):
            if is_dir:
                raise DirectoryNotFoundError(f"Directory not found: {path!r}")
            else:
                raise FileNotFoundError(f"File not found: {path!r}")
            
        # Check if path is correct type
        if is_dir and not os.path.isdir(path):
            raise DirectoryNotFoundError(f"Path is not a directory: {path!r}")
        elif not is_dir and not os.path.isfile(path):
            raise FileNotFoundError(f"Path is not a file: {path!r}")
            
        # Check if path is accessible
        try:
            if is_dir:
                os.listdir(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    f.read(1)
        except OSError as e:
            if e.errno == 13:  # Permission denied
                raise PathSecurityError(f"Permission denied accessing path: {path!r}")
            raise
            
        # Normalize path to catch directory traversal attempts
        norm_path = os.path.realpath(path)
        base_dir = os.path.realpath(".")
        
        # Check if normalized path starts with base directory
        rel_path = os.path.relpath(norm_path, base_dir)
        if rel_path.startswith(".."):
            raise PathSecurityError(f"Path {path!r} is outside the base directory")
            
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


def validate_schema_file(path: str) -> dict:
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
        return schema
    except json.JSONDecodeError as e:
        raise InvalidJSONError(f"Invalid JSON: {e.msg}")


def collect_file_variables(args: argparse.Namespace) -> Dict[str, TemplateValue]:
    """Collect file-related variables from command line arguments.
    
    This function handles all file-related argument types:
    - Single files (--file)
    - Multiple files (--files)
    - Directories (--dir)
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary mapping variable names to FileInfo objects or lists
        
    Raises:
        ValueError: If file mappings are invalid
        FileNotFoundError: If files do not exist
        DirectoryNotFoundError: If directories do not exist
        PathSecurityError: If paths violate security constraints
    """
    try:
        return collect_files(
            file_args=args.file,
            files_args=args.files,
            dir_args=args.dir,
            recursive=args.recursive,
            allowed_extensions=None,  # We'll handle extensions elsewhere if needed
            load_content=False,  # Keep content loading lazy
            allowed_dirs=args.allowed_dir,
        )
    except (FileNotFoundError, DirectoryNotFoundError, PathSecurityError):
        raise  # Let these propagate unchanged
    except Exception as e:
        raise ValueError(f"Error collecting files: {e}")


def collect_simple_variables(args) -> Dict[str, str]:
    """Collect simple string variables from --var arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary mapping variable names to string values
        
    Raises:
        VariableNameError: If a variable name is invalid or duplicate
    """
    variables = {}
    all_names = set()
    
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
                raise VariableNameError(f"Invalid variable mapping format: {mapping}. Expected name=value")
                
    return variables


def collect_json_variables(args) -> Dict[str, Any]:
    """Collect JSON variables from --json-var arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary mapping variable names to parsed JSON values
        
    Raises:
        VariableNameError: If a variable name is invalid or duplicate
        InvalidJSONError: If a JSON value is invalid
    """
    variables = {}
    all_names = set()
    
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
                    raise InvalidJSONError(f"Invalid JSON value for {name}: {str(e)}")
            except ValueError:
                raise VariableNameError(f"Invalid JSON variable mapping format: {mapping}. Expected name=json")
                
    return variables


def create_template_context(args: argparse.Namespace, security_manager: SecurityManager) -> Dict[str, TemplateValue]:
    """Create template context from CLI arguments."""
    # Collect variables from different sources
    file_variables = collect_file_variables(args)
    simple_variables = collect_simple_variables(args)
    json_variables = collect_json_variables(args)
    
    # Check for naming conflicts
    all_names = set()
    for name in file_variables:
        if name in all_names:
            raise VariableNameError(f"Duplicate variable name: {name}")
        all_names.add(name)
        
    for name in simple_variables:
        if name in all_names:
            raise VariableNameError(f"Duplicate variable name: {name}")
        all_names.add(name)
        
    for name in json_variables:
        if name in all_names:
            raise VariableNameError(f"Duplicate variable name: {name}")
        all_names.add(name)
    
    # Combine all variables into template context
    template_context = {
        **file_variables,    # FileInfo objects and lists
        **simple_variables,  # Simple string values
        **json_variables     # Parsed JSON structures
    }
    
    # Apply security checks only to FileInfo objects
    for name, value in template_context.items():
        if isinstance(value, FileInfo):
            if not security_manager.is_allowed_file(Path(value.path)):
                raise PathSecurityError(f"File access denied: {value.path}")
        elif isinstance(value, list) and value and isinstance(value[0], FileInfo):
            # Check each FileInfo in a list
            for file_info in value:
                if not security_manager.is_allowed_file(Path(file_info.path)):
                    raise PathSecurityError(f"File access denied: {file_info.path}")
    
    return template_context


async def _main() -> ExitCode:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Make structured OpenAI API calls from the command line."
    )
    
    # Add version argument
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit"
    )
    
    # Add task argument
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task template string or @file"
    )

    # File input arguments
    parser.add_argument('--file', action='append', metavar='NAME=PATH',
                      help='Map file to variable name. Format: name=path. Can be specified multiple times.')
    parser.add_argument('--files', action='append', metavar='NAME=PATTERN',
                      help='Map glob pattern to variable name. Format: name=pattern. Can be specified multiple times.')
    parser.add_argument('--dir', action='append', metavar='NAME=PATH',
                      help='Map directory to variable name. Format: name=path. Can be specified multiple times.')
    parser.add_argument('--recursive', action='store_true',
                      help='Recursively process directories')
    parser.add_argument('--ext',
                      help='Comma-separated list of file extensions to include')

    # Variable arguments
    parser.add_argument('--var', action='append', metavar='NAME=VALUE',
                      help='Set variable to string value. Format: name=value. Can be specified multiple times.')
    parser.add_argument('--json-var', action='append', metavar='NAME=JSON',
                      help='Set variable to JSON value. Format: name=json. Can be specified multiple times.')
    parser.add_argument('--value', action='append', metavar='NAME=VALUE',
                      help='Value mapping in name=value format. Can be specified multiple times.')
    
    # Model arguments  
    parser.add_argument('--model', default='gpt-4o-2024-08-06',
                      help=('OpenAI model to use. Supported models:\n'
                            '- gpt-4o: 128K context, 16K output\n'
                            '- gpt-4o-mini: 128K context, 16K output\n'
                            '- o1: 200K context, 100K output'))
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Temperature for sampling (default: 0.0)')
    parser.add_argument('--max-tokens', type=int,
                      help=('Maximum number of tokens to generate. Set to 0 or negative to disable '
                            'token limit checks. Defaults to model-specific limit.'))
    parser.add_argument('--top-p', type=float, default=1.0,
                      help='Top-p sampling parameter (default: 1.0)')
    parser.add_argument('--frequency-penalty', type=float, default=0.0,
                      help='Frequency penalty parameter (default: 0.0)')
    parser.add_argument('--presence-penalty', type=float, default=0.0,
                      help='Presence penalty parameter (default: 0.0)')
    parser.add_argument('--timeout', type=float, default=60.0,
                      help='Timeout in seconds for API calls (default: 60.0)')
    
    # System prompt arguments
    parser.add_argument('--system-prompt',
                      help='System prompt string or @file. Overrides task template system prompt.')
    parser.add_argument('--ignore-task-sysprompt', action='store_true',
                      help='Ignore system prompt in task template even if present.')
    
    # Output arguments
    parser.add_argument('--output-file',
                      help='Write JSON output to this file instead of stdout')
    parser.add_argument('--schema-file', required=True,
                      help='Path to JSON schema file defining the response structure')
    parser.add_argument('--validate-schema', action='store_true',
                      help='Validate the JSON schema file and response')
    parser.add_argument('--no-progress', action='store_true',
                      help='Disable progress indicators')
    parser.add_argument('--dry-run', action='store_true',
                      help='Print request without sending')
    
    # Other arguments
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--api-key',
                      help=('OpenAI API key. Overrides OPENAI_API_KEY environment variable. '
                            'Warning: Key might be visible in process list or shell history.'))
    parser.add_argument(
        "--allowed-dir",
        action="append",
        type=str,
        help="Additional directory to allow file access or a file containing a list of allowed directories (using @ notation) (can be used multiple times)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger("ostruct")
    
    # Initialize security manager with current directory as base
    security_manager = SecurityManager(Path.cwd())
    
    # Process allowed directories
    if args.allowed_dir:
        for allowed_dir in args.allowed_dir:
            if allowed_dir.startswith('@'):
                # Read allowed directories from file
                allowed_file = Path(allowed_dir[1:])
                try:
                    security_manager.add_allowed_dirs_from_file(allowed_file)
                except PathSecurityError as e:
                    logger.error(str(e))
                    return ExitCode.SECURITY_ERROR
                except OSError as e:
                    logger.error(f"Could not read allowed directories file: {e}")
                    return ExitCode.IO_ERROR
            else:
                # Add single allowed directory
                try:
                    security_manager.add_allowed_dir(Path(allowed_dir))
                except OSError as e:
                    logger.error(f"Invalid allowed directory path: {e}")
                    return ExitCode.IO_ERROR
    
    # Create template context from arguments with security checks
    try:
        template_context = create_template_context(args, security_manager)
    except PathSecurityError as e:
        logger.error(str(e))
        return ExitCode.SECURITY_ERROR
    except VariableError as e:  # Add specific handling for variable errors
        logger.error(str(e))
        return ExitCode.DATA_ERROR
    except OSError as e:
        logger.error(f"File access error: {e}")
        return ExitCode.IO_ERROR
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return ExitCode.DATA_ERROR
    
    # Create Jinja2 environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("."),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        line_statement_prefix="#",
        line_comment_prefix="##",
        undefined=jinja2.StrictUndefined,
        extensions=['jinja2.ext.do', 'jinja2.ext.loopcontrols']
    )
    
    # Load and validate task template
    try:
        task_template = validate_task_template(args.task)
    except TaskTemplateSyntaxError as e:
        logger.error(f"Template syntax error: {e}")
        return ExitCode.VALIDATION_ERROR
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
            args.ignore_task_sysprompt
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
        {"role": "user", "content": user_message}
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
    client = AsyncOpenAI(
        api_key=api_key,
        timeout=args.timeout
    )
    
    # Create progress context
    progress = ProgressContext(
        show_progress=not args.no_progress,
        output_file=args.output_file
    )
    
    # Make API request
    try:
        if not supports_structured_output(args.model):
            logger.error(
                f"Model '{args.model}' does not support structured output"
            )
            return ExitCode.API_ERROR
            
        # Create output schema model
        output_model = create_dynamic_model(schema) if args.validate_schema else create_model(
            "DynamicModel",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            **{"result": (Any, ...)}
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
            presence_penalty=args.presence_penalty
        ):
            # Each chunk is already a structured response
            response = chunk
        
        if response is None:
            raise EmptyResponseError("No response received from API")
        
        # Write response to file if requested
        if args.output_file:
            try:
                with open(args.output_file, 'w') as f:
                    json.dump(response, f, indent=2)
            except OSError as e:
                raise ValueError(f"Could not write to output file: {e}")
        
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
    
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return ExitCode.INTERNAL_ERROR


async def main() -> ExitCode:
    """Async main entry point for the CLI."""
    try:
        return await _main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return ExitCode.INTERRUPTED
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return ExitCode.UNKNOWN_ERROR


def cli_main() -> int:
    """Synchronous entry point for command line usage."""
    try:
        return int(asyncio.run(main()))
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return int(ExitCode.UNKNOWN_ERROR)


if __name__ == "__main__":
    sys.exit(cli_main())

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
]
