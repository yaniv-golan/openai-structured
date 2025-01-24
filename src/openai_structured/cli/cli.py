"""Command-line interface for making structured OpenAI API calls."""

import argparse
import asyncio
import json
import logging
import os
import sys
from enum import IntEnum, Enum
if sys.version_info >= (3, 11):
    from enum import StrEnum
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
from pydantic import (
    BaseModel,
    ConfigDict,
    create_model,
    Field,
    TypeAdapter,
    EmailStr,
    AnyUrl,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaMode
import annotated_types
from datetime import datetime, date, time

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
    ModelCreationError,
    FieldDefinitionError,
    NestedModelError,
    ModelValidationError,
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


def create_dynamic_model(
    schema: Dict[str, Any],
    base_name: str = "DynamicModel",
    show_schema: bool = False,
    debug_validation: bool = False,
) -> Type[BaseModel]:
    """Create a Pydantic model from a JSON schema.
    
    Args:
        schema: JSON schema dict, can be wrapped in {"schema": ...} format
        base_name: Base name for the model
        show_schema: Whether to show the generated schema
        debug_validation: Whether to enable validation debugging
        
    Returns:
        Generated Pydantic model class
        
    Raises:
        ModelCreationError: When model creation fails
        SchemaValidationError: When schema is invalid
    """
    if debug_validation:
        logger.info("Creating dynamic model from schema:")
        logger.info(json.dumps(schema, indent=2))
    
    # Validate schema is a dictionary
    if not isinstance(schema, dict):
        if debug_validation:
            logger.error("Schema must be a dictionary, got %s", type(schema))
        raise SchemaValidationError("Schema must be a dictionary")
    
    # Handle our wrapper format if present
    if "schema" in schema:
        if debug_validation:
            logger.info("Found schema wrapper, extracting inner schema")
            logger.info("Original schema: %s", json.dumps(schema, indent=2))
        inner_schema = schema["schema"]
        if not isinstance(inner_schema, dict):
            if debug_validation:
                logger.error("Inner schema must be a dictionary, got %s", type(inner_schema))
            raise SchemaValidationError("Inner schema must be a dictionary")
        if debug_validation:
            logger.info("Using inner schema:")
            logger.info(json.dumps(inner_schema, indent=2))
        schema = inner_schema
    
    # Ensure schema has type field
    if "type" not in schema:
        if debug_validation:
            logger.info("Schema missing type field, assuming object type")
        schema["type"] = "object"
    
    # Validate root schema is object type
    if schema["type"] != "object":
        if debug_validation:
            logger.error("Schema type must be 'object', got %s", schema["type"])
        raise SchemaValidationError("Root schema must be of type 'object'")
    
    # Ensure schema has properties
    if "properties" not in schema:
        if debug_validation:
            logger.info("Schema missing properties field, using empty dict")
        schema["properties"] = {}
    
    if debug_validation:
        logger.info("Required fields: %s", schema.get("required", []))
        logger.info("Properties:")
        for prop_name, prop_schema in schema.get("properties", {}).items():
            logger.info("  %s:", prop_name)
            logger.info("    Type: %s", prop_schema.get("type"))
            logger.info("    Required: %s", prop_name in schema.get("required", []))
            logger.info("    Schema: %s", json.dumps(prop_schema, indent=2))
    
    if debug_validation:
        logger.info("Validated schema structure: %s", json.dumps(schema, indent=2))

    def _get_type_with_constraints(field_schema: dict) -> tuple:
        """Get Python type with Pydantic v2 constraints."""
        if debug_validation:
            logger.info("Processing field schema for type constraints:")
            logger.info(json.dumps(field_schema, indent=2))
        
        field_type = field_schema.get("type")
        if debug_validation:
            logger.info("Field type: %s", field_type)

        constraints = []
        field_args = {
            "description": field_schema.get("description"),
            "title": field_schema.get("title"),
            "json_schema_extra": {k: v for k, v in field_schema.items() 
                                if k not in {"type", "description", "title", "default", 
                                           "minimum", "maximum", "exclusiveMinimum", 
                                           "exclusiveMaximum", "multipleOf", "minLength",
                                           "maxLength", "pattern", "format", "enum"}}
        }
        
        if "default" in field_schema:
            field_args["default"] = field_schema["default"]
            if debug_validation:
                logger.info("Default value: %s", field_schema["default"])
        
        if field_type == "string":
            if debug_validation:
                logger.info("Processing string type constraints")
            base_type = str
            if "minLength" in field_schema:
                constraints.append(annotated_types.MinLen(field_schema["minLength"]))
                if debug_validation:
                    logger.info("Added minLength constraint: %d", field_schema["minLength"])
            if "maxLength" in field_schema:
                constraints.append(annotated_types.MaxLen(field_schema["maxLength"]))
                if debug_validation:
                    logger.info("Added maxLength constraint: %d", field_schema["maxLength"])
            if "pattern" in field_schema:
                constraints.append(annotated_types.Pattern(field_schema["pattern"]))
                if debug_validation:
                    logger.info("Added pattern constraint: %s", field_schema["pattern"])
            if "format" in field_schema:
                if debug_validation:
                    logger.info("Processing string format: %s", field_schema["format"])
                # Handle special string formats
                if field_schema["format"] == "date-time":
                    base_type = datetime
                elif field_schema["format"] == "date":
                    base_type = date
                elif field_schema["format"] == "time":
                    base_type = time
                elif field_schema["format"] == "email":
                    base_type = EmailStr
                elif field_schema["format"] == "uri":
                    base_type = AnyUrl
                if debug_validation:
                    logger.info("Using base type: %s", base_type.__name__)
        
        elif field_type in ("integer", "number"):
            if debug_validation:
                logger.info("Processing numeric type constraints")
            base_type = int if field_type == "integer" else float
            if "minimum" in field_schema:
                constraints.append(annotated_types.Ge(field_schema["minimum"]))
                if debug_validation:
                    logger.info("Added minimum constraint: %s", field_schema["minimum"])
            if "maximum" in field_schema:
                constraints.append(annotated_types.Le(field_schema["maximum"]))
                if debug_validation:
                    logger.info("Added maximum constraint: %s", field_schema["maximum"])
            if "exclusiveMinimum" in field_schema:
                constraints.append(annotated_types.Gt(field_schema["exclusiveMinimum"]))
                if debug_validation:
                    logger.info("Added exclusiveMinimum constraint: %s", field_schema["exclusiveMinimum"])
            if "exclusiveMaximum" in field_schema:
                constraints.append(annotated_types.Lt(field_schema["exclusiveMaximum"]))
                if debug_validation:
                    logger.info("Added exclusiveMaximum constraint: %s", field_schema["exclusiveMaximum"])
            if "multipleOf" in field_schema:
                constraints.append(annotated_types.MultipleOf(field_schema["multipleOf"]))
                if debug_validation:
                    logger.info("Added multipleOf constraint: %s", field_schema["multipleOf"])
        
        elif field_type == "boolean":
            if debug_validation:
                logger.info("Processing boolean type")
            base_type = bool
        
        elif field_type == "null":
            if debug_validation:
                logger.info("Processing null type")
            return (None, Field(**field_args))
        
        elif field_type == "array":
            if debug_validation:
                logger.info("Processing array type")
            items = field_schema.get("items", {})
            if items.get("type") == "object":
                if debug_validation:
                    logger.info("Array items are objects, creating nested model")
                # Create nested model for array items
                try:
                    item_model = create_dynamic_model(
                        items,
                        base_name=f"{base_name}Item",
                        show_schema=show_schema,
                        debug_validation=debug_validation
                    )
                    base_type = List[item_model]
                    if debug_validation:
                        logger.info("Created array item model: %s", item_model.__name__)
                except Exception as e:
                    if debug_validation:
                        logger.error("Failed to create array item model: %s", str(e))
                    return (Any, Field(**field_args))
            else:
                if debug_validation:
                    logger.info("Array items are primitive types")
                # Handle primitive array types
                item_type, _ = _get_type_with_constraints(items)
                base_type = List[item_type]
                if debug_validation:
                    logger.info("Using array type: List[%s]", getattr(item_type, "__name__", str(item_type)))
            
            # Add array constraints
            if "minItems" in field_schema:
                constraints.append(annotated_types.MinLen(field_schema["minItems"]))
                if debug_validation:
                    logger.info("Added minItems constraint: %d", field_schema["minItems"])
            if "maxItems" in field_schema:
                constraints.append(annotated_types.MaxLen(field_schema["maxItems"]))
                if debug_validation:
                    logger.info("Added maxItems constraint: %d", field_schema["maxItems"])
            if field_schema.get("uniqueItems", False):
                base_type = Set[item_type if 'item_type' in locals() else item_model]
                if debug_validation:
                    logger.info("Using Set type for uniqueItems constraint")
        
        elif field_type == "object":
            if debug_validation:
                logger.info("Processing object type")
            if "additionalProperties" in field_schema:
                if debug_validation:
                    logger.info("Object has additionalProperties")
                if isinstance(field_schema["additionalProperties"], dict):
                    value_type, _ = _get_type_with_constraints(field_schema["additionalProperties"])
                    base_type = Dict[str, value_type]
                    if debug_validation:
                        logger.info("Using Dict[str, %s] for additionalProperties", 
                                  getattr(value_type, "__name__", str(value_type)))
                else:
                    base_type = Dict[str, Any]
                    if debug_validation:
                        logger.info("Using Dict[str, Any] for additionalProperties")
            else:
                if debug_validation:
                    logger.info("Creating nested object model")
                try:
                    base_type = create_dynamic_model(
                        field_schema,
                        base_name=f"{base_name}Object",
                        show_schema=show_schema,
                        debug_validation=debug_validation
                    )
                    if debug_validation:
                        logger.info("Created nested object model: %s", base_type.__name__)
                except Exception as e:
                    if debug_validation:
                        logger.error("Failed to create object model: %s", str(e))
                    return (Any, Field(**field_args))
        
        else:
            if debug_validation:
                logger.info("Unknown type %s, using Any", field_type)
            return (Any, Field(**field_args))

        # Handle enum values if present
        if "enum" in field_schema:
            if debug_validation:
                logger.info("Processing enum values: %s", field_schema["enum"])
                logger.info("Current base_name: %s", base_name)
                logger.info("Field name: %s", field_name)
                logger.info("Field schema: %s", json.dumps(field_schema, indent=2))
                logger.info("Field args: %s", field_args)
                
            enum_name = f"{base_name}{field_name.title()}Enum"
            return _create_enum_type(
                enum_name=enum_name,
                values=field_schema["enum"],
                field_args=field_args,
                debug_validation=debug_validation
            )

        # Create field with constraints
        if constraints:
            if debug_validation:
                logger.info("Creating annotated type with %d constraints", len(constraints))
                for constraint in constraints:
                    logger.info("  Constraint: %s", constraint)
            return (Annotated[base_type, *constraints], Field(**field_args))
        
        if debug_validation:
            logger.info("Using base type without constraints: %s", 
                       getattr(base_type, "__name__", str(base_type)))
            logger.info("Field args: %s", field_args)
        return (base_type, Field(**field_args))

    def create_field_definition(field_schema: dict, field_name: str) -> tuple:
        """Create a field definition with proper typing and validation."""
        if debug_validation:
            logger.debug("Creating field definition for %s", field_name)
            logger.debug("Field schema: %s", json.dumps(field_schema, indent=2))
        
        try:
            field_type = field_schema.get("type")
            if not field_type:
                return (Any, Field(description=field_schema.get("description")))

            # Handle array types
            if field_type == "array":
                try:
                    items = field_schema.get("items", {})
                    if items.get("type") == "object":
                        try:
                            item_model = create_dynamic_model(
                                items,  # Pass full items schema
                                f"{base_name}{field_name.title()}Item",
                                show_schema,
                                debug_validation
                            )
                        except ModelCreationError as e:
                            raise NestedModelError(
                                f"{base_name}{field_name.title()}Item",
                                field_name,
                                str(e)
                            ) from e
                        
                        constraints = {}
                        if "minItems" in field_schema:
                            constraints["min_length"] = field_schema["minItems"]
                        if "maxItems" in field_schema:
                            constraints["max_length"] = field_schema["maxItems"]
                        if field_schema.get("uniqueItems", False):
                            return (
                                set[item_model],
                                Field(description=field_schema.get("description"), **constraints)
                            )
                        return (
                            List[item_model],
                            Field(description=field_schema.get("description"), **constraints)
                        )
                    else:
                        item_type, item_field = _get_type_with_constraints(items)
                        constraints = {}
                        if "minItems" in field_schema:
                            constraints["min_length"] = field_schema["minItems"]
                        if "maxItems" in field_schema:
                            constraints["max_length"] = field_schema["maxItems"]
                        return (
                            List[item_type],
                            Field(description=field_schema.get("description"), **constraints)
                        )
                except Exception as e:
                    raise FieldDefinitionError(
                        field_name,
                        "array",
                        f"Failed to create array field: {str(e)}"
                    ) from e
            
            # Handle object types
            elif field_type == "object":
                try:
                    if "additionalProperties" in field_schema:
                        if isinstance(field_schema["additionalProperties"], dict):
                            value_type, value_field = _get_type_with_constraints(field_schema["additionalProperties"])
                            return (
                                Dict[str, value_type],
                                Field(description=field_schema.get("description"))
                            )
                    nested_model = create_dynamic_model(
                        field_schema,  # Pass full field schema
                        f"{base_name}{field_name.title()}",
                        show_schema,
                        debug_validation
                    )
                    return (
                        nested_model,
                        Field(description=field_schema.get("description"))
                    )
                except ModelCreationError as e:
                    raise NestedModelError(
                        f"{base_name}{field_name.title()}",
                        field_name,
                        str(e)
                    ) from e
                except Exception as e:
                    raise FieldDefinitionError(
                        field_name,
                        "object",
                        f"Failed to create object field: {str(e)}"
                    ) from e
            
            # Handle basic types with constraints
            else:
                try:
                    python_type, field = _get_type_with_constraints(field_schema)
                    return (python_type, field)
                except Exception as e:
                    raise FieldDefinitionError(
                        field_name,
                        field_type,
                        f"Failed to create basic field: {str(e)}"
                    ) from e
                    
        except (FieldDefinitionError, NestedModelError):
            raise
        except Exception as e:
            if debug_validation:
                logger.error("Failed to create field definition for %s: %s", field_name, str(e))
            raise FieldDefinitionError(
                field_name,
                field_schema.get("type", "unknown"),
                f"Unexpected error: {str(e)}"
            ) from e

    # Create field definitions with proper naming
    if debug_validation:
        logger.info("Processing schema properties for %s", base_name)
        logger.info("Schema: %s", json.dumps(schema, indent=2))
        logger.info("Required fields: %s", schema.get("required", []))
        
    field_definitions = {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    validation_errors = []
    
    for field_name, field_schema in properties.items():
        try:
            if debug_validation:
                logger.info("Processing field %s:", field_name)
                logger.info("  Schema: %s", json.dumps(field_schema, indent=2))
                
            python_type, field = create_field_definition(field_schema, field_name)
            
            # Handle optional fields
            if field_name not in required:
                if debug_validation:
                    logger.info("Field %s is optional, wrapping in Optional", field_name)
                python_type = Optional[python_type]
            else:
                if debug_validation:
                    logger.info("Field %s is required", field_name)
            
            # Create field definition
            field_definitions[field_name] = (python_type, field)
            
            if debug_validation:
                logger.info("Successfully created field definition:")
                logger.info("  Name: %s", field_name)
                logger.info("  Type: %s", python_type)
                logger.info("  Required: %s", field_name in required)
                logger.info("  Field config:")
                logger.info("    Description: %s", field.description)
                logger.info("    Title: %s", field.title)
                logger.info("    JSON Schema Extra: %s", field.json_schema_extra)
                
        except (FieldDefinitionError, NestedModelError) as e:
            if debug_validation:
                logger.error("Error creating field %s:", field_name)
                logger.error("  Error type: %s", type(e).__name__)
                logger.error("  Error message: %s", str(e))
            validation_errors.append(str(e))
    
    if validation_errors:
        if debug_validation:
            logger.error("Model validation failed with %d errors:", len(validation_errors))
            for error in validation_errors:
                logger.error("  - %s", error)
        raise ModelValidationError(base_name, validation_errors)

    # Create model with configuration
    try:
        model_config = ConfigDict(
            title=schema.get("title", base_name),
            description=schema.get("description"),
            extra="forbid" if schema.get("additionalProperties") is False else "allow",
            strict=True,
            frozen=schema.get("readOnly", False),
            validate_default=True,
            use_enum_values=True,
            json_schema_mode='serialization',
            arbitrary_types_allowed=True,
            json_schema_extra={k: v for k, v in schema.items() 
                            if k not in {"type", "properties", "required", "title", 
                                       "description", "additionalProperties", "readOnly"}}
        )

        if debug_validation:
            logger.info("Creating model with configuration:")
            logger.info("  Title: %s", model_config.get("title"))
            logger.info("  Description: %s", model_config.get("description"))
            logger.info("  Extra: %s", model_config.get("extra"))
            logger.info("  Strict: %s", model_config.get("strict"))
            logger.info("  Frozen: %s", model_config.get("frozen"))
            logger.info("  Validate Default: %s", model_config.get("validate_default"))
            logger.info("  Use Enum Values: %s", model_config.get("use_enum_values"))
            logger.info("  JSON Schema Mode: %s", model_config.get("json_schema_mode"))
            logger.info("  Arbitrary Types Allowed: %s", model_config.get("arbitrary_types_allowed"))
            logger.info("  JSON Schema Extra: %s", model_config.get("json_schema_extra"))
            logger.info("Full configuration: %s", json.dumps(model_config, default=str))
            logger.info("Creating model with fields:")
            for name, (type_, field) in field_definitions.items():
                logger.info("  Field: %s", name)
                logger.info("    Type: %s", type_)
                logger.info("    Required: %s", name in required)
                logger.info("    Field config:")
                logger.info("      Description: %s", field.description)
                logger.info("      Title: %s", field.title)
                logger.info("      JSON Schema Extra: %s", field.json_schema_extra)

        model = create_model(
            base_name,
            __config__=model_config,
            **{name: (type_, field) for name, (type_, field) in field_definitions.items()}
        )

        if debug_validation:
            logger.info("Successfully created model: %s", model.__name__)
            logger.info("Model config: %s", dict(model.model_config))
            logger.info("Model schema: %s", json.dumps(model.model_json_schema(), indent=2))

        # Validate the model's JSON schema
        try:
            if debug_validation:
                logger.info("Validating model schema")
            schema = TypeAdapter(model).json_schema()
            if debug_validation:
                logger.info("Model schema validation successful")
                logger.info("Generated schema: %s", json.dumps(schema, indent=2))
            if show_schema or debug_validation:
                logger.info("Generated model schema for %s:", base_name)
                logger.info(json.dumps(schema, indent=2))
        except ValidationError as e:
            if debug_validation:
                logger.error("Schema validation failed:")
                logger.error("  Error type: %s", type(e).__name__)
                logger.error("  Error message: %s", str(e))
                if hasattr(e, 'errors'):
                    logger.error("  Validation errors:")
                    for error in e.errors():
                        logger.error("    - %s", error)
            raise ModelValidationError(base_name, [str(e)])
        except Exception as e:
            if debug_validation:
                logger.error("Unexpected error during schema validation:")
                logger.error("  Error type: %s", type(e).__name__)
                logger.error("  Error message: %s", str(e))
            raise ModelValidationError(base_name, [f"Schema validation failed: {str(e)}"])

        return model

    except Exception as e:
        if debug_validation:
            logger.error("Failed to create model:")
            logger.error("  Error type: %s", type(e).__name__)
            logger.error("  Error message: %s", str(e))
            if hasattr(e, '__cause__'):
                logger.error("  Caused by: %s", str(e.__cause__))
            if hasattr(e, '__context__'):
                logger.error("  Context: %s", str(e.__context__))
            if hasattr(e, '__traceback__'):
                import traceback
                logger.error("  Traceback:\n%s", ''.join(traceback.format_tb(e.__traceback__)))
        raise ModelCreationError(f"Failed to create model '{base_name}': {str(e)}")


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


def validate_schema_file(
    path: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Validate a JSON schema file.
    
    Args:
        path: Path to the schema file
        verbose: Whether to enable verbose logging
        
    Returns:
        The validated schema
        
    Raises:
        SchemaFileError: When file cannot be read
        InvalidJSONError: When file contains invalid JSON
        SchemaValidationError: When schema is invalid
    """
    if verbose:
        logger.info("Validating schema file: %s", path)
        
    try:
        with open(path) as f:
            schema = json.load(f)
    except FileNotFoundError:
        raise SchemaFileError(f"Schema file not found: {path}")
    except json.JSONDecodeError as e:
        raise InvalidJSONError(f"Invalid JSON in schema file: {e}")
    except Exception as e:
        raise SchemaFileError(f"Failed to read schema file: {e}")
        
    # Pre-validation structure checks
    if verbose:
        logger.info("Performing pre-validation structure checks")
        logger.debug("Loaded schema: %s", json.dumps(schema, indent=2))
        
    if not isinstance(schema, dict):
        if verbose:
            logger.error("Schema is not a dictionary: %s", type(schema).__name__)
        raise SchemaValidationError("Schema must be a JSON object")

    # Validate schema structure
    if "schema" in schema:
        if verbose:
            logger.debug("Found schema wrapper, validating inner schema")
        inner_schema = schema["schema"]
        if not isinstance(inner_schema, dict):
            if verbose:
                logger.error("Inner schema is not a dictionary: %s", type(inner_schema).__name__)
            raise SchemaValidationError("Inner schema must be a JSON object")
        if verbose:
            logger.debug("Inner schema validated successfully")
    else:
        if verbose:
            logger.debug("No schema wrapper found, using schema as-is")

    # Return the full schema including wrapper
    return schema


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


def _create_enum_type(enum_name: str, values: list[str], field_args: dict[str, Any], debug_validation: bool = False) -> tuple[Type[Enum], Field]:
    """Create a properly initialized string enum with version-aware implementation"""
    try:
        if debug_validation:
            logger.info("Creating enum type with name: %s", enum_name)
            logger.info("Values to process: %s", values)
        
        # Create member definitions with uppercase names
        member_defs = [(v.upper(), v) for v in values]
        
        # Version-aware enum creation
        if sys.version_info >= (3, 11):
            enum_type = StrEnum(enum_name, member_defs, module=__name__)
            if debug_validation:
                logger.info("Using Python 3.11+ native StrEnum")
        else:
            enum_type = Enum(enum_name, member_defs, module=__name__, type=str)
            if debug_validation:
                logger.info("Using str-mixed Enum for Python < 3.11")
        
        # Core validation assertions
        assert issubclass(enum_type, str), "Enum must inherit from str"
        assert issubclass(enum_type, Enum), "Must be proper Enum"
        
        if debug_validation:
            # Verify inheritance
            logger.info("Verifying enum inheritance:")
            logger.info("Is Enum: %s", issubclass(enum_type, Enum))
            logger.info("Is str: %s", issubclass(enum_type, str))
            
            # Verify member access
            test_value = values[0]
            test_member = getattr(enum_type, test_value.upper())
            logger.info("Member value test: %s == %s", test_member.value, test_value)
            
            # Verify instance creation
            test_instance = enum_type(test_value)
            logger.info("Instance creation test: %s", test_instance)
            logger.info("Instance value: %s", test_instance.value)
            
            # Version-specific checks
            if sys.version_info >= (3, 11):
                logger.info("StrEnum verification: %s", isinstance(test_instance, StrEnum))
            else:
                logger.info("str mixin verification: %s", isinstance(test_instance, str))
        
        return (enum_type, Field(**field_args))
        
    except Exception as e:
        if debug_validation:
            logger.error("Failed to create enum type: %s", str(e))
            logger.error("Exception type: %s", type(e).__name__)
            if hasattr(e, '__cause__'):
                logger.error("Caused by: %s", str(e.__cause__))
        raise


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Debug output options
    debug_group = parser.add_argument_group("Debug Output Options")
    debug_group.add_argument(
        "--show-model-schema",
        action="store_true",
        help="Display the generated Pydantic model schema",
    )
    debug_group.add_argument(
        "--debug-validation",
        action="store_true",
        help="Show detailed schema validation debugging information",
    )
    debug_group.add_argument(
        "--verbose-schema",
        action="store_true",
        help="Enable verbose schema debugging output",
    )
    debug_group.add_argument(
        "--progress-level",
        choices=["none", "basic", "detailed"],
        default="basic",
        help="Set the level of progress reporting (default: basic)",
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
    """Main CLI function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = os.path.expanduser("~/.ostruct/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "ostruct.log")
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )
    
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

    # Initialize template context
    template_context = None
    
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

    # Ensure template_context was created successfully
    if template_context is None:
        logger.error("Failed to create template context")
        return ExitCode.SECURITY_ERROR

    # Load and validate schema
    try:
        logger.debug("[_main] Loading schema from %s", args.schema_file)
        schema = validate_schema_file(args.schema_file, verbose=args.verbose_schema)
        logger.debug("[_main] Creating output model")
        output_model = create_dynamic_model(
            schema,
            base_name="OutputModel",
            show_schema=args.show_model_schema,
            debug_validation=args.debug_validation
        )
        logger.debug("[_main] Successfully created output model")
    except (SchemaFileError, InvalidJSONError, SchemaValidationError) as e:
        logger.error(str(e))
        return ExitCode.SCHEMA_ERROR
    except ModelCreationError as e:
        logger.error(f"Model creation error: {e}")
        return ExitCode.SCHEMA_ERROR
    except Exception as e:
        logger.error(f"Unexpected error creating model: {e}")
        return ExitCode.SCHEMA_ERROR

    # Load and validate task template
    try:
        task_template = validate_task_template(args.task)
    except TaskTemplateVariableError as e:
        logger.error(str(e))
        return ExitCode.DATA_ERROR
    except TaskTemplateSyntaxError as e:
        logger.error(str(e))
        return ExitCode.DATA_ERROR
    except FileNotFoundError as e:
        logger.error(str(e))
        return ExitCode.FILE_NOT_FOUND
    except PathSecurityError as e:
        logger.error(str(e))
        return ExitCode.SECURITY_ERROR

    # Create Jinja environment and render template
    try:
        env = create_jinja_env()
        user_message = render_template(
            task_template,
            template_context,
            env,
            progress_enabled=args.progress_level != "none",
        )
    except ValueError as e:
        logger.error(str(e))
        return ExitCode.DATA_ERROR

    # Validate model support
    try:
        supports_structured_output(args.model)
    except ModelNotSupportedError as e:
        logger.error(str(e))
        return ExitCode.DATA_ERROR
    except ModelVersionError as e:
        logger.error(str(e))
        return ExitCode.DATA_ERROR

    # Estimate token usage
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": user_message},
    ]
    total_tokens = estimate_tokens_for_chat(messages, args.model)
    context_limit = get_context_window_limit(args.model)

    if total_tokens > context_limit:
        logger.error(
            f"Total tokens ({total_tokens}) exceeds model context limit ({context_limit})"
        )
        return ExitCode.DATA_ERROR

    # Handle dry run mode
    if args.dry_run:
        logger.info("*** DRY RUN MODE - No API call will be made ***\n")
        logger.info("System Prompt:\n%s\n", args.system_prompt)
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
        with ProgressContext(
            description="Processing API response",
            level=args.progress_level,
        ) as progress:
            async for chunk in async_openai_structured_stream(
                client=client,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                system_prompt=args.system_prompt,
                user_prompt=user_message,
                output_schema=output_model,
                timeout=args.timeout
            ):
                # Debug logging
                logger.debug("Received chunk: %s", chunk)
                logger.debug("Chunk type: %s", type(chunk))
                
                # Write output
                dumped = chunk.model_dump(mode='json')
                logger.debug("Dumped chunk: %s", dumped)
                
                if args.output_file:
                    with open(args.output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(dumped, indent=2))
                        f.write("\n")
                else:
                    progress.print_output(json.dumps(dumped, indent=2))
                
                progress.update()

    except StreamInterruptedError as e:
        logger.error(f"Stream interrupted: {e}")
        return ExitCode.API_ERROR
    except StreamBufferError as e:
        logger.error(f"Stream buffer error: {e}")
        return ExitCode.API_ERROR
    except StreamParseError as e:
        logger.error(f"Stream parse error: {e}")
        return ExitCode.API_ERROR
    except APIResponseError as e:
        logger.error(f"API response error: {e}")
        return ExitCode.API_ERROR
    except EmptyResponseError as e:
        logger.error(f"Empty response error: {e}")
        return ExitCode.API_ERROR
    except InvalidResponseFormatError as e:
        logger.error(f"Invalid response format: {e}")
        return ExitCode.API_ERROR
    except (APIConnectionError, InternalServerError) as e:
        logger.error(f"API connection error: {e}")
        return ExitCode.API_ERROR
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        return ExitCode.API_ERROR
    except BadRequestError as e:
        logger.error(f"Bad request: {e}")
        return ExitCode.API_ERROR
    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        return ExitCode.API_ERROR
    except OpenAIClientError as e:
        logger.error(f"OpenAI client error: {e}")
        return ExitCode.API_ERROR
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return ExitCode.INTERNAL_ERROR

    return ExitCode.SUCCESS


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
        sys.exit(ExitCode.INTERNAL_ERROR.value)


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
