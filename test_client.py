import json
from openai import OpenAI
from openai_structured.client import openai_structured_stream
from pydantic import BaseModel, create_model
from typing import Type, Any, List, Dict

def read_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

def create_model_from_json_schema(schema_name: str, schema_json: dict) -> Type[BaseModel]:
    """Create a Pydantic model from JSON schema with nested object support."""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": List,
        "object": Dict
    }

    def create_field_definition(field_schema: dict) -> tuple:
        field_type = field_schema.get("type")
        if not field_type:
            return (Any, ...)

        if field_type == "array":
            items = field_schema.get("items", {})
            if items.get("type") == "object":
                # Create a nested model for array items
                nested_model = create_model_from_json_schema(
                    f"{schema_name}Item",
                    {"properties": items.get("properties", {})}
                )
                return (List[nested_model], ...)
            else:
                item_type = type_mapping.get(items.get("type", "string"), Any)
                return (List[item_type], ...)
        elif field_type == "object":
            # Create a nested model for object properties
            nested_model = create_model_from_json_schema(
                f"{schema_name}Nested",
                {"properties": field_schema.get("properties", {})}
            )
            return (nested_model, ...)
        else:
            return (type_mapping.get(field_type, Any), ...)

    field_definitions = {}
    properties = schema_json.get("properties", {})
    for field_name, field_schema in properties.items():
        field_definitions[field_name] = create_field_definition(field_schema)

    return create_model(schema_name, **field_definitions)

def create_model_from_schema(schema: dict) -> Type[BaseModel]:
    """Create a Pydantic model from our schema format."""
    if 'schema' not in schema:
        raise ValueError("Schema missing 'schema' key")
    
    return create_model_from_json_schema("ReviewModel", schema['schema'])

def main():
    # Initialize OpenAI client
    client = OpenAI()

    # Read the system prompt, schema and demo file
    system_prompt = read_file('examples/use_cases/01_code_review/system_prompt.txt')
    schema = json.loads(read_file('examples/use_cases/01_code_review/schema.json'))
    demo_file = read_file('examples/use_cases/01_code_review/sample_input/demo.py')

    # Create the review model from schema
    try:
        ReviewModel = create_model_from_schema(schema)
        print("Created model:", ReviewModel)
        print("Model schema:", ReviewModel.model_json_schema())
    except Exception as e:
        print(f"Failed to create model from schema: {type(e).__name__}: {str(e)}")
        return

    # Create the same user prompt as seen in the logs
    user_prompt = '''Analyze the following code files for potential issues. For each file, provide a detailed review focusing on security, code quality, performance, and documentation.

For each file, identify:
1. Any security vulnerabilities or risks
2. Code quality issues and potential improvements
3. Performance concerns and optimization opportunities
4. Documentation and maintainability issues

Provide your review in a structured format with:
- A list of specific issues found in each file
- The severity and category of each issue
- Line numbers where applicable
- Clear recommendations for fixing each issue
- A summary assessment for each file
- An overall summary of the entire codebase

Files to review:

## File: examples/use_cases/01_code_review/sample_input/demo.py

{}'''.format(demo_file)

    try:
        # Use the openai_structured_stream function from client.py
        stream = openai_structured_stream(
            client=client,
            model="gpt-4o-2024-08-06",
            output_schema=ReviewModel,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0
        )

        # Process the stream
        for chunk in stream:
            print("Received chunk:", chunk)

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Caused by: {type(e.__cause__).__name__}: {str(e.__cause__)}")

if __name__ == "__main__":
    main() 