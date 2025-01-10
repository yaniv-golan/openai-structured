"""Client for making structured OpenAI API calls."""

import json
from typing import AsyncGenerator, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .errors import (
    APIResponseError,
    EmptyResponseError,
    InvalidResponseFormatError,
    ModelNotSupportedError,
)

T = TypeVar("T", bound=BaseModel)

SUPPORTED_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo-preview",
]

DEFAULT_SYSTEM_PROMPT = """
Extract structured information from the user's input.
Respond with valid JSON that matches the specified schema.
Do not include any other text in your response.
"""


def openai_structured_call(
    client: OpenAI,
    model: str,
    output_schema: Type[T],
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> T:
    """Make a structured OpenAI API call.

    Args:
        client: OpenAI client instance
        model: Model to use (e.g., "gpt-4")
        output_schema: Pydantic model class for output structure
        user_prompt: User input to process
        system_prompt: Optional system prompt (default: schema-focused prompt)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: None)

    Returns:
        Instance of output_schema containing structured response

    Raises:
        ModelNotSupportedError: If model is not supported
        APIResponseError: If API call fails
        InvalidResponseFormatError: If response is not valid JSON
        EmptyResponseError: If response is empty
    """
    if model not in SUPPORTED_MODELS:
        raise ModelNotSupportedError(
            f"Model {model} not supported. Use one of: {SUPPORTED_MODELS}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                system_prompt
                or DEFAULT_SYSTEM_PROMPT
                + f"\nSchema: {output_schema.model_json_schema()}"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise APIResponseError(f"API call failed: {str(e)}") from e

    if not response.choices:
        raise EmptyResponseError("No choices in response")

    try:
        content = response.choices[0].message.content
        if not content:
            raise EmptyResponseError("Empty response content")

        data = json.loads(content)
        return output_schema.model_validate(data)
    except json.JSONDecodeError as e:
        raise InvalidResponseFormatError(
            f"Invalid JSON in response: {str(e)}"
        ) from e
    except Exception as e:
        raise InvalidResponseFormatError(
            f"Failed to parse response: {str(e)}"
        ) from e


async def openai_structured_stream(
    client: OpenAI,
    model: str,
    output_schema: Type[T],
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[T, None]:
    """Stream structured responses from OpenAI API.

    Args:
        client: OpenAI client instance
        model: Model to use (e.g., "gpt-4")
        output_schema: Pydantic model class for output structure
        user_prompt: User input to process
        system_prompt: Optional system prompt (default: schema-focused)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: None)

    Yields:
        Instances of output_schema containing structured responses

    Raises:
        ModelNotSupportedError: If model is not supported
        APIResponseError: If API call fails
        InvalidResponseFormatError: If response is not valid JSON
    """
    if model not in SUPPORTED_MODELS:
        raise ModelNotSupportedError(
            f"Model {model} not supported. Use one of: {SUPPORTED_MODELS}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                system_prompt
                or DEFAULT_SYSTEM_PROMPT
                + f"\nSchema: {output_schema.model_json_schema()}"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            stream=True,
        )
    except Exception as e:
        raise APIResponseError(f"API call failed: {str(e)}") from e

    buffer = ""
    async for chunk in stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if not content:
            continue

        buffer += content
        try:
            data = json.loads(buffer)
            yield output_schema.model_validate(data)
            buffer = ""
        except json.JSONDecodeError:
            continue
        except Exception as e:
            raise InvalidResponseFormatError(
                f"Failed to parse response: {str(e)}"
            ) from e
