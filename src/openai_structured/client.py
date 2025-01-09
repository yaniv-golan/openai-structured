# src/openai_structured/client.py
import json
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Type

from openai import (APIConnectionError, APIError, AuthenticationError,
                    BadRequestError, InternalServerError, OpenAI,
                    RateLimitError)
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Constants
OPENAI_API_SUPPORTED_MODELS = {
    "gpt-4o": "2024-08-06",
    "gpt-4o-mini": "2024-07-18",
    "o1": "2024-12-17",
}
DEFAULT_TEMPERATURE = 0.2

# Import custom exceptions from errors.py
from .errors import (APIResponseError, EmptyResponseError,
                     InvalidResponseFormatError, ModelNotSupportedError,
                     OpenAIClientError)


# Helper Functions
def _is_model_supported(model_name: str) -> bool:
    for base_model, min_version in OPENAI_API_SUPPORTED_MODELS.items():
        if base_model in model_name:
            try:
                version = model_name.split(f"{base_model}-")[-1]
                return version >= min_version
            except IndexError:
                return False
    return False


def _create_chat_messages(
    system_prompt: str, user_prompt: str
) -> List[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _log_request(logger_opt: Optional[logging.Logger], **kwargs: Any) -> None:
    if logger_opt:
        logger_opt.debug(
            "OpenAI API Request",
            extra={
                "payload": {
                    "component": "openai",
                    "operation": "structured_call",
                    **kwargs,
                }
            },
        )


def _log_response(logger_opt: Optional[logging.Logger], **kwargs: Any) -> None:
    if logger_opt:
        logger_opt.debug(
            "OpenAI API Response",
            extra={
                "payload": {
                    "component": "openai",
                    "operation": "structured_call",
                    **kwargs,
                }
            },
        )


def _log_error(
    logger_opt: Optional[logging.Logger], message: str, error: Exception, **kwargs: Any
) -> None:
    if logger_opt:
        logger_opt.error(
            message,
            exc_info=True,
            extra={
                "payload": {
                    "component": "openai",
                    "operation": "structured_call",
                    "error": str(error),
                    **kwargs,
                }
            },
        )


def _parse_json_response(
    content: Optional[str], output_schema: Type[BaseModel]
) -> BaseModel:
    if not content:
        raise EmptyResponseError("OpenAI API returned empty response")
    try:
        return output_schema.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError) as e:
        raise InvalidResponseFormatError(
            f"Invalid response format from OpenAI API: {e}"
        ) from e


def openai_structured_call(
    client: OpenAI,
    model: str,
    output_schema: Type[BaseModel],
    user_prompt: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> BaseModel:
    """
    Effortlessly call the OpenAI API for structured output.

    Args:
        client: Initialized OpenAI client.
        model: OpenAI model name (e.g., "gpt-4o-2024-08-06").
        output_schema: Pydantic model defining the expected output structure.
        user_prompt: The user's request.
        system_prompt: System instructions for the model.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        top_p: Top-p sampling.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        logger: Optional custom logger.

    Returns:
        An instance of the `output_schema` Pydantic model.

    Raises:
        ModelNotSupportedError: If the model doesn't support structured output.
        APIResponseError: For errors from the OpenAI API.
        OpenAIClientError: For general client-related errors.
    """

    if not _is_model_supported(model):
        raise ModelNotSupportedError(
            f"Model '{model}' does not support structured output."
        )

    messages = _create_chat_messages(system_prompt, user_prompt)

    try:
        _log_request(
            logger,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            output_schema=output_schema.__name__,
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        _log_response(
            logger,
            response_id=response.id,
            model=response.model,
            usage=response.usage.dict() if response.usage else None,
        )

        return _parse_json_response(response.choices[0].message.content, output_schema)

    except RateLimitError as e:
        _log_error(logger, "OpenAI API rate limit exceeded.", e)
        raise APIResponseError(
            f"OpenAI API rate limit exceeded.", response_id=getattr(e, "response", None)
        ) from e
    except (
        AuthenticationError,
        BadRequestError,
        APIConnectionError,
        InternalServerError,
    ) as e:
        _log_error(logger, f"OpenAI API error.", e)
        raise APIResponseError(
            f"OpenAI API error: {e}", response_id=getattr(e, "response", None)
        ) from e
    except Exception as e:
        _log_error(logger, f"An unexpected error occurred.", e)
        raise OpenAIClientError(f"Unexpected error during OpenAI API call: {e}") from e


async def openai_structured_stream(
    client: OpenAI,
    model: str,
    output_schema: Type[BaseModel],
    user_prompt: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> AsyncIterable[BaseModel]:
    """
    Asynchronously stream structured output from the OpenAI API.

    Args:
        client: Initialized OpenAI client.
        model: OpenAI model name (e.g., "gpt-4o-2024-08-06").
        output_schema: Pydantic model defining the expected output structure.
        user_prompt: The user's request.
        system_prompt: System instructions for the model.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        top_p: Top-p sampling.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        logger: Optional custom logger.

    Yields:
        Instances of the `output_schema` Pydantic model as they become available.

    Raises:
        ModelNotSupportedError: If the model doesn't support structured output.
        APIResponseError: For errors from the OpenAI API.
        OpenAIClientError: For general client-related errors.
    """

    if not _is_model_supported(model):
        raise ModelNotSupportedError(
            f"Model '{model}' does not support structured output."
        )

    messages = _create_chat_messages(system_prompt, user_prompt)
    collected_content = ""

    _log_request(
        logger,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        streaming=True,
        output_schema=output_schema.__name__,
    )

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
                collected_content += delta
                # Attempt to parse and yield if valid JSON is formed
                try:
                    model_instance = output_schema.model_validate_json(
                        collected_content
                    )
                    yield model_instance
                    collected_content = ""  # Reset after yielding
                except (ValidationError, json.JSONDecodeError):
                    pass  # Continue accumulating content

    except (
        AuthenticationError,
        BadRequestError,
        APIConnectionError,
        InternalServerError,
    ) as e:
        _log_error(logger, f"OpenAI API error during streaming.", e)
        raise APIResponseError(
            f"OpenAI API error during streaming: {e}",
            response_id=getattr(e, "response", None),
        ) from e
    except Exception as e:
        _log_error(logger, f"An unexpected error occurred during streaming.", e)
        raise OpenAIClientError(
            f"Unexpected error during OpenAI API streaming call: {e}"
        ) from e
