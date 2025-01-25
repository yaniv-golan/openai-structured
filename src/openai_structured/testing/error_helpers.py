"""Error helpers for testing.

This module provides utilities for simulating various error conditions in tests.
"""

from typing import Optional, Type, Union
from unittest.mock import MagicMock

class MockAPIError(Exception):
    """Mock API error with status code and message."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code

def create_error_response(
    error: Union[str, Exception, Type[Exception]],
    status_code: Optional[int] = None,
    delay: Optional[float] = None
) -> MagicMock:
    """Create a mock that raises an error when accessed.
    
    Args:
        error: String message, exception instance, or exception class
        status_code: Optional HTTP status code
        delay: Optional delay before error occurs (in seconds)
        
    Returns:
        MagicMock configured to raise the specified error
    """
    if isinstance(error, str):
        exc = MockAPIError(error, status_code)
    elif isinstance(error, type) and issubclass(error, Exception):
        exc = error("Mock API Error")
    else:
        exc = error
    
    mock = MagicMock()
    mock.side_effect = exc
    return mock

def create_rate_limit_response(
    max_requests: int = 3,
    reset_after: float = 60
) -> MagicMock:
    """Create a mock that simulates rate limiting.
    
    Args:
        max_requests: Number of requests before rate limit
        reset_after: Time until rate limit resets (seconds)
        
    Returns:
        MagicMock that raises rate limit error after max_requests
    """
    mock = MagicMock()
    call_count = 0
    
    def rate_limited(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > max_requests:
            raise MockAPIError(
                f"Rate limit exceeded. Reset in {reset_after} seconds",
                status_code=429
            )
        return MagicMock(choices=[MagicMock(message=MagicMock(content="test"))])
    
    mock.side_effect = rate_limited
    return mock

def create_timeout_response(
    timeout_after: float = 5.0
) -> MagicMock:
    """Create a mock that simulates network timeouts.
    
    Args:
        timeout_after: Time until timeout occurs (seconds)
        
    Returns:
        MagicMock that raises timeout error after specified duration
    """
    mock = MagicMock()
    mock.side_effect = TimeoutError(f"Request timed out after {timeout_after} seconds")
    return mock 