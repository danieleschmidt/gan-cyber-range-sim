"""Timeout management utilities."""

import asyncio
import time
from typing import Callable, Any, TypeVar, Optional
from functools import wraps

T = TypeVar('T')


class TimeoutError(Exception):
    """Custom timeout error."""
    pass


class TimeoutManager:
    """Manages timeout operations."""
    
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
    
    async def with_timeout(self, coro, timeout: Optional[float] = None):
        """Execute coroutine with timeout."""
        timeout = timeout or self.default_timeout
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")


def with_timeout(timeout: float):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
        return wrapper
    return decorator