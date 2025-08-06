"""Retry patterns with exponential backoff and jitter."""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Optional, Type, Union, TypeVar, List
from dataclasses import dataclass
from functools import wraps

T = TypeVar('T')


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retriable_exceptions: tuple = (Exception,)
    non_retriable_exceptions: tuple = ()


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts. Last error: {last_exception}")


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """Calculate exponential backoff delay with optional jitter."""
    delay = base_delay * (exponential_base ** (attempt - 1))
    delay = min(delay, max_delay)
    
    if jitter:
        # Add jitter to avoid thundering herd
        jitter_range = delay * 0.1
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)  # Ensure non-negative
    
    return delay


async def async_retry(
    func: Callable[..., T],
    policy: RetryPolicy,
    *args,
    **kwargs
) -> T:
    """Execute async function with retry logic."""
    logger = logging.getLogger(f"retry.{func.__name__}")
    last_exception = None
    
    for attempt in range(1, policy.max_attempts + 1):
        try:
            if attempt > 1:
                delay = exponential_backoff(
                    attempt - 1,
                    policy.base_delay,
                    policy.exponential_base,
                    policy.max_delay,
                    policy.jitter
                )
                logger.debug(f"Retry attempt {attempt} after {delay:.2f}s delay")
                await asyncio.sleep(delay)
            
            result = await func(*args, **kwargs)
            
            if attempt > 1:
                logger.info(f"Success on retry attempt {attempt}")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Check if exception is non-retriable
            if any(isinstance(e, exc_type) for exc_type in policy.non_retriable_exceptions):
                logger.warning(f"Non-retriable exception: {e}")
                raise e
            
            # Check if exception is retriable
            if not any(isinstance(e, exc_type) for exc_type in policy.retriable_exceptions):
                logger.warning(f"Non-retriable exception type: {e}")
                raise e
            
            if attempt < policy.max_attempts:
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying...")
            else:
                logger.error(f"All {policy.max_attempts} attempts failed")
    
    raise RetryExhaustedError(policy.max_attempts, last_exception)


def sync_retry(
    func: Callable[..., T],
    policy: RetryPolicy,
    *args,
    **kwargs
) -> T:
    """Execute sync function with retry logic."""
    logger = logging.getLogger(f"retry.{func.__name__}")
    last_exception = None
    
    for attempt in range(1, policy.max_attempts + 1):
        try:
            if attempt > 1:
                delay = exponential_backoff(
                    attempt - 1,
                    policy.base_delay,
                    policy.exponential_base,
                    policy.max_delay,
                    policy.jitter
                )
                logger.debug(f"Retry attempt {attempt} after {delay:.2f}s delay")
                time.sleep(delay)
            
            result = func(*args, **kwargs)
            
            if attempt > 1:
                logger.info(f"Success on retry attempt {attempt}")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Check if exception is non-retriable
            if any(isinstance(e, exc_type) for exc_type in policy.non_retriable_exceptions):
                logger.warning(f"Non-retriable exception: {e}")
                raise e
            
            # Check if exception is retriable
            if not any(isinstance(e, exc_type) for exc_type in policy.retriable_exceptions):
                logger.warning(f"Non-retriable exception type: {e}")
                raise e
            
            if attempt < policy.max_attempts:
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying...")
            else:
                logger.error(f"All {policy.max_attempts} attempts failed")
    
    raise RetryExhaustedError(policy.max_attempts, last_exception)


def with_retry(policy: RetryPolicy = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic to a function."""
    if policy is None:
        policy = RetryPolicy()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                return await async_retry(func, policy, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                return sync_retry(func, policy, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# Predefined retry policies for common scenarios

# Network operations - retry on timeouts and connection errors
NETWORK_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    retriable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,  # Network-related OS errors
    ),
    non_retriable_exceptions=(
        ValueError,
        TypeError,
        KeyError,
    )
)

# LLM API calls - retry on rate limits and temporary failures
LLM_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=1.5,
    retriable_exceptions=(
        ConnectionError,
        TimeoutError,
        # Add specific LLM API exceptions here
    )
)

# Database operations - retry on temporary connection issues
DATABASE_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    retriable_exceptions=(
        ConnectionError,
        TimeoutError,
        # Add database-specific exceptions
    ),
    non_retriable_exceptions=(
        ValueError,
        TypeError,
        # Add constraint violation exceptions
    )
)

# File operations - retry on temporary I/O errors
FILE_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=0.1,
    max_delay=5.0,
    retriable_exceptions=(
        IOError,
        OSError,
        PermissionError,  # Might be temporary
    ),
    non_retriable_exceptions=(
        FileNotFoundError,  # Usually permanent
        IsADirectoryError,  # Usually permanent
    )
)


class RetryMetrics:
    """Collect retry metrics for monitoring."""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.retry_counts: List[int] = []
        self.exception_counts: dict = {}
    
    def record_attempt(self, attempt: int, success: bool, exception: Optional[Exception] = None):
        """Record a retry attempt."""
        self.total_attempts += attempt
        
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        self.retry_counts.append(attempt)
        
        if exception:
            exc_type = type(exception).__name__
            self.exception_counts[exc_type] = self.exception_counts.get(exc_type, 0) + 1
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        total_operations = self.successful_operations + self.failed_operations
        
        if total_operations == 0:
            return {
                "total_operations": 0,
                "success_rate": 0.0,
                "average_attempts": 0.0,
                "max_attempts": 0,
                "exception_counts": {}
            }
        
        return {
            "total_operations": total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.successful_operations / total_operations,
            "average_attempts": sum(self.retry_counts) / len(self.retry_counts),
            "max_attempts": max(self.retry_counts),
            "total_attempts": self.total_attempts,
            "exception_counts": self.exception_counts.copy()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.total_attempts = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.retry_counts.clear()
        self.exception_counts.clear()


# Global metrics instance
retry_metrics = RetryMetrics()


def with_retry_metrics(policy: RetryPolicy = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that adds retry logic with metrics collection."""
    if policy is None:
        policy = RetryPolicy()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                attempt_count = 0
                try:
                    result = await async_retry(func, policy, *args, **kwargs)
                    # Count successful attempts (this is approximate)
                    attempt_count = 1  # Minimum one attempt
                    retry_metrics.record_attempt(attempt_count, True)
                    return result
                except RetryExhaustedError as e:
                    retry_metrics.record_attempt(e.attempts, False, e.last_exception)
                    raise
                except Exception as e:
                    retry_metrics.record_attempt(1, False, e)
                    raise
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                attempt_count = 0
                try:
                    result = sync_retry(func, policy, *args, **kwargs)
                    attempt_count = 1  # Minimum one attempt
                    retry_metrics.record_attempt(attempt_count, True)
                    return result
                except RetryExhaustedError as e:
                    retry_metrics.record_attempt(e.attempts, False, e.last_exception)
                    raise
                except Exception as e:
                    retry_metrics.record_attempt(1, False, e)
                    raise
            
            return sync_wrapper
    
    return decorator