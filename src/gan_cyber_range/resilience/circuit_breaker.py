"""Circuit breaker pattern implementation for fault tolerance."""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime, timedelta

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5           # Failures to trip breaker
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    success_threshold: int = 3           # Successes to close from half-open
    timeout: float = 30.0               # Request timeout
    expected_exception: type = Exception  # Exception type to count as failure


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker(Generic[T]):
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
        
        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_request_time: Optional[float] = None
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_rejected = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        async with self._lock:
            self.total_requests += 1
            self.last_request_time = time.time()
            
            # Check if circuit should transition states
            await self._check_state_transition()
            
            # If circuit is open, reject immediately
            if self.state == CircuitBreakerState.OPEN:
                self.total_rejected += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Last failure: {self.last_failure_time}"
                )
        
        # Execute the function with timeout
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=self.config.timeout
                )
            else:
                # Run sync function in executor with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=self.config.timeout
                )
            
            await self._on_success()
            return result
            
        except asyncio.TimeoutError:
            self.total_timeouts += 1
            await self._on_failure("timeout")
            raise
            
        except self.config.expected_exception as e:
            await self._on_failure(str(e))
            raise
            
        except Exception as e:
            # Unexpected exceptions don't count as failures
            self.logger.warning(f"Unexpected exception in circuit breaker '{self.name}': {e}")
            raise
    
    async def _check_state_transition(self) -> None:
        """Check if circuit breaker should change state."""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if we should try half-open
            if (self.last_failure_time and 
                current_time - self.last_failure_time >= self.config.recovery_timeout):
                self.logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Check if we should close (enough successes)
            if self.success_count >= self.config.success_threshold:
                self.logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
    
    async def _on_success(self) -> None:
        """Handle successful execution."""
        async with self._lock:
            self.total_successes += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                self.logger.debug(
                    f"Circuit breaker '{self.name}' success in HALF_OPEN: "
                    f"{self.success_count}/{self.config.success_threshold}"
                )
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    async def _on_failure(self, error: str) -> None:
        """Handle failed execution."""
        async with self._lock:
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            self.logger.warning(
                f"Circuit breaker '{self.name}' failure #{self.failure_count}: {error}"
            )
            
            # Check if we should open the circuit
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                
                self.logger.error(
                    f"Circuit breaker '{self.name}' transitioning to OPEN after "
                    f"{self.failure_count} failures"
                )
                self.state = CircuitBreakerState.OPEN
                
            elif (self.state == CircuitBreakerState.HALF_OPEN):
                # Any failure in half-open goes back to open
                self.logger.warning(
                    f"Circuit breaker '{self.name}' back to OPEN from HALF_OPEN"
                )
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        current_time = time.time()
        uptime = (current_time - self.last_request_time) if self.last_request_time else 0
        
        success_rate = (
            self.total_successes / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_timeouts": self.total_timeouts,
            "total_rejected": self.total_rejected,
            "success_rate": success_rate,
            "last_failure_time": self.last_failure_time,
            "last_request_time": self.last_request_time,
            "uptime_seconds": uptime,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.logger.info(f"Manually resetting circuit breaker '{self.name}'")
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
    
    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        self.logger.warning(f"Manually opening circuit breaker '{self.name}'")
        self.state = CircuitBreakerState.OPEN
        self.last_failure_time = time.time()


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger("CircuitBreakerRegistry")
    
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
            self.logger.info(f"Created circuit breaker: {name}")
        
        return self.breakers[name]
    
    def remove_breaker(self, name: str) -> None:
        """Remove a circuit breaker."""
        if name in self.breakers:
            del self.breakers[name]
            self.logger.info(f"Removed circuit breaker: {name}")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
        self.logger.info("Reset all circuit breakers")


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def with_circuit_breaker(
    name: str, 
    config: CircuitBreakerConfig = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to wrap function with circuit breaker."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker = circuit_breaker_registry.get_breaker(name, config)
        
        async def async_wrapper(*args, **kwargs) -> T:
            return await breaker.call(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs) -> T:
            # Convert sync function to async for circuit breaker
            async def async_func():
                return func(*args, **kwargs)
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a task
                return loop.run_until_complete(breaker.call(async_func))
            else:
                # Create new event loop
                return asyncio.run(breaker.call(async_func))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator