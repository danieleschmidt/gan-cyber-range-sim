"""Rate limiting utilities."""

import asyncio
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int
    tokens: float
    fill_rate: float
    last_update: float
    
    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.fill_rate = fill_rate
        self.last_update = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_update = now


class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, rate: float, burst: Optional[int] = None):
        self.bucket = TokenBucket(burst or int(rate), rate)
    
    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary."""
        while not self.bucket.consume(tokens):
            await asyncio.sleep(0.1)  # Wait a bit before retrying