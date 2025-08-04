"""Caching system for performance optimization."""

import asyncio
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import hashlib


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    hit_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def touch(self) -> None:
        """Update last accessed time and increment hit count."""
        self.last_accessed = datetime.now()
        self.hit_count += 1


class CacheBackend(ABC):
    """Abstract cache backend."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            entry.touch()
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        async with self._lock:
            # Calculate expiration
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
            
            # Estimate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = len(str(value))
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            # Evict if necessary
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()
            
            self._cache[key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        import fnmatch
        async with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at
        )
        del self._cache[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.size_bytes for entry in self._cache.values())
        total_hits = sum(entry.hit_count for entry in self._cache.values())
        
        return {
            "entries": len(self._cache),
            "max_size": self.max_size,
            "total_size_bytes": total_size,
            "total_hits": total_hits,
            "hit_rate": total_hits / max(len(self._cache), 1)
        }


class RedisCache(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._redis = None
    
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import aioredis
                self._redis = aioredis.from_url(self.redis_url)
            except ImportError:
                raise ImportError("aioredis is required for Redis cache backend")
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        redis = await self._get_redis()
        try:
            data = await redis.get(key)
            if data is None:
                return None
            
            # Deserialize
            try:
                return pickle.loads(data)
            except:
                return json.loads(data.decode())
        except Exception:
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        redis = await self._get_redis()
        try:
            # Serialize
            try:
                data = pickle.dumps(value)
            except:
                data = json.dumps(value).encode()
            
            # Set with TTL
            ttl = ttl or self.default_ttl
            if ttl > 0:
                await redis.setex(key, ttl, data)
            else:
                await redis.set(key, data)
            
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        redis = await self._get_redis()
        try:
            result = await redis.delete(key)
            return result > 0
        except Exception:
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        redis = await self._get_redis()
        try:
            await redis.flushdb()
            return True
        except Exception:
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        redis = await self._get_redis()
        try:
            keys = await redis.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception:
            return []


class CacheManager:
    """High-level cache manager with multiple backends and strategies."""
    
    def __init__(
        self,
        primary_backend: CacheBackend,
        secondary_backend: Optional[CacheBackend] = None,
        enable_metrics: bool = True
    ):
        self.primary_backend = primary_backend
        self.secondary_backend = secondary_backend
        self.enable_metrics = enable_metrics
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.total_get_time = 0.0
        self.total_set_time = 0.0
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with fallback."""
        start_time = time.time()
        
        try:
            # Try primary backend
            value = await self.primary_backend.get(key)
            if value is not None:
                if self.enable_metrics:
                    self.hits += 1
                    self.total_get_time += time.time() - start_time
                return value
            
            # Try secondary backend if available
            if self.secondary_backend:
                value = await self.secondary_backend.get(key)
                if value is not None:
                    # Populate primary cache
                    asyncio.create_task(self.primary_backend.set(key, value))
                    if self.enable_metrics:
                        self.hits += 1
                        self.total_get_time += time.time() - start_time
                    return value
            
            # Cache miss
            if self.enable_metrics:
                self.misses += 1
                self.total_get_time += time.time() - start_time
            
            return default
        
        except Exception:
            if self.enable_metrics:
                self.errors += 1
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        start_time = time.time()
        
        try:
            # Set in primary backend
            primary_success = await self.primary_backend.set(key, value, ttl)
            
            # Set in secondary backend if available
            secondary_success = True
            if self.secondary_backend:
                secondary_success = await self.secondary_backend.set(key, value, ttl)
            
            if self.enable_metrics:
                self.total_set_time += time.time() - start_time
            
            return primary_success and secondary_success
        
        except Exception:
            if self.enable_metrics:
                self.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            primary_success = await self.primary_backend.delete(key)
            
            secondary_success = True
            if self.secondary_backend:
                secondary_success = await self.secondary_backend.delete(key)
            
            return primary_success and secondary_success
        
        except Exception:
            if self.enable_metrics:
                self.errors += 1
            return False
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """Get value from cache or set it using factory function."""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Generate value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        # Cache the value
        await self.set(key, value, ttl)
        return value
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        try:
            # Get keys from primary backend
            keys = await self.primary_backend.keys(pattern)
            
            # Delete from both backends
            deleted_count = 0
            for key in keys:
                if await self.delete(key):
                    deleted_count += 1
            
            return deleted_count
        
        except Exception:
            if self.enable_metrics:
                self.errors += 1
            return 0
    
    def get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create deterministic key from arguments
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]  # Use SHA-256, truncate for readability
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        avg_get_time = self.total_get_time / total_requests if total_requests > 0 else 0
        avg_set_time = self.total_set_time / max(self.hits, 1)
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "avg_get_time_ms": avg_get_time * 1000,
            "avg_set_time_ms": avg_set_time * 1000
        }
    
    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.total_get_time = 0.0
        self.total_set_time = 0.0


def cache_result(
    cache_manager: CacheManager,
    key_prefix: str,
    ttl: int = 3600,
    include_args: bool = True
):
    """Decorator to cache function results."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if include_args:
                cache_key = cache_manager.get_cache_key(key_prefix, *args, **kwargs)
            else:
                cache_key = key_prefix
            
            # Try to get from cache
            result = await cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(cache_key, result, ttl)
            return result
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in asyncio
            async def _async_wrapper():
                return await async_wrapper(*args, **kwargs)
            
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_async_wrapper())
            except RuntimeError:
                # No event loop, run sync version
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class AdaptiveCache:
    """Adaptive cache that adjusts behavior based on usage patterns."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.key_priorities: Dict[str, float] = {}
        self.monitoring_enabled = True
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get with adaptive behavior."""
        if self.monitoring_enabled:
            self._record_access(key)
        
        return await self.cache_manager.get(key, default)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set with adaptive TTL."""
        if ttl is None:
            ttl = self._calculate_adaptive_ttl(key)
        
        return await self.cache_manager.set(key, value, ttl)
    
    def _record_access(self, key: str) -> None:
        """Record key access for pattern analysis."""
        now = datetime.now()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(now)
        
        # Keep only recent accesses (last hour)
        cutoff = now - timedelta(hours=1)
        self.access_patterns[key] = [
            access for access in self.access_patterns[key]
            if access > cutoff
        ]
        
        # Update priority
        self.key_priorities[key] = len(self.access_patterns[key])
    
    def _calculate_adaptive_ttl(self, key: str) -> int:
        """Calculate adaptive TTL based on access patterns."""
        base_ttl = 3600  # 1 hour
        
        # Get access frequency
        frequency = self.key_priorities.get(key, 1)
        
        # High frequency items get longer TTL
        if frequency > 10:
            return base_ttl * 4  # 4 hours
        elif frequency > 5:
            return base_ttl * 2  # 2 hours
        else:
            return base_ttl  # 1 hour
    
    def get_hottest_keys(self, limit: int = 10) -> List[tuple]:
        """Get most frequently accessed keys."""
        sorted_keys = sorted(
            self.key_priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_keys[:limit]