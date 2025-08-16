"""Advanced performance optimization and caching system."""

import asyncio
import time
import hashlib
import pickle
import logging
from typing import Any, Dict, List, Optional, Callable, Union, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
from collections import defaultdict, OrderedDict
import threading


T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    value: T
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    def touch(self):
        """Update last accessed time and increment count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class CacheManager:
    """Advanced multi-level cache manager with intelligent eviction."""
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl_seconds: Optional[int] = 3600,
        cleanup_interval_seconds: int = 300
    ):
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanups": 0
        }
        
        self.logger = logging.getLogger("cache_manager")
        self._cleanup_task = None
        self._running = False
    
    def start_cleanup_task(self):
        """Start background cleanup task."""
        if not self._running:
            self._running = True
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            except RuntimeError:
                # No event loop running, cleanup will be manual
                pass
    
    def stop_cleanup_task(self):
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                self._cleanup_expired()
                await asyncio.sleep(self.cleanup_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(10)
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._stats["cleanups"] += 1
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _evict_lru(self):
        """Evict least recently used entries when cache is full."""
        with self._lock:
            while len(self._cache) >= self.max_size:
                # Remove oldest entry (LRU)
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1
            
            return entry.value
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ):
        """Set value in cache."""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds
        
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
            
            # Evict if necessary
            self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=0,
                ttl_seconds=ttl_seconds
            )
            
            self._cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "cleanups": 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "stats": self._stats.copy(),
                "memory_usage_estimate": self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        try:
            # Sample a few entries to estimate average size
            sample_size = min(10, len(self._cache))
            if sample_size == 0:
                return 0
            
            sample_entries = list(self._cache.values())[:sample_size]
            total_size = sum(
                len(pickle.dumps(entry.value, protocol=pickle.HIGHEST_PROTOCOL))
                for entry in sample_entries
            )
            
            avg_size = total_size / sample_size
            return int(avg_size * len(self._cache))
            
        except Exception:
            return 0


def cached(
    ttl_seconds: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching function results."""
    
    if cache_manager is None:
        cache_manager = global_cache
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl_seconds)
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: cache_manager.clear()
        wrapper.cache_info = lambda: cache_manager.get_stats()
        
        return wrapper
    
    return decorator


def async_cached(
    ttl_seconds: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching async function results."""
    
    if cache_manager is None:
        cache_manager = global_cache
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function name and arguments."""
    # Create deterministic key from function name and arguments
    key_data = {
        "func": func_name,
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    
    key_str = str(key_data)
    return hashlib.md5(key_str.encode()).hexdigest()


class QueryOptimizer:
    """Database query optimization and caching."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or global_cache
        self.query_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0,
            "avg_time": 0,
            "last_executed": None
        })
        self.logger = logging.getLogger("query_optimizer")
    
    def optimized_query(
        self,
        query: str,
        params: Optional[Dict] = None,
        ttl_seconds: int = 300
    ):
        """Decorator for optimizing database queries."""
        
        def decorator(query_func):
            @wraps(query_func)
            async def wrapper(*args, **kwargs):
                # Generate cache key from query and params
                cache_key = self._generate_query_key(query, params)
                
                # Try cache first
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    self.logger.debug(f"Query cache hit: {query[:50]}...")
                    return cached_result
                
                # Execute query with timing
                start_time = time.time()
                try:
                    result = await query_func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Update statistics
                    stats = self.query_stats[query]
                    stats["count"] += 1
                    stats["total_time"] += execution_time
                    stats["avg_time"] = stats["total_time"] / stats["count"]
                    stats["last_executed"] = datetime.utcnow()
                    
                    # Cache result
                    self.cache_manager.set(cache_key, result, ttl_seconds)
                    
                    self.logger.debug(
                        f"Query executed in {execution_time:.3f}s: {query[:50]}..."
                    )
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.error(
                        f"Query failed after {execution_time:.3f}s: {query[:50]}... Error: {e}"
                    )
                    raise
            
            return wrapper
        
        return decorator
    
    def _generate_query_key(self, query: str, params: Optional[Dict]) -> str:
        """Generate cache key for query."""
        key_data = {"query": query, "params": params or {}}
        key_str = str(key_data)
        return f"query_{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        return dict(self.query_stats)
    
    def get_slow_queries(self, threshold_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """Get queries that are performing slowly."""
        slow_queries = []
        
        for query, stats in self.query_stats.items():
            if stats["avg_time"] > threshold_seconds:
                slow_queries.append({
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "avg_time": stats["avg_time"],
                    "count": stats["count"],
                    "total_time": stats["total_time"]
                })
        
        return sorted(slow_queries, key=lambda x: x["avg_time"], reverse=True)


class ResourceOptimizer:
    """System resource optimization."""
    
    def __init__(self):
        self.resource_stats = defaultdict(list)
        self.optimization_history = []
        self.logger = logging.getLogger("resource_optimizer")
    
    def monitor_resource_usage(
        self,
        component_name: str,
        cpu_percent: float,
        memory_mb: float,
        io_ops_per_sec: int
    ):
        """Record resource usage for optimization analysis."""
        timestamp = datetime.utcnow()
        
        self.resource_stats[component_name].append({
            "timestamp": timestamp,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "io_ops_per_sec": io_ops_per_sec
        })
        
        # Keep only last 1000 entries per component
        if len(self.resource_stats[component_name]) > 1000:
            self.resource_stats[component_name] = self.resource_stats[component_name][-1000:]
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get resource optimization recommendations."""
        recommendations = []
        
        for component, stats in self.resource_stats.items():
            if len(stats) < 10:  # Need minimum data points
                continue
            
            recent_stats = stats[-100:]  # Last 100 data points
            
            avg_cpu = sum(s["cpu_percent"] for s in recent_stats) / len(recent_stats)
            avg_memory = sum(s["memory_mb"] for s in recent_stats) / len(recent_stats)
            avg_io = sum(s["io_ops_per_sec"] for s in recent_stats) / len(recent_stats)
            
            # Generate recommendations based on usage patterns
            if avg_cpu > 80:
                recommendations.append({
                    "component": component,
                    "type": "scale_up",
                    "resource": "cpu",
                    "current_usage": avg_cpu,
                    "recommendation": "Increase CPU allocation or scale horizontally",
                    "priority": "high"
                })
            
            elif avg_cpu < 20:
                recommendations.append({
                    "component": component,
                    "type": "scale_down",
                    "resource": "cpu",
                    "current_usage": avg_cpu,
                    "recommendation": "Reduce CPU allocation to save costs",
                    "priority": "low"
                })
            
            if avg_memory > 1024:  # > 1GB
                recommendations.append({
                    "component": component,
                    "type": "optimize",
                    "resource": "memory",
                    "current_usage": avg_memory,
                    "recommendation": "Investigate memory usage patterns and optimize",
                    "priority": "medium"
                })
            
            if avg_io > 10000:  # High I/O
                recommendations.append({
                    "component": component,
                    "type": "optimize",
                    "resource": "io",
                    "current_usage": avg_io,
                    "recommendation": "Consider caching or I/O optimization",
                    "priority": "medium"
                })
        
        return recommendations
    
    def optimize_memory_usage(self):
        """Trigger memory optimization actions."""
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear global caches if memory pressure is high
        if hasattr(global_cache, 'clear'):
            cache_stats = global_cache.get_stats()
            if cache_stats.get('memory_usage_estimate', 0) > 100 * 1024 * 1024:  # 100MB
                global_cache.clear()
                self.logger.info("Cleared global cache due to memory pressure")
        
        optimization_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "memory_optimization",
            "objects_collected": collected,
            "action": "garbage_collection"
        }
        
        self.optimization_history.append(optimization_event)
        
        return optimization_event


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.query_optimizer = QueryOptimizer(self.cache_manager)
        self.resource_optimizer = ResourceOptimizer()
        self.logger = logging.getLogger("performance_optimizer")
        
        # Start cache cleanup
        self.cache_manager.start_cleanup_task()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "cache": self.cache_manager.get_stats(),
            "queries": {
                "stats": self.query_optimizer.get_query_stats(),
                "slow_queries": self.query_optimizer.get_slow_queries()
            },
            "resources": {
                "recommendations": self.resource_optimizer.get_optimization_recommendations(),
                "optimization_history": self.resource_optimizer.optimization_history[-10:]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def run_optimization_cycle(self):
        """Run a complete optimization cycle."""
        self.logger.info("Starting performance optimization cycle")
        
        # Memory optimization
        memory_result = self.resource_optimizer.optimize_memory_usage()
        
        # Cache cleanup
        self.cache_manager._cleanup_expired()
        
        # Generate recommendations
        recommendations = self.resource_optimizer.get_optimization_recommendations()
        
        self.logger.info(
            f"Optimization cycle completed. "
            f"Memory: {memory_result['objects_collected']} objects collected. "
            f"Recommendations: {len(recommendations)}"
        )
        
        return {
            "memory_optimization": memory_result,
            "recommendations": recommendations,
            "cache_stats": self.cache_manager.get_stats()
        }
    
    def shutdown(self):
        """Shutdown performance optimizer."""
        self.cache_manager.stop_cleanup_task()


# Global instances
global_cache = CacheManager()
global_performance_optimizer = PerformanceOptimizer()


# Convenience functions
def clear_all_caches():
    """Clear all global caches."""
    global_cache.clear()


def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics."""
    return global_performance_optimizer.get_performance_report()