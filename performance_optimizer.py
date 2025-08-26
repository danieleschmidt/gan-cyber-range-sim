#!/usr/bin/env python3
"""Performance optimization and scaling for GAN Cyber Range Simulator."""

import sys
import os
import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import concurrent.futures
import threading
from collections import deque
import hashlib
import uuid

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from minimal_test import MockCyberRange, MockRedTeamAgent, MockBlueTeamAgent
from robust_cyber_range import RobustLogger, SystemHealth


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    requests_per_second: float = 0.0
    avg_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    concurrent_simulations: int = 0
    total_simulations: int = 0
    errors_per_minute: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'requests_per_second': self.requests_per_second,
            'avg_response_time_ms': self.avg_response_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'cache_hit_rate': self.cache_hit_rate,
            'concurrent_simulations': self.concurrent_simulations,
            'total_simulations': self.total_simulations,
            'errors_per_minute': self.errors_per_minute
        }


class InMemoryCache:
    """High-performance in-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        self.logger = RobustLogger("Cache")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        return (time.time() - entry['timestamp']) > self.ttl_seconds
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if (current_time - entry['timestamp']) > self.ttl_seconds
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
    
    def _evict_lru(self) -> None:
        """Remove least recently used entry."""
        if self.access_order:
            lru_key = self.access_order.popleft()
            if lru_key in self.cache:
                del self.cache[lru_key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if self._is_expired(entry):
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.miss_count += 1
                    return None
                
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.hit_count += 1
                return entry['value']
            
            self.miss_count += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        with self.lock:
            # Clean expired entries periodically
            if len(self.cache) % 100 == 0:
                self._evict_expired()
            
            # Evict LRU if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            # Store new entry
            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl_seconds
            }


class ConnectionPool:
    """Connection pool for managing concurrent simulations."""
    
    def __init__(self, max_connections: int = 10, timeout_seconds: int = 30):
        self.max_connections = max_connections
        self.timeout_seconds = timeout_seconds
        self.active_connections = set()
        self.available_connections = asyncio.Queue(maxsize=max_connections)
        self.lock = asyncio.Lock()
        self.logger = RobustLogger("ConnectionPool")
        
        # Pre-populate with connection placeholders
        for i in range(max_connections):
            self.available_connections.put_nowait(f"conn_{i}")
    
    async def acquire(self) -> str:
        """Acquire a connection from the pool."""
        try:
            connection = await asyncio.wait_for(
                self.available_connections.get(),
                timeout=self.timeout_seconds
            )
            
            async with self.lock:
                self.active_connections.add(connection)
            
            return connection
            
        except asyncio.TimeoutError:
            raise RuntimeError("Connection pool timeout - all connections busy")
    
    async def release(self, connection: str) -> None:
        """Release a connection back to the pool."""
        async with self.lock:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
        
        try:
            self.available_connections.put_nowait(connection)
        except asyncio.QueueFull:
            self.logger.warning(f"Queue full when releasing connection: {connection}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'max_connections': self.max_connections,
            'active_connections': len(self.active_connections),
            'available_connections': self.available_connections.qsize(),
            'timeout_seconds': self.timeout_seconds
        }


class LoadBalancer:
    """Load balancer for distributing simulation workload."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.logger = RobustLogger("LoadBalancer")
    
    async def submit_task(self, task_id: str, coro) -> str:
        """Submit a task for load-balanced execution."""
        self.logger.info(f"Submitting task: {task_id}")
        
        # Create task
        task = asyncio.create_task(coro)
        self.active_tasks[task_id] = {
            'task': task,
            'start_time': time.time(),
            'status': 'running'
        }
        
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result from a submitted task."""
        if task_id not in self.active_tasks:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        task_info = self.active_tasks[task_id]
        task = task_info['task']
        
        try:
            if timeout:
                result = await asyncio.wait_for(task, timeout=timeout)
            else:
                result = await task
            
            task_info['status'] = 'completed'
            task_info['end_time'] = time.time()
            task_info['duration'] = task_info['end_time'] - task_info['start_time']
            
            self.completed_tasks += 1
            self.logger.info(f"Task completed: {task_id} in {task_info['duration']:.2f}s")
            
            return result
            
        except Exception as e:
            task_info['status'] = 'failed'
            task_info['error'] = str(e)
            self.failed_tasks += 1
            self.logger.error(f"Task failed: {task_id}", exception=e)
            raise
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        if task_id not in self.active_tasks:
            return {'status': 'unknown'}
        
        return self.active_tasks[task_id].copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        active_count = len([t for t in self.active_tasks.values() if t['status'] == 'running'])
        
        return {
            'max_workers': self.max_workers,
            'active_tasks': active_count,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'total_tasks': len(self.active_tasks)
        }
    
    def cleanup_completed_tasks(self) -> None:
        """Remove completed tasks to free memory."""
        completed_ids = [
            task_id for task_id, info in self.active_tasks.items()
            if info['status'] in ['completed', 'failed']
        ]
        
        for task_id in completed_ids:
            del self.active_tasks[task_id]


class AutoScaler:
    """Auto-scaling based on load and performance metrics."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.metrics_window = deque(maxlen=60)  # 1 minute of metrics
        self.last_scale_time = time.time()
        self.scale_cooldown = 30  # 30 seconds between scaling decisions
        self.logger = RobustLogger("AutoScaler")
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for scaling decisions."""
        self.metrics_window.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
    
    def should_scale_up(self) -> bool:
        """Determine if we should scale up."""
        if len(self.metrics_window) < 10:  # Need enough data
            return False
        
        if self.current_instances >= self.max_instances:
            return False
        
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Check recent metrics
        recent_metrics = list(self.metrics_window)[-10:]  # Last 10 data points
        
        # Scale up if:
        # - High CPU usage consistently
        # - High response times
        # - Many concurrent simulations
        
        avg_cpu = sum(m['metrics'].cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_response = sum(m['metrics'].avg_response_time_ms for m in recent_metrics) / len(recent_metrics)
        max_concurrent = max(m['metrics'].concurrent_simulations for m in recent_metrics)
        
        if (avg_cpu > 70 or avg_response > 1000 or max_concurrent > self.current_instances * 2):
            return True
        
        return False
    
    def should_scale_down(self) -> bool:
        """Determine if we should scale down."""
        if len(self.metrics_window) < 10:
            return False
        
        if self.current_instances <= self.min_instances:
            return False
        
        if time.time() - self.last_scale_time < self.scale_cooldown * 2:  # Longer cooldown for scale down
            return False
        
        # Check recent metrics
        recent_metrics = list(self.metrics_window)[-10:]
        
        avg_cpu = sum(m['metrics'].cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_response = sum(m['metrics'].avg_response_time_ms for m in recent_metrics) / len(recent_metrics)
        max_concurrent = max(m['metrics'].concurrent_simulations for m in recent_metrics)
        
        if (avg_cpu < 30 and avg_response < 500 and max_concurrent < self.current_instances):
            return True
        
        return False
    
    def scale_up(self) -> int:
        """Scale up by one instance."""
        if self.current_instances < self.max_instances:
            self.current_instances += 1
            self.last_scale_time = time.time()
            self.logger.info(f"Scaled up to {self.current_instances} instances")
        
        return self.current_instances
    
    def scale_down(self) -> int:
        """Scale down by one instance."""
        if self.current_instances > self.min_instances:
            self.current_instances -= 1
            self.last_scale_time = time.time()
            self.logger.info(f"Scaled down to {self.current_instances} instances")
        
        return self.current_instances
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'metrics_points': len(self.metrics_window),
            'last_scale_time': self.last_scale_time,
            'time_since_last_scale': time.time() - self.last_scale_time
        }


class PerformanceOptimizedCyberRange:
    """High-performance cyber range with caching, pooling, and auto-scaling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = RobustLogger("OptimizedCyberRange")
        
        # Performance components
        self.cache = InMemoryCache(max_size=1000, ttl_seconds=300)
        self.connection_pool = ConnectionPool(max_connections=20)
        self.load_balancer = LoadBalancer(max_workers=8)
        self.auto_scaler = AutoScaler(min_instances=1, max_instances=5)
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=1000)
        self.start_time = time.time()
        self.simulation_count = 0
        
        self.logger.info("Performance-optimized cyber range initialized", config=config)
    
    async def run_simulation_optimized(self, simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run simulation with performance optimizations."""
        start_time = time.time()
        simulation_id = str(uuid.uuid4())
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(simulation_config)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                self.logger.info(f"Cache hit for simulation: {simulation_id}")
                cached_result['cache_hit'] = True
                cached_result['simulation_id'] = simulation_id
                return cached_result
            
            # Acquire connection from pool
            connection = await self.connection_pool.acquire()
            
            try:
                # Submit to load balancer
                task_id = await self.load_balancer.submit_task(
                    simulation_id,
                    self._execute_simulation(simulation_config, connection)
                )
                
                # Get result with timeout
                result = await self.load_balancer.get_task_result(
                    task_id, 
                    timeout=simulation_config.get('duration', 0.1) * 3600 + 60
                )
                
                # Cache result
                result['cache_hit'] = False
                result['execution_time_ms'] = (time.time() - start_time) * 1000
                self.cache.set(cache_key, result)
                
                # Update metrics
                self._update_performance_metrics(result)
                
                return result
                
            finally:
                await self.connection_pool.release(connection)
        
        except Exception as e:
            self.logger.error(f"Optimized simulation failed: {simulation_id}", exception=e)
            raise
    
    async def _execute_simulation(self, config: Dict[str, Any], connection: str) -> Dict[str, Any]:
        """Execute the actual simulation."""
        self.logger.info(f"Executing simulation on connection: {connection}")
        
        # Create mock components
        cyber_range = MockCyberRange(vulnerable_services=config.get('services', ['webapp']))
        red_team = MockRedTeamAgent(skill_level=config.get('red_skill', 'advanced'))
        blue_team = MockBlueTeamAgent(skill_level=config.get('blue_skill', 'advanced'))
        
        # Run simulation
        results = await cyber_range.simulate(
            red_team=red_team,
            blue_team=blue_team,
            duration_hours=config.get('duration', 0.1)
        )
        
        # Format results
        return {
            'simulation_id': results.simulation_id,
            'duration': str(results.duration),
            'total_attacks': results.total_attacks,
            'services_compromised': results.services_compromised,
            'attacks_blocked': results.attacks_blocked,
            'compromise_rate': results.compromise_rate,
            'defense_effectiveness': results.defense_effectiveness,
            'connection_used': connection,
            'agent_stats': {
                'red_team': red_team.get_stats(),
                'blue_team': blue_team.get_stats()
            }
        }
    
    def _generate_cache_key(self, config: Dict[str, Any]) -> str:
        """Generate cache key for simulation config."""
        # Create deterministic key from config
        key_parts = [
            str(config.get('services', [])),
            str(config.get('duration', 0.1)),
            str(config.get('red_skill', 'advanced')),
            str(config.get('blue_skill', 'advanced'))
        ]
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_performance_metrics(self, result: Dict[str, Any]) -> None:
        """Update performance metrics."""
        current_time = time.time()
        
        metrics = PerformanceMetrics(
            avg_response_time_ms=result.get('execution_time_ms', 0),
            concurrent_simulations=len(self.load_balancer.active_tasks),
            total_simulations=self.simulation_count + 1,
            cache_hit_rate=self.cache.get_stats()['hit_rate']
        )
        
        # Record metrics for auto-scaling
        self.auto_scaler.record_metrics(metrics)
        
        # Store in history
        self.metrics_history.append({
            'timestamp': current_time,
            'metrics': metrics
        })
        
        # Auto-scaling decisions
        if self.auto_scaler.should_scale_up():
            self.auto_scaler.scale_up()
        elif self.auto_scaler.should_scale_down():
            self.auto_scaler.scale_down()
        
        self.simulation_count += 1
    
    async def run_concurrent_simulations(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple simulations concurrently."""
        self.logger.info(f"Running {len(configs)} concurrent simulations")
        
        # Create tasks for all simulations
        tasks = []
        for i, config in enumerate(configs):
            task = asyncio.create_task(
                self.run_simulation_optimized(config),
                name=f"simulation_{i}"
            )
            tasks.append(task)
        
        # Wait for all to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and exceptions
            successful_results = []
            failed_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Simulation {i} failed", exception=result)
                    failed_count += 1
                else:
                    successful_results.append(result)
            
            self.logger.info(f"Concurrent execution complete: {len(successful_results)} succeeded, {failed_count} failed")
            return successful_results
            
        except Exception as e:
            self.logger.error("Concurrent simulation execution failed", exception=e)
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'total_simulations': self.simulation_count,
            'simulations_per_hour': (self.simulation_count / uptime) * 3600 if uptime > 0 else 0,
            'cache_stats': self.cache.get_stats(),
            'connection_pool_stats': self.connection_pool.get_stats(),
            'load_balancer_stats': self.load_balancer.get_stats(),
            'auto_scaler_stats': self.auto_scaler.get_stats(),
            'current_instances': self.auto_scaler.current_instances,
            'metrics_history_size': len(self.metrics_history)
        }
    
    async def health_check(self) -> SystemHealth:
        """Perform health check with performance considerations."""
        health = SystemHealth()
        health.last_check = datetime.now()
        health.uptime_seconds = time.time() - self.start_time
        
        try:
            # Check cache
            cache_stats = self.cache.get_stats()
            if cache_stats['hit_rate'] > 0.5:
                health.components['cache'] = 'healthy'
            elif cache_stats['hit_rate'] > 0.2:
                health.components['cache'] = 'degraded'
            else:
                health.components['cache'] = 'unhealthy'
            
            # Check connection pool
            pool_stats = self.connection_pool.get_stats()
            if pool_stats['available_connections'] > 0:
                health.components['connection_pool'] = 'healthy'
            else:
                health.components['connection_pool'] = 'degraded'
                health.errors.append('No available connections')
            
            # Check load balancer
            lb_stats = self.load_balancer.get_stats()
            if lb_stats['failed_tasks'] / max(lb_stats['total_tasks'], 1) < 0.1:  # Less than 10% failure rate
                health.components['load_balancer'] = 'healthy'
            else:
                health.components['load_balancer'] = 'degraded'
                health.errors.append(f"High failure rate: {lb_stats['failed_tasks']}/{lb_stats['total_tasks']}")
            
            # Overall status
            component_statuses = list(health.components.values())
            if 'unhealthy' in component_statuses:
                health.status = 'unhealthy'
            elif 'degraded' in component_statuses:
                health.status = 'degraded'
            else:
                health.status = 'healthy'
            
            return health
            
        except Exception as e:
            health.status = 'unhealthy'
            health.errors.append(f"Health check failed: {e}")
            return health


# Performance testing and demonstration
async def performance_demo():
    """Demonstrate performance optimization features."""
    print("🚀 PERFORMANCE-OPTIMIZED GAN CYBER RANGE DEMONSTRATION")
    print("="*70)
    
    # Initialize optimized cyber range
    config = {
        'services': ['webapp', 'database'],
        'duration': 0.02,  # Very short for demo
        'red_skill': 'advanced',
        'blue_skill': 'advanced'
    }
    
    optimized_range = PerformanceOptimizedCyberRange(config)
    
    print("🏗️ Testing single simulation with caching...")
    
    # Run same simulation twice to test caching
    start_time = time.time()
    result1 = await optimized_range.run_simulation_optimized(config)
    time1 = time.time() - start_time
    
    start_time = time.time()
    result2 = await optimized_range.run_simulation_optimized(config)
    time2 = time.time() - start_time
    
    print(f"   First run: {time1:.3f}s (cache_hit: {result1.get('cache_hit', False)})")
    print(f"   Second run: {time2:.3f}s (cache_hit: {result2.get('cache_hit', False)})")
    print(f"   Cache speedup: {time1/time2:.1f}x faster" if time2 > 0 else "   Cache working!")
    
    print("\n🔄 Testing concurrent simulations...")
    
    # Test concurrent execution
    concurrent_configs = [
        {**config, 'services': ['webapp']},
        {**config, 'services': ['database']}, 
        {**config, 'services': ['api-gateway']},
        {**config, 'services': ['webapp', 'database']}
    ]
    
    start_time = time.time()
    concurrent_results = await optimized_range.run_concurrent_simulations(concurrent_configs)
    concurrent_time = time.time() - start_time
    
    print(f"   Concurrent execution: {len(concurrent_results)} simulations in {concurrent_time:.3f}s")
    print(f"   Average per simulation: {concurrent_time/len(concurrent_results):.3f}s")
    
    print("\n📊 Performance Statistics:")
    stats = optimized_range.get_performance_stats()
    
    print(f"   Total simulations: {stats['total_simulations']}")
    print(f"   Simulations per hour: {stats['simulations_per_hour']:.1f}")
    print(f"   Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
    print(f"   Active connections: {stats['connection_pool_stats']['active_connections']}")
    print(f"   Auto-scaler instances: {stats['current_instances']}")
    
    print("\n🏥 Health Check:")
    health = await optimized_range.health_check()
    print(f"   Overall status: {health.status}")
    print(f"   Components: {health.components}")
    
    if health.errors:
        print(f"   Errors: {health.errors}")
    
    print("\n✅ Performance optimization demonstration complete!")
    print(f"📈 System successfully handled {stats['total_simulations']} simulations")


if __name__ == "__main__":
    asyncio.run(performance_demo())