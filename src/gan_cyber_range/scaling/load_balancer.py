"""Load balancing and traffic distribution."""

import asyncio
import random
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    CONSISTENT_HASH = "consistent_hash"


class BackendStatus(Enum):
    """Backend server status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"


@dataclass
class Backend:
    """Backend server representation."""
    id: str
    host: str
    port: int
    weight: int = 100
    status: BackendStatus = BackendStatus.HEALTHY
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        """Get backend URL."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.failed_requests / self.total_requests)
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times[-100:])  # Last 100 requests
    
    def is_available(self) -> bool:
        """Check if backend is available for requests."""
        return self.status in [BackendStatus.HEALTHY, BackendStatus.OVERLOADED]
    
    def record_request(self, response_time: float, success: bool) -> None:
        """Record request metrics."""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        
        self.response_times.append(response_time)
        # Keep only recent response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-500:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "url": self.url,
            "weight": self.weight,
            "status": self.status.value,
            "active_connections": self.active_connections,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "metadata": self.metadata
        }


class HealthChecker:
    """Health checker for backend servers."""
    
    def __init__(
        self,
        check_interval: int = 30,
        timeout: float = 5.0,
        failure_threshold: int = 3,
        success_threshold: int = 2
    ):
        self.check_interval = check_interval
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        
        self.failure_counts: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}
        self.running = False
        self.check_task: Optional[asyncio.Task] = None
    
    async def start(self, backends: List[Backend]) -> None:
        """Start health checking."""
        if self.running:
            return
        
        self.running = True
        self.check_task = asyncio.create_task(self._health_check_loop(backends))
    
    async def stop(self) -> None:
        """Stop health checking."""
        self.running = False
        
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self, backends: List[Backend]) -> None:
        """Main health check loop."""
        while self.running:
            try:
                # Check all backends concurrently
                tasks = [self._check_backend(backend) for backend in backends]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health check loop error: {e}")
                await asyncio.sleep(5)
    
    async def _check_backend(self, backend: Backend) -> None:
        """Check individual backend health."""
        try:
            # Perform health check (HTTP request)
            start_time = time.time()
            
            # This would be replaced with actual HTTP client
            # For now, simulate health check
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Simulate some failures based on success rate
            is_healthy = random.random() < 0.95  # 95% success rate
            
            response_time = time.time() - start_time
            
            if is_healthy:
                self._handle_health_check_success(backend)
            else:
                self._handle_health_check_failure(backend)
            
            backend.last_health_check = datetime.now()
            
        except Exception as e:
            self._handle_health_check_failure(backend)
    
    def _handle_health_check_success(self, backend: Backend) -> None:
        """Handle successful health check."""
        backend_id = backend.id
        
        # Reset failure count
        self.failure_counts[backend_id] = 0
        
        # Increment success count
        self.success_counts[backend_id] = self.success_counts.get(backend_id, 0) + 1
        
        # Mark as healthy if enough successes
        if (backend.status != BackendStatus.HEALTHY and 
            self.success_counts[backend_id] >= self.success_threshold):
            backend.status = BackendStatus.HEALTHY
            self.success_counts[backend_id] = 0
    
    def _handle_health_check_failure(self, backend: Backend) -> None:
        """Handle failed health check."""
        backend_id = backend.id
        
        # Reset success count
        self.success_counts[backend_id] = 0
        
        # Increment failure count
        self.failure_counts[backend_id] = self.failure_counts.get(backend_id, 0) + 1
        
        # Mark as unhealthy if enough failures
        if (backend.status == BackendStatus.HEALTHY and 
            self.failure_counts[backend_id] >= self.failure_threshold):
            backend.status = BackendStatus.UNHEALTHY


class LoadBalancer:
    """Load balancer with multiple strategies and health checking."""
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        enable_health_checks: bool = True,
        health_check_interval: int = 30
    ):
        self.strategy = strategy
        self.backends: List[Backend] = []
        self.current_index = 0
        
        # Health checking
        self.health_checker = HealthChecker(check_interval=health_check_interval) if enable_health_checks else None
        
        # Metrics
        self.total_requests = 0
        self.failed_requests = 0
        self.request_times: List[float] = []
        
        # Strategy-specific state
        self.hash_ring: Dict[int, str] = {}  # For consistent hashing
        self.ring_size = 360  # Virtual nodes for consistent hashing
    
    def add_backend(
        self,
        backend_id: str,
        host: str,
        port: int,
        weight: int = 100,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add backend server."""
        backend = Backend(
            id=backend_id,
            host=host,
            port=port,
            weight=weight,
            metadata=metadata or {}
        )
        
        self.backends.append(backend)
        
        # Update consistent hash ring if using that strategy
        if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            self._update_hash_ring()
    
    def remove_backend(self, backend_id: str) -> bool:
        """Remove backend server."""
        for i, backend in enumerate(self.backends):
            if backend.id == backend_id:
                self.backends.pop(i)
                
                # Update consistent hash ring if using that strategy
                if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                    self._update_hash_ring()
                
                return True
        return False
    
    def update_backend_weight(self, backend_id: str, weight: int) -> bool:
        """Update backend weight."""
        for backend in self.backends:
            if backend.id == backend_id:
                backend.weight = weight
                
                # Update consistent hash ring if using that strategy
                if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                    self._update_hash_ring()
                
                return True
        return False
    
    def set_backend_status(self, backend_id: str, status: BackendStatus) -> bool:
        """Set backend status."""
        for backend in self.backends:
            if backend.id == backend_id:
                backend.status = status
                return True
        return False
    
    async def start(self) -> None:
        """Start load balancer services."""
        if self.health_checker:
            await self.health_checker.start(self.backends)
    
    async def stop(self) -> None:
        """Stop load balancer services."""
        if self.health_checker:
            await self.health_checker.stop()
    
    def get_backend(self, client_key: Optional[str] = None) -> Optional[Backend]:
        """Get next backend using current strategy."""
        available_backends = [b for b in self.backends if b.is_available()]
        
        if not available_backends:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(available_backends)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(available_backends)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(available_backends)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(available_backends)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random(available_backends)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random(available_backends)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash(available_backends, client_key or "default")
        else:
            return self._round_robin(available_backends)
    
    def _round_robin(self, backends: List[Backend]) -> Backend:
        """Round robin selection."""
        backend = backends[self.current_index % len(backends)]
        self.current_index = (self.current_index + 1) % len(backends)
        return backend
    
    def _weighted_round_robin(self, backends: List[Backend]) -> Backend:
        """Weighted round robin selection."""
        total_weight = sum(b.weight for b in backends)
        if total_weight == 0:
            return self._round_robin(backends)
        
        # Create weighted list
        weighted_backends = []
        for backend in backends:
            count = max(1, backend.weight * 10 // total_weight)
            weighted_backends.extend([backend] * count)
        
        if not weighted_backends:
            return backends[0]
        
        backend = weighted_backends[self.current_index % len(weighted_backends)]
        self.current_index = (self.current_index + 1) % len(weighted_backends)
        return backend
    
    def _least_connections(self, backends: List[Backend]) -> Backend:
        """Least connections selection."""
        return min(backends, key=lambda b: b.active_connections)
    
    def _least_response_time(self, backends: List[Backend]) -> Backend:
        """Least response time selection."""
        return min(backends, key=lambda b: b.avg_response_time)
    
    def _random(self, backends: List[Backend]) -> Backend:
        """Random selection."""
        return random.choice(backends)
    
    def _weighted_random(self, backends: List[Backend]) -> Backend:
        """Weighted random selection."""
        total_weight = sum(b.weight for b in backends)
        if total_weight == 0:
            return self._random(backends)
        
        r = random.randint(1, total_weight)
        current_weight = 0
        
        for backend in backends:
            current_weight += backend.weight
            if r <= current_weight:
                return backend
        
        return backends[-1]  # Fallback
    
    def _consistent_hash(self, backends: List[Backend], client_key: str) -> Backend:
        """Consistent hash selection."""
        if not self.hash_ring:
            self._update_hash_ring()
        
        if not self.hash_ring:
            return self._random(backends)
        
        # Hash the client key
        hash_value = hash(client_key) % self.ring_size
        
        # Find the next backend in the ring
        for i in range(self.ring_size):
            pos = (hash_value + i) % self.ring_size
            if pos in self.hash_ring:
                backend_id = self.hash_ring[pos]
                # Find the backend
                for backend in backends:
                    if backend.id == backend_id:
                        return backend
        
        # Fallback to random
        return self._random(backends)
    
    def _update_hash_ring(self) -> None:
        """Update consistent hash ring."""
        self.hash_ring.clear()
        
        for backend in self.backends:
            if not backend.is_available():
                continue
            
            # Add virtual nodes based on weight
            virtual_nodes = max(1, backend.weight // 10)
            for i in range(virtual_nodes):
                hash_key = f"{backend.id}:{i}"
                hash_value = hash(hash_key) % self.ring_size
                self.hash_ring[hash_value] = backend.id
    
    async def execute_request(
        self,
        request_handler: Callable[[Backend], Any],
        client_key: Optional[str] = None,
        max_retries: int = 3
    ) -> Any:
        """Execute request with load balancing and retry logic."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            backend = self.get_backend(client_key)
            if not backend:
                raise RuntimeError("No available backends")
            
            start_time = time.time()
            
            try:
                # Track active connection
                backend.active_connections += 1
                
                # Execute request
                if asyncio.iscoroutinefunction(request_handler):
                    result = await request_handler(backend)
                else:
                    result = request_handler(backend)
                
                # Record success
                response_time = time.time() - start_time
                backend.record_request(response_time, True)
                
                self.total_requests += 1
                self.request_times.append(response_time)
                
                return result
                
            except Exception as e:
                # Record failure
                response_time = time.time() - start_time
                backend.record_request(response_time, False)
                
                self.total_requests += 1
                self.failed_requests += 1
                
                last_exception = e
                
                # Mark backend as overloaded if too many failures
                if backend.success_rate < 0.5:
                    backend.status = BackendStatus.OVERLOADED
                
            finally:
                # Decrease active connections
                backend.active_connections = max(0, backend.active_connections - 1)
        
        # All retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Request failed after all retries")
    
    def get_backend_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all backends."""
        return [backend.to_dict() for backend in self.backends]
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        available_backends = len([b for b in self.backends if b.is_available()])
        
        avg_response_time = (
            statistics.mean(self.request_times[-1000:])  # Last 1000 requests
            if self.request_times else 0
        )
        
        success_rate = (
            1.0 - (self.failed_requests / self.total_requests)
            if self.total_requests > 0 else 1.0
        )
        
        return {
            "strategy": self.strategy.value,
            "total_backends": len(self.backends),
            "available_backends": available_backends,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time * 1000,
            "health_checks_enabled": self.health_checker is not None
        }
    
    def rebalance_backends(self) -> None:
        """Rebalance backend weights based on performance."""
        if len(self.backends) < 2:
            return
        
        # Calculate performance scores
        for backend in self.backends:
            if backend.total_requests == 0:
                continue
            
            # Performance score based on success rate and response time
            success_weight = 0.7
            response_time_weight = 0.3
            
            success_score = backend.success_rate
            
            # Normalize response time (lower is better)
            max_response_time = max(b.avg_response_time for b in self.backends)
            if max_response_time > 0:
                response_time_score = 1.0 - (backend.avg_response_time / max_response_time)
            else:
                response_time_score = 1.0
            
            performance_score = (success_weight * success_score + 
                               response_time_weight * response_time_score)
            
            # Adjust weight based on performance
            new_weight = max(10, int(performance_score * 100))
            backend.weight = new_weight
        
        # Update hash ring if using consistent hashing
        if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            self._update_hash_ring()