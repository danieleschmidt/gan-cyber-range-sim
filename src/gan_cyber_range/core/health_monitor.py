"""System health monitoring and diagnostics."""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a system component."""
    component_name: str
    status: HealthStatus
    message: str
    last_check: datetime
    metrics: Dict[str, Any]
    dependencies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component_name": self.component_name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "metrics": self.metrics,
            "dependencies": self.dependencies
        }


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(
        self, 
        name: str, 
        check_interval: int = 30,
        timeout: int = 10,
        dependencies: Optional[List[str]] = None
    ):
        self.name = name
        self.check_interval = check_interval
        self.timeout = timeout
        self.dependencies = dependencies or []
        self.last_status = HealthStatus.UNKNOWN
        self.last_check = None
        
    async def check_health(self) -> ComponentHealth:
        """Perform health check - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement check_health")
    
    async def run_check(self) -> ComponentHealth:
        """Run health check with timeout and error handling."""
        try:
            check_task = asyncio.create_task(self.check_health())
            health = await asyncio.wait_for(check_task, timeout=self.timeout)
            self.last_status = health.status
            self.last_check = datetime.utcnow()
            return health
            
        except asyncio.TimeoutError:
            return ComponentHealth(
                component_name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.timeout}s",
                last_check=datetime.utcnow(),
                metrics={"timeout": True},
                dependencies=self.dependencies
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                last_check=datetime.utcnow(),
                metrics={"error": str(e)},
                dependencies=self.dependencies
            )


class SystemResourceCheck(HealthCheck):
    """Check system resource utilization."""
    
    def __init__(
        self, 
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0,
        **kwargs
    ):
        super().__init__("system_resources", **kwargs)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def check_health(self) -> ComponentHealth:
        """Check system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # Determine status based on thresholds
        status = HealthStatus.HEALTHY
        messages = []
        
        if cpu_percent > self.cpu_threshold:
            status = HealthStatus.WARNING
            messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
        if memory.percent > self.memory_threshold:
            status = HealthStatus.CRITICAL if memory.percent > 95 else HealthStatus.WARNING
            messages.append(f"High memory usage: {memory.percent:.1f}%")
            
        if disk.percent > self.disk_threshold:
            status = HealthStatus.CRITICAL
            messages.append(f"Low disk space: {disk.percent:.1f}% used")
        
        message = "; ".join(messages) if messages else "System resources normal"
        
        return ComponentHealth(
            component_name=self.name,
            status=status,
            message=message,
            last_check=datetime.utcnow(),
            metrics=metrics,
            dependencies=self.dependencies
        )


class DatabaseCheck(HealthCheck):
    """Check database connectivity and performance."""
    
    def __init__(self, connection_string: str, **kwargs):
        super().__init__("database", **kwargs)
        self.connection_string = connection_string
    
    async def check_health(self) -> ComponentHealth:
        """Check database health."""
        start_time = time.time()
        
        try:
            # Simulated database check - would use actual DB connection
            await asyncio.sleep(0.1)  # Simulate query time
            
            query_time = time.time() - start_time
            
            status = HealthStatus.HEALTHY
            if query_time > 1.0:
                status = HealthStatus.WARNING
            elif query_time > 2.0:
                status = HealthStatus.CRITICAL
                
            return ComponentHealth(
                component_name=self.name,
                status=status,
                message=f"Database responsive (query time: {query_time:.3f}s)",
                last_check=datetime.utcnow(),
                metrics={
                    "query_time_seconds": query_time,
                    "connection_active": True
                },
                dependencies=self.dependencies
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                last_check=datetime.utcnow(),
                metrics={
                    "connection_active": False,
                    "error": str(e)
                },
                dependencies=self.dependencies
            )


class KubernetesCheck(HealthCheck):
    """Check Kubernetes cluster health."""
    
    def __init__(self, namespace: str = "default", **kwargs):
        super().__init__("kubernetes", **kwargs)
        self.namespace = namespace
    
    async def check_health(self) -> ComponentHealth:
        """Check Kubernetes cluster status."""
        try:
            # Simulated K8s check - would use kubernetes client
            pod_count = 5  # Simulated
            running_pods = 5  # Simulated
            
            metrics = {
                "total_pods": pod_count,
                "running_pods": running_pods,
                "healthy_ratio": running_pods / pod_count if pod_count > 0 else 0,
                "namespace": self.namespace
            }
            
            if running_pods == pod_count:
                status = HealthStatus.HEALTHY
                message = f"All {pod_count} pods running in {self.namespace}"
            elif running_pods / pod_count >= 0.8:
                status = HealthStatus.WARNING
                message = f"{running_pods}/{pod_count} pods running in {self.namespace}"
            else:
                status = HealthStatus.CRITICAL
                message = f"Only {running_pods}/{pod_count} pods running in {self.namespace}"
            
            return ComponentHealth(
                component_name=self.name,
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                metrics=metrics,
                dependencies=self.dependencies
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Kubernetes check failed: {str(e)}",
                last_check=datetime.utcnow(),
                metrics={"error": str(e)},
                dependencies=self.dependencies
            )


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: Dict[str, List[ComponentHealth]] = {}
        self.alert_callbacks: List[Callable] = []
        self.running = False
        self.logger = logging.getLogger(__name__)
        
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        self.health_history[health_check.name] = []
        
    def register_alert_callback(self, callback: Callable):
        """Register callback for health alerts."""
        self.alert_callbacks.append(callback)
        
    async def run_all_checks(self) -> Dict[str, ComponentHealth]:
        """Run all registered health checks."""
        results = {}
        
        # Run checks concurrently
        tasks = {
            name: check.run_check() 
            for name, check in self.health_checks.items()
        }
        
        completed = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for (name, _), result in zip(tasks.items(), completed):
            if isinstance(result, Exception):
                # Create error health result
                result = ComponentHealth(
                    component_name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check exception: {str(result)}",
                    last_check=datetime.utcnow(),
                    metrics={"exception": str(result)},
                    dependencies=[]
                )
            
            results[name] = result
            
            # Store in history (keep last 100 entries)
            self.health_history[name].append(result)
            if len(self.health_history[name]) > 100:
                self.health_history[name].pop(0)
                
            # Trigger alerts for critical status
            if result.status == HealthStatus.CRITICAL:
                await self._trigger_alerts(result)
        
        return results
    
    async def _trigger_alerts(self, health: ComponentHealth):
        """Trigger alert callbacks for critical health issues."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(health)
                else:
                    callback(health)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self.running = True
        self.logger.info("Starting system health monitoring")
        
        while self.running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        self.logger.info("Stopping system health monitoring")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.health_history:
            return {"status": "unknown", "components": {}}
        
        latest_health = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, history in self.health_history.items():
            if history:
                latest = history[-1]
                latest_health[name] = latest.to_dict()
                
                # Determine overall status (worst case)
                if latest.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif latest.status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.WARNING
        
        return {
            "status": overall_status.value,
            "components": latest_health,
            "last_updated": datetime.utcnow().isoformat(),
            "monitoring_active": self.running
        }
    
    def get_health_trends(self, component: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health trends for a component over time."""
        if component not in self.health_history:
            return []
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_health = [
            h.to_dict() 
            for h in self.health_history[component]
            if h.last_check >= cutoff
        ]
        
        return recent_health


# Default health checks
def create_default_health_monitor() -> SystemHealthMonitor:
    """Create system health monitor with default checks."""
    monitor = SystemHealthMonitor()
    
    # Add default health checks
    monitor.register_health_check(SystemResourceCheck())
    monitor.register_health_check(KubernetesCheck())
    
    return monitor