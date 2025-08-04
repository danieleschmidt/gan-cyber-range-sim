"""Health checking and system monitoring."""

import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata or {}
        }


class HealthChecker:
    """System health monitoring and checking."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.logger = logging.getLogger("HealthChecker")
        self.running = False
        self._check_task: Optional[asyncio.Task] = None
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not found",
                timestamp=datetime.now(),
                response_time_ms=0.0
            )
        
        start_time = datetime.now()
        try:
            check_func = self.checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if isinstance(result, HealthCheckResult):
                result.response_time_ms = response_time
                return result
            elif isinstance(result, dict):
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus(result.get("status", "healthy")),
                    message=result.get("message", "OK"),
                    timestamp=datetime.now(),
                    response_time_ms=response_time,
                    metadata=result.get("metadata")
                )
            else:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    timestamp=datetime.now(),
                    response_time_ms=response_time
                )
        
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=response_time
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        # Run checks concurrently
        tasks = []
        for name in self.checks:
            task = asyncio.create_task(self.run_check(name))
            tasks.append((name, task))
        
        # Collect results
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
                self.results[name] = result
            except Exception as e:
                self.logger.error(f"Failed to run check '{name}': {e}")
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Execution failed: {str(e)}",
                    timestamp=datetime.now(),
                    response_time_ms=0.0
                )
        
        return results
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting health monitoring")
        
        self._check_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        overall_status = self.get_overall_health()
        
        summary = {
            "overall_status": overall_status.value,
            "total_checks": len(self.checks),
            "last_check": max([r.timestamp for r in self.results.values()]).isoformat() if self.results else None,
            "checks": {}
        }
        
        for name, result in self.results.items():
            summary["checks"][name] = result.to_dict()
        
        return summary
    
    def register_default_checks(self) -> None:
        """Register default system health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("cpu_usage", self._check_cpu_usage)
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check overall system resources."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = "unhealthy"
                message = "Critical resource usage"
            elif cpu_percent > 75 or memory.percent > 75 or disk.percent > 80:
                status = "degraded"
                message = "High resource usage"
            else:
                status = "healthy"
                message = "Resources within normal limits"
            
            return {
                "status": status,
                "message": message,
                "metadata": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent
                }
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Failed to check system resources: {e}"
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space."""
        try:
            disk = psutil.disk_usage('/')
            percent_used = disk.percent
            
            if percent_used > 95:
                status = "unhealthy"
                message = f"Critical disk usage: {percent_used:.1f}%"
            elif percent_used > 85:
                status = "degraded"
                message = f"High disk usage: {percent_used:.1f}%"
            else:
                status = "healthy"
                message = f"Disk usage: {percent_used:.1f}%"
            
            return {
                "status": status,
                "message": message,
                "metadata": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent_used": percent_used
                }
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Failed to check disk space: {e}"
            }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            percent_used = memory.percent
            
            if percent_used > 95:
                status = "unhealthy"
                message = f"Critical memory usage: {percent_used:.1f}%"
            elif percent_used > 85:
                status = "degraded"
                message = f"High memory usage: {percent_used:.1f}%"
            else:
                status = "healthy"
                message = f"Memory usage: {percent_used:.1f}%"
            
            return {
                "status": status,
                "message": message,
                "metadata": {
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "percent_used": percent_used
                }
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Failed to check memory usage: {e}"
            }
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            if cpu_percent > 95:
                status = "unhealthy"
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 85:
                status = "degraded"
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = "healthy"
                message = f"CPU usage: {cpu_percent:.1f}%"
            
            return {
                "status": status,
                "message": message,
                "metadata": {
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "load_avg_1m": load_avg[0],
                    "load_avg_5m": load_avg[1],
                    "load_avg_15m": load_avg[2]
                }
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Failed to check CPU usage: {e}"
            }