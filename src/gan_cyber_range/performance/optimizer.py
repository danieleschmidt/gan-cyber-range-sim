"""Performance optimization and auto-scaling."""

import asyncio
import time
import psutil
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics


class OptimizationLevel(Enum):
    """Optimization levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    active_tasks: int
    queue_size: int
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_io_read": self.disk_io_read,
            "disk_io_write": self.disk_io_write,
            "network_io_sent": self.network_io_sent,
            "network_io_recv": self.network_io_recv,
            "active_tasks": self.active_tasks,
            "queue_size": self.queue_size,
            "response_time_ms": self.response_time_ms,
            "throughput_rps": self.throughput_rps,
            "error_rate": self.error_rate
        }


@dataclass
class OptimizationRule:
    """Performance optimization rule."""
    name: str
    condition: Callable[[PerformanceMetrics], bool]
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: int = 100
    cooldown_seconds: int = 60
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    def can_trigger(self) -> bool:
        """Check if rule can be triggered (not in cooldown)."""
        if self.last_triggered is None:
            return True
        
        return datetime.now() - self.last_triggered > timedelta(seconds=self.cooldown_seconds)
    
    def trigger(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger the optimization rule."""
        self.last_triggered = datetime.now()
        self.trigger_count += 1
        return self.action(current_config)


class PerformanceOptimizer:
    """Automatic performance optimization system."""
    
    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
        monitoring_interval: int = 30,
        metrics_history_size: int = 100
    ):
        self.optimization_level = optimization_level
        self.monitoring_interval = monitoring_interval
        self.metrics_history_size = metrics_history_size
        
        # Metrics storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_rules: List[OptimizationRule] = []
        
        # Current configuration
        self.current_config = {
            "max_workers": 10,
            "queue_size": 1000,
            "cache_size": 1000,
            "connection_pool_size": 20,
            "timeout_seconds": 30,
            "retry_attempts": 3
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Setup default optimization rules
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default optimization rules."""
        # CPU-based scaling rules
        self.add_rule(
            name="scale_up_cpu_high",
            condition=lambda m: m.cpu_percent > 80,
            action=lambda c: {**c, "max_workers": min(c["max_workers"] * 2, 50)},
            priority=90,
            cooldown_seconds=120
        )
        
        self.add_rule(
            name="scale_down_cpu_low",
            condition=lambda m: m.cpu_percent < 20,
            action=lambda c: {**c, "max_workers": max(c["max_workers"] // 2, 2)},
            priority=80,
            cooldown_seconds=300
        )
        
        # Memory-based optimization
        self.add_rule(
            name="reduce_cache_memory_high",
            condition=lambda m: m.memory_percent > 85,
            action=lambda c: {**c, "cache_size": max(c["cache_size"] // 2, 100)},
            priority=95,
            cooldown_seconds=60
        )
        
        # Queue-based scaling
        self.add_rule(
            name="increase_workers_queue_full",
            condition=lambda m: m.queue_size > 800 and m.active_tasks < m.queue_size * 0.1,
            action=lambda c: {**c, "max_workers": min(c["max_workers"] + 5, 50)},
            priority=85,
            cooldown_seconds=60
        )
        
        # Response time optimization
        self.add_rule(
            name="optimize_timeout_slow_response",
            condition=lambda m: m.response_time_ms > 5000,
            action=lambda c: {**c, "timeout_seconds": min(c["timeout_seconds"] + 10, 120)},
            priority=70,
            cooldown_seconds=180
        )
        
        # Error rate optimization
        self.add_rule(
            name="increase_retries_high_error_rate",
            condition=lambda m: m.error_rate > 0.1,
            action=lambda c: {**c, "retry_attempts": min(c["retry_attempts"] + 1, 10)},
            priority=75,
            cooldown_seconds=300
        )
    
    def add_rule(
        self,
        name: str,
        condition: Callable[[PerformanceMetrics], bool],
        action: Callable[[Dict[str, Any]], Dict[str, Any]],
        priority: int = 100,
        cooldown_seconds: int = 60
    ) -> None:
        """Add optimization rule."""
        rule = OptimizationRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            cooldown_seconds=cooldown_seconds
        )
        
        self.optimization_rules.append(rule)
        # Sort by priority (higher first)
        self.optimization_rules.sort(key=lambda r: r.priority, reverse=True)
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring and optimization."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                self._store_metrics(metrics)
                
                # Run optimization
                await self._optimize_performance(metrics)
                
                # Wait for next cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Application metrics (would be provided by actual components)
        active_tasks = 0  # Would get from task pool
        queue_size = 0    # Would get from task queue
        response_time_ms = 100.0  # Would calculate from request timings
        throughput_rps = 10.0     # Would calculate from request counts
        error_rate = 0.01         # Would calculate from error counts
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            network_io_sent=network_io.bytes_sent if network_io else 0,
            network_io_recv=network_io.bytes_recv if network_io else 0,
            active_tasks=active_tasks,
            queue_size=queue_size,
            response_time_ms=response_time_ms,
            throughput_rps=throughput_rps,
            error_rate=error_rate
        )
    
    def _store_metrics(self, metrics: PerformanceMetrics) -> None:
        """Store metrics in history."""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.metrics_history_size:
            self.metrics_history = self.metrics_history[-self.metrics_history_size:]
    
    async def _optimize_performance(self, current_metrics: PerformanceMetrics) -> None:
        """Run optimization based on current metrics."""
        original_config = self.current_config.copy()
        
        # Apply optimization rules
        for rule in self.optimization_rules:
            if not rule.can_trigger():
                continue
            
            if rule.condition(current_metrics):
                new_config = rule.trigger(self.current_config)
                
                # Apply configuration change
                await self._apply_configuration(new_config)
                
                print(f"Optimization rule '{rule.name}' triggered")
                print(f"Config change: {original_config} -> {new_config}")
                
                # Only apply one rule per cycle
                break
    
    async def _apply_configuration(self, new_config: Dict[str, Any]) -> None:
        """Apply new configuration."""
        # This would integrate with actual components to apply changes
        self.current_config = new_config
        
        # In a real implementation, this would:
        # - Update task pool settings
        # - Resize caches
        # - Adjust connection pools
        # - Update timeout values
        # etc.
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        avg_cpu = statistics.mean(m.cpu_percent for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_percent for m in recent_metrics)
        avg_response_time = statistics.mean(m.response_time_ms for m in recent_metrics)
        avg_throughput = statistics.mean(m.throughput_rps for m in recent_metrics)
        avg_error_rate = statistics.mean(m.error_rate for m in recent_metrics)
        
        # Calculate trends
        if len(recent_metrics) >= 2:
            cpu_trend = recent_metrics[-1].cpu_percent - recent_metrics[0].cpu_percent
            memory_trend = recent_metrics[-1].memory_percent - recent_metrics[0].memory_percent
            response_time_trend = recent_metrics[-1].response_time_ms - recent_metrics[0].response_time_ms
        else:
            cpu_trend = memory_trend = response_time_trend = 0
        
        return {
            "current": recent_metrics[-1].to_dict(),
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "response_time_ms": avg_response_time,
                "throughput_rps": avg_throughput,
                "error_rate": avg_error_rate
            },
            "trends": {
                "cpu_trend": cpu_trend,
                "memory_trend": memory_trend,
                "response_time_trend": response_time_trend
            },
            "configuration": self.current_config,
            "optimization_level": self.optimization_level.value,
            "monitoring_active": self.monitoring_active
        }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization rule trigger history."""
        history = []
        
        for rule in self.optimization_rules:
            if rule.trigger_count > 0:
                history.append({
                    "name": rule.name,
                    "priority": rule.priority,
                    "trigger_count": rule.trigger_count,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
                    "cooldown_seconds": rule.cooldown_seconds
                })
        
        return sorted(history, key=lambda x: x["trigger_count"], reverse=True)
    
    def predict_scaling_needs(self) -> Dict[str, Any]:
        """Predict future scaling needs based on historical data."""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_metrics = self.metrics_history[-20:]  # Last 20 measurements
        
        # Calculate growth rates
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        throughput_values = [m.throughput_rps for m in recent_metrics]
        
        # Simple linear trend analysis
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            x = list(range(len(values)))
            y = values
            n = len(values)
            
            # Linear regression slope
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            return slope
        
        cpu_trend = calculate_trend(cpu_values)
        memory_trend = calculate_trend(memory_values)
        throughput_trend = calculate_trend(throughput_values)
        
        # Predict values in next 10 cycles
        prediction_horizon = 10
        predicted_cpu = cpu_values[-1] + (cpu_trend * prediction_horizon)
        predicted_memory = memory_values[-1] + (memory_trend * prediction_horizon)
        predicted_throughput = throughput_values[-1] + (throughput_trend * prediction_horizon)
        
        # Generate recommendations
        recommendations = []
        
        if predicted_cpu > 70:
            recommendations.append({
                "action": "increase_workers",
                "reason": f"CPU predicted to reach {predicted_cpu:.1f}%",
                "urgency": "high" if predicted_cpu > 80 else "medium"
            })
        
        if predicted_memory > 80:
            recommendations.append({
                "action": "reduce_cache_size",
                "reason": f"Memory predicted to reach {predicted_memory:.1f}%",
                "urgency": "high" if predicted_memory > 90 else "medium"
            })
        
        if throughput_trend > 0 and predicted_throughput > throughput_values[-1] * 1.5:
            recommendations.append({
                "action": "scale_infrastructure",
                "reason": f"Throughput growing rapidly: {throughput_trend:.2f} RPS/cycle",
                "urgency": "medium"
            })
        
        return {
            "predictions": {
                "cpu_percent": predicted_cpu,
                "memory_percent": predicted_memory,
                "throughput_rps": predicted_throughput
            },
            "trends": {
                "cpu_trend": cpu_trend,
                "memory_trend": memory_trend,
                "throughput_trend": throughput_trend
            },
            "recommendations": recommendations,
            "confidence": min(len(recent_metrics) / 20.0, 1.0)  # Confidence based on data points
        }
    
    def create_custom_rule(
        self,
        name: str,
        metric_threshold: Dict[str, Any],
        action_config: Dict[str, Any],
        priority: int = 50
    ) -> None:
        """Create custom optimization rule."""
        def condition(metrics: PerformanceMetrics) -> bool:
            for metric_name, threshold in metric_threshold.items():
                metric_value = getattr(metrics, metric_name, 0)
                operator = threshold.get("operator", "greater_than")
                value = threshold.get("value", 0)
                
                if operator == "greater_than" and metric_value <= value:
                    return False
                elif operator == "less_than" and metric_value >= value:
                    return False
                elif operator == "equals" and metric_value != value:
                    return False
            
            return True
        
        def action(current_config: Dict[str, Any]) -> Dict[str, Any]:
            new_config = current_config.copy()
            for config_key, change in action_config.items():
                if config_key in new_config:
                    if change.get("operation") == "multiply":
                        new_config[config_key] = int(new_config[config_key] * change["value"])
                    elif change.get("operation") == "add":
                        new_config[config_key] = new_config[config_key] + change["value"]
                    elif change.get("operation") == "set":
                        new_config[config_key] = change["value"]
            
            return new_config
        
        self.add_rule(name, condition, action, priority)