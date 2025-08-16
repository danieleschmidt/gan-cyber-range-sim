"""Intelligent auto-scaling system for cyber range components."""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import statistics


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Scaling trigger types."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    REQUEST_RATE = "request_rate"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    request_rate_per_second: float
    queue_length: int
    response_time_ms: float
    active_connections: int
    custom_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "request_rate_per_second": self.request_rate_per_second,
            "queue_length": self.queue_length,
            "response_time_ms": self.response_time_ms,
            "active_connections": self.active_connections,
            "custom_metrics": self.custom_metrics
        }


@dataclass
class ScalingPolicy:
    """Configuration for scaling behavior."""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown_seconds: int = 300  # 5 minutes
    scale_down_cooldown_seconds: int = 600  # 10 minutes
    metrics_window_minutes: int = 5
    scale_up_factor: float = 1.5  # Scale up by 50%
    scale_down_factor: float = 0.7  # Scale down by 30%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_cpu_percent": self.target_cpu_percent,
            "target_memory_percent": self.target_memory_percent,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "scale_up_cooldown_seconds": self.scale_up_cooldown_seconds,
            "scale_down_cooldown_seconds": self.scale_down_cooldown_seconds,
            "metrics_window_minutes": self.metrics_window_minutes,
            "scale_up_factor": self.scale_up_factor,
            "scale_down_factor": self.scale_down_factor
        }


class AutoScaler:
    """Intelligent auto-scaling engine."""
    
    def __init__(
        self,
        component_name: str,
        scaling_policy: ScalingPolicy,
        metrics_collector: Callable[[], ScalingMetrics],
        scaler_executor: Callable[[int], bool]  # Function to execute scaling
    ):
        self.component_name = component_name
        self.policy = scaling_policy
        self.metrics_collector = metrics_collector
        self.scaler_executor = scaler_executor
        
        self.current_replicas = scaling_policy.min_replicas
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_history: List[Dict[str, Any]] = []
        self.last_scale_up = None
        self.last_scale_down = None
        
        self.running = False
        self.logger = logging.getLogger(f"autoscaler.{component_name}")
        
    async def start(self, check_interval_seconds: int = 30):
        """Start auto-scaling monitoring."""
        self.running = True
        self.logger.info(f"Starting auto-scaler for {self.component_name}")
        
        while self.running:
            try:
                await self._scaling_cycle()
                await asyncio.sleep(check_interval_seconds)
            except Exception as e:
                self.logger.error(f"Auto-scaling cycle error: {e}")
                await asyncio.sleep(10)  # Brief pause on error
    
    def stop(self):
        """Stop auto-scaling."""
        self.running = False
        self.logger.info(f"Stopping auto-scaler for {self.component_name}")
    
    async def _scaling_cycle(self):
        """Execute one scaling evaluation cycle."""
        # Collect current metrics
        try:
            current_metrics = self.metrics_collector()
            self.metrics_history.append(current_metrics)
            
            # Keep only recent metrics
            cutoff_time = datetime.utcnow() - timedelta(
                minutes=self.policy.metrics_window_minutes * 2
            )
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return
        
        # Analyze metrics for scaling decision
        scaling_decision = self._analyze_scaling_need()
        
        if scaling_decision["action"] != ScalingDirection.STABLE:
            await self._execute_scaling(scaling_decision)
    
    def _analyze_scaling_need(self) -> Dict[str, Any]:
        """Analyze metrics to determine scaling need."""
        if len(self.metrics_history) < 2:
            return {"action": ScalingDirection.STABLE, "reason": "Insufficient metrics"}
        
        # Get recent metrics for analysis
        window_start = datetime.utcnow() - timedelta(
            minutes=self.policy.metrics_window_minutes
        )
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= window_start
        ]
        
        if not recent_metrics:
            return {"action": ScalingDirection.STABLE, "reason": "No recent metrics"}
        
        # Calculate average metrics over window
        avg_cpu = statistics.mean(m.cpu_usage_percent for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_usage_percent for m in recent_metrics)
        avg_response_time = statistics.mean(m.response_time_ms for m in recent_metrics)
        avg_queue_length = statistics.mean(m.queue_length for m in recent_metrics)
        
        # Check cooldown periods
        now = datetime.utcnow()
        if self.last_scale_up:
            time_since_scale_up = (now - self.last_scale_up).total_seconds()
            if time_since_scale_up < self.policy.scale_up_cooldown_seconds:
                return {
                    "action": ScalingDirection.STABLE,
                    "reason": f"Scale-up cooldown active ({time_since_scale_up:.0f}s remaining)"
                }
        
        if self.last_scale_down:
            time_since_scale_down = (now - self.last_scale_down).total_seconds()
            if time_since_scale_down < self.policy.scale_down_cooldown_seconds:
                return {
                    "action": ScalingDirection.STABLE,
                    "reason": f"Scale-down cooldown active ({time_since_scale_down:.0f}s remaining)"
                }
        
        # Evaluate scaling triggers
        scale_up_reasons = []
        scale_down_reasons = []
        
        if avg_cpu > self.policy.scale_up_threshold:
            scale_up_reasons.append(f"High CPU usage: {avg_cpu:.1f}%")
        elif avg_cpu < self.policy.scale_down_threshold:
            scale_down_reasons.append(f"Low CPU usage: {avg_cpu:.1f}%")
            
        if avg_memory > self.policy.scale_up_threshold:
            scale_up_reasons.append(f"High memory usage: {avg_memory:.1f}%")
        elif avg_memory < self.policy.scale_down_threshold:
            scale_down_reasons.append(f"Low memory usage: {avg_memory:.1f}%")
        
        if avg_response_time > 5000:  # > 5 seconds
            scale_up_reasons.append(f"High response time: {avg_response_time:.0f}ms")
        
        if avg_queue_length > 100:
            scale_up_reasons.append(f"High queue length: {avg_queue_length:.0f}")
        
        # Make scaling decision
        if scale_up_reasons and self.current_replicas < self.policy.max_replicas:
            target_replicas = min(
                int(self.current_replicas * self.policy.scale_up_factor),
                self.policy.max_replicas
            )
            return {
                "action": ScalingDirection.UP,
                "target_replicas": target_replicas,
                "current_replicas": self.current_replicas,
                "reasons": scale_up_reasons,
                "metrics": {
                    "avg_cpu": avg_cpu,
                    "avg_memory": avg_memory,
                    "avg_response_time": avg_response_time,
                    "avg_queue_length": avg_queue_length
                }
            }
        
        elif scale_down_reasons and self.current_replicas > self.policy.min_replicas:
            # Only scale down if ALL metrics suggest it
            if len(scale_down_reasons) >= 2:  # At least 2 metrics suggest scale down
                target_replicas = max(
                    int(self.current_replicas * self.policy.scale_down_factor),
                    self.policy.min_replicas
                )
                return {
                    "action": ScalingDirection.DOWN,
                    "target_replicas": target_replicas,
                    "current_replicas": self.current_replicas,
                    "reasons": scale_down_reasons,
                    "metrics": {
                        "avg_cpu": avg_cpu,
                        "avg_memory": avg_memory,
                        "avg_response_time": avg_response_time,
                        "avg_queue_length": avg_queue_length
                    }
                }
        
        return {
            "action": ScalingDirection.STABLE,
            "reason": "Metrics within acceptable range",
            "metrics": {
                "avg_cpu": avg_cpu,
                "avg_memory": avg_memory,
                "avg_response_time": avg_response_time,
                "avg_queue_length": avg_queue_length
            }
        }
    
    async def _execute_scaling(self, decision: Dict[str, Any]):
        """Execute scaling action."""
        action = decision["action"]
        target_replicas = decision.get("target_replicas", self.current_replicas)
        
        self.logger.info(
            f"Scaling {action.value}: {self.current_replicas} -> {target_replicas} "
            f"replicas. Reasons: {', '.join(decision.get('reasons', []))}"
        )
        
        try:
            # Execute scaling through provided function
            success = self.scaler_executor(target_replicas)
            
            if success:
                # Update state
                old_replicas = self.current_replicas
                self.current_replicas = target_replicas
                
                if action == ScalingDirection.UP:
                    self.last_scale_up = datetime.utcnow()
                else:
                    self.last_scale_down = datetime.utcnow()
                
                # Record scaling event
                scaling_event = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": action.value,
                    "old_replicas": old_replicas,
                    "new_replicas": target_replicas,
                    "reasons": decision.get("reasons", []),
                    "metrics": decision.get("metrics", {}),
                    "success": True
                }
                self.scaling_history.append(scaling_event)
                
                self.logger.info(f"Scaling completed successfully")
            else:
                self.logger.error("Scaling execution failed")
                scaling_event = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": action.value,
                    "old_replicas": self.current_replicas,
                    "target_replicas": target_replicas,
                    "success": False,
                    "error": "Scaling execution failed"
                }
                self.scaling_history.append(scaling_event)
                
        except Exception as e:
            self.logger.error(f"Scaling execution error: {e}")
            scaling_event = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action.value,
                "error": str(e),
                "success": False
            }
            self.scaling_history.append(scaling_event)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current auto-scaler status."""
        recent_metrics = []
        if self.metrics_history:
            # Get last 10 metrics
            recent_metrics = [
                m.to_dict() for m in self.metrics_history[-10:]
            ]
        
        recent_scaling = []
        if self.scaling_history:
            # Get last 10 scaling events
            recent_scaling = self.scaling_history[-10:]
        
        return {
            "component_name": self.component_name,
            "running": self.running,
            "current_replicas": self.current_replicas,
            "policy": self.policy.to_dict(),
            "recent_metrics": recent_metrics,
            "recent_scaling_events": recent_scaling,
            "last_scale_up": self.last_scale_up.isoformat() if self.last_scale_up else None,
            "last_scale_down": self.last_scale_down.isoformat() if self.last_scale_down else None
        }


class HorizontalPodAutoscaler:
    """Kubernetes-specific horizontal pod autoscaler."""
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.auto_scalers: Dict[str, AutoScaler] = {}
        self.logger = logging.getLogger("k8s_hpa")
    
    def register_deployment(
        self,
        deployment_name: str,
        scaling_policy: ScalingPolicy,
        metrics_collector: Callable[[], ScalingMetrics]
    ):
        """Register a deployment for auto-scaling."""
        
        def k8s_scaler(target_replicas: int) -> bool:
            """Execute Kubernetes scaling - placeholder implementation."""
            try:
                # This would use kubernetes client to scale deployment
                self.logger.info(
                    f"Scaling K8s deployment {deployment_name} to {target_replicas} replicas"
                )
                # kubectl scale deployment deployment_name --replicas=target_replicas
                return True
            except Exception as e:
                self.logger.error(f"K8s scaling failed: {e}")
                return False
        
        auto_scaler = AutoScaler(
            component_name=deployment_name,
            scaling_policy=scaling_policy,
            metrics_collector=metrics_collector,
            scaler_executor=k8s_scaler
        )
        
        self.auto_scalers[deployment_name] = auto_scaler
        return auto_scaler
    
    async def start_all(self, check_interval_seconds: int = 30):
        """Start all registered auto-scalers."""
        if not self.auto_scalers:
            self.logger.warning("No auto-scalers registered")
            return
        
        tasks = [
            scaler.start(check_interval_seconds)
            for scaler in self.auto_scalers.values()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop_all(self):
        """Stop all auto-scalers."""
        for scaler in self.auto_scalers.values():
            scaler.stop()
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of all auto-scalers."""
        return {
            "namespace": self.namespace,
            "auto_scalers": {
                name: scaler.get_status()
                for name, scaler in self.auto_scalers.items()
            }
        }


# Example usage and testing
async def example_usage():
    """Example of how to use the auto-scaler."""
    
    # Define scaling policy
    policy = ScalingPolicy(
        min_replicas=2,
        max_replicas=20,
        target_cpu_percent=70.0,
        scale_up_threshold=80.0,
        scale_down_threshold=30.0
    )
    
    # Mock metrics collector
    def collect_metrics() -> ScalingMetrics:
        import random
        return ScalingMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=random.uniform(20, 90),
            memory_usage_percent=random.uniform(30, 85),
            request_rate_per_second=random.uniform(10, 1000),
            queue_length=random.randint(0, 200),
            response_time_ms=random.uniform(100, 8000),
            active_connections=random.randint(50, 500),
            custom_metrics={}
        )
    
    # Mock scaling executor
    def execute_scaling(target_replicas: int) -> bool:
        print(f"Scaling to {target_replicas} replicas")
        return True
    
    # Create and start auto-scaler
    scaler = AutoScaler(
        component_name="api-gateway",
        scaling_policy=policy,
        metrics_collector=collect_metrics,
        scaler_executor=execute_scaling
    )
    
    # Run for 5 minutes as example
    await asyncio.wait_for(scaler.start(check_interval_seconds=10), timeout=300)


if __name__ == "__main__":
    asyncio.run(example_usage())