"""Auto-scaling system for quality gates based on load and performance."""

import asyncio
import logging
import time
import psutil
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque, defaultdict
import json

from .monitoring import PerformanceSnapshot, MetricsCollector
from ..core.error_handling import CyberRangeError, ErrorSeverity


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingStrategy(Enum):
    """Scaling strategies."""
    REACTIVE = "reactive"          # React to current load
    PREDICTIVE = "predictive"      # Predict future load
    ADAPTIVE = "adaptive"          # Learn from patterns
    CONSERVATIVE = "conservative"   # Scale slowly and safely
    AGGRESSIVE = "aggressive"      # Scale quickly


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    cpu_utilization: float
    memory_utilization: float
    queue_length: int
    response_time: float
    throughput: float
    error_rate: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "queue_length": self.queue_length,
            "response_time": self.response_time,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: datetime
    direction: ScalingDirection
    trigger_reason: str
    from_instances: int
    to_instances: int
    metrics_before: ScalingMetrics
    metrics_after: Optional[ScalingMetrics] = None
    success: bool = True
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "trigger_reason": self.trigger_reason,
            "from_instances": self.from_instances,
            "to_instances": self.to_instances,
            "metrics_before": self.metrics_before.to_dict(),
            "metrics_after": self.metrics_after.to_dict() if self.metrics_after else None,
            "success": self.success,
            "duration": self.duration
        }


class LoadPredictor:
    """Predicts future load based on historical patterns."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.patterns: Dict[str, List[float]] = defaultdict(list)
        self.logger = logging.getLogger("load_predictor")
    
    def record_metrics(self, metrics: ScalingMetrics):
        """Record metrics for prediction."""
        self.metrics_history.append(metrics)
        
        # Extract patterns by hour of day
        hour = metrics.timestamp.hour
        self.patterns[f"hour_{hour}"].append(metrics.cpu_utilization)
        
        # Extract patterns by day of week
        day = metrics.timestamp.weekday()
        self.patterns[f"day_{day}"].append(metrics.cpu_utilization)
        
        # Keep pattern history manageable
        for key in self.patterns:
            if len(self.patterns[key]) > 100:
                self.patterns[key] = self.patterns[key][-100:]
    
    def predict_load(self, minutes_ahead: int = 15) -> Tuple[float, float]:
        """Predict CPU and memory load for specified minutes ahead."""
        if len(self.metrics_history) < 10:
            # Not enough data, return current metrics
            if self.metrics_history:
                latest = self.metrics_history[-1]
                return latest.cpu_utilization, latest.memory_utilization
            return 50.0, 50.0
        
        now = datetime.now()
        future_time = now + timedelta(minutes=minutes_ahead)
        
        # Get pattern for predicted time
        future_hour = future_time.hour
        future_day = future_time.weekday()
        
        # Predict based on patterns
        hour_pattern = self.patterns.get(f"hour_{future_hour}", [])
        day_pattern = self.patterns.get(f"day_{future_day}", [])
        
        # Calculate trend from recent data
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 data points
        if len(recent_metrics) >= 2:
            recent_cpu = [m.cpu_utilization for m in recent_metrics]
            recent_memory = [m.memory_utilization for m in recent_metrics]
            
            # Simple linear trend
            cpu_trend = (recent_cpu[-1] - recent_cpu[0]) / len(recent_cpu)
            memory_trend = (recent_memory[-1] - recent_memory[0]) / len(recent_memory)
        else:
            cpu_trend = 0
            memory_trend = 0
        
        # Combine pattern-based and trend-based predictions
        if hour_pattern:
            predicted_cpu = statistics.mean(hour_pattern) + (cpu_trend * minutes_ahead)
        else:
            predicted_cpu = recent_metrics[-1].cpu_utilization + (cpu_trend * minutes_ahead)
        
        if day_pattern:
            predicted_memory = statistics.mean([m.memory_utilization for m in recent_metrics])
        else:
            predicted_memory = recent_metrics[-1].memory_utilization + (memory_trend * minutes_ahead)
        
        # Clamp predictions to reasonable bounds
        predicted_cpu = max(0.0, min(100.0, predicted_cpu))
        predicted_memory = max(0.0, min(100.0, predicted_memory))
        
        return predicted_cpu, predicted_memory
    
    def get_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy based on historical data."""
        if len(self.metrics_history) < 30:
            return 0.5  # Default accuracy
        
        # Compare predictions with actual values
        accuracies = []
        metrics_list = list(self.metrics_history)
        
        for i in range(15, len(metrics_list) - 15):  # Test on middle portion
            # Use data up to point i to predict i+15
            historical_data = metrics_list[:i]
            actual_metrics = metrics_list[i + 15]
            
            # Simple prediction based on trend
            if len(historical_data) >= 10:
                recent_cpu = [m.cpu_utilization for m in historical_data[-10:]]
                trend = (recent_cpu[-1] - recent_cpu[0]) / len(recent_cpu)
                predicted = recent_cpu[-1] + (trend * 15)
                
                # Calculate accuracy (percentage error)
                error = abs(predicted - actual_metrics.cpu_utilization) / max(actual_metrics.cpu_utilization, 1.0)
                accuracy = max(0.0, 1.0 - error)
                accuracies.append(accuracy)
        
        return statistics.mean(accuracies) if accuracies else 0.5


class QualityGateAutoScaler:
    """Auto-scaling system for quality gate execution."""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.strategy = strategy
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger("quality_gate_autoscaler")
        
        # Scaling configuration
        self.scale_up_threshold = 80.0    # CPU/Memory threshold to scale up
        self.scale_down_threshold = 30.0  # CPU/Memory threshold to scale down
        self.scale_up_cooldown = 300      # 5 minutes
        self.scale_down_cooldown = 600    # 10 minutes
        
        # Scaling history and state
        self.scaling_events: deque = deque(maxlen=100)
        self.last_scale_time = datetime.min
        self.load_predictor = LoadPredictor()
        
        # Performance tracking
        self.current_metrics = ScalingMetrics(
            cpu_utilization=0.0,
            memory_utilization=0.0,
            queue_length=0,
            response_time=0.0,
            throughput=0.0,
            error_rate=0.0,
            timestamp=datetime.now()
        )
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_active = False
        
        # Scaling thresholds based on strategy
        self._configure_strategy()
    
    def _configure_strategy(self):
        """Configure scaling parameters based on strategy."""
        if self.strategy == ScalingStrategy.CONSERVATIVE:
            self.scale_up_threshold = 85.0
            self.scale_down_threshold = 20.0
            self.scale_up_cooldown = 600    # 10 minutes
            self.scale_down_cooldown = 900  # 15 minutes
        elif self.strategy == ScalingStrategy.AGGRESSIVE:
            self.scale_up_threshold = 70.0
            self.scale_down_threshold = 40.0
            self.scale_up_cooldown = 180    # 3 minutes
            self.scale_down_cooldown = 300  # 5 minutes
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            self.scale_up_threshold = 75.0
            self.scale_down_threshold = 35.0
            self.scale_up_cooldown = 240    # 4 minutes
            self.scale_down_cooldown = 480  # 8 minutes
        # REACTIVE and ADAPTIVE use default values
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start auto-scaling monitoring."""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self.logger.info(f"Starting auto-scaling monitoring (strategy: {self.strategy.value})")
        
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
    
    async def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Auto-scaling monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect current metrics
                current_metrics = await self._collect_metrics()
                self.current_metrics = current_metrics
                
                # Record for prediction
                self.load_predictor.record_metrics(current_metrics)
                
                # Make scaling decision
                scaling_decision = await self._make_scaling_decision(current_metrics)
                
                # Execute scaling if needed
                if scaling_decision != ScalingDirection.STABLE:
                    await self._execute_scaling(scaling_decision, current_metrics)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory from psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Simulated queue length and response time
            # In a real implementation, these would come from the quality gate system
            queue_length = 0  # Would be actual queue size
            response_time = 0.0  # Would be actual response time
            throughput = 0.0  # Would be actual throughput
            error_rate = 0.0  # Would be actual error rate
            
            # If metrics collector is available, get more detailed info
            if self.metrics_collector:
                # Get recent performance trend
                trend = self.metrics_collector.get_performance_trend(5)  # Last 5 minutes
                if trend:
                    cpu_percent = trend.get("cpu", {}).get("current", cpu_percent)
                    memory_percent = trend.get("memory", {}).get("current", memory_percent)
            
            return ScalingMetrics(
                cpu_utilization=cpu_percent,
                memory_utilization=memory_percent,
                queue_length=queue_length,
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to collect metrics: {e}")
            return ScalingMetrics(
                cpu_utilization=0.0,
                memory_utilization=0.0,
                queue_length=0,
                response_time=0.0,
                throughput=0.0,
                error_rate=0.0,
                timestamp=datetime.now()
            )
    
    async def _make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDirection:
        """Make scaling decision based on current metrics and strategy."""
        now = datetime.now()
        
        # Check cooldown periods
        time_since_last_scale = (now - self.last_scale_time).total_seconds()
        
        # Determine if we should consider scaling
        can_scale_up = (
            time_since_last_scale >= self.scale_up_cooldown and
            self.current_instances < self.max_instances
        )
        can_scale_down = (
            time_since_last_scale >= self.scale_down_cooldown and
            self.current_instances > self.min_instances
        )
        
        # Strategy-specific decision logic
        if self.strategy == ScalingStrategy.PREDICTIVE:
            return await self._predictive_scaling_decision(metrics, can_scale_up, can_scale_down)
        elif self.strategy == ScalingStrategy.ADAPTIVE:
            return await self._adaptive_scaling_decision(metrics, can_scale_up, can_scale_down)
        else:
            return await self._reactive_scaling_decision(metrics, can_scale_up, can_scale_down)
    
    async def _reactive_scaling_decision(
        self,
        metrics: ScalingMetrics,
        can_scale_up: bool,
        can_scale_down: bool
    ) -> ScalingDirection:
        """Reactive scaling based on current metrics."""
        cpu_util = metrics.cpu_utilization
        memory_util = metrics.memory_utilization
        
        # Scale up if CPU or memory is high
        if can_scale_up and (cpu_util > self.scale_up_threshold or memory_util > self.scale_up_threshold):
            self.logger.info(f"Reactive scale up: CPU={cpu_util:.1f}%, Memory={memory_util:.1f}%")
            return ScalingDirection.UP
        
        # Scale down if both CPU and memory are low
        if can_scale_down and cpu_util < self.scale_down_threshold and memory_util < self.scale_down_threshold:
            self.logger.info(f"Reactive scale down: CPU={cpu_util:.1f}%, Memory={memory_util:.1f}%")
            return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    async def _predictive_scaling_decision(
        self,
        metrics: ScalingMetrics,
        can_scale_up: bool,
        can_scale_down: bool
    ) -> ScalingDirection:
        """Predictive scaling based on forecasted load."""
        current_cpu = metrics.cpu_utilization
        current_memory = metrics.memory_utilization
        
        # Predict load 15 minutes ahead
        predicted_cpu, predicted_memory = self.load_predictor.predict_load(15)
        
        # Factor in prediction accuracy
        accuracy = self.load_predictor.get_prediction_accuracy()
        confidence_threshold = 0.7
        
        if accuracy > confidence_threshold:
            # Use predictions with high confidence
            if can_scale_up and (predicted_cpu > self.scale_up_threshold or predicted_memory > self.scale_up_threshold):
                self.logger.info(
                    f"Predictive scale up: Current CPU={current_cpu:.1f}%, "
                    f"Predicted CPU={predicted_cpu:.1f}% (accuracy={accuracy:.2f})"
                )
                return ScalingDirection.UP
            
            if can_scale_down and predicted_cpu < self.scale_down_threshold and predicted_memory < self.scale_down_threshold:
                self.logger.info(
                    f"Predictive scale down: Current CPU={current_cpu:.1f}%, "
                    f"Predicted CPU={predicted_cpu:.1f}% (accuracy={accuracy:.2f})"
                )
                return ScalingDirection.DOWN
        else:
            # Fall back to reactive scaling with low confidence
            return await self._reactive_scaling_decision(metrics, can_scale_up, can_scale_down)
        
        return ScalingDirection.STABLE
    
    async def _adaptive_scaling_decision(
        self,
        metrics: ScalingMetrics,
        can_scale_up: bool,
        can_scale_down: bool
    ) -> ScalingDirection:
        """Adaptive scaling that learns from past scaling events."""
        # Analyze recent scaling events to adapt thresholds
        recent_events = [e for e in self.scaling_events if 
                        (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        # Adjust thresholds based on recent success/failure patterns
        scale_up_threshold = self.scale_up_threshold
        scale_down_threshold = self.scale_down_threshold
        
        if recent_events:
            successful_scale_ups = [e for e in recent_events 
                                   if e.direction == ScalingDirection.UP and e.success]
            unsuccessful_scale_ups = [e for e in recent_events 
                                     if e.direction == ScalingDirection.UP and not e.success]
            
            # If recent scale-ups were unsuccessful, raise threshold
            if len(unsuccessful_scale_ups) > len(successful_scale_ups):
                scale_up_threshold += 5.0
                self.logger.debug("Adapting: Raising scale-up threshold due to recent failures")
            
            # If we're scaling up too frequently, raise threshold
            if len([e for e in recent_events if e.direction == ScalingDirection.UP]) > 2:
                scale_up_threshold += 2.0
                self.logger.debug("Adapting: Raising scale-up threshold due to frequent scaling")
        
        # Apply adaptive thresholds
        cpu_util = metrics.cpu_utilization
        memory_util = metrics.memory_utilization
        
        if can_scale_up and (cpu_util > scale_up_threshold or memory_util > scale_up_threshold):
            self.logger.info(f"Adaptive scale up: CPU={cpu_util:.1f}%, Memory={memory_util:.1f}% (threshold={scale_up_threshold:.1f})")
            return ScalingDirection.UP
        
        if can_scale_down and cpu_util < scale_down_threshold and memory_util < scale_down_threshold:
            self.logger.info(f"Adaptive scale down: CPU={cpu_util:.1f}%, Memory={memory_util:.1f}%")
            return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    async def _execute_scaling(self, direction: ScalingDirection, metrics: ScalingMetrics):
        """Execute scaling action."""
        start_time = time.time()
        old_instances = self.current_instances
        
        if direction == ScalingDirection.UP:
            new_instances = min(self.current_instances + 1, self.max_instances)
            trigger_reason = f"High resource utilization: CPU={metrics.cpu_utilization:.1f}%, Memory={metrics.memory_utilization:.1f}%"
        else:  # ScalingDirection.DOWN
            new_instances = max(self.current_instances - 1, self.min_instances)
            trigger_reason = f"Low resource utilization: CPU={metrics.cpu_utilization:.1f}%, Memory={metrics.memory_utilization:.1f}%"
        
        if new_instances == old_instances:
            return  # No change needed
        
        # Simulate scaling action
        success = await self._perform_scaling_action(old_instances, new_instances)
        
        execution_time = time.time() - start_time
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=datetime.now(),
            direction=direction,
            trigger_reason=trigger_reason,
            from_instances=old_instances,
            to_instances=new_instances,
            metrics_before=metrics,
            success=success,
            duration=execution_time
        )
        
        if success:
            self.current_instances = new_instances
            self.last_scale_time = datetime.now()
            self.logger.info(f"Scaling {direction.value}: {old_instances} -> {new_instances} instances")
        else:
            self.logger.error(f"Scaling {direction.value} failed: {old_instances} -> {new_instances}")
        
        self.scaling_events.append(event)
        
        # Wait a bit and collect post-scaling metrics
        await asyncio.sleep(30)  # Wait 30 seconds for scaling to take effect
        post_metrics = await self._collect_metrics()
        event.metrics_after = post_metrics
    
    async def _perform_scaling_action(self, from_instances: int, to_instances: int) -> bool:
        """Perform the actual scaling action."""
        try:
            # In a real implementation, this would:
            # 1. Update container orchestrator (Kubernetes, Docker Swarm, etc.)
            # 2. Wait for new instances to be ready
            # 3. Update load balancer configuration
            # 4. Verify scaling success
            
            # Simulate scaling delay
            scaling_delay = abs(to_instances - from_instances) * 5  # 5 seconds per instance
            await asyncio.sleep(min(scaling_delay, 30))  # Cap at 30 seconds
            
            # Simulate 95% success rate
            import random
            return random.random() > 0.05
            
        except Exception as e:
            self.logger.error(f"Scaling action failed: {e}")
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "strategy": self.strategy.value,
            "current_metrics": self.current_metrics.to_dict(),
            "scaling_thresholds": {
                "scale_up": self.scale_up_threshold,
                "scale_down": self.scale_down_threshold
            },
            "cooldown_periods": {
                "scale_up": self.scale_up_cooldown,
                "scale_down": self.scale_down_cooldown
            },
            "last_scale_time": self.last_scale_time.isoformat(),
            "recent_events": len(self.scaling_events),
            "prediction_accuracy": self.load_predictor.get_prediction_accuracy()
        }
    
    def get_scaling_insights(self) -> Dict[str, Any]:
        """Get insights from scaling history."""
        if not self.scaling_events:
            return {"status": "no_data"}
        
        events = list(self.scaling_events)
        
        # Basic statistics
        total_events = len(events)
        successful_events = len([e for e in events if e.success])
        scale_up_events = len([e for e in events if e.direction == ScalingDirection.UP])
        scale_down_events = len([e for e in events if e.direction == ScalingDirection.DOWN])
        
        # Recent performance
        recent_events = [e for e in events if 
                        (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        avg_duration = statistics.mean([e.duration for e in events]) if events else 0
        
        # Scaling efficiency
        efficiency_scores = []
        for event in events:
            if event.metrics_after:
                before_util = max(event.metrics_before.cpu_utilization, event.metrics_before.memory_utilization)
                after_util = max(event.metrics_after.cpu_utilization, event.metrics_after.memory_utilization)
                
                if event.direction == ScalingDirection.UP:
                    # Good if utilization decreased after scaling up
                    efficiency = max(0, (before_util - after_util) / before_util) if before_util > 0 else 0
                else:
                    # Good if utilization didn't increase too much after scaling down
                    efficiency = max(0, 1 - ((after_util - before_util) / 100)) if after_util > before_util else 1
                
                efficiency_scores.append(efficiency)
        
        avg_efficiency = statistics.mean(efficiency_scores) if efficiency_scores else 0
        
        return {
            "status": "success",
            "total_events": total_events,
            "success_rate": successful_events / total_events if total_events > 0 else 0,
            "scale_up_events": scale_up_events,
            "scale_down_events": scale_down_events,
            "recent_events_count": len(recent_events),
            "average_scaling_duration": avg_duration,
            "scaling_efficiency": avg_efficiency,
            "prediction_accuracy": self.load_predictor.get_prediction_accuracy()
        }
    
    def export_scaling_data(self, file_path: str):
        """Export scaling data for analysis."""
        data = {
            "scaling_status": self.get_scaling_status(),
            "scaling_insights": self.get_scaling_insights(),
            "scaling_events": [event.to_dict() for event in self.scaling_events],
            "configuration": {
                "strategy": self.strategy.value,
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "thresholds": {
                    "scale_up": self.scale_up_threshold,
                    "scale_down": self.scale_down_threshold
                }
            }
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"Scaling data exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export scaling data: {e}")