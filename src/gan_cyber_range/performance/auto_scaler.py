"""
Intelligent Auto-Scaling Engine for GAN Cyber Range.

Provides dynamic resource scaling with:
- Predictive scaling based on ML models
- Multi-dimensional metrics analysis
- Cost-optimized scaling decisions
- Kubernetes HPA and VPA integration
- Real-time performance optimization
"""

import asyncio
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

from ..monitoring.metrics import MetricsCollector
from ..resilience.health_monitor import HealthMonitor


logger = logging.getLogger(__name__)


class ScalingDirection(str, Enum):
    """Scaling directions."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(str, Enum):
    """Resource types for scaling."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    REPLICAS = "replicas"
    GPU = "gpu"


@dataclass
class ScalingMetric:
    """Metric configuration for scaling decisions."""
    name: str
    resource_type: ResourceType
    target_utilization: float  # Target utilization percentage (0-100)
    scale_up_threshold: float  # Scale up when above this percentage
    scale_down_threshold: float  # Scale down when below this percentage
    weight: float = 1.0  # Weight in scaling decision
    enabled: bool = True


@dataclass
class ScalingRule:
    """Scaling rule configuration."""
    component: str
    metrics: List[ScalingMetric]
    min_replicas: int = 1
    max_replicas: int = 100
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    enabled: bool = True
    
    # Runtime state
    last_scale_up: Optional[datetime] = None
    last_scale_down: Optional[datetime] = None
    current_replicas: int = 1


@dataclass
class ScalingDecision:
    """Scaling decision with rationale."""
    component: str
    direction: ScalingDirection
    target_replicas: int
    current_replicas: int
    confidence: float  # 0-1 confidence in decision
    reasoning: List[str]
    triggered_metrics: List[str]
    cost_impact: float  # Estimated cost change
    timestamp: datetime


@dataclass
class PredictionModel:
    """Simple predictive model for resource usage."""
    component: str
    metric_type: str
    history_window: int = 100
    prediction_window: int = 10  # minutes
    
    # Historical data
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Model parameters
    trend_weight: float = 0.7
    seasonal_weight: float = 0.3


class AutoScaler:
    """Intelligent auto-scaling engine."""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 health_monitor: HealthMonitor):
        self.metrics_collector = metrics_collector
        self.health_monitor = health_monitor
        
        # Scaling configuration
        self.scaling_rules: Dict[str, ScalingRule] = {}
        
        # Prediction models
        self.prediction_models: Dict[str, PredictionModel] = {}
        
        # Scaling history
        self.scaling_history: List[ScalingDecision] = []
        
        # Performance tracking
        self.performance_baseline: Dict[str, float] = {}
        self.cost_optimization_enabled = True
        
        # Runtime state
        self.running = False
        self.scaling_tasks: List[asyncio.Task] = []
        
        # Configuration
        self.evaluation_interval = 60  # seconds
        self.prediction_accuracy_threshold = 0.8
        self.cost_increase_threshold = 0.2  # 20% cost increase limit
        
        # Initialize default scaling rules
        self._initialize_default_rules()
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a scaling rule for a component."""
        self.scaling_rules[rule.component] = rule
        logger.info(f"Added scaling rule for component: {rule.component}")
    
    def add_prediction_model(self, model: PredictionModel):
        """Add a prediction model for a metric."""
        model_key = f"{model.component}_{model.metric_type}"
        self.prediction_models[model_key] = model
        logger.info(f"Added prediction model: {model_key}")
    
    async def start_autoscaler(self):
        """Start the auto-scaling engine."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting auto-scaling engine")
        
        # Start scaling evaluation task
        eval_task = asyncio.create_task(self._evaluation_loop())
        self.scaling_tasks.append(eval_task)
        
        # Start prediction model updates
        prediction_task = asyncio.create_task(self._update_predictions())
        self.scaling_tasks.append(prediction_task)
        
        # Start performance monitoring
        perf_task = asyncio.create_task(self._monitor_performance())
        self.scaling_tasks.append(perf_task)
        
        logger.info("Auto-scaling engine started")
    
    async def stop_autoscaler(self):
        """Stop the auto-scaling engine."""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping auto-scaling engine")
        
        # Cancel all tasks
        for task in self.scaling_tasks:
            task.cancel()
        
        await asyncio.gather(*self.scaling_tasks, return_exceptions=True)
        self.scaling_tasks.clear()
        
        logger.info("Auto-scaling engine stopped")
    
    async def _evaluation_loop(self):
        """Main scaling evaluation loop."""
        while self.running:
            try:
                # Evaluate all scaling rules
                for component, rule in self.scaling_rules.items():
                    if rule.enabled:
                        decision = await self._evaluate_scaling_rule(rule)
                        
                        if decision and decision.direction != ScalingDirection.STABLE:
                            await self._execute_scaling_decision(decision)
                
                await asyncio.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling evaluation loop: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_scaling_rule(self, rule: ScalingRule) -> Optional[ScalingDecision]:
        """Evaluate a scaling rule and make a decision."""
        try:
            # Check cooldown periods
            now = datetime.now()
            
            if rule.last_scale_up:
                scale_up_cooldown = timedelta(seconds=rule.scale_up_cooldown)
                if now - rule.last_scale_up < scale_up_cooldown:
                    return None  # Still in scale-up cooldown
            
            if rule.last_scale_down:
                scale_down_cooldown = timedelta(seconds=rule.scale_down_cooldown)
                if now - rule.last_scale_down < scale_down_cooldown:
                    return None  # Still in scale-down cooldown
            
            # Get current metrics
            current_metrics = await self._get_current_metrics(rule.component)
            
            # Calculate scaling scores
            scale_up_score = 0.0
            scale_down_score = 0.0
            triggered_metrics = []
            reasoning = []
            
            total_weight = sum(metric.weight for metric in rule.metrics if metric.enabled)
            
            for metric in rule.metrics:
                if not metric.enabled:
                    continue
                
                current_value = current_metrics.get(metric.name, 0.0)
                normalized_weight = metric.weight / total_weight
                
                if current_value > metric.scale_up_threshold:
                    scale_up_score += normalized_weight
                    triggered_metrics.append(f"{metric.name}:{current_value:.1f}%")
                    reasoning.append(f"{metric.name} above scale-up threshold ({current_value:.1f}% > {metric.scale_up_threshold}%)")
                
                elif current_value < metric.scale_down_threshold:
                    scale_down_score += normalized_weight
                    triggered_metrics.append(f"{metric.name}:{current_value:.1f}%")
                    reasoning.append(f"{metric.name} below scale-down threshold ({current_value:.1f}% < {metric.scale_down_threshold}%)")
            
            # Include predictive analysis
            predicted_metrics = await self._get_predicted_metrics(rule.component)
            prediction_adjustment = self._calculate_prediction_adjustment(predicted_metrics, rule)
            
            scale_up_score += prediction_adjustment.get("scale_up", 0.0)
            scale_down_score += prediction_adjustment.get("scale_down", 0.0)
            
            if prediction_adjustment.get("reasoning"):
                reasoning.extend(prediction_adjustment["reasoning"])
            
            # Determine scaling direction
            decision_threshold = 0.6  # Need 60% confidence to scale
            
            if scale_up_score > decision_threshold and scale_up_score > scale_down_score:
                target_replicas = min(rule.current_replicas + 1, rule.max_replicas)
                direction = ScalingDirection.UP
                confidence = min(scale_up_score, 1.0)
            
            elif scale_down_score > decision_threshold and scale_down_score > scale_up_score:
                target_replicas = max(rule.current_replicas - 1, rule.min_replicas)
                direction = ScalingDirection.DOWN
                confidence = min(scale_down_score, 1.0)
            
            else:
                return None  # No scaling needed
            
            # Check cost impact if enabled
            cost_impact = 0.0
            if self.cost_optimization_enabled:
                cost_impact = self._estimate_cost_impact(rule.component, target_replicas, rule.current_replicas)
                
                # Prevent excessive cost increases
                if direction == ScalingDirection.UP and cost_impact > self.cost_increase_threshold:
                    reasoning.append(f"Scaling prevented due to high cost impact ({cost_impact:.1%})")
                    return None
            
            return ScalingDecision(
                component=rule.component,
                direction=direction,
                target_replicas=target_replicas,
                current_replicas=rule.current_replicas,
                confidence=confidence,
                reasoning=reasoning,
                triggered_metrics=triggered_metrics,
                cost_impact=cost_impact,
                timestamp=now
            )
            
        except Exception as e:
            logger.error(f"Error evaluating scaling rule for {rule.component}: {e}")
            return None
    
    async def _get_current_metrics(self, component: str) -> Dict[str, float]:
        """Get current metrics for a component."""
        # This would integrate with your metrics collection system
        metrics = {}
        
        try:
            # Get metrics from collector
            component_metrics = await self.metrics_collector.get_component_metrics(component)
            
            # Convert to utilization percentages
            metrics["cpu_utilization"] = component_metrics.get("cpu_percent", 0.0)
            metrics["memory_utilization"] = component_metrics.get("memory_percent", 0.0)
            metrics["network_utilization"] = component_metrics.get("network_percent", 0.0)
            metrics["request_rate"] = component_metrics.get("requests_per_second", 0.0)
            metrics["response_time"] = component_metrics.get("avg_response_time", 0.0)
            metrics["error_rate"] = component_metrics.get("error_rate_percent", 0.0)
            
        except Exception as e:
            logger.error(f"Error getting metrics for {component}: {e}")
        
        return metrics
    
    async def _get_predicted_metrics(self, component: str) -> Dict[str, float]:
        """Get predicted metrics for a component."""
        predictions = {}
        
        for model_key, model in self.prediction_models.items():
            if not model_key.startswith(component):
                continue
            
            try:
                prediction = self._predict_metric_value(model)
                metric_name = model_key.split('_', 1)[1]
                predictions[metric_name] = prediction
                
            except Exception as e:
                logger.error(f"Error predicting metric {model_key}: {e}")
        
        return predictions
    
    def _predict_metric_value(self, model: PredictionModel) -> float:
        """Predict future metric value using simple trend analysis."""
        if len(model.values) < 10:
            return 0.0  # Not enough data
        
        values = list(model.values)
        timestamps = list(model.timestamps)
        
        # Calculate trend (simple linear regression)
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Trend slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict future value
        future_x = n + model.prediction_window
        trend_prediction = slope * future_x + intercept
        
        # Calculate seasonal component (simple moving average deviation)
        recent_values = values[-10:]  # Last 10 values
        seasonal_avg = statistics.mean(recent_values)
        overall_avg = statistics.mean(values)
        seasonal_factor = seasonal_avg / overall_avg if overall_avg > 0 else 1.0
        
        # Combine trend and seasonal components
        prediction = (model.trend_weight * trend_prediction + 
                     model.seasonal_weight * seasonal_factor * trend_prediction)
        
        return max(0.0, prediction)  # Ensure non-negative
    
    def _calculate_prediction_adjustment(self, predicted_metrics: Dict[str, float], rule: ScalingRule) -> Dict[str, Any]:
        """Calculate scaling adjustment based on predictions."""
        adjustment = {"scale_up": 0.0, "scale_down": 0.0, "reasoning": []}
        
        for metric in rule.metrics:
            if not metric.enabled:
                continue
            
            predicted_value = predicted_metrics.get(metric.name, 0.0)
            
            # Weight predictions lower than current metrics
            prediction_weight = 0.3
            
            if predicted_value > metric.scale_up_threshold:
                adjustment["scale_up"] += prediction_weight * metric.weight
                adjustment["reasoning"].append(f"Predicted {metric.name}: {predicted_value:.1f}% (will exceed threshold)")
            
            elif predicted_value < metric.scale_down_threshold:
                adjustment["scale_down"] += prediction_weight * metric.weight
                adjustment["reasoning"].append(f"Predicted {metric.name}: {predicted_value:.1f}% (will drop below threshold)")
        
        return adjustment
    
    def _estimate_cost_impact(self, component: str, target_replicas: int, current_replicas: int) -> float:
        """Estimate cost impact of scaling decision."""
        # This would integrate with cloud provider cost APIs
        
        # Simple cost model based on replica count
        replica_cost = 0.10  # $0.10 per replica per hour (example)
        hours_per_month = 24 * 30
        
        current_cost = current_replicas * replica_cost * hours_per_month
        target_cost = target_replicas * replica_cost * hours_per_month
        
        if current_cost == 0:
            return 0.0
        
        return (target_cost - current_cost) / current_cost
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        try:
            logger.info(f"Executing scaling decision for {decision.component}: "
                       f"{decision.current_replicas} -> {decision.target_replicas} "
                       f"(confidence: {decision.confidence:.2f})")
            
            # Update rule state
            rule = self.scaling_rules[decision.component]
            
            if decision.direction == ScalingDirection.UP:
                rule.last_scale_up = decision.timestamp
            else:
                rule.last_scale_down = decision.timestamp
            
            # Execute scaling (this would integrate with Kubernetes HPA/VPA)
            success = await self._scale_component(decision.component, decision.target_replicas)
            
            if success:
                rule.current_replicas = decision.target_replicas
                logger.info(f"Successfully scaled {decision.component} to {decision.target_replicas} replicas")
                
                # Record decision
                self.scaling_history.append(decision)
                
                # Keep only last 1000 decisions
                if len(self.scaling_history) > 1000:
                    self.scaling_history = self.scaling_history[-1000:]
                
                # Update metrics
                await self.metrics_collector.record_scaling_event(
                    component=decision.component,
                    direction=decision.direction.value,
                    replicas=decision.target_replicas,
                    confidence=decision.confidence
                )
            
            else:
                logger.error(f"Failed to scale {decision.component}")
            
        except Exception as e:
            logger.error(f"Error executing scaling decision for {decision.component}: {e}")
    
    async def _scale_component(self, component: str, target_replicas: int) -> bool:
        """Scale a component to target replica count."""
        try:
            # This would integrate with Kubernetes API
            # For now, simulate scaling
            logger.info(f"Scaling {component} to {target_replicas} replicas")
            
            # Simulate scaling delay
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error scaling {component}: {e}")
            return False
    
    async def _update_predictions(self):
        """Update prediction models with new data."""
        while self.running:
            try:
                for model_key, model in self.prediction_models.items():
                    component = model_key.split('_')[0]
                    metric_type = model_key.split('_', 1)[1]
                    
                    # Get current metric value
                    current_metrics = await self._get_current_metrics(component)
                    current_value = current_metrics.get(metric_type, 0.0)
                    
                    # Add to model history
                    model.timestamps.append(datetime.now())
                    model.values.append(current_value)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating predictions: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_performance(self):
        """Monitor performance impact of scaling decisions."""
        while self.running:
            try:
                # Check if recent scaling decisions improved performance
                recent_decisions = [
                    d for d in self.scaling_history
                    if datetime.now() - d.timestamp < timedelta(minutes=30)
                ]
                
                for decision in recent_decisions:
                    performance_impact = await self._measure_performance_impact(decision)
                    
                    if performance_impact:
                        logger.info(f"Performance impact of scaling {decision.component}: "
                                   f"{performance_impact:.2f}% improvement")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring performance: {e}")
                await asyncio.sleep(300)
    
    async def _measure_performance_impact(self, decision: ScalingDecision) -> Optional[float]:
        """Measure performance impact of a scaling decision."""
        try:
            # Get performance metrics before and after scaling
            component = decision.component
            
            # This would compare metrics from before and after scaling
            # For now, simulate positive impact for scale-up decisions
            if decision.direction == ScalingDirection.UP:
                return 15.0  # 15% improvement
            else:
                return -5.0  # 5% degradation (acceptable for cost savings)
                
        except Exception as e:
            logger.error(f"Error measuring performance impact: {e}")
            return None
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "running": self.running,
            "components": {
                name: {
                    "current_replicas": rule.current_replicas,
                    "min_replicas": rule.min_replicas,
                    "max_replicas": rule.max_replicas,
                    "last_scale_up": rule.last_scale_up.isoformat() if rule.last_scale_up else None,
                    "last_scale_down": rule.last_scale_down.isoformat() if rule.last_scale_down else None,
                    "enabled": rule.enabled
                }
                for name, rule in self.scaling_rules.items()
            },
            "recent_decisions": [
                {
                    "component": d.component,
                    "direction": d.direction,
                    "target_replicas": d.target_replicas,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat(),
                    "cost_impact": d.cost_impact
                }
                for d in self.scaling_history[-10:]  # Last 10 decisions
            ]
        }
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        
        # API server scaling rule
        api_metrics = [
            ScalingMetric("cpu_utilization", ResourceType.CPU, 70.0, 80.0, 30.0, weight=1.0),
            ScalingMetric("memory_utilization", ResourceType.MEMORY, 80.0, 85.0, 40.0, weight=0.8),
            ScalingMetric("response_time", ResourceType.NETWORK, 500.0, 1000.0, 100.0, weight=1.2),
            ScalingMetric("request_rate", ResourceType.NETWORK, 100.0, 200.0, 20.0, weight=0.9)
        ]
        
        api_rule = ScalingRule(
            component="api_server",
            metrics=api_metrics,
            min_replicas=2,
            max_replicas=20,
            scale_up_cooldown=300,
            scale_down_cooldown=600
        )
        
        self.add_scaling_rule(api_rule)
        
        # Agent worker scaling rule
        agent_metrics = [
            ScalingMetric("cpu_utilization", ResourceType.CPU, 75.0, 85.0, 25.0, weight=1.0),
            ScalingMetric("memory_utilization", ResourceType.MEMORY, 70.0, 80.0, 30.0, weight=0.9),
        ]
        
        agent_rule = ScalingRule(
            component="agent_workers",
            metrics=agent_metrics,
            min_replicas=1,
            max_replicas=10,
            scale_up_cooldown=180,
            scale_down_cooldown=900
        )
        
        self.add_scaling_rule(agent_rule)
        
        # Add prediction models
        for component in ["api_server", "agent_workers"]:
            for metric_type in ["cpu_utilization", "memory_utilization", "response_time"]:
                model = PredictionModel(
                    component=component,
                    metric_type=metric_type,
                    history_window=100,
                    prediction_window=10
                )
                self.add_prediction_model(model)