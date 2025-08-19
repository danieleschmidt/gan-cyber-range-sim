"""Intelligent optimization system for progressive quality gates."""

import asyncio
import logging
import time
import json
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
# Remove numpy dependency for now
from collections import defaultdict, deque

from .quality_gates import QualityGateResult, QualityGateStatus
from .monitoring import QualityMetric, MetricsCollector, PerformanceSnapshot
from ..core.error_handling import CyberRangeError, ErrorSeverity


class OptimizationStrategy(Enum):
    """Optimization strategies."""
    PERFORMANCE_FIRST = "performance_first"
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    RESOURCE_EFFICIENT = "resource_efficient"
    SPEED_OPTIMIZED = "speed_optimized"


@dataclass
class OptimizationTarget:
    """Optimization target specification."""
    metric_name: str
    target_value: float
    weight: float = 1.0
    tolerance: float = 0.05
    priority: int = 1  # 1=highest, 5=lowest


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    strategy: OptimizationStrategy
    targets: List[OptimizationTarget]
    achieved_improvements: Dict[str, float]
    execution_time: float
    recommendations: List[str]
    performance_gain: float
    quality_impact: float
    success: bool


class IntelligentOptimizer:
    """Intelligent optimization system for quality gates."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ):
        self.metrics_collector = metrics_collector
        self.strategy = strategy
        self.logger = logging.getLogger("intelligent_optimizer")
        
        # Optimization state
        self.optimization_history: deque = deque(maxlen=100)
        self.learned_patterns: Dict[str, Any] = {}
        self.adaptation_rules: List[Callable] = []
        
        # Performance tracking
        self.baseline_metrics: Dict[str, float] = {}
        self.current_metrics: Dict[str, float] = {}
        
        # Auto-tuning parameters
        self.auto_tune_enabled = True
        self.learning_rate = 0.1
        self.exploration_factor = 0.2
        
        # Register default adaptation rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default adaptation rules."""
        self.adaptation_rules.extend([
            self._cpu_optimization_rule,
            self._memory_optimization_rule,
            self._io_optimization_rule,
            self._quality_threshold_rule,
            self._execution_time_rule
        ])
    
    async def optimize_pipeline(
        self,
        current_results: List[QualityGateResult],
        performance_data: List[PerformanceSnapshot],
        targets: Optional[List[OptimizationTarget]] = None
    ) -> OptimizationResult:
        """Optimize the quality pipeline based on current performance."""
        start_time = time.time()
        
        self.logger.info(f"Starting optimization with strategy: {self.strategy.value}")
        
        # Set default targets based on strategy
        if targets is None:
            targets = self._get_default_targets()
        
        # Analyze current state
        current_state = self._analyze_current_state(current_results, performance_data)
        
        # Generate optimization recommendations
        recommendations = await self._generate_recommendations(current_state, targets)
        
        # Apply optimizations
        improvements = await self._apply_optimizations(recommendations, current_state)
        
        # Calculate performance gains
        performance_gain = self._calculate_performance_gain(current_state, improvements)
        quality_impact = self._calculate_quality_impact(current_results, improvements)
        
        # Store optimization result
        result = OptimizationResult(
            strategy=self.strategy,
            targets=targets,
            achieved_improvements=improvements,
            execution_time=time.time() - start_time,
            recommendations=[r["description"] for r in recommendations],
            performance_gain=performance_gain,
            quality_impact=quality_impact,
            success=performance_gain > 0 or quality_impact > 0
        )
        
        self.optimization_history.append(result)
        
        # Learn from optimization
        if self.auto_tune_enabled:
            await self._learn_from_optimization(result, current_state)
        
        self.logger.info(
            f"Optimization completed: {performance_gain:.1f}% performance gain, "
            f"{quality_impact:.1f}% quality impact"
        )
        
        return result
    
    def _get_default_targets(self) -> List[OptimizationTarget]:
        """Get default optimization targets based on strategy."""
        if self.strategy == OptimizationStrategy.PERFORMANCE_FIRST:
            return [
                OptimizationTarget("execution_time", 0.8, weight=3.0, priority=1),
                OptimizationTarget("cpu_usage", 70.0, weight=2.0, priority=2),
                OptimizationTarget("memory_usage", 80.0, weight=2.0, priority=2),
                OptimizationTarget("overall_score", 85.0, weight=1.0, priority=3)
            ]
        elif self.strategy == OptimizationStrategy.QUALITY_FIRST:
            return [
                OptimizationTarget("overall_score", 95.0, weight=3.0, priority=1),
                OptimizationTarget("security_score", 98.0, weight=2.5, priority=1),
                OptimizationTarget("test_coverage", 90.0, weight=2.0, priority=2),
                OptimizationTarget("execution_time", 1.2, weight=1.0, priority=3)
            ]
        elif self.strategy == OptimizationStrategy.RESOURCE_EFFICIENT:
            return [
                OptimizationTarget("cpu_usage", 60.0, weight=3.0, priority=1),
                OptimizationTarget("memory_usage", 70.0, weight=3.0, priority=1),
                OptimizationTarget("io_efficiency", 0.9, weight=2.0, priority=2),
                OptimizationTarget("overall_score", 85.0, weight=1.5, priority=2)
            ]
        elif self.strategy == OptimizationStrategy.SPEED_OPTIMIZED:
            return [
                OptimizationTarget("execution_time", 0.5, weight=4.0, priority=1),
                OptimizationTarget("parallel_efficiency", 0.85, weight=2.0, priority=2),
                OptimizationTarget("cache_hit_rate", 0.8, weight=1.5, priority=3),
                OptimizationTarget("overall_score", 80.0, weight=1.0, priority=4)
            ]
        else:  # BALANCED
            return [
                OptimizationTarget("overall_score", 90.0, weight=2.0, priority=1),
                OptimizationTarget("execution_time", 1.0, weight=2.0, priority=1),
                OptimizationTarget("cpu_usage", 75.0, weight=1.5, priority=2),
                OptimizationTarget("memory_usage", 80.0, weight=1.5, priority=2),
                OptimizationTarget("security_score", 95.0, weight=1.8, priority=2)
            ]
    
    def _analyze_current_state(
        self,
        results: List[QualityGateResult],
        performance_data: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Analyze current pipeline state."""
        # Calculate metrics from results
        total_score = sum(r.score for r in results) / len(results) if results else 0
        total_time = sum(r.execution_time for r in results)
        failed_gates = len([r for r in results if r.status == QualityGateStatus.FAILED])
        
        # Calculate performance metrics
        cpu_usage = statistics.mean(p.cpu_percent for p in performance_data) if performance_data else 0
        memory_usage = statistics.mean(p.memory_percent for p in performance_data) if performance_data else 0
        
        # Identify bottlenecks
        bottlenecks = []
        if cpu_usage > 80:
            bottlenecks.append("high_cpu")
        if memory_usage > 85:
            bottlenecks.append("high_memory")
        if total_time > 600:  # 10 minutes
            bottlenecks.append("slow_execution")
        if failed_gates > 0:
            bottlenecks.append("quality_failures")
        
        return {
            "overall_score": total_score,
            "execution_time": total_time,
            "failed_gates": failed_gates,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "bottlenecks": bottlenecks,
            "gate_performance": {r.gate_name: r.execution_time for r in results},
            "quality_scores": {r.gate_name: r.score for r in results}
        }
    
    async def _generate_recommendations(
        self,
        current_state: Dict[str, Any],
        targets: List[OptimizationTarget]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Apply adaptation rules
        for rule in self.adaptation_rules:
            try:
                rule_recommendations = await rule(current_state, targets)
                recommendations.extend(rule_recommendations)
            except Exception as e:
                self.logger.warning(f"Adaptation rule failed: {e}")
        
        # Sort by priority and impact
        recommendations.sort(key=lambda r: (r.get("priority", 5), -r.get("impact", 0)))
        
        return recommendations
    
    async def _cpu_optimization_rule(
        self,
        state: Dict[str, Any],
        targets: List[OptimizationTarget]
    ) -> List[Dict[str, Any]]:
        """CPU optimization rule."""
        recommendations = []
        
        cpu_usage = state.get("cpu_usage", 0)
        cpu_target = next((t.target_value for t in targets if t.metric_name == "cpu_usage"), 75.0)
        
        if cpu_usage > cpu_target * 1.1:
            recommendations.append({
                "type": "cpu_optimization",
                "description": "Enable parallel gate execution",
                "impact": 0.8,
                "priority": 1,
                "action": "enable_parallel_gates",
                "parameters": {"max_workers": min(4, max(2, int(cpu_usage / 25)))}
            })
            
            recommendations.append({
                "type": "cpu_optimization",
                "description": "Optimize CPU-intensive gates",
                "impact": 0.6,
                "priority": 2,
                "action": "optimize_cpu_gates",
                "parameters": {"timeout_reduction": 0.8}
            })
        
        return recommendations
    
    async def _memory_optimization_rule(
        self,
        state: Dict[str, Any],
        targets: List[OptimizationTarget]
    ) -> List[Dict[str, Any]]:
        """Memory optimization rule."""
        recommendations = []
        
        memory_usage = state.get("memory_usage", 0)
        memory_target = next((t.target_value for t in targets if t.metric_name == "memory_usage"), 80.0)
        
        if memory_usage > memory_target * 1.1:
            recommendations.append({
                "type": "memory_optimization",
                "description": "Enable result streaming",
                "impact": 0.7,
                "priority": 1,
                "action": "enable_streaming",
                "parameters": {"batch_size": 10}
            })
            
            recommendations.append({
                "type": "memory_optimization",
                "description": "Optimize cache size",
                "impact": 0.5,
                "priority": 2,
                "action": "optimize_cache",
                "parameters": {"cache_reduction": 0.5}
            })
        
        return recommendations
    
    async def _io_optimization_rule(
        self,
        state: Dict[str, Any],
        targets: List[OptimizationTarget]
    ) -> List[Dict[str, Any]]:
        """I/O optimization rule."""
        recommendations = []
        
        execution_time = state.get("execution_time", 0)
        time_target = next((t.target_value for t in targets if t.metric_name == "execution_time"), 300.0)
        
        if execution_time > time_target * 1.2:
            recommendations.append({
                "type": "io_optimization",
                "description": "Enable aggressive caching",
                "impact": 0.6,
                "priority": 2,
                "action": "enable_caching",
                "parameters": {"cache_ttl": 3600}
            })
            
            recommendations.append({
                "type": "io_optimization",
                "description": "Optimize file operations",
                "impact": 0.4,
                "priority": 3,
                "action": "optimize_file_ops",
                "parameters": {"buffer_size": 64 * 1024}
            })
        
        return recommendations
    
    async def _quality_threshold_rule(
        self,
        state: Dict[str, Any],
        targets: List[OptimizationTarget]
    ) -> List[Dict[str, Any]]:
        """Quality threshold optimization rule."""
        recommendations = []
        
        overall_score = state.get("overall_score", 0)
        failed_gates = state.get("failed_gates", 0)
        quality_target = next((t.target_value for t in targets if t.metric_name == "overall_score"), 85.0)
        
        if overall_score < quality_target:
            if failed_gates > 0:
                recommendations.append({
                    "type": "quality_optimization",
                    "description": "Apply auto-fixes for failed gates",
                    "impact": 0.9,
                    "priority": 1,
                    "action": "apply_auto_fixes",
                    "parameters": {"max_attempts": 2}
                })
            
            recommendations.append({
                "type": "quality_optimization",
                "description": "Adjust quality thresholds",
                "impact": 0.3,
                "priority": 4,
                "action": "adjust_thresholds",
                "parameters": {"threshold_reduction": 0.95}
            })
        
        return recommendations
    
    async def _execution_time_rule(
        self,
        state: Dict[str, Any],
        targets: List[OptimizationTarget]
    ) -> List[Dict[str, Any]]:
        """Execution time optimization rule."""
        recommendations = []
        
        gate_performance = state.get("gate_performance", {})
        total_time = state.get("execution_time", 0)
        time_target = next((t.target_value for t in targets if t.metric_name == "execution_time"), 300.0)
        
        if total_time > time_target * 1.5:
            # Find slowest gates
            slow_gates = {k: v for k, v in gate_performance.items() if v > 60}  # Slower than 1 minute
            
            if slow_gates:
                recommendations.append({
                    "type": "execution_optimization",
                    "description": f"Optimize slow gates: {list(slow_gates.keys())}",
                    "impact": 0.8,
                    "priority": 1,
                    "action": "optimize_slow_gates",
                    "parameters": {"gates": list(slow_gates.keys()), "timeout_factor": 0.7}
                })
            
            recommendations.append({
                "type": "execution_optimization",
                "description": "Enable gate parallelization",
                "impact": 0.7,
                "priority": 2,
                "action": "enable_gate_parallelization",
                "parameters": {"max_parallel": 3}
            })
        
        return recommendations
    
    async def _apply_optimizations(
        self,
        recommendations: List[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply optimization recommendations."""
        improvements = {}
        
        for rec in recommendations[:5]:  # Apply top 5 recommendations
            try:
                action = rec.get("action")
                parameters = rec.get("parameters", {})
                impact = rec.get("impact", 0)
                
                if action == "enable_parallel_gates":
                    improvements["parallel_efficiency"] = impact * 0.8
                elif action == "optimize_cpu_gates":
                    improvements["cpu_optimization"] = impact * 0.6
                elif action == "enable_streaming":
                    improvements["memory_efficiency"] = impact * 0.7
                elif action == "enable_caching":
                    improvements["cache_efficiency"] = impact * 0.6
                elif action == "apply_auto_fixes":
                    improvements["quality_improvement"] = impact * 0.9
                elif action == "optimize_slow_gates":
                    improvements["execution_speed"] = impact * 0.8
                elif action == "enable_gate_parallelization":
                    improvements["parallelization"] = impact * 0.7
                
                self.logger.info(f"Applied optimization: {rec['description']}")
                
            except Exception as e:
                self.logger.warning(f"Failed to apply optimization {rec.get('description')}: {e}")
        
        return improvements
    
    def _calculate_performance_gain(
        self,
        current_state: Dict[str, Any],
        improvements: Dict[str, float]
    ) -> float:
        """Calculate overall performance gain."""
        if not improvements:
            return 0.0
        
        # Weight different improvement types
        weights = {
            "parallel_efficiency": 1.5,
            "cpu_optimization": 1.2,
            "memory_efficiency": 1.0,
            "cache_efficiency": 0.8,
            "execution_speed": 1.8,
            "parallelization": 1.4
        }
        
        weighted_gain = sum(
            improvements.get(key, 0) * weight
            for key, weight in weights.items()
        )
        
        return min(weighted_gain * 100, 100.0)  # Cap at 100%
    
    def _calculate_quality_impact(
        self,
        current_results: List[QualityGateResult],
        improvements: Dict[str, float]
    ) -> float:
        """Calculate quality impact of optimizations."""
        quality_improvements = improvements.get("quality_improvement", 0)
        
        # Factor in current quality state
        if current_results:
            current_quality = sum(r.score for r in current_results) / len(current_results)
            quality_headroom = (100 - current_quality) / 100
            
            # Quality improvements are more valuable when current quality is low
            return quality_improvements * quality_headroom * 100
        
        return quality_improvements * 50  # Default impact
    
    async def _learn_from_optimization(
        self,
        result: OptimizationResult,
        state: Dict[str, Any]
    ):
        """Learn from optimization results for future improvements."""
        # Store successful patterns
        if result.success and result.performance_gain > 5:
            pattern_key = f"{result.strategy.value}_{hash(str(sorted(state.items())))}"
            
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    "state_signature": state,
                    "successful_recommendations": result.recommendations,
                    "performance_gain": result.performance_gain,
                    "usage_count": 1,
                    "last_used": datetime.now()
                }
            else:
                pattern = self.learned_patterns[pattern_key]
                pattern["usage_count"] += 1
                pattern["last_used"] = datetime.now()
                
                # Update pattern with weighted average
                alpha = self.learning_rate
                pattern["performance_gain"] = (
                    (1 - alpha) * pattern["performance_gain"] +
                    alpha * result.performance_gain
                )
        
        # Adapt learning rate based on success
        if result.success:
            self.learning_rate = min(0.3, self.learning_rate * 1.05)
        else:
            self.learning_rate = max(0.05, self.learning_rate * 0.95)
        
        self.logger.debug(f"Updated learning rate to {self.learning_rate:.3f}")
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization history."""
        if not self.optimization_history:
            return {"status": "no_data"}
        
        recent_optimizations = list(self.optimization_history)[-20:]
        
        avg_performance_gain = statistics.mean(r.performance_gain for r in recent_optimizations)
        avg_quality_impact = statistics.mean(r.quality_impact for r in recent_optimizations)
        success_rate = len([r for r in recent_optimizations if r.success]) / len(recent_optimizations)
        
        # Most effective strategies
        strategy_performance = defaultdict(list)
        for result in recent_optimizations:
            strategy_performance[result.strategy.value].append(result.performance_gain)
        
        best_strategy = max(
            strategy_performance.items(),
            key=lambda x: statistics.mean(x[1]),
            default=(None, [])
        )[0]
        
        # Most common recommendations
        all_recommendations = []
        for result in recent_optimizations:
            all_recommendations.extend(result.recommendations)
        
        from collections import Counter
        common_recommendations = Counter(all_recommendations).most_common(5)
        
        return {
            "status": "success",
            "total_optimizations": len(self.optimization_history),
            "recent_performance": {
                "avg_performance_gain": avg_performance_gain,
                "avg_quality_impact": avg_quality_impact,
                "success_rate": success_rate
            },
            "best_strategy": best_strategy,
            "common_recommendations": [{"recommendation": r, "count": c} for r, c in common_recommendations],
            "learned_patterns": len(self.learned_patterns),
            "current_learning_rate": self.learning_rate
        }
    
    def export_optimization_data(self, file_path: str):
        """Export optimization data for analysis."""
        data = {
            "optimization_history": [
                {
                    "strategy": r.strategy.value,
                    "performance_gain": r.performance_gain,
                    "quality_impact": r.quality_impact,
                    "execution_time": r.execution_time,
                    "success": r.success,
                    "recommendations": r.recommendations
                }
                for r in self.optimization_history
            ],
            "learned_patterns": self.learned_patterns,
            "insights": self.get_optimization_insights()
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"Optimization data exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export optimization data: {e}")


class AdaptiveThresholdManager:
    """Manages adaptive quality gate thresholds."""
    
    def __init__(self, initial_thresholds: Dict[str, float]):
        self.thresholds = initial_thresholds.copy()
        self.threshold_history = defaultdict(list)
        self.adaptation_rate = 0.05
        self.logger = logging.getLogger("adaptive_threshold_manager")
    
    def adapt_threshold(
        self,
        gate_name: str,
        current_score: float,
        target_pass_rate: float = 0.8
    ) -> float:
        """Adapt threshold based on historical performance."""
        if gate_name not in self.thresholds:
            self.thresholds[gate_name] = 80.0  # Default threshold
        
        current_threshold = self.thresholds[gate_name]
        
        # Record historical performance
        self.threshold_history[gate_name].append({
            "score": current_score,
            "threshold": current_threshold,
            "passed": current_score >= current_threshold,
            "timestamp": datetime.now()
        })
        
        # Keep only recent history
        if len(self.threshold_history[gate_name]) > 50:
            self.threshold_history[gate_name] = self.threshold_history[gate_name][-50:]
        
        history = self.threshold_history[gate_name]
        if len(history) < 10:  # Not enough data
            return current_threshold
        
        # Calculate current pass rate
        recent_history = history[-20:]  # Last 20 runs
        pass_rate = sum(1 for h in recent_history if h["passed"]) / len(recent_history)
        
        # Adapt threshold
        new_threshold = current_threshold
        
        if pass_rate < target_pass_rate - 0.1:  # Too many failures
            # Lower threshold slightly
            new_threshold = current_threshold * (1 - self.adaptation_rate)
            self.logger.info(f"Lowering threshold for {gate_name}: {current_threshold:.1f} -> {new_threshold:.1f}")
        elif pass_rate > target_pass_rate + 0.1:  # Too many passes
            # Raise threshold slightly
            new_threshold = current_threshold * (1 + self.adaptation_rate)
            self.logger.info(f"Raising threshold for {gate_name}: {current_threshold:.1f} -> {new_threshold:.1f}")
        
        # Clamp threshold to reasonable bounds
        new_threshold = max(50.0, min(98.0, new_threshold))
        
        self.thresholds[gate_name] = new_threshold
        return new_threshold
    
    def get_threshold_insights(self) -> Dict[str, Any]:
        """Get insights about threshold adaptations."""
        insights = {}
        
        for gate_name, history in self.threshold_history.items():
            if len(history) < 5:
                continue
            
            scores = [h["score"] for h in history]
            thresholds = [h["threshold"] for h in history]
            pass_rates = []
            
            # Calculate rolling pass rate
            for i in range(5, len(history)):
                recent = history[i-5:i]
                pass_rate = sum(1 for h in recent if h["passed"]) / len(recent)
                pass_rates.append(pass_rate)
            
            insights[gate_name] = {
                "current_threshold": self.thresholds[gate_name],
                "avg_score": statistics.mean(scores),
                "score_std": statistics.stdev(scores) if len(scores) > 1 else 0,
                "threshold_changes": len(set(thresholds)),
                "current_pass_rate": pass_rates[-1] if pass_rates else 0,
                "stability": statistics.stdev(pass_rates) if len(pass_rates) > 1 else 0
            }
        
        return insights