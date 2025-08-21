"""Adaptive threshold management for dynamic quality gates."""

import asyncio
import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict, deque

from .quality_gates import QualityGateResult, QualityGateStatus
from ..core.error_handling import CyberRangeError, ErrorSeverity


class AdaptationStrategy(Enum):
    """Threshold adaptation strategies."""
    CONSERVATIVE = "conservative"  # Slow adaptation, prefer stability
    BALANCED = "balanced"         # Moderate adaptation
    AGGRESSIVE = "aggressive"     # Fast adaptation, prefer optimization
    CONTEXT_AWARE = "context_aware"  # Adapt based on project context


@dataclass
class ThresholdHistory:
    """Historical threshold data for analysis."""
    metric_name: str
    timestamp: datetime
    threshold: float
    actual_value: float
    result_status: QualityGateStatus
    adaptation_reason: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationRule:
    """Rule for threshold adaptation."""
    metric_name: str
    min_threshold: float
    max_threshold: float
    adaptation_rate: float
    confidence_threshold: float
    stability_window: int
    context_factors: List[str] = field(default_factory=list)


class StatisticalAnalyzer:
    """Statistical analysis for threshold optimization."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.data_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_measurement(self, metric_name: str, value: float):
        """Add measurement for statistical analysis."""
        self.data_windows[metric_name].append((datetime.now(), value))
    
    def calculate_statistics(self, metric_name: str) -> Dict[str, float]:
        """Calculate statistical measures for metric."""
        values = [point[1] for point in self.data_windows[metric_name]]
        
        if len(values) < 5:
            return {}
        
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "percentile_25": np.percentile(values, 25),
            "percentile_75": np.percentile(values, 75),
            "percentile_90": np.percentile(values, 90),
            "percentile_95": np.percentile(values, 95),
            "sample_size": len(values)
        }
    
    def detect_outliers(self, metric_name: str, z_threshold: float = 2.0) -> List[Tuple[datetime, float]]:
        """Detect outliers using z-score analysis."""
        data = list(self.data_windows[metric_name])
        if len(data) < 10:
            return []
        
        values = [point[1] for point in data]
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        if std_dev == 0:
            return []
        
        outliers = []
        for timestamp, value in data:
            z_score = abs((value - mean) / std_dev)
            if z_score > z_threshold:
                outliers.append((timestamp, value))
        
        return outliers
    
    def analyze_stability(self, metric_name: str) -> Dict[str, Any]:
        """Analyze metric stability over time."""
        values = [point[1] for point in self.data_windows[metric_name]]
        
        if len(values) < 10:
            return {"stable": False, "reason": "insufficient_data"}
        
        # Calculate coefficient of variation
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        cv = (std_dev / mean) * 100 if mean != 0 else float('inf')
        
        # Check for trends
        recent_half = values[len(values)//2:]
        earlier_half = values[:len(values)//2]
        
        recent_mean = statistics.mean(recent_half)
        earlier_mean = statistics.mean(earlier_half)
        
        trend_change = abs(recent_mean - earlier_mean) / earlier_mean * 100 if earlier_mean != 0 else 0
        
        # Stability criteria
        is_stable = cv < 10.0 and trend_change < 5.0
        
        return {
            "stable": is_stable,
            "coefficient_of_variation": cv,
            "trend_change_percent": trend_change,
            "recent_mean": recent_mean,
            "earlier_mean": earlier_mean,
            "stability_score": max(0, 100 - cv - trend_change)
        }


class ContextAnalyzer:
    """Analyzes project context for threshold adaptation."""
    
    def __init__(self):
        self.logger = logging.getLogger("context_analyzer")
        
    async def analyze_project_context(self) -> Dict[str, Any]:
        """Analyze current project context."""
        context = {
            "project_phase": await self._detect_project_phase(),
            "team_size": await self._estimate_team_size(),
            "code_complexity": await self._analyze_code_complexity(),
            "deployment_frequency": await self._analyze_deployment_frequency(),
            "error_rate": await self._calculate_error_rate(),
            "performance_requirements": await self._assess_performance_requirements()
        }
        
        return context
    
    async def _detect_project_phase(self) -> str:
        """Detect current project development phase."""
        # Check git history and file patterns
        try:
            # Look for indicators
            has_tests = Path("tests").exists()
            has_ci = Path(".github/workflows").exists() or Path(".gitlab-ci.yml").exists()
            has_prod_config = Path("docker-compose.yml").exists() or Path("Dockerfile").exists()
            
            if has_prod_config and has_ci and has_tests:
                return "production"
            elif has_tests and has_ci:
                return "development"
            elif has_tests:
                return "testing"
            else:
                return "initial"
                
        except Exception as e:
            self.logger.debug(f"Could not detect project phase: {e}")
            return "unknown"
    
    async def _estimate_team_size(self) -> str:
        """Estimate team size from git history."""
        # Simplified - would analyze git contributors
        return "medium"  # small, medium, large
    
    async def _analyze_code_complexity(self) -> str:
        """Analyze codebase complexity."""
        try:
            # Count files and estimate complexity
            python_files = list(Path(".").rglob("*.py"))
            file_count = len(python_files)
            
            if file_count > 100:
                return "high"
            elif file_count > 50:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    async def _analyze_deployment_frequency(self) -> str:
        """Analyze deployment frequency patterns."""
        return "medium"  # daily, weekly, monthly
    
    async def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        return 2.0  # Placeholder percentage
    
    async def _assess_performance_requirements(self) -> str:
        """Assess performance requirements."""
        return "standard"  # low, standard, high, critical


class AdaptiveThresholdManager:
    """Manages adaptive thresholds for quality gates."""
    
    def __init__(
        self,
        adaptation_strategy: AdaptationStrategy = AdaptationStrategy.BALANCED,
        min_samples: int = 20,
        confidence_threshold: float = 0.8,
        save_interval: int = 300
    ):
        self.adaptation_strategy = adaptation_strategy
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.save_interval = save_interval
        
        self.logger = logging.getLogger("adaptive_threshold_manager")
        self.statistical_analyzer = StatisticalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        
        # Threshold management
        self.current_thresholds: Dict[str, float] = {}
        self.threshold_history: List[ThresholdHistory] = []
        self.adaptation_rules: Dict[str, AdaptationRule] = self._initialize_adaptation_rules()
        
        # Performance tracking
        self.adaptation_count = 0
        self.last_adaptation_time = datetime.now()
        
    def _initialize_adaptation_rules(self) -> Dict[str, AdaptationRule]:
        """Initialize adaptation rules for different metrics."""
        return {
            "test_coverage": AdaptationRule(
                metric_name="test_coverage",
                min_threshold=70.0,
                max_threshold=95.0,
                adaptation_rate=0.1,
                confidence_threshold=0.8,
                stability_window=20,
                context_factors=["project_phase", "team_size"]
            ),
            "security_scan": AdaptationRule(
                metric_name="security_scan",
                min_threshold=85.0,
                max_threshold=98.0,
                adaptation_rate=0.05,
                confidence_threshold=0.9,
                stability_window=30,
                context_factors=["project_phase", "error_rate"]
            ),
            "performance_benchmark": AdaptationRule(
                metric_name="performance_benchmark",
                min_threshold=70.0,
                max_threshold=95.0,
                adaptation_rate=0.15,
                confidence_threshold=0.75,
                stability_window=15,
                context_factors=["performance_requirements", "deployment_frequency"]
            ),
            "code_quality": AdaptationRule(
                metric_name="code_quality",
                min_threshold=75.0,
                max_threshold=95.0,
                adaptation_rate=0.1,
                confidence_threshold=0.8,
                stability_window=25,
                context_factors=["team_size", "code_complexity"]
            ),
            "compliance_check": AdaptationRule(
                metric_name="compliance_check",
                min_threshold=90.0,
                max_threshold=98.0,
                adaptation_rate=0.02,
                confidence_threshold=0.95,
                stability_window=50,
                context_factors=["project_phase"]
            )
        }
    
    async def initialize_thresholds(self, initial_thresholds: Dict[str, float]):
        """Initialize threshold values."""
        self.current_thresholds = initial_thresholds.copy()
        self.logger.info(f"Initialized thresholds: {self.current_thresholds}")
    
    async def update_measurement(self, metric_name: str, value: float, result: QualityGateResult):
        """Update with new measurement and potentially adapt threshold."""
        # Add to statistical analysis
        self.statistical_analyzer.add_measurement(metric_name, value)
        
        # Record in history
        history_entry = ThresholdHistory(
            metric_name=metric_name,
            timestamp=datetime.now(),
            threshold=self.current_thresholds.get(metric_name, 0.0),
            actual_value=value,
            result_status=result.status,
            adaptation_reason="measurement_update"
        )
        self.threshold_history.append(history_entry)
        
        # Consider threshold adaptation
        await self._consider_adaptation(metric_name)
    
    async def _consider_adaptation(self, metric_name: str):
        """Consider whether to adapt threshold for metric."""
        if metric_name not in self.adaptation_rules:
            return
        
        rule = self.adaptation_rules[metric_name]
        
        # Check if we have enough samples
        stats = self.statistical_analyzer.calculate_statistics(metric_name)
        if not stats or stats.get("sample_size", 0) < self.min_samples:
            return
        
        # Check stability
        stability = self.statistical_analyzer.analyze_stability(metric_name)
        if not stability.get("stable", False) and self.adaptation_strategy == AdaptationStrategy.CONSERVATIVE:
            return
        
        # Get project context
        context = await self.context_analyzer.analyze_project_context()
        
        # Calculate optimal threshold
        optimal_threshold = await self._calculate_optimal_threshold(metric_name, stats, context)
        
        # Decide if adaptation is needed
        current_threshold = self.current_thresholds.get(metric_name, rule.min_threshold)
        
        if abs(optimal_threshold - current_threshold) > (current_threshold * 0.05):  # 5% change threshold
            await self._adapt_threshold(metric_name, optimal_threshold, context, stats)
    
    async def _calculate_optimal_threshold(
        self, 
        metric_name: str, 
        stats: Dict[str, float], 
        context: Dict[str, Any]
    ) -> float:
        """Calculate optimal threshold based on statistics and context."""
        rule = self.adaptation_rules[metric_name]
        
        # Base threshold from statistical analysis
        if self.adaptation_strategy == AdaptationStrategy.CONSERVATIVE:
            # Use 25th percentile for conservative approach
            base_threshold = stats.get("percentile_25", rule.min_threshold)
        elif self.adaptation_strategy == AdaptationStrategy.AGGRESSIVE:
            # Use 75th percentile for aggressive approach
            base_threshold = stats.get("percentile_75", rule.max_threshold)
        else:  # BALANCED or CONTEXT_AWARE
            # Use median for balanced approach
            base_threshold = stats.get("median", (rule.min_threshold + rule.max_threshold) / 2)
        
        # Apply context adjustments
        adjusted_threshold = await self._apply_context_adjustments(
            metric_name, base_threshold, context
        )
        
        # Ensure within bounds
        final_threshold = max(rule.min_threshold, min(rule.max_threshold, adjusted_threshold))
        
        return final_threshold
    
    async def _apply_context_adjustments(
        self, 
        metric_name: str, 
        base_threshold: float, 
        context: Dict[str, Any]
    ) -> float:
        """Apply context-based adjustments to threshold."""
        rule = self.adaptation_rules[metric_name]
        adjusted_threshold = base_threshold
        
        for factor in rule.context_factors:
            factor_value = context.get(factor, "unknown")
            
            if factor == "project_phase":
                if factor_value == "initial":
                    adjusted_threshold *= 0.9  # Lower expectations for initial phase
                elif factor_value == "production":
                    adjusted_threshold *= 1.1  # Higher expectations for production
            
            elif factor == "team_size":
                if factor_value == "small":
                    adjusted_threshold *= 0.95  # Slightly lower for small teams
                elif factor_value == "large":
                    adjusted_threshold *= 1.05  # Slightly higher for large teams
            
            elif factor == "code_complexity":
                if factor_value == "high":
                    adjusted_threshold *= 0.9  # Lower for complex codebases
                elif factor_value == "low":
                    adjusted_threshold *= 1.1  # Higher for simple codebases
            
            elif factor == "performance_requirements":
                if factor_value == "critical":
                    adjusted_threshold *= 1.15  # Much higher for critical performance
                elif factor_value == "low":
                    adjusted_threshold *= 0.85  # Lower for non-critical performance
            
            elif factor == "error_rate":
                error_rate = context.get("error_rate", 0)
                if error_rate > 5.0:
                    adjusted_threshold *= 0.9  # Lower thresholds if high error rate
            
            elif factor == "deployment_frequency":
                if factor_value == "daily":
                    adjusted_threshold *= 1.05  # Higher for frequent deployments
                elif factor_value == "monthly":
                    adjusted_threshold *= 0.95  # Lower for infrequent deployments
        
        return adjusted_threshold
    
    async def _adapt_threshold(
        self, 
        metric_name: str, 
        new_threshold: float, 
        context: Dict[str, Any], 
        stats: Dict[str, float]
    ):
        """Adapt threshold for metric."""
        old_threshold = self.current_thresholds.get(metric_name, 0.0)
        rule = self.adaptation_rules[metric_name]
        
        # Apply adaptation rate to smooth changes
        if self.adaptation_strategy == AdaptationStrategy.CONSERVATIVE:
            adaptation_rate = rule.adaptation_rate * 0.5
        elif self.adaptation_strategy == AdaptationStrategy.AGGRESSIVE:
            adaptation_rate = rule.adaptation_rate * 2.0
        else:
            adaptation_rate = rule.adaptation_rate
        
        # Calculate adapted threshold
        threshold_change = (new_threshold - old_threshold) * adaptation_rate
        adapted_threshold = old_threshold + threshold_change
        
        # Update threshold
        self.current_thresholds[metric_name] = adapted_threshold
        self.adaptation_count += 1
        self.last_adaptation_time = datetime.now()
        
        # Record adaptation
        history_entry = ThresholdHistory(
            metric_name=metric_name,
            timestamp=datetime.now(),
            threshold=adapted_threshold,
            actual_value=stats.get("mean", 0.0),
            result_status=QualityGateStatus.PASSED,  # Placeholder
            adaptation_reason=f"adaptive_optimization_{self.adaptation_strategy.value}",
            context=context
        )
        self.threshold_history.append(history_entry)
        
        self.logger.info(
            f"Adapted threshold for {metric_name}: {old_threshold:.1f}% → {adapted_threshold:.1f}% "
            f"(target: {new_threshold:.1f}%)"
        )
    
    def get_current_threshold(self, metric_name: str) -> Optional[float]:
        """Get current threshold for metric."""
        return self.current_thresholds.get(metric_name)
    
    def get_threshold_history(self, metric_name: str, days: int = 7) -> List[ThresholdHistory]:
        """Get threshold history for metric."""
        cutoff_time = datetime.now() - timedelta(days=days)
        return [
            entry for entry in self.threshold_history
            if entry.metric_name == metric_name and entry.timestamp >= cutoff_time
        ]
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of threshold adaptations."""
        recent_adaptations = [
            entry for entry in self.threshold_history
            if entry.timestamp >= datetime.now() - timedelta(days=1)
            and "adaptive_optimization" in entry.adaptation_reason
        ]
        
        return {
            "strategy": self.adaptation_strategy.value,
            "total_adaptations": self.adaptation_count,
            "last_adaptation": self.last_adaptation_time.isoformat(),
            "recent_adaptations_24h": len(recent_adaptations),
            "current_thresholds": self.current_thresholds.copy(),
            "metrics_tracked": list(self.adaptation_rules.keys())
        }
    
    async def save_thresholds(self, file_path: str = "adaptive_thresholds.json"):
        """Save current thresholds and history to file."""
        data = {
            "strategy": self.adaptation_strategy.value,
            "thresholds": self.current_thresholds,
            "adaptation_count": self.adaptation_count,
            "last_update": datetime.now().isoformat(),
            "history": [
                {
                    "metric_name": entry.metric_name,
                    "timestamp": entry.timestamp.isoformat(),
                    "threshold": entry.threshold,
                    "actual_value": entry.actual_value,
                    "status": entry.result_status.value,
                    "reason": entry.adaptation_reason,
                    "context": entry.context
                }
                for entry in self.threshold_history[-100:]  # Save last 100 entries
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved adaptive thresholds to {file_path}")
    
    async def load_thresholds(self, file_path: str = "adaptive_thresholds.json"):
        """Load thresholds and history from file."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            self.current_thresholds = data.get("thresholds", {})
            self.adaptation_count = data.get("adaptation_count", 0)
            
            # Load history
            for entry_data in data.get("history", []):
                entry = ThresholdHistory(
                    metric_name=entry_data["metric_name"],
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    threshold=entry_data["threshold"],
                    actual_value=entry_data["actual_value"],
                    result_status=QualityGateStatus(entry_data["status"]),
                    adaptation_reason=entry_data["reason"],
                    context=entry_data.get("context", {})
                )
                self.threshold_history.append(entry)
            
            self.logger.info(f"Loaded adaptive thresholds from {file_path}")
            
        except FileNotFoundError:
            self.logger.info(f"No existing threshold file found at {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load thresholds from {file_path}: {e}")
    
    async def start_background_adaptation(self):
        """Start background adaptation process."""
        self.logger.info("Starting background threshold adaptation")
        
        while True:
            try:
                await asyncio.sleep(self.save_interval)
                
                # Periodically save thresholds
                await self.save_thresholds()
                
                # Clean old history (keep last 30 days)
                cutoff_time = datetime.now() - timedelta(days=30)
                self.threshold_history = [
                    entry for entry in self.threshold_history
                    if entry.timestamp >= cutoff_time
                ]
                
            except Exception as e:
                self.logger.error(f"Background adaptation error: {e}")
                await asyncio.sleep(60)
    
    def reset_adaptation(self, metric_name: str = None):
        """Reset adaptation for specific metric or all metrics."""
        if metric_name:
            if metric_name in self.current_thresholds:
                rule = self.adaptation_rules[metric_name]
                self.current_thresholds[metric_name] = (rule.min_threshold + rule.max_threshold) / 2
                self.logger.info(f"Reset threshold for {metric_name}")
        else:
            for name, rule in self.adaptation_rules.items():
                self.current_thresholds[name] = (rule.min_threshold + rule.max_threshold) / 2
            self.logger.info("Reset all thresholds to default values")