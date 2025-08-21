"""Predictive Quality Assurance system with advanced anomaly detection."""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from collections import defaultdict, deque
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .quality_gates import QualityGateResult, QualityGateStatus
from .ml_optimizer import MLQualityOptimizer, MLPrediction, MLFeature
from .realtime_monitor import RealTimeQualityMonitor, QualityAlert, MonitoringLevel
from ..core.error_handling import CyberRangeError, ErrorSeverity


class PredictionHorizon(Enum):
    """Time horizons for quality predictions."""
    IMMEDIATE = "immediate"      # Next 1-6 hours
    SHORT_TERM = "short_term"    # Next 1-3 days
    MEDIUM_TERM = "medium_term"  # Next 1-2 weeks
    LONG_TERM = "long_term"      # Next 1-3 months


class AnomalyType(Enum):
    """Types of anomalies in quality metrics."""
    STATISTICAL = "statistical"          # Statistical outliers
    TREND = "trend"                      # Unexpected trend changes
    PATTERN = "pattern"                  # Pattern deviations
    PERFORMANCE = "performance"          # Performance anomalies
    CORRELATION = "correlation"          # Cross-metric correlation anomalies
    CYCLICAL = "cyclical"               # Cyclical pattern violations


@dataclass
class QualityPrediction:
    """Quality prediction with uncertainty estimation."""
    metric_name: str
    horizon: PredictionHorizon
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    prediction_timestamp: datetime
    target_timestamp: datetime
    contributing_factors: List[str]
    risk_level: str  # low, medium, high, critical
    recommended_actions: List[str]
    
    @property
    def uncertainty(self) -> float:
        """Calculate prediction uncertainty."""
        return (self.confidence_interval[1] - self.confidence_interval[0]) / 2
    
    @property
    def is_high_risk(self) -> bool:
        """Check if prediction indicates high risk."""
        return self.risk_level in ["high", "critical"]


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    metric_name: str
    anomaly_type: AnomalyType
    severity: float  # 0-1 scale
    timestamp: datetime
    value: float
    expected_range: Tuple[float, float]
    deviation_score: float
    context: Dict[str, Any]
    explanation: str
    suggested_investigation: List[str]


@dataclass
class QualityRisk:
    """Quality risk assessment."""
    metric_name: str
    current_risk_level: str
    predicted_risk_level: str
    risk_factors: List[str]
    impact_assessment: str
    mitigation_strategies: List[str]
    monitoring_recommendations: List[str]
    escalation_triggers: List[str]


class StatisticalAnomalyDetector:
    """Statistical anomaly detection using multiple methods."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 0.05):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.data_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
    def add_measurement(self, metric_name: str, value: float, timestamp: datetime = None):
        """Add measurement for anomaly detection."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.data_windows[metric_name].append((timestamp, value))
        self._update_baseline_stats(metric_name)
    
    def _update_baseline_stats(self, metric_name: str):
        """Update baseline statistics for metric."""
        values = [point[1] for point in self.data_windows[metric_name]]
        
        if len(values) < 10:
            return
        
        self.baseline_stats[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "iqr": np.percentile(values, 75) - np.percentile(values, 25),
            "min": np.min(values),
            "max": np.max(values)
        }
    
    def detect_statistical_anomalies(self, metric_name: str, value: float) -> List[AnomalyDetection]:
        """Detect statistical anomalies using multiple methods."""
        anomalies = []
        
        if metric_name not in self.baseline_stats:
            return anomalies
        
        stats = self.baseline_stats[metric_name]
        timestamp = datetime.now()
        
        # Z-score method
        if stats["std"] > 0:
            z_score = abs((value - stats["mean"]) / stats["std"])
            if z_score > stats.norm.ppf(1 - self.sensitivity / 2):  # Two-tailed test
                anomalies.append(AnomalyDetection(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=min(1.0, z_score / 5.0),
                    timestamp=timestamp,
                    value=value,
                    expected_range=(stats["mean"] - 2*stats["std"], stats["mean"] + 2*stats["std"]),
                    deviation_score=z_score,
                    context={"method": "z_score", "z_score": z_score},
                    explanation=f"Value {value:.2f} is {z_score:.2f} standard deviations from mean",
                    suggested_investigation=["Check data collection", "Review recent changes"]
                ))
        
        # IQR method
        iqr = stats["iqr"]
        if iqr > 0:
            lower_bound = stats["q25"] - 1.5 * iqr
            upper_bound = stats["q75"] + 1.5 * iqr
            
            if value < lower_bound or value > upper_bound:
                deviation = max(lower_bound - value, value - upper_bound, 0)
                severity = min(1.0, deviation / (iqr * 2))
                
                anomalies.append(AnomalyDetection(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    timestamp=timestamp,
                    value=value,
                    expected_range=(lower_bound, upper_bound),
                    deviation_score=deviation,
                    context={"method": "iqr", "iqr": iqr},
                    explanation=f"Value {value:.2f} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                    suggested_investigation=["Check for outliers", "Investigate unusual conditions"]
                ))
        
        # Modified Z-score using median
        mad = np.median([abs(x - stats["median"]) for x in [point[1] for point in self.data_windows[metric_name]]])
        if mad > 0:
            modified_z_score = 0.6745 * (value - stats["median"]) / mad
            if abs(modified_z_score) > 3.5:
                anomalies.append(AnomalyDetection(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=min(1.0, abs(modified_z_score) / 10.0),
                    timestamp=timestamp,
                    value=value,
                    expected_range=(stats["median"] - 3.5*mad/0.6745, stats["median"] + 3.5*mad/0.6745),
                    deviation_score=abs(modified_z_score),
                    context={"method": "modified_z_score", "mad": mad},
                    explanation=f"Modified Z-score {modified_z_score:.2f} indicates anomaly",
                    suggested_investigation=["Check for system changes", "Review metric definition"]
                ))
        
        return anomalies
    
    def detect_trend_anomalies(self, metric_name: str) -> List[AnomalyDetection]:
        """Detect anomalies in trends."""
        anomalies = []
        
        if len(self.data_windows[metric_name]) < 20:
            return anomalies
        
        values = [point[1] for point in self.data_windows[metric_name]]
        timestamps = [point[0] for point in self.data_windows[metric_name]]
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Check for significant trend changes
        if abs(r_value) > 0.5 and p_value < 0.05:  # Significant trend
            recent_values = values[-5:]
            recent_x = x[-5:]
            
            # Check if recent values deviate from trend
            expected_values = slope * recent_x + intercept
            deviations = [abs(actual - expected) for actual, expected in zip(recent_values, expected_values)]
            avg_deviation = np.mean(deviations)
            
            if avg_deviation > np.std(values):
                anomalies.append(AnomalyDetection(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.TREND,
                    severity=min(1.0, avg_deviation / (2 * np.std(values))),
                    timestamp=timestamps[-1],
                    value=values[-1],
                    expected_range=(expected_values[-1] - np.std(values), expected_values[-1] + np.std(values)),
                    deviation_score=avg_deviation,
                    context={"slope": slope, "r_value": r_value, "p_value": p_value},
                    explanation=f"Recent values deviate from established trend (slope: {slope:.4f})",
                    suggested_investigation=["Analyze trend change causes", "Review process modifications"]
                ))
        
        return anomalies


class PatternAnomalyDetector:
    """Pattern-based anomaly detection using clustering and sequence analysis."""
    
    def __init__(self, pattern_window: int = 24, min_pattern_length: int = 3):
        self.pattern_window = pattern_window
        self.min_pattern_length = min_pattern_length
        self.learned_patterns: Dict[str, List[List[float]]] = defaultdict(list)
        self.clusterer = DBSCAN(eps=0.5, min_samples=3)
        
    def learn_patterns(self, metric_name: str, time_series: List[Tuple[datetime, float]]):
        """Learn normal patterns from historical data."""
        if len(time_series) < self.pattern_window * 3:
            return
        
        # Extract patterns of fixed length
        values = [point[1] for point in time_series]
        patterns = []
        
        for i in range(len(values) - self.pattern_window + 1):
            pattern = values[i:i + self.pattern_window]
            # Normalize pattern
            if np.std(pattern) > 0:
                normalized_pattern = (np.array(pattern) - np.mean(pattern)) / np.std(pattern)
                patterns.append(normalized_pattern.tolist())
        
        self.learned_patterns[metric_name] = patterns
    
    def detect_pattern_anomalies(self, metric_name: str, recent_values: List[float]) -> List[AnomalyDetection]:
        """Detect pattern anomalies."""
        anomalies = []
        
        if (metric_name not in self.learned_patterns or 
            len(recent_values) < self.min_pattern_length):
            return anomalies
        
        learned_patterns = self.learned_patterns[metric_name]
        if not learned_patterns:
            return anomalies
        
        # Normalize recent pattern
        if np.std(recent_values) == 0:
            return anomalies
        
        normalized_recent = (np.array(recent_values) - np.mean(recent_values)) / np.std(recent_values)
        
        # Find similarity to learned patterns
        similarities = []
        for pattern in learned_patterns:
            if len(pattern) == len(normalized_recent):
                # Calculate correlation
                correlation = np.corrcoef(normalized_recent, pattern)[0, 1]
                if not np.isnan(correlation):
                    similarities.append(correlation)
        
        if similarities:
            max_similarity = max(similarities)
            avg_similarity = np.mean(similarities)
            
            # If current pattern is very different from learned patterns
            if max_similarity < 0.3 and avg_similarity < 0.1:
                anomalies.append(AnomalyDetection(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.PATTERN,
                    severity=1.0 - max_similarity,
                    timestamp=datetime.now(),
                    value=recent_values[-1],
                    expected_range=(min(recent_values), max(recent_values)),
                    deviation_score=1.0 - max_similarity,
                    context={"max_similarity": max_similarity, "avg_similarity": avg_similarity},
                    explanation=f"Current pattern similarity {max_similarity:.3f} below threshold",
                    suggested_investigation=["Analyze pattern change", "Check for configuration changes"]
                ))
        
        return anomalies


class CorrelationAnomalyDetector:
    """Detect anomalies in cross-metric correlations."""
    
    def __init__(self, correlation_window: int = 50):
        self.correlation_window = correlation_window
        self.correlation_history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=correlation_window)
        )
        self.expected_correlations: Dict[Tuple[str, str], float] = {}
        
    def update_correlations(self, metrics: Dict[str, float]):
        """Update correlation tracking."""
        metric_names = list(metrics.keys())
        
        # Calculate correlations between all pairs
        for i, metric1 in enumerate(metric_names):
            for j, metric2 in enumerate(metric_names[i+1:], i+1):
                pair = (metric1, metric2)
                
                # Store correlation values for this pair
                if len(self.correlation_history[pair]) >= 2:
                    # Get recent values for both metrics
                    recent_values1 = [metrics[metric1]]
                    recent_values2 = [metrics[metric2]]
                    
                    # Add to history for correlation calculation
                    self.correlation_history[pair].append((recent_values1[0], recent_values2[0]))
                else:
                    self.correlation_history[pair].append((metrics[metric1], metrics[metric2]))
        
        # Update expected correlations
        self._update_expected_correlations()
    
    def _update_expected_correlations(self):
        """Update expected correlation values."""
        for pair, values in self.correlation_history.items():
            if len(values) >= 10:
                values1 = [v[0] for v in values]
                values2 = [v[1] for v in values]
                
                if np.std(values1) > 0 and np.std(values2) > 0:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    if not np.isnan(correlation):
                        self.expected_correlations[pair] = correlation
    
    def detect_correlation_anomalies(self, metrics: Dict[str, float]) -> List[AnomalyDetection]:
        """Detect correlation anomalies."""
        anomalies = []
        
        for pair, expected_corr in self.expected_correlations.items():
            metric1, metric2 = pair
            
            if metric1 not in metrics or metric2 not in metrics:
                continue
            
            # Get recent correlation
            if len(self.correlation_history[pair]) >= 5:
                recent_values = list(self.correlation_history[pair])[-5:]
                values1 = [v[0] for v in recent_values]
                values2 = [v[1] for v in recent_values]
                
                if np.std(values1) > 0 and np.std(values2) > 0:
                    recent_corr = np.corrcoef(values1, values2)[0, 1]
                    
                    if not np.isnan(recent_corr):
                        correlation_change = abs(recent_corr - expected_corr)
                        
                        # Significant correlation change
                        if correlation_change > 0.3:
                            anomalies.append(AnomalyDetection(
                                metric_name=f"{metric1}_vs_{metric2}",
                                anomaly_type=AnomalyType.CORRELATION,
                                severity=min(1.0, correlation_change),
                                timestamp=datetime.now(),
                                value=recent_corr,
                                expected_range=(expected_corr - 0.2, expected_corr + 0.2),
                                deviation_score=correlation_change,
                                context={
                                    "expected_correlation": expected_corr,
                                    "recent_correlation": recent_corr,
                                    "metrics": [metric1, metric2]
                                },
                                explanation=f"Correlation changed from {expected_corr:.3f} to {recent_corr:.3f}",
                                suggested_investigation=[
                                    f"Investigate {metric1} behavior",
                                    f"Investigate {metric2} behavior",
                                    "Check for independent system changes"
                                ]
                            ))
        
        return anomalies


class PredictiveQualityAssurance:
    """Advanced predictive quality assurance system."""
    
    def __init__(
        self,
        prediction_horizons: List[PredictionHorizon] = None,
        anomaly_sensitivity: float = 0.05,
        risk_threshold: float = 0.7
    ):
        if prediction_horizons is None:
            prediction_horizons = [
                PredictionHorizon.IMMEDIATE,
                PredictionHorizon.SHORT_TERM,
                PredictionHorizon.MEDIUM_TERM
            ]
        
        self.prediction_horizons = prediction_horizons
        self.anomaly_sensitivity = anomaly_sensitivity
        self.risk_threshold = risk_threshold
        
        self.logger = logging.getLogger("predictive_qa")
        
        # Component systems
        self.ml_optimizer = MLQualityOptimizer()
        self.realtime_monitor = RealTimeQualityMonitor()
        
        # Anomaly detectors
        self.statistical_detector = StatisticalAnomalyDetector(sensitivity=anomaly_sensitivity)
        self.pattern_detector = PatternAnomalyDetector()
        self.correlation_detector = CorrelationAnomalyDetector()
        
        # Prediction and anomaly storage
        self.predictions: Dict[str, List[QualityPrediction]] = defaultdict(list)
        self.anomalies: List[AnomalyDetection] = []
        self.risk_assessments: Dict[str, QualityRisk] = {}
        
        # Performance tracking
        self.prediction_accuracy: Dict[str, List[float]] = defaultdict(list)
        
    async def initialize(self, metrics: List[str]):
        """Initialize predictive QA system."""
        await self.ml_optimizer.initialize_models(metrics)
        
        # Load existing models and data
        await self.ml_optimizer.load_models()
        
        self.logger.info(f"Initialized predictive QA for {len(metrics)} metrics")
    
    async def update_measurements(
        self,
        measurements: Dict[str, float],
        context: Dict[str, Any] = None
    ):
        """Update with new measurements and trigger analysis."""
        timestamp = datetime.now()
        context = context or {}
        
        # Update anomaly detectors
        for metric_name, value in measurements.items():
            self.statistical_detector.add_measurement(metric_name, value, timestamp)
        
        self.correlation_detector.update_correlations(measurements)
        
        # Detect anomalies
        await self._detect_all_anomalies(measurements)
        
        # Update predictions
        await self._update_predictions(measurements, context)
        
        # Update risk assessments
        await self._update_risk_assessments(measurements)
        
        self.logger.debug(f"Updated measurements for {len(measurements)} metrics")
    
    async def _detect_all_anomalies(self, measurements: Dict[str, float]):
        """Detect anomalies using all detection methods."""
        new_anomalies = []
        
        # Statistical anomalies
        for metric_name, value in measurements.items():
            stat_anomalies = self.statistical_detector.detect_statistical_anomalies(metric_name, value)
            new_anomalies.extend(stat_anomalies)
            
            trend_anomalies = self.statistical_detector.detect_trend_anomalies(metric_name)
            new_anomalies.extend(trend_anomalies)
        
        # Pattern anomalies
        for metric_name in measurements.keys():
            if len(self.statistical_detector.data_windows[metric_name]) >= 10:
                recent_values = [point[1] for point in list(self.statistical_detector.data_windows[metric_name])[-10:]]
                pattern_anomalies = self.pattern_detector.detect_pattern_anomalies(metric_name, recent_values)
                new_anomalies.extend(pattern_anomalies)
        
        # Correlation anomalies
        corr_anomalies = self.correlation_detector.detect_correlation_anomalies(measurements)
        new_anomalies.extend(corr_anomalies)
        
        # Store new anomalies
        self.anomalies.extend(new_anomalies)
        
        # Clean old anomalies (keep last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.anomalies = [a for a in self.anomalies if a.timestamp >= cutoff_time]
        
        # Log significant anomalies
        high_severity_anomalies = [a for a in new_anomalies if a.severity > 0.7]
        if high_severity_anomalies:
            self.logger.warning(f"Detected {len(high_severity_anomalies)} high-severity anomalies")
    
    async def _update_predictions(self, measurements: Dict[str, float], context: Dict[str, Any]):
        """Update quality predictions."""
        for metric_name in measurements.keys():
            for horizon in self.prediction_horizons:
                try:
                    prediction = await self._generate_prediction(metric_name, horizon, context)
                    if prediction:
                        self.predictions[metric_name].append(prediction)
                        
                        # Limit stored predictions
                        self.predictions[metric_name] = self.predictions[metric_name][-50:]
                        
                except Exception as e:
                    self.logger.debug(f"Prediction failed for {metric_name} at {horizon.value}: {e}")
    
    async def _generate_prediction(
        self,
        metric_name: str,
        horizon: PredictionHorizon,
        context: Dict[str, Any]
    ) -> Optional[QualityPrediction]:
        """Generate prediction for metric and horizon."""
        # Extract features for ML prediction
        features = await self._extract_prediction_features(metric_name, context)
        
        # Get ML predictions
        ml_prediction = await self.ml_optimizer.predict_performance(metric_name, features)
        if not ml_prediction:
            return None
        
        # Calculate time delta for horizon
        time_deltas = {
            PredictionHorizon.IMMEDIATE: timedelta(hours=3),
            PredictionHorizon.SHORT_TERM: timedelta(days=2),
            PredictionHorizon.MEDIUM_TERM: timedelta(weeks=1),
            PredictionHorizon.LONG_TERM: timedelta(weeks=8)
        }
        
        target_time = datetime.now() + time_deltas[horizon]
        
        # Adjust prediction based on horizon and uncertainty
        base_prediction = ml_prediction.predicted_value
        uncertainty_factor = self._calculate_uncertainty_factor(horizon, metric_name)
        
        # Calculate confidence interval
        uncertainty = base_prediction * uncertainty_factor
        confidence_interval = (
            max(0, base_prediction - uncertainty),
            min(100, base_prediction + uncertainty)
        )
        
        # Assess risk level
        risk_level = self._assess_risk_level(base_prediction, confidence_interval, horizon)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metric_name, base_prediction, risk_level, horizon)
        
        return QualityPrediction(
            metric_name=metric_name,
            horizon=horizon,
            predicted_value=base_prediction,
            confidence_interval=confidence_interval,
            confidence_level=ml_prediction.confidence,
            prediction_timestamp=datetime.now(),
            target_timestamp=target_time,
            contributing_factors=ml_prediction.features_used[:5],  # Top 5 features
            risk_level=risk_level,
            recommended_actions=recommendations
        )
    
    async def _extract_prediction_features(self, metric_name: str, context: Dict[str, Any]) -> List[MLFeature]:
        """Extract features for prediction."""
        features = []
        
        # Current metric value
        if len(self.statistical_detector.data_windows[metric_name]) > 0:
            recent_values = [point[1] for point in self.statistical_detector.data_windows[metric_name]]
            features.append(MLFeature(
                name=f"{metric_name}_current",
                value=recent_values[-1] if recent_values else 0.0,
                feature_type="current_value"
            ))
            
            # Trend features
            if len(recent_values) >= 5:
                recent_trend = np.mean(recent_values[-5:]) - np.mean(recent_values[-10:-5]) if len(recent_values) >= 10 else 0
                features.append(MLFeature(
                    name=f"{metric_name}_trend",
                    value=recent_trend,
                    feature_type="trend"
                ))
        
        # Context features
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features.append(MLFeature(
                    name=f"context_{key}",
                    value=float(value),
                    feature_type="context"
                ))
        
        # Anomaly features
        recent_anomalies = [a for a in self.anomalies if a.metric_name == metric_name and 
                          a.timestamp >= datetime.now() - timedelta(hours=6)]
        features.append(MLFeature(
            name=f"{metric_name}_recent_anomalies",
            value=float(len(recent_anomalies)),
            feature_type="anomaly_count"
        ))
        
        if recent_anomalies:
            avg_severity = np.mean([a.severity for a in recent_anomalies])
            features.append(MLFeature(
                name=f"{metric_name}_anomaly_severity",
                value=avg_severity,
                feature_type="anomaly_severity"
            ))
        
        return features
    
    def _calculate_uncertainty_factor(self, horizon: PredictionHorizon, metric_name: str) -> float:
        """Calculate uncertainty factor based on horizon and metric characteristics."""
        base_uncertainty = {
            PredictionHorizon.IMMEDIATE: 0.05,
            PredictionHorizon.SHORT_TERM: 0.15,
            PredictionHorizon.MEDIUM_TERM: 0.25,
            PredictionHorizon.LONG_TERM: 0.40
        }[horizon]
        
        # Adjust based on metric volatility
        if metric_name in self.statistical_detector.baseline_stats:
            stats = self.statistical_detector.baseline_stats[metric_name]
            cv = stats["std"] / stats["mean"] if stats["mean"] != 0 else 0
            volatility_factor = min(2.0, 1.0 + cv)
            base_uncertainty *= volatility_factor
        
        return base_uncertainty
    
    def _assess_risk_level(
        self,
        predicted_value: float,
        confidence_interval: Tuple[float, float],
        horizon: PredictionHorizon
    ) -> str:
        """Assess risk level based on prediction."""
        lower_bound, upper_bound = confidence_interval
        uncertainty = (upper_bound - lower_bound) / 2
        
        # Risk factors
        risk_score = 0.0
        
        # Low predicted value risk
        if predicted_value < 70:
            risk_score += 0.4
        elif predicted_value < 80:
            risk_score += 0.2
        
        # High uncertainty risk
        if uncertainty > 15:
            risk_score += 0.3
        elif uncertainty > 10:
            risk_score += 0.15
        
        # Horizon risk (longer horizons are riskier)
        horizon_risk = {
            PredictionHorizon.IMMEDIATE: 0.0,
            PredictionHorizon.SHORT_TERM: 0.1,
            PredictionHorizon.MEDIUM_TERM: 0.2,
            PredictionHorizon.LONG_TERM: 0.3
        }[horizon]
        risk_score += horizon_risk
        
        # Classify risk level
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.3:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(
        self,
        metric_name: str,
        predicted_value: float,
        risk_level: str,
        horizon: PredictionHorizon
    ) -> List[str]:
        """Generate recommendations based on prediction."""
        recommendations = []
        
        if risk_level in ["high", "critical"]:
            recommendations.extend([
                f"Monitor {metric_name} closely",
                "Consider preventive actions",
                "Review recent changes that might impact quality"
            ])
        
        if predicted_value < 70:
            recommendations.extend([
                f"Immediate attention required for {metric_name}",
                "Investigate root causes",
                "Implement corrective measures"
            ])
        
        if horizon in [PredictionHorizon.MEDIUM_TERM, PredictionHorizon.LONG_TERM]:
            recommendations.extend([
                "Plan proactive improvements",
                "Schedule quality review sessions",
                "Consider architectural changes"
            ])
        
        # Metric-specific recommendations
        metric_recommendations = {
            "test_coverage": [
                "Add automated test generation",
                "Review untested code paths",
                "Improve integration test coverage"
            ],
            "security_scan": [
                "Schedule security audit",
                "Update security dependencies",
                "Review access controls"
            ],
            "performance_benchmark": [
                "Profile application performance",
                "Optimize slow queries",
                "Review resource utilization"
            ]
        }
        
        if metric_name in metric_recommendations:
            recommendations.extend(metric_recommendations[metric_name][:2])
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    async def _update_risk_assessments(self, measurements: Dict[str, float]):
        """Update risk assessments for all metrics."""
        for metric_name in measurements.keys():
            try:
                risk_assessment = await self._assess_metric_risk(metric_name)
                if risk_assessment:
                    self.risk_assessments[metric_name] = risk_assessment
            except Exception as e:
                self.logger.debug(f"Risk assessment failed for {metric_name}: {e}")
    
    async def _assess_metric_risk(self, metric_name: str) -> Optional[QualityRisk]:
        """Assess comprehensive risk for metric."""
        # Get recent predictions
        recent_predictions = [p for p in self.predictions[metric_name] 
                            if p.prediction_timestamp >= datetime.now() - timedelta(hours=1)]
        
        if not recent_predictions:
            return None
        
        # Current risk level
        current_value = 0.0
        if len(self.statistical_detector.data_windows[metric_name]) > 0:
            current_value = list(self.statistical_detector.data_windows[metric_name])[-1][1]
        
        current_risk = self._assess_risk_level(current_value, (current_value-5, current_value+5), PredictionHorizon.IMMEDIATE)
        
        # Predicted risk level (worst case from predictions)
        predicted_risks = [p.risk_level for p in recent_predictions]
        risk_hierarchy = ["low", "medium", "high", "critical"]
        predicted_risk = max(predicted_risks, key=lambda x: risk_hierarchy.index(x)) if predicted_risks else "low"
        
        # Risk factors
        risk_factors = []
        
        # Recent anomalies
        recent_anomalies = [a for a in self.anomalies if a.metric_name == metric_name and 
                          a.timestamp >= datetime.now() - timedelta(hours=6)]
        if recent_anomalies:
            risk_factors.append(f"Recent anomalies detected ({len(recent_anomalies)})")
        
        # Trend analysis
        if len(self.statistical_detector.data_windows[metric_name]) >= 10:
            recent_values = [point[1] for point in list(self.statistical_detector.data_windows[metric_name])[-10:]]
            trend = np.mean(recent_values[-5:]) - np.mean(recent_values[-10:-5])
            if trend < -2:
                risk_factors.append("Negative trend detected")
        
        # High uncertainty predictions
        high_uncertainty_preds = [p for p in recent_predictions if p.uncertainty > 10]
        if high_uncertainty_preds:
            risk_factors.append("High prediction uncertainty")
        
        return QualityRisk(
            metric_name=metric_name,
            current_risk_level=current_risk,
            predicted_risk_level=predicted_risk,
            risk_factors=risk_factors,
            impact_assessment=self._assess_impact(metric_name, predicted_risk),
            mitigation_strategies=self._generate_mitigation_strategies(metric_name, predicted_risk),
            monitoring_recommendations=self._generate_monitoring_recommendations(metric_name),
            escalation_triggers=self._generate_escalation_triggers(metric_name, predicted_risk)
        )
    
    def _assess_impact(self, metric_name: str, risk_level: str) -> str:
        """Assess impact of quality degradation."""
        impact_map = {
            ("test_coverage", "critical"): "High risk of production bugs and failures",
            ("test_coverage", "high"): "Moderate risk of undetected issues",
            ("security_scan", "critical"): "Critical security vulnerabilities exposed",
            ("security_scan", "high"): "Security risks requiring immediate attention",
            ("performance_benchmark", "critical"): "Severe performance degradation expected",
            ("performance_benchmark", "high"): "Performance issues likely to impact users",
        }
        
        return impact_map.get((metric_name, risk_level), f"Quality degradation in {metric_name}")
    
    def _generate_mitigation_strategies(self, metric_name: str, risk_level: str) -> List[str]:
        """Generate mitigation strategies."""
        strategies = {
            "test_coverage": [
                "Implement automated test generation",
                "Mandatory code review for test coverage",
                "Block deployments below coverage threshold"
            ],
            "security_scan": [
                "Emergency security patch deployment",
                "Temporary access restrictions",
                "Security audit and penetration testing"
            ],
            "performance_benchmark": [
                "Performance optimization sprint",
                "Resource scaling and optimization",
                "Load testing and capacity planning"
            ]
        }
        
        base_strategies = strategies.get(metric_name, ["Monitor closely", "Investigate root causes"])
        
        if risk_level in ["high", "critical"]:
            base_strategies.insert(0, "Immediate escalation and response")
        
        return base_strategies[:4]
    
    def _generate_monitoring_recommendations(self, metric_name: str) -> List[str]:
        """Generate monitoring recommendations."""
        return [
            f"Increase monitoring frequency for {metric_name}",
            "Set up real-time alerts for threshold breaches",
            "Implement predictive alerting",
            "Schedule regular quality reviews"
        ]
    
    def _generate_escalation_triggers(self, metric_name: str, risk_level: str) -> List[str]:
        """Generate escalation triggers."""
        base_triggers = [
            f"{metric_name} drops below critical threshold",
            "Multiple consecutive prediction failures",
            "Anomaly detection confidence > 90%"
        ]
        
        if risk_level == "critical":
            base_triggers.insert(0, "Immediate escalation required")
        
        return base_triggers
    
    def get_prediction_summary(self, metric_name: str = None) -> Dict[str, Any]:
        """Get summary of predictions."""
        if metric_name:
            predictions = self.predictions.get(metric_name, [])
        else:
            predictions = [p for pred_list in self.predictions.values() for p in pred_list]
        
        if not predictions:
            return {"total_predictions": 0}
        
        recent_predictions = [p for p in predictions if 
                            p.prediction_timestamp >= datetime.now() - timedelta(hours=24)]
        
        high_risk_predictions = [p for p in recent_predictions if p.is_high_risk]
        
        return {
            "total_predictions": len(predictions),
            "recent_predictions_24h": len(recent_predictions),
            "high_risk_predictions": len(high_risk_predictions),
            "average_confidence": np.mean([p.confidence_level for p in recent_predictions]) if recent_predictions else 0,
            "risk_distribution": {
                "low": len([p for p in recent_predictions if p.risk_level == "low"]),
                "medium": len([p for p in recent_predictions if p.risk_level == "medium"]),
                "high": len([p for p in recent_predictions if p.risk_level == "high"]),
                "critical": len([p for p in recent_predictions if p.risk_level == "critical"])
            }
        }
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        recent_anomalies = [a for a in self.anomalies if 
                          a.timestamp >= datetime.now() - timedelta(hours=24)]
        
        return {
            "total_anomalies": len(self.anomalies),
            "recent_anomalies_24h": len(recent_anomalies),
            "high_severity_anomalies": len([a for a in recent_anomalies if a.severity > 0.7]),
            "anomaly_types": {
                anomaly_type.value: len([a for a in recent_anomalies if a.anomaly_type == anomaly_type])
                for anomaly_type in AnomalyType
            },
            "affected_metrics": len(set(a.metric_name for a in recent_anomalies))
        }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of risk assessments."""
        high_risk_metrics = [name for name, risk in self.risk_assessments.items() 
                           if risk.predicted_risk_level in ["high", "critical"]]
        
        return {
            "total_metrics_assessed": len(self.risk_assessments),
            "high_risk_metrics": len(high_risk_metrics),
            "critical_risk_metrics": len([name for name, risk in self.risk_assessments.items() 
                                        if risk.predicted_risk_level == "critical"]),
            "metrics_with_risk_factors": len([risk for risk in self.risk_assessments.values() 
                                            if risk.risk_factors]),
            "high_risk_metric_names": high_risk_metrics
        }
    
    async def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "prediction_summary": self.get_prediction_summary(),
            "anomaly_summary": self.get_anomaly_summary(),
            "risk_summary": self.get_risk_summary(),
            "ml_model_status": self.ml_optimizer.get_model_status(),
            "recommendations": await self._generate_global_recommendations()
        }
    
    async def _generate_global_recommendations(self) -> List[str]:
        """Generate global recommendations based on all analysis."""
        recommendations = []
        
        # High-risk metrics
        high_risk_metrics = [name for name, risk in self.risk_assessments.items() 
                           if risk.predicted_risk_level in ["high", "critical"]]
        
        if high_risk_metrics:
            recommendations.append(f"Immediate attention required for: {', '.join(high_risk_metrics)}")
        
        # Recent anomalies
        recent_anomalies = [a for a in self.anomalies if 
                          a.timestamp >= datetime.now() - timedelta(hours=6)]
        
        if len(recent_anomalies) > 5:
            recommendations.append("High anomaly activity detected - investigate system changes")
        
        # Model training status
        model_status = self.ml_optimizer.get_model_status()
        if model_status["trained_models"] < model_status["total_metrics"] * 0.5:
            recommendations.append("Insufficient ML model training - collect more data")
        
        return recommendations[:5]