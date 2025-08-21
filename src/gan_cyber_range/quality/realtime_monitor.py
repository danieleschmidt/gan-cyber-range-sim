"""Real-time quality monitoring system with live feedback and alerts."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
import websockets
from collections import defaultdict, deque

from .quality_gates import QualityGateResult, QualityGateStatus
from ..core.error_handling import CyberRangeError, ErrorSeverity
from ..monitoring.metrics import MetricsCollector


class MonitoringLevel(Enum):
    """Monitoring sensitivity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass 
class QualityMetric:
    """Real-time quality metric data point."""
    name: str
    value: float
    timestamp: datetime
    threshold: float
    status: QualityGateStatus
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_degraded(self) -> bool:
        """Check if metric indicates quality degradation."""
        return self.value < self.threshold * 0.8


@dataclass
class QualityAlert:
    """Quality alert with context and severity."""
    id: str
    metric_name: str
    severity: MonitoringLevel
    message: str
    timestamp: datetime
    current_value: float
    threshold: float
    historical_trend: List[float]
    suggested_actions: List[str]
    auto_fixable: bool = False


class QualityTrend:
    """Tracks quality trends over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_points: deque = deque(maxlen=window_size)
        
    def add_point(self, value: float, timestamp: datetime = None):
        """Add a data point to trend analysis."""
        if timestamp is None:
            timestamp = datetime.now()
        self.data_points.append((timestamp, value))
    
    def get_trend_direction(self) -> str:
        """Analyze trend direction."""
        if len(self.data_points) < 5:
            return "insufficient_data"
            
        recent_values = [point[1] for point in list(self.data_points)[-5:]]
        older_values = [point[1] for point in list(self.data_points)[-10:-5]] if len(self.data_points) >= 10 else []
        
        if not older_values:
            return "stable"
            
        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "degrading"
        else:
            return "stable"
    
    def predict_next_value(self) -> Optional[float]:
        """Simple linear prediction of next value."""
        if len(self.data_points) < 3:
            return None
            
        values = [point[1] for point in self.data_points]
        n = len(values)
        
        # Simple linear regression
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        intercept = (y_sum - slope * x_sum) / n
        
        return slope * n + intercept


class RealTimeQualityMonitor:
    """Real-time quality monitoring with live feedback."""
    
    def __init__(
        self,
        update_interval: float = 30.0,
        alert_threshold_degradation: float = 0.15,
        enable_websocket: bool = True,
        websocket_port: int = 8765
    ):
        self.update_interval = update_interval
        self.alert_threshold_degradation = alert_threshold_degradation
        self.enable_websocket = enable_websocket
        self.websocket_port = websocket_port
        
        self.logger = logging.getLogger("realtime_quality_monitor")
        self.metrics_collector = MetricsCollector()
        
        # Monitoring state
        self.active_metrics: Dict[str, QualityMetric] = {}
        self.quality_trends: Dict[str, QualityTrend] = defaultdict(QualityTrend)
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.subscribers: Set[websockets.WebSocketServerProtocol] = set()
        
        # Monitoring configuration
        self.monitoring_rules = self._configure_monitoring_rules()
        self.auto_fix_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.last_update = datetime.now()
        self.update_count = 0
        
    def _configure_monitoring_rules(self) -> Dict[str, Dict]:
        """Configure monitoring rules for different quality metrics."""
        return {
            "test_coverage": {
                "threshold": 85.0,
                "alert_level": MonitoringLevel.HIGH,
                "auto_fix": True,
                "check_interval": 60.0
            },
            "security_score": {
                "threshold": 90.0,
                "alert_level": MonitoringLevel.CRITICAL,
                "auto_fix": False,
                "check_interval": 30.0
            },
            "performance_score": {
                "threshold": 80.0,
                "alert_level": MonitoringLevel.MEDIUM,
                "auto_fix": True,
                "check_interval": 45.0
            },
            "code_quality": {
                "threshold": 85.0,
                "alert_level": MonitoringLevel.MEDIUM,
                "auto_fix": True,
                "check_interval": 120.0
            },
            "compliance_score": {
                "threshold": 95.0,
                "alert_level": MonitoringLevel.HIGH,
                "auto_fix": False,
                "check_interval": 300.0
            },
            "build_success_rate": {
                "threshold": 95.0,
                "alert_level": MonitoringLevel.HIGH,
                "auto_fix": True,
                "check_interval": 60.0
            },
            "deployment_health": {
                "threshold": 98.0,
                "alert_level": MonitoringLevel.CRITICAL,
                "auto_fix": False,
                "check_interval": 30.0
            }
        }
    
    async def start_monitoring(self):
        """Start real-time quality monitoring."""
        self.logger.info("Starting real-time quality monitoring")
        
        # Start monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._trend_analyzer()),
        ]
        
        # Start websocket server if enabled
        if self.enable_websocket:
            monitoring_tasks.append(
                asyncio.create_task(self._start_websocket_server())
            )
        
        try:
            await asyncio.gather(*monitoring_tasks)
        except Exception as e:
            self.logger.error(f"Monitoring failed: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Main monitoring loop for continuous quality assessment."""
        while True:
            try:
                start_time = time.time()
                
                # Collect current quality metrics
                await self._collect_quality_metrics()
                
                # Analyze metrics and detect issues
                await self._analyze_metrics()
                
                # Update trends
                self._update_trends()
                
                # Broadcast updates to subscribers
                await self._broadcast_updates()
                
                # Performance tracking
                execution_time = time.time() - start_time
                self.update_count += 1
                self.last_update = datetime.now()
                
                self.logger.debug(
                    f"Monitoring cycle {self.update_count} completed in {execution_time:.2f}s"
                )
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry
    
    async def _collect_quality_metrics(self):
        """Collect current quality metrics from various sources."""
        current_time = datetime.now()
        
        for metric_name, config in self.monitoring_rules.items():
            try:
                # Check if it's time to update this metric
                if (metric_name in self.active_metrics and 
                    (current_time - self.active_metrics[metric_name].timestamp).seconds < config["check_interval"]):
                    continue
                
                # Collect metric based on type
                value = await self._collect_metric_value(metric_name)
                
                if value is not None:
                    status = self._determine_metric_status(value, config["threshold"])
                    
                    metric = QualityMetric(
                        name=metric_name,
                        value=value,
                        timestamp=current_time,
                        threshold=config["threshold"],
                        status=status,
                        source="realtime_monitor",
                        tags={"monitoring_level": config["alert_level"].value}
                    )
                    
                    self.active_metrics[metric_name] = metric
                    
                    # Add to metrics collector for persistence
                    await self.metrics_collector.record_metric(
                        name=f"quality.{metric_name}",
                        value=value,
                        tags=metric.tags
                    )
                    
            except Exception as e:
                self.logger.warning(f"Failed to collect metric {metric_name}: {e}")
    
    async def _collect_metric_value(self, metric_name: str) -> Optional[float]:
        """Collect specific metric value."""
        if metric_name == "test_coverage":
            return await self._get_test_coverage()
        elif metric_name == "security_score":
            return await self._get_security_score()
        elif metric_name == "performance_score":
            return await self._get_performance_score()
        elif metric_name == "code_quality":
            return await self._get_code_quality_score()
        elif metric_name == "compliance_score":
            return await self._get_compliance_score()
        elif metric_name == "build_success_rate":
            return await self._get_build_success_rate()
        elif metric_name == "deployment_health":
            return await self._get_deployment_health()
        else:
            self.logger.warning(f"Unknown metric: {metric_name}")
            return None
    
    async def _get_test_coverage(self) -> Optional[float]:
        """Get current test coverage percentage."""
        try:
            # Check for existing coverage report
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    data = json.load(f)
                return data["totals"]["percent_covered"]
        except Exception as e:
            self.logger.debug(f"Could not get test coverage: {e}")
        return None
    
    async def _get_security_score(self) -> Optional[float]:
        """Get current security score."""
        # Simplified security score calculation
        base_score = 95.0
        
        # Check for known security files/issues
        security_issues = 0
        
        # Check bandit report if exists
        bandit_file = Path("bandit_report.json")
        if bandit_file.exists():
            try:
                with open(bandit_file) as f:
                    data = json.load(f)
                high_issues = len([r for r in data.get("results", []) if r.get("issue_severity") == "HIGH"])
                security_issues += high_issues * 10
            except Exception:
                pass
        
        return max(0.0, base_score - security_issues)
    
    async def _get_performance_score(self) -> Optional[float]:
        """Get current performance score."""
        # Simplified performance assessment
        return 85.0  # Placeholder - would integrate with actual performance metrics
    
    async def _get_code_quality_score(self) -> Optional[float]:
        """Get current code quality score."""
        # Simplified code quality assessment
        return 88.0  # Placeholder - would integrate with linting tools
    
    async def _get_compliance_score(self) -> Optional[float]:
        """Get current compliance score."""
        # Check for required compliance files
        required_files = ["LICENSE", "README.md", "SECURITY.md", "CODE_OF_CONDUCT.md"]
        existing_files = sum(1 for f in required_files if Path(f).exists())
        return (existing_files / len(required_files)) * 100.0
    
    async def _get_build_success_rate(self) -> Optional[float]:
        """Get build success rate."""
        return 98.0  # Placeholder - would integrate with CI/CD metrics
    
    async def _get_deployment_health(self) -> Optional[float]:
        """Get deployment health score."""
        return 99.0  # Placeholder - would integrate with deployment monitoring
    
    def _determine_metric_status(self, value: float, threshold: float) -> QualityGateStatus:
        """Determine metric status based on value and threshold."""
        if value >= threshold:
            return QualityGateStatus.PASSED
        elif value >= threshold * 0.9:
            return QualityGateStatus.WARNING
        else:
            return QualityGateStatus.FAILED
    
    async def _analyze_metrics(self):
        """Analyze metrics for alerts and anomalies."""
        for metric_name, metric in self.active_metrics.items():
            # Check for immediate issues
            if metric.status == QualityGateStatus.FAILED:
                await self._create_alert(metric, MonitoringLevel.HIGH)
            elif metric.is_degraded:
                await self._create_alert(metric, MonitoringLevel.MEDIUM)
            
            # Check trend analysis
            trend = self.quality_trends[metric_name]
            if trend.get_trend_direction() == "degrading":
                await self._create_trend_alert(metric, trend)
    
    async def _create_alert(self, metric: QualityMetric, severity: MonitoringLevel):
        """Create quality alert."""
        alert_id = f"{metric.name}_{int(time.time())}"
        
        if alert_id in self.active_alerts:
            return  # Alert already exists
        
        suggested_actions = self._get_suggested_actions(metric)
        auto_fixable = metric.name in self.auto_fix_handlers
        
        alert = QualityAlert(
            id=alert_id,
            metric_name=metric.name,
            severity=severity,
            message=f"Quality degradation detected in {metric.name}: {metric.value:.1f}% (threshold: {metric.threshold:.1f}%)",
            timestamp=metric.timestamp,
            current_value=metric.value,
            threshold=metric.threshold,
            historical_trend=self._get_historical_values(metric.name),
            suggested_actions=suggested_actions,
            auto_fixable=auto_fixable
        )
        
        self.active_alerts[alert_id] = alert
        
        self.logger.warning(f"Quality alert created: {alert.message}")
        
        # Attempt auto-fix if available and enabled
        if auto_fixable and severity in [MonitoringLevel.HIGH, MonitoringLevel.CRITICAL]:
            await self._attempt_auto_fix(alert)
    
    async def _create_trend_alert(self, metric: QualityMetric, trend: QualityTrend):
        """Create alert based on trend analysis."""
        prediction = trend.predict_next_value()
        if prediction and prediction < metric.threshold * 0.8:
            alert_id = f"{metric.name}_trend_{int(time.time())}"
            
            alert = QualityAlert(
                id=alert_id,
                metric_name=metric.name,
                severity=MonitoringLevel.MEDIUM,
                message=f"Degrading trend detected in {metric.name}. Predicted value: {prediction:.1f}%",
                timestamp=metric.timestamp,
                current_value=metric.value,
                threshold=metric.threshold,
                historical_trend=self._get_historical_values(metric.name),
                suggested_actions=["Monitor closely", "Consider proactive improvements"],
                auto_fixable=False
            )
            
            self.active_alerts[alert_id] = alert
            self.logger.info(f"Trend alert created: {alert.message}")
    
    def _get_suggested_actions(self, metric: QualityMetric) -> List[str]:
        """Get suggested actions for metric improvement."""
        actions = {
            "test_coverage": [
                "Add unit tests for uncovered code",
                "Review and improve integration tests",
                "Generate test stubs for new features"
            ],
            "security_score": [
                "Review and fix security vulnerabilities",
                "Update dependencies with security patches",
                "Run security audit tools"
            ],
            "performance_score": [
                "Profile and optimize slow functions",
                "Review database query performance",
                "Implement caching strategies"
            ],
            "code_quality": [
                "Run code formatters (black, isort)",
                "Fix linting issues",
                "Address type checking errors"
            ],
            "compliance_score": [
                "Add missing documentation files",
                "Update license information",
                "Review security policies"
            ]
        }
        
        return actions.get(metric.name, ["Monitor and investigate"])
    
    def _get_historical_values(self, metric_name: str) -> List[float]:
        """Get historical values for trend analysis."""
        trend = self.quality_trends[metric_name]
        return [point[1] for point in trend.data_points]
    
    def _update_trends(self):
        """Update trend data for all metrics."""
        for metric_name, metric in self.active_metrics.items():
            self.quality_trends[metric_name].add_point(
                metric.value, 
                metric.timestamp
            )
    
    async def _attempt_auto_fix(self, alert: QualityAlert):
        """Attempt automatic fix for quality issue."""
        if alert.metric_name in self.auto_fix_handlers:
            try:
                self.logger.info(f"Attempting auto-fix for {alert.metric_name}")
                await self.auto_fix_handlers[alert.metric_name](alert)
            except Exception as e:
                self.logger.error(f"Auto-fix failed for {alert.metric_name}: {e}")
    
    async def _alert_processor(self):
        """Process and manage active alerts."""
        while True:
            try:
                current_time = datetime.now()
                
                # Clean up old alerts (older than 1 hour)
                expired_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if (current_time - alert.timestamp) > timedelta(hours=1)
                ]
                
                for alert_id in expired_alerts:
                    del self.active_alerts[alert_id]
                
                # Log active alert summary
                if self.active_alerts:
                    critical_count = len([a for a in self.active_alerts.values() if a.severity == MonitoringLevel.CRITICAL])
                    high_count = len([a for a in self.active_alerts.values() if a.severity == MonitoringLevel.HIGH])
                    
                    if critical_count > 0 or high_count > 0:
                        self.logger.warning(
                            f"Active quality alerts: {critical_count} critical, {high_count} high priority"
                        )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Alert processor error: {e}")
                await asyncio.sleep(60)
    
    async def _trend_analyzer(self):
        """Analyze trends and generate insights."""
        while True:
            try:
                # Analyze trends for all metrics
                for metric_name, trend in self.quality_trends.items():
                    direction = trend.get_trend_direction()
                    
                    if direction == "degrading":
                        self.logger.warning(f"Degrading trend detected in {metric_name}")
                    elif direction == "improving":
                        self.logger.info(f"Quality improvement detected in {metric_name}")
                
                await asyncio.sleep(600)  # Analyze every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Trend analyzer error: {e}")
                await asyncio.sleep(120)
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates."""
        async def handle_client(websocket, path):
            self.subscribers.add(websocket)
            self.logger.info(f"WebSocket client connected: {websocket.remote_address}")
            
            try:
                # Send initial state
                await websocket.send(json.dumps({
                    "type": "initial_state",
                    "metrics": {name: {
                        "value": metric.value,
                        "status": metric.status.value,
                        "threshold": metric.threshold,
                        "timestamp": metric.timestamp.isoformat()
                    } for name, metric in self.active_metrics.items()},
                    "alerts": {alert_id: {
                        "metric_name": alert.metric_name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    } for alert_id, alert in self.active_alerts.items()}
                }))
                
                # Wait for client to disconnect
                await websocket.wait_closed()
                
            except Exception as e:
                self.logger.debug(f"WebSocket client error: {e}")
            finally:
                self.subscribers.discard(websocket)
                self.logger.info("WebSocket client disconnected")
        
        self.logger.info(f"Starting WebSocket server on port {self.websocket_port}")
        await websockets.serve(handle_client, "localhost", self.websocket_port)
    
    async def _broadcast_updates(self):
        """Broadcast updates to WebSocket subscribers."""
        if not self.subscribers:
            return
        
        update_data = {
            "type": "metrics_update",
            "timestamp": datetime.now().isoformat(),
            "metrics": {name: {
                "value": metric.value,
                "status": metric.status.value,
                "trend": self.quality_trends[name].get_trend_direction()
            } for name, metric in self.active_metrics.items()},
            "new_alerts": list(self.active_alerts.keys())[-5:]  # Last 5 alerts
        }
        
        # Send to all subscribers
        disconnected = set()
        for websocket in self.subscribers:
            try:
                await websocket.send(json.dumps(update_data))
            except Exception:
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        self.subscribers -= disconnected
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get current data for monitoring dashboard."""
        return {
            "metrics": {name: {
                "value": metric.value,
                "status": metric.status.value,
                "threshold": metric.threshold,
                "timestamp": metric.timestamp.isoformat(),
                "trend": self.quality_trends[name].get_trend_direction(),
                "is_degraded": metric.is_degraded
            } for name, metric in self.active_metrics.items()},
            
            "alerts": {alert_id: {
                "metric_name": alert.metric_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "auto_fixable": alert.auto_fixable,
                "suggested_actions": alert.suggested_actions
            } for alert_id, alert in self.active_alerts.items()},
            
            "system_health": {
                "monitoring_active": True,
                "last_update": self.last_update.isoformat(),
                "update_count": self.update_count,
                "subscriber_count": len(self.subscribers)
            }
        }
    
    async def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.logger.info("Stopping real-time quality monitoring")
        # Implementation would clean up tasks and resources