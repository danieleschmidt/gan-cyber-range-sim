"""Advanced monitoring and observability for quality gates."""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import psutil
import threading
from collections import defaultdict, deque

from ..core.error_handling import CyberRangeError, ErrorSeverity


@dataclass
class QualityMetric:
    """Individual quality metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    threshold: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "threshold": self.threshold
        }


@dataclass
class PerformanceSnapshot:
    """System performance snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_usage_percent": self.disk_usage_percent,
            "network_io": self.network_io,
            "process_count": self.process_count,
            "load_average": self.load_average
        }


class MetricsCollector:
    """Collects and aggregates quality metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.performance_history: deque = deque(maxlen=max_history)
        self.logger = logging.getLogger("metrics_collector")
        self._collection_lock = threading.Lock()
        
    def record_metric(self, metric: QualityMetric):
        """Record a quality metric."""
        with self._collection_lock:
            self.metrics[metric.name].append(metric)
            self.logger.debug(f"Recorded metric: {metric.name} = {metric.value}")
    
    def record_performance_snapshot(self) -> PerformanceSnapshot:
        """Record current system performance."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_average = [0.0, 0.0, 0.0]
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                network_io=network_stats,
                process_count=process_count,
                load_average=load_average
            )
            
            with self._collection_lock:
                self.performance_history.append(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.warning(f"Failed to collect performance metrics: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                network_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def get_metric_summary(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, float]:
        """Get summary statistics for a metric over specified duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        with self._collection_lock:
            recent_metrics = [
                m for m in self.metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
            "latest": values[-1]
        }
    
    def get_performance_trend(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get performance trend over specified duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        with self._collection_lock:
            recent_snapshots = [
                s for s in self.performance_history
                if s.timestamp >= cutoff_time
            ]
        
        if not recent_snapshots:
            return {}
        
        # Calculate trends
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        disk_values = [s.disk_usage_percent for s in recent_snapshots]
        
        return {
            "duration_minutes": duration_minutes,
            "snapshot_count": len(recent_snapshots),
            "cpu": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values),
                "current": cpu_values[-1]
            },
            "memory": {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": sum(memory_values) / len(memory_values),
                "current": memory_values[-1]
            },
            "disk": {
                "min": min(disk_values),
                "max": max(disk_values),
                "avg": sum(disk_values) / len(disk_values),
                "current": disk_values[-1]
            },
            "latest_snapshot": recent_snapshots[-1].to_dict()
        }


class QualityMonitor:
    """Real-time quality monitoring system."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.metrics_collector = metrics_collector
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "quality_score": 70.0
        }
        self.logger = logging.getLogger("quality_monitor")
        self.alert_handlers: List[Callable] = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    def register_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """Register an alert handler."""
        self.alert_handlers.append(handler)
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring."""
        if self._monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        self.logger.info(f"Starting quality monitoring (interval: {interval_seconds}s)")
        
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Quality monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Collect performance snapshot
                snapshot = self.metrics_collector.record_performance_snapshot()
                
                # Check for alerts
                await self._check_alerts(snapshot)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _check_alerts(self, snapshot: PerformanceSnapshot):
        """Check for alert conditions."""
        alerts = []
        
        # CPU alert
        if snapshot.cpu_percent > self.alert_thresholds.get("cpu_percent", 80.0):
            alerts.append({
                "type": "high_cpu",
                "message": f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                "value": snapshot.cpu_percent,
                "threshold": self.alert_thresholds["cpu_percent"]
            })
        
        # Memory alert
        if snapshot.memory_percent > self.alert_thresholds.get("memory_percent", 85.0):
            alerts.append({
                "type": "high_memory",
                "message": f"High memory usage: {snapshot.memory_percent:.1f}%",
                "value": snapshot.memory_percent,
                "threshold": self.alert_thresholds["memory_percent"]
            })
        
        # Disk alert
        if snapshot.disk_usage_percent > self.alert_thresholds.get("disk_usage_percent", 90.0):
            alerts.append({
                "type": "high_disk",
                "message": f"High disk usage: {snapshot.disk_usage_percent:.1f}%",
                "value": snapshot.disk_usage_percent,
                "threshold": self.alert_thresholds["disk_usage_percent"]
            })
        
        # Send alerts
        for alert in alerts:
            self.logger.warning(f"Alert: {alert['message']}")
            for handler in self.alert_handlers:
                try:
                    await handler(alert["type"], alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")


class QualityDashboard:
    """Real-time quality dashboard."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        monitor: QualityMonitor,
        update_interval: int = 5
    ):
        self.metrics_collector = metrics_collector
        self.monitor = monitor
        self.update_interval = update_interval
        self.logger = logging.getLogger("quality_dashboard")
        self._dashboard_data = {}
        self._running = False
    
    async def start_dashboard(self):
        """Start the dashboard."""
        self._running = True
        self.logger.info("Starting quality dashboard")
        
        # Start background update task
        asyncio.create_task(self._update_dashboard())
    
    async def stop_dashboard(self):
        """Stop the dashboard."""
        self._running = False
        self.logger.info("Quality dashboard stopped")
    
    async def _update_dashboard(self):
        """Update dashboard data."""
        while self._running:
            try:
                # Collect current metrics
                performance_trend = self.metrics_collector.get_performance_trend(60)
                
                # Update dashboard data
                self._dashboard_data = {
                    "timestamp": datetime.now().isoformat(),
                    "performance": performance_trend,
                    "alerts": {
                        "thresholds": self.monitor.alert_thresholds,
                        "active_alerts": []  # Would be populated from monitoring
                    },
                    "quality_metrics": {
                        name: self.metrics_collector.get_metric_summary(name, 60)
                        for name in ["test_coverage", "security_score", "code_quality"]
                        if name in self.metrics_collector.metrics
                    }
                }
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Dashboard update failed: {e}")
                await asyncio.sleep(self.update_interval)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self._dashboard_data.copy()
    
    def export_dashboard_data(self, file_path: str):
        """Export dashboard data to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self._dashboard_data, f, indent=2)
            self.logger.info(f"Dashboard data exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export dashboard data: {e}")


class QualityTrendAnalyzer:
    """Analyzes quality trends and provides insights."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger("quality_trend_analyzer")
    
    def analyze_quality_trend(
        self,
        metric_name: str,
        duration_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze quality trend for a specific metric."""
        cutoff_time = datetime.now() - timedelta(hours=duration_hours)
        
        with self.metrics_collector._collection_lock:
            recent_metrics = [
                m for m in self.metrics_collector.metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]
        
        if len(recent_metrics) < 2:
            return {"status": "insufficient_data", "message": "Not enough data for trend analysis"}
        
        # Calculate trend
        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope (trend direction)
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Determine trend direction
        if slope > 0.1:
            trend_direction = "improving"
        elif slope < -0.1:
            trend_direction = "declining"
        else:
            trend_direction = "stable"
        
        # Calculate variance
        variance = sum((v - y_mean) ** 2 for v in values) / n if n > 0 else 0
        stability = "stable" if variance < 10 else "volatile"
        
        return {
            "status": "success",
            "metric_name": metric_name,
            "duration_hours": duration_hours,
            "data_points": n,
            "trend_direction": trend_direction,
            "slope": slope,
            "stability": stability,
            "variance": variance,
            "current_value": values[-1],
            "average_value": y_mean,
            "min_value": min(values),
            "max_value": max(values),
            "time_range": {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat()
            }
        }
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics_analysis": {},
            "performance_analysis": {},
            "recommendations": []
        }
        
        # Analyze each metric
        for metric_name in self.metrics_collector.metrics.keys():
            analysis = self.analyze_quality_trend(metric_name, 24)
            report["metrics_analysis"][metric_name] = analysis
            
            # Generate recommendations based on trends
            if analysis.get("trend_direction") == "declining":
                report["recommendations"].append(
                    f"Investigate declining trend in {metric_name}"
                )
            elif analysis.get("stability") == "volatile":
                report["recommendations"].append(
                    f"Address volatility in {metric_name} metric"
                )
        
        # Performance analysis
        performance_trend = self.metrics_collector.get_performance_trend(60)
        report["performance_analysis"] = performance_trend
        
        # Performance recommendations
        if performance_trend.get("cpu", {}).get("avg", 0) > 70:
            report["recommendations"].append("Consider CPU optimization")
        
        if performance_trend.get("memory", {}).get("avg", 0) > 80:
            report["recommendations"].append("Consider memory optimization")
        
        return report


# Default alert handlers
async def console_alert_handler(alert_type: str, alert_data: Dict[str, Any]):
    """Console alert handler."""
    print(f"🚨 ALERT [{alert_type.upper()}]: {alert_data['message']}")


async def file_alert_handler(alert_type: str, alert_data: Dict[str, Any]):
    """File alert handler."""
    alert_file = Path("quality_alerts.log")
    
    alert_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": alert_type,
        "data": alert_data
    }
    
    try:
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert_entry) + '\n')
    except Exception as e:
        logging.error(f"Failed to write alert to file: {e}")


# Global monitoring components
_global_metrics_collector = None
_global_monitor = None
_global_dashboard = None


def get_global_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def get_global_monitor() -> QualityMonitor:
    """Get global quality monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        collector = get_global_metrics_collector()
        _global_monitor = QualityMonitor(collector)
        # Register default alert handlers
        _global_monitor.register_alert_handler(console_alert_handler)
        _global_monitor.register_alert_handler(file_alert_handler)
    return _global_monitor


def get_global_dashboard() -> QualityDashboard:
    """Get global quality dashboard instance."""
    global _global_dashboard
    if _global_dashboard is None:
        collector = get_global_metrics_collector()
        monitor = get_global_monitor()
        _global_dashboard = QualityDashboard(collector, monitor)
    return _global_dashboard