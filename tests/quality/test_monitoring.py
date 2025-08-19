"""Tests for quality monitoring system."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from gan_cyber_range.quality.monitoring import (
    QualityMetric,
    MetricsCollector,
    QualityMonitor,
    QualityDashboard,
    QualityTrendAnalyzer,
    PerformanceSnapshot
)


class TestQualityMetric:
    """Test QualityMetric functionality."""
    
    def test_metric_creation(self):
        """Test QualityMetric creation."""
        timestamp = datetime.now()
        metric = QualityMetric(
            name="test_coverage",
            value=85.5,
            unit="percent",
            timestamp=timestamp,
            tags={"component": "tests"},
            threshold=80.0
        )
        
        assert metric.name == "test_coverage"
        assert metric.value == 85.5
        assert metric.unit == "percent"
        assert metric.timestamp == timestamp
        assert metric.tags == {"component": "tests"}
        assert metric.threshold == 80.0
    
    def test_metric_to_dict(self):
        """Test metric serialization to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        metric = QualityMetric(
            name="security_score",
            value=92.0,
            unit="percent",
            timestamp=timestamp
        )
        
        data = metric.to_dict()
        
        assert data["name"] == "security_score"
        assert data["value"] == 92.0
        assert data["unit"] == "percent"
        assert data["timestamp"] == "2024-01-01T12:00:00"
        assert data["tags"] == {}
        assert data["threshold"] is None


class TestPerformanceSnapshot:
    """Test PerformanceSnapshot functionality."""
    
    def test_snapshot_creation(self):
        """Test PerformanceSnapshot creation."""
        timestamp = datetime.now()
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            cpu_percent=75.5,
            memory_percent=60.2,
            disk_usage_percent=45.0,
            network_io={"bytes_sent": 1024, "bytes_recv": 2048},
            process_count=150,
            load_average=[1.5, 1.2, 1.0]
        )
        
        assert snapshot.timestamp == timestamp
        assert snapshot.cpu_percent == 75.5
        assert snapshot.memory_percent == 60.2
        assert snapshot.disk_usage_percent == 45.0
        assert snapshot.network_io == {"bytes_sent": 1024, "bytes_recv": 2048}
        assert snapshot.process_count == 150
        assert snapshot.load_average == [1.5, 1.2, 1.0]
    
    def test_snapshot_to_dict(self):
        """Test snapshot serialization to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            cpu_percent=80.0,
            memory_percent=70.0,
            disk_usage_percent=50.0,
            network_io={},
            process_count=100,
            load_average=[1.0]
        )
        
        data = snapshot.to_dict()
        
        assert data["timestamp"] == "2024-01-01T12:00:00"
        assert data["cpu_percent"] == 80.0
        assert data["memory_percent"] == 70.0


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_collector_creation(self):
        """Test MetricsCollector creation."""
        collector = MetricsCollector(max_history=500)
        
        assert collector.max_history == 500
        assert len(collector.metrics) == 0
        assert len(collector.performance_history) == 0
    
    def test_record_metric(self):
        """Test recording quality metrics."""
        collector = MetricsCollector()
        
        metric = QualityMetric(
            name="test_coverage",
            value=85.0,
            unit="percent",
            timestamp=datetime.now()
        )
        
        collector.record_metric(metric)
        
        assert len(collector.metrics["test_coverage"]) == 1
        assert collector.metrics["test_coverage"][0] == metric
    
    def test_metric_history_limit(self):
        """Test metric history size limiting."""
        collector = MetricsCollector(max_history=3)
        
        # Add more metrics than the limit
        for i in range(5):
            metric = QualityMetric(
                name="test_metric",
                value=float(i),
                unit="count",
                timestamp=datetime.now()
            )
            collector.record_metric(metric)
        
        # Should only keep the last 3
        assert len(collector.metrics["test_metric"]) == 3
        assert collector.metrics["test_metric"][-1].value == 4.0  # Last value
    
    @patch('gan_cyber_range.quality.monitoring.psutil')
    def test_record_performance_snapshot(self, mock_psutil):
        """Test recording performance snapshots."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 75.0
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
        mock_psutil.disk_usage.return_value = Mock(used=50, total=100)
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1024, bytes_recv=2048, 
            packets_sent=10, packets_recv=20
        )
        mock_psutil.pids.return_value = list(range(150))
        mock_psutil.getloadavg.return_value = [1.0, 1.2, 1.5]
        
        collector = MetricsCollector()
        snapshot = collector.record_performance_snapshot()
        
        assert snapshot.cpu_percent == 75.0
        assert snapshot.memory_percent == 60.0
        assert snapshot.disk_usage_percent == 50.0
        assert snapshot.process_count == 150
        assert len(collector.performance_history) == 1
    
    def test_get_metric_summary(self):
        """Test metric summary calculation."""
        collector = MetricsCollector()
        
        # Add test metrics
        base_time = datetime.now()
        values = [80.0, 85.0, 90.0, 75.0, 95.0]
        
        for i, value in enumerate(values):
            metric = QualityMetric(
                name="test_score",
                value=value,
                unit="percent",
                timestamp=base_time + timedelta(minutes=i)
            )
            collector.record_metric(metric)
        
        summary = collector.get_metric_summary("test_score", 60)
        
        assert summary["count"] == 5
        assert summary["min"] == 75.0
        assert summary["max"] == 95.0
        assert summary["avg"] == 85.0  # (80+85+90+75+95)/5
        assert summary["latest"] == 95.0
    
    def test_get_performance_trend(self):
        """Test performance trend calculation."""
        collector = MetricsCollector()
        
        # Add performance snapshots
        base_time = datetime.now()
        for i in range(3):
            snapshot = PerformanceSnapshot(
                timestamp=base_time + timedelta(minutes=i),
                cpu_percent=70.0 + i * 5,  # 70, 75, 80
                memory_percent=60.0 + i * 2,  # 60, 62, 64
                disk_usage_percent=50.0,
                network_io={},
                process_count=100,
                load_average=[1.0]
            )
            collector.performance_history.append(snapshot)
        
        trend = collector.get_performance_trend(60)
        
        assert trend["snapshot_count"] == 3
        assert trend["cpu"]["min"] == 70.0
        assert trend["cpu"]["max"] == 80.0
        assert trend["cpu"]["avg"] == 75.0
        assert trend["cpu"]["current"] == 80.0


class TestQualityMonitor:
    """Test QualityMonitor functionality."""
    
    def test_monitor_creation(self):
        """Test QualityMonitor creation."""
        collector = MetricsCollector()
        monitor = QualityMonitor(
            collector,
            alert_thresholds={"cpu_percent": 85.0, "memory_percent": 90.0}
        )
        
        assert monitor.metrics_collector == collector
        assert monitor.alert_thresholds["cpu_percent"] == 85.0
        assert monitor.alert_thresholds["memory_percent"] == 90.0
        assert len(monitor.alert_handlers) == 0
    
    def test_register_alert_handler(self):
        """Test registering alert handlers."""
        collector = MetricsCollector()
        monitor = QualityMonitor(collector)
        
        handler = Mock()
        monitor.register_alert_handler(handler)
        
        assert len(monitor.alert_handlers) == 1
        assert monitor.alert_handlers[0] == handler
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        collector = MetricsCollector()
        monitor = QualityMonitor(collector)
        
        # Start monitoring
        await monitor.start_monitoring(interval_seconds=0.1)
        assert monitor._monitoring
        assert monitor._monitor_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        assert not monitor._monitoring
    
    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Test alert generation for threshold violations."""
        collector = MetricsCollector()
        monitor = QualityMonitor(
            collector,
            alert_thresholds={"cpu_percent": 75.0, "memory_percent": 80.0}
        )
        
        alert_handler = Mock()
        monitor.register_alert_handler(alert_handler)
        
        # Create snapshot that exceeds thresholds
        high_usage_snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # Above 75.0 threshold
            memory_percent=90.0,  # Above 80.0 threshold
            disk_usage_percent=50.0,
            network_io={},
            process_count=100,
            load_average=[1.0]
        )
        
        await monitor._check_alerts(high_usage_snapshot)
        
        # Should have called alert handler twice (CPU and memory alerts)
        assert alert_handler.call_count == 2


class TestQualityDashboard:
    """Test QualityDashboard functionality."""
    
    def test_dashboard_creation(self):
        """Test QualityDashboard creation."""
        collector = MetricsCollector()
        monitor = QualityMonitor(collector)
        dashboard = QualityDashboard(collector, monitor, update_interval=1)
        
        assert dashboard.metrics_collector == collector
        assert dashboard.monitor == monitor
        assert dashboard.update_interval == 1
        assert not dashboard._running
    
    @pytest.mark.asyncio
    async def test_dashboard_lifecycle(self):
        """Test dashboard start and stop."""
        collector = MetricsCollector()
        monitor = QualityMonitor(collector)
        dashboard = QualityDashboard(collector, monitor, update_interval=0.1)
        
        # Start dashboard
        await dashboard.start_dashboard()
        assert dashboard._running
        
        # Let it update once
        await asyncio.sleep(0.2)
        
        # Check dashboard data
        data = dashboard.get_dashboard_data()
        assert "timestamp" in data
        assert "performance" in data
        assert "alerts" in data
        
        # Stop dashboard
        await dashboard.stop_dashboard()
        assert not dashboard._running
    
    def test_export_dashboard_data(self, tmp_path):
        """Test exporting dashboard data."""
        collector = MetricsCollector()
        monitor = QualityMonitor(collector)
        dashboard = QualityDashboard(collector, monitor)
        
        # Set some test data
        dashboard._dashboard_data = {
            "timestamp": "2024-01-01T12:00:00",
            "test_data": "test_value"
        }
        
        export_file = tmp_path / "dashboard_export.json"
        dashboard.export_dashboard_data(str(export_file))
        
        assert export_file.exists()
        
        import json
        with open(export_file) as f:
            data = json.load(f)
        
        assert data["timestamp"] == "2024-01-01T12:00:00"
        assert data["test_data"] == "test_value"


class TestQualityTrendAnalyzer:
    """Test QualityTrendAnalyzer functionality."""
    
    def test_analyzer_creation(self):
        """Test QualityTrendAnalyzer creation."""
        collector = MetricsCollector()
        analyzer = QualityTrendAnalyzer(collector)
        
        assert analyzer.metrics_collector == collector
    
    def test_analyze_quality_trend_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        collector = MetricsCollector()
        analyzer = QualityTrendAnalyzer(collector)
        
        # Add only one metric
        metric = QualityMetric(
            name="test_score",
            value=85.0,
            unit="percent",
            timestamp=datetime.now()
        )
        collector.record_metric(metric)
        
        analysis = analyzer.analyze_quality_trend("test_score", 24)
        
        assert analysis["status"] == "insufficient_data"
        assert "Not enough data" in analysis["message"]
    
    def test_analyze_quality_trend_improving(self):
        """Test trend analysis for improving metrics."""
        collector = MetricsCollector()
        analyzer = QualityTrendAnalyzer(collector)
        
        # Add metrics with improving trend
        base_time = datetime.now() - timedelta(hours=2)
        values = [70.0, 75.0, 80.0, 85.0, 90.0]  # Improving
        
        for i, value in enumerate(values):
            metric = QualityMetric(
                name="improving_score",
                value=value,
                unit="percent",
                timestamp=base_time + timedelta(minutes=30 * i)
            )
            collector.record_metric(metric)
        
        analysis = analyzer.analyze_quality_trend("improving_score", 24)
        
        assert analysis["status"] == "success"
        assert analysis["trend_direction"] == "improving"
        assert analysis["current_value"] == 90.0
        assert analysis["min_value"] == 70.0
        assert analysis["max_value"] == 90.0
    
    def test_analyze_quality_trend_declining(self):
        """Test trend analysis for declining metrics."""
        collector = MetricsCollector()
        analyzer = QualityTrendAnalyzer(collector)
        
        # Add metrics with declining trend
        base_time = datetime.now() - timedelta(hours=2)
        values = [90.0, 85.0, 80.0, 75.0, 70.0]  # Declining
        
        for i, value in enumerate(values):
            metric = QualityMetric(
                name="declining_score",
                value=value,
                unit="percent",
                timestamp=base_time + timedelta(minutes=30 * i)
            )
            collector.record_metric(metric)
        
        analysis = analyzer.analyze_quality_trend("declining_score", 24)
        
        assert analysis["status"] == "success"
        assert analysis["trend_direction"] == "declining"
        assert analysis["current_value"] == 70.0
    
    def test_generate_quality_report(self):
        """Test comprehensive quality report generation."""
        collector = MetricsCollector()
        analyzer = QualityTrendAnalyzer(collector)
        
        # Add some test metrics
        base_time = datetime.now() - timedelta(hours=1)
        for i in range(5):
            metric = QualityMetric(
                name="test_coverage",
                value=80.0 + i,
                unit="percent",
                timestamp=base_time + timedelta(minutes=10 * i)
            )
            collector.record_metric(metric)
        
        # Add performance snapshots
        for i in range(3):
            snapshot = PerformanceSnapshot(
                timestamp=base_time + timedelta(minutes=20 * i),
                cpu_percent=70.0,
                memory_percent=60.0,
                disk_usage_percent=50.0,
                network_io={},
                process_count=100,
                load_average=[1.0]
            )
            collector.performance_history.append(snapshot)
        
        report = analyzer.generate_quality_report()
        
        assert "generated_at" in report
        assert "metrics_analysis" in report
        assert "performance_analysis" in report
        assert "recommendations" in report
        
        # Check metrics analysis
        assert "test_coverage" in report["metrics_analysis"]
        coverage_analysis = report["metrics_analysis"]["test_coverage"]
        assert coverage_analysis["status"] == "success"
        assert coverage_analysis["trend_direction"] == "improving"


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_pipeline(self):
        """Test complete monitoring pipeline."""
        # Create components
        collector = MetricsCollector(max_history=10)
        monitor = QualityMonitor(collector, alert_thresholds={"cpu_percent": 90.0})
        dashboard = QualityDashboard(collector, monitor, update_interval=0.1)
        analyzer = QualityTrendAnalyzer(collector)
        
        # Add some test data
        for i in range(5):
            metric = QualityMetric(
                name="integration_test",
                value=80.0 + i,
                unit="percent",
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            collector.record_metric(metric)
        
        # Start monitoring briefly
        await monitor.start_monitoring(interval_seconds=0.1)
        await dashboard.start_dashboard()
        
        # Let it run
        await asyncio.sleep(0.2)
        
        # Check results
        dashboard_data = dashboard.get_dashboard_data()
        assert "timestamp" in dashboard_data
        
        summary = collector.get_metric_summary("integration_test", 60)
        assert summary["count"] == 5
        
        analysis = analyzer.analyze_quality_trend("integration_test", 1)
        assert analysis["status"] == "success"
        
        # Cleanup
        await monitor.stop_monitoring()
        await dashboard.stop_dashboard()


if __name__ == "__main__":
    pytest.main([__file__])