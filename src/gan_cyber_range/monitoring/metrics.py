"""Metrics collection and monitoring."""

import time
from datetime import datetime
from typing import Any, Dict, List
from dataclasses import dataclass, field
from prometheus_client import Counter, Histogram, Gauge, start_http_server


@dataclass
class Metric:
    """Represents a metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    type: str = "gauge"


class MetricsCollector:
    """Collects and exposes metrics for cyber range simulation."""
    
    def __init__(self, enable_prometheus: bool = True, prometheus_port: int = 8000):
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        self.metrics_buffer: List[Metric] = []
        
        # Prometheus metrics
        if enable_prometheus:
            self._setup_prometheus_metrics()
            start_http_server(prometheus_port)
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        # Simulation metrics
        self.simulation_duration = Histogram(
            "cyber_range_simulation_duration_seconds",
            "Duration of cyber range simulations"
        )
        
        self.attacks_total = Counter(
            "cyber_range_attacks_total",
            "Total number of attacks",
            ["agent_type", "attack_type", "target"]
        )
        
        self.attacks_successful = Counter(
            "cyber_range_attacks_successful_total",
            "Total number of successful attacks",
            ["agent_type", "attack_type", "target"]
        )
        
        self.defenses_total = Counter(
            "cyber_range_defenses_total",
            "Total number of defensive actions",
            ["agent_type", "defense_type", "target"]
        )
        
        self.defenses_successful = Counter(
            "cyber_range_defenses_successful_total",
            "Total number of successful defensive actions",
            ["agent_type", "defense_type", "target"]
        )
        
        # System metrics
        self.services_compromised = Gauge(
            "cyber_range_services_compromised",
            "Number of compromised services"
        )
        
        self.services_total = Gauge(
            "cyber_range_services_total",
            "Total number of services"
        )
        
        self.agent_success_rate = Gauge(
            "cyber_range_agent_success_rate",
            "Agent success rate",
            ["agent_name", "agent_type"]
        )
        
        # Performance metrics
        self.action_execution_time = Histogram(
            "cyber_range_action_execution_seconds",
            "Time taken to execute actions",
            ["agent_type", "action_type"]
        )
    
    def record_attack(self, agent_name: str, attack_type: str, target: str, success: bool, execution_time: float = None) -> None:
        """Record an attack action."""
        labels = {"agent_type": "red_team", "attack_type": attack_type, "target": target}
        
        self.attacks_total.labels(**labels).inc()
        if success:
            self.attacks_successful.labels(**labels).inc()
        
        if execution_time and self.enable_prometheus:
            self.action_execution_time.labels(agent_type="red_team", action_type=attack_type).observe(execution_time)
        
        # Buffer for custom analytics
        metric = Metric(
            name="attack",
            value=1.0 if success else 0.0,
            labels={**labels, "agent_name": agent_name, "success": str(success)}
        )
        self.metrics_buffer.append(metric)
    
    def record_defense(self, agent_name: str, defense_type: str, target: str, success: bool, execution_time: float = None) -> None:
        """Record a defensive action."""
        labels = {"agent_type": "blue_team", "defense_type": defense_type, "target": target}
        
        self.defenses_total.labels(**labels).inc()
        if success:
            self.defenses_successful.labels(**labels).inc()
        
        if execution_time and self.enable_prometheus:
            self.action_execution_time.labels(agent_type="blue_team", action_type=defense_type).observe(execution_time)
        
        # Buffer for custom analytics
        metric = Metric(
            name="defense",
            value=1.0 if success else 0.0,
            labels={**labels, "agent_name": agent_name, "success": str(success)}
        )
        self.metrics_buffer.append(metric)
    
    def record_simulation_duration(self, duration_seconds: float) -> None:
        """Record simulation duration."""
        if self.enable_prometheus:
            self.simulation_duration.observe(duration_seconds)
        
        metric = Metric(
            name="simulation_duration",
            value=duration_seconds,
            type="histogram"
        )
        self.metrics_buffer.append(metric)
    
    def update_service_metrics(self, services_total: int, services_compromised: int) -> None:
        """Update service-related metrics."""
        if self.enable_prometheus:
            self.services_total.set(services_total)
            self.services_compromised.set(services_compromised)
        
        self.metrics_buffer.append(Metric("services_total", services_total))
        self.metrics_buffer.append(Metric("services_compromised", services_compromised))
    
    def update_agent_success_rate(self, agent_name: str, agent_type: str, success_rate: float) -> None:
        """Update agent success rate."""
        if self.enable_prometheus:
            self.agent_success_rate.labels(agent_name=agent_name, agent_type=agent_type).set(success_rate)
        
        metric = Metric(
            name="agent_success_rate",
            value=success_rate,
            labels={"agent_name": agent_name, "agent_type": agent_type}
        )
        self.metrics_buffer.append(metric)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_buffer:
            return {}
        
        summary = {
            "total_metrics": len(self.metrics_buffer),
            "metrics_by_type": {},
            "recent_metrics": self.metrics_buffer[-10:],
            "collection_period": {
                "start": min(m.timestamp for m in self.metrics_buffer).isoformat(),
                "end": max(m.timestamp for m in self.metrics_buffer).isoformat()
            }
        }
        
        # Count metrics by type
        for metric in self.metrics_buffer:
            metric_type = metric.name
            if metric_type not in summary["metrics_by_type"]:
                summary["metrics_by_type"][metric_type] = 0
            summary["metrics_by_type"][metric_type] += 1
        
        return summary
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        if format_type == "json":
            import json
            metrics_data = []
            for metric in self.metrics_buffer:
                metrics_data.append({
                    "name": metric.name,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "labels": metric.labels,
                    "type": metric.type
                })
            return json.dumps(metrics_data, indent=2)
        
        elif format_type == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["name", "value", "timestamp", "labels", "type"])
            
            for metric in self.metrics_buffer:
                writer.writerow([
                    metric.name,
                    metric.value,
                    metric.timestamp.isoformat(),
                    str(metric.labels),
                    metric.type
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def clear_buffer(self, keep_last: int = 100) -> None:
        """Clear metrics buffer, optionally keeping recent metrics."""
        if keep_last > 0:
            self.metrics_buffer = self.metrics_buffer[-keep_last:]
        else:
            self.metrics_buffer.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from collected metrics."""
        if not self.metrics_buffer:
            return {}
        
        # Calculate attack success rates
        attack_metrics = [m for m in self.metrics_buffer if m.name == "attack"]
        total_attacks = len(attack_metrics)
        successful_attacks = len([m for m in attack_metrics if m.value == 1.0])
        
        # Calculate defense success rates
        defense_metrics = [m for m in self.metrics_buffer if m.name == "defense"]
        total_defenses = len(defense_metrics)
        successful_defenses = len([m for m in defense_metrics if m.value == 1.0])
        
        return {
            "attack_stats": {
                "total": total_attacks,
                "successful": successful_attacks,
                "success_rate": successful_attacks / total_attacks if total_attacks > 0 else 0
            },
            "defense_stats": {
                "total": total_defenses,
                "successful": successful_defenses,
                "success_rate": successful_defenses / total_defenses if total_defenses > 0 else 0
            },
            "overall_stats": {
                "total_actions": total_attacks + total_defenses,
                "defense_effectiveness": successful_defenses / total_attacks if total_attacks > 0 else 1.0
            }
        }