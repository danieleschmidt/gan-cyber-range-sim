"""Real-time security dashboard and monitoring system."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


@dataclass
class SecurityMetric:
    """Security metric data point."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str  # threats, incidents, network, systems
    tags: Dict[str, str]
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, critical


@dataclass
class ThreatIndicator:
    """Real-time threat indicator."""
    id: str
    type: str
    value: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    source: str
    severity: str


@dataclass
class SecurityAlert:
    """Security alert for dashboard."""
    id: str
    title: str
    severity: str
    source: str
    timestamp: datetime
    affected_assets: List[str]
    status: str  # new, acknowledged, investigating, resolved
    indicators: List[str]


class SecurityDashboard:
    """Real-time security monitoring dashboard."""
    
    def __init__(self):
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.current_metrics = {}
        self.active_alerts = []
        self.threat_indicators = {}
        self.connected_clients = set()
        self.logger = logging.getLogger("SecurityDashboard")
        
        # Initialize FastAPI app
        self.app = FastAPI(title="GAN Cyber Range Security Dashboard")
        self.setup_routes()
        
        # Metric thresholds and configurations
        self.metric_configs = {
            "threat_detection_rate": {
                "unit": "threats/hour",
                "warning_threshold": 10.0,
                "critical_threshold": 25.0,
                "category": "threats"
            },
            "incident_response_time": {
                "unit": "minutes",
                "warning_threshold": 30.0,
                "critical_threshold": 60.0,
                "category": "incidents"
            },
            "network_anomalies": {
                "unit": "anomalies/hour",
                "warning_threshold": 5.0,
                "critical_threshold": 15.0,
                "category": "network"
            },
            "system_compromise_risk": {
                "unit": "risk_score",
                "warning_threshold": 0.7,
                "critical_threshold": 0.9,
                "category": "systems"
            },
            "failed_auth_attempts": {
                "unit": "attempts/hour",
                "warning_threshold": 50.0,
                "critical_threshold": 100.0,
                "category": "threats"
            },
            "data_exfiltration_risk": {
                "unit": "risk_score",
                "warning_threshold": 0.6,
                "critical_threshold": 0.8,
                "category": "threats"
            },
            "honeypot_interactions": {
                "unit": "interactions/hour",
                "warning_threshold": 3.0,
                "critical_threshold": 10.0,
                "category": "threats"
            },
            "patch_compliance": {
                "unit": "percentage",
                "warning_threshold": 85.0,
                "critical_threshold": 70.0,
                "category": "systems",
                "invert_thresholds": True  # Lower values are worse
            }
        }
    
    def setup_routes(self):
        """Setup FastAPI routes for the dashboard."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            return self.get_dashboard_html()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_handler(websocket)
        
        @self.app.get("/api/metrics/current")
        async def get_current_metrics():
            return {"metrics": self.get_current_metrics_summary()}
        
        @self.app.get("/api/metrics/historical/{metric_name}")
        async def get_metric_history(metric_name: str, hours: int = 24):
            return {"data": self.get_metric_history(metric_name, hours)}
        
        @self.app.get("/api/alerts/active")
        async def get_active_alerts():
            return {"alerts": [asdict(alert) for alert in self.active_alerts]}
        
        @self.app.get("/api/threats/indicators")
        async def get_threat_indicators():
            return {"indicators": [asdict(indicator) for indicator in self.threat_indicators.values()]}
        
        @self.app.get("/api/dashboard/summary")
        async def get_dashboard_summary():
            return await self.get_dashboard_summary()
    
    async def websocket_handler(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates."""
        await websocket.accept()
        self.connected_clients.add(websocket)
        
        try:
            # Send initial dashboard state
            initial_data = await self.get_dashboard_summary()
            await websocket.send_json(initial_data)
            
            # Keep connection alive and handle client messages
            while True:
                try:
                    # Wait for client message or timeout
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                    
                    # Handle client requests
                    if data.get("type") == "subscribe_metric":
                        metric_name = data.get("metric_name")
                        if metric_name in self.current_metrics:
                            metric_data = {
                                "type": "metric_update",
                                "metric": metric_name,
                                "data": self.current_metrics[metric_name]
                            }
                            await websocket.send_json(metric_data)
                
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
                
        except WebSocketDisconnect:
            pass
        finally:
            self.connected_clients.discard(websocket)
    
    async def update_metric(
        self,
        name: str,
        value: float,
        tags: Dict[str, str] = None,
        timestamp: datetime = None
    ) -> None:
        """Update a security metric and broadcast to connected clients."""
        timestamp = timestamp or datetime.now()
        tags = tags or {}
        
        config = self.metric_configs.get(name, {})
        
        # Determine status based on thresholds
        status = "normal"
        if config.get("invert_thresholds", False):
            # For metrics where lower values are worse (e.g., patch compliance)
            if value < config.get("critical_threshold", 0):
                status = "critical"
            elif value < config.get("warning_threshold", 0):
                status = "warning"
        else:
            # For metrics where higher values are worse
            if value > config.get("critical_threshold", float('inf')):
                status = "critical"
            elif value > config.get("warning_threshold", float('inf')):
                status = "warning"
        
        metric = SecurityMetric(
            name=name,
            value=value,
            unit=config.get("unit", "count"),
            timestamp=timestamp,
            category=config.get("category", "general"),
            tags=tags,
            threshold=config.get("warning_threshold"),
            status=status
        )
        
        # Store in buffer
        self.metrics_buffer[name].append(metric)
        self.current_metrics[name] = metric
        
        # Broadcast to connected clients
        await self.broadcast_metric_update(metric)
        
        # Log critical metrics
        if status == "critical":
            self.logger.critical(f"CRITICAL METRIC: {name} = {value} {metric.unit}")
        elif status == "warning":
            self.logger.warning(f"WARNING METRIC: {name} = {value} {metric.unit}")
    
    async def add_security_alert(
        self,
        alert_id: str,
        title: str,
        severity: str,
        source: str,
        affected_assets: List[str] = None,
        indicators: List[str] = None
    ) -> None:
        """Add a new security alert to the dashboard."""
        alert = SecurityAlert(
            id=alert_id,
            title=title,
            severity=severity,
            source=source,
            timestamp=datetime.now(),
            affected_assets=affected_assets or [],
            status="new",
            indicators=indicators or []
        )
        
        self.active_alerts.append(alert)
        
        # Keep only recent alerts (last 100)
        self.active_alerts = sorted(self.active_alerts, key=lambda x: x.timestamp, reverse=True)[:100]
        
        # Broadcast to connected clients
        await self.broadcast_alert_update(alert)
        
        self.logger.warning(f"NEW ALERT [{severity.upper()}]: {title}")
    
    async def add_threat_indicator(
        self,
        indicator_id: str,
        indicator_type: str,
        value: str,
        confidence: float,
        source: str,
        severity: str
    ) -> None:
        """Add or update a threat indicator."""
        now = datetime.now()
        
        if indicator_id in self.threat_indicators:
            # Update existing indicator
            indicator = self.threat_indicators[indicator_id]
            indicator.last_seen = now
            indicator.confidence = max(indicator.confidence, confidence)
        else:
            # Create new indicator
            indicator = ThreatIndicator(
                id=indicator_id,
                type=indicator_type,
                value=value,
                confidence=confidence,
                first_seen=now,
                last_seen=now,
                source=source,
                severity=severity
            )
            self.threat_indicators[indicator_id] = indicator
        
        # Broadcast to connected clients
        await self.broadcast_threat_indicator_update(indicator)
    
    async def broadcast_metric_update(self, metric: SecurityMetric) -> None:
        """Broadcast metric update to all connected clients."""
        message = {
            "type": "metric_update",
            "data": {
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "status": metric.status,
                "category": metric.category
            }
        }
        
        await self.broadcast_to_clients(message)
    
    async def broadcast_alert_update(self, alert: SecurityAlert) -> None:
        """Broadcast new alert to all connected clients."""
        message = {
            "type": "new_alert",
            "data": {
                "id": alert.id,
                "title": alert.title,
                "severity": alert.severity,
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "affected_assets": alert.affected_assets,
                "status": alert.status
            }
        }
        
        await self.broadcast_to_clients(message)
    
    async def broadcast_threat_indicator_update(self, indicator: ThreatIndicator) -> None:
        """Broadcast threat indicator update to all connected clients."""
        message = {
            "type": "threat_indicator_update",
            "data": {
                "id": indicator.id,
                "type": indicator.type,
                "value": indicator.value,
                "confidence": indicator.confidence,
                "severity": indicator.severity,
                "last_seen": indicator.last_seen.isoformat()
            }
        }
        
        await self.broadcast_to_clients(message)
    
    async def broadcast_to_clients(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected WebSocket clients."""
        if not self.connected_clients:
            return
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send_json(message)
            except:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metric values."""
        summary = {
            "threats": {},
            "incidents": {},
            "network": {},
            "systems": {}
        }
        
        for name, metric in self.current_metrics.items():
            summary[metric.category][name] = {
                "value": metric.value,
                "unit": metric.unit,
                "status": metric.status,
                "timestamp": metric.timestamp.isoformat()
            }
        
        return summary
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric."""
        if metric_name not in self.metrics_buffer:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        historical_data = []
        for metric in self.metrics_buffer[metric_name]:
            if metric.timestamp > cutoff_time:
                historical_data.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.value,
                    "status": metric.status
                })
        
        return historical_data
    
    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary."""
        now = datetime.now()
        
        # Calculate summary statistics
        critical_alerts = len([a for a in self.active_alerts if a.severity == "critical"])
        high_alerts = len([a for a in self.active_alerts if a.severity == "high"])
        
        critical_metrics = len([m for m in self.current_metrics.values() if m.status == "critical"])
        warning_metrics = len([m for m in self.current_metrics.values() if m.status == "warning"])
        
        high_confidence_indicators = len([
            i for i in self.threat_indicators.values()
            if i.confidence >= 0.8
        ])
        
        # Calculate trend data (last 4 hours vs previous 4 hours)
        threat_trend = await self.calculate_threat_trend()
        
        return {
            "timestamp": now.isoformat(),
            "overview": {
                "security_status": self.calculate_overall_security_status(),
                "critical_alerts": critical_alerts,
                "high_alerts": high_alerts,
                "critical_metrics": critical_metrics,
                "warning_metrics": warning_metrics,
                "active_indicators": len(self.threat_indicators),
                "high_confidence_indicators": high_confidence_indicators
            },
            "metrics": self.get_current_metrics_summary(),
            "recent_alerts": [
                {
                    "id": alert.id,
                    "title": alert.title,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.active_alerts[:5]  # Last 5 alerts
            ],
            "threat_trends": threat_trend,
            "system_health": await self.calculate_system_health()
        }
    
    def calculate_overall_security_status(self) -> str:
        """Calculate overall security status based on current metrics and alerts."""
        critical_alerts = len([a for a in self.active_alerts if a.severity == "critical"])
        critical_metrics = len([m for m in self.current_metrics.values() if m.status == "critical"])
        
        if critical_alerts > 0 or critical_metrics > 0:
            return "critical"
        
        high_alerts = len([a for a in self.active_alerts if a.severity == "high"])
        warning_metrics = len([m for m in self.current_metrics.values() if m.status == "warning"])
        
        if high_alerts > 2 or warning_metrics > 3:
            return "degraded"
        
        if high_alerts > 0 or warning_metrics > 0:
            return "warning"
        
        return "normal"
    
    async def calculate_threat_trend(self) -> Dict[str, Any]:
        """Calculate threat trends over time."""
        now = datetime.now()
        
        # Current period (last 4 hours)
        current_start = now - timedelta(hours=4)
        current_alerts = len([
            a for a in self.active_alerts
            if a.timestamp > current_start
        ])
        
        # Previous period (4-8 hours ago)
        previous_start = now - timedelta(hours=8)
        previous_end = now - timedelta(hours=4)
        previous_alerts = len([
            a for a in self.active_alerts
            if previous_start < a.timestamp <= previous_end
        ])
        
        # Calculate trend
        if previous_alerts == 0:
            trend = "stable" if current_alerts == 0 else "increasing"
            change_percent = 0
        else:
            change_percent = ((current_alerts - previous_alerts) / previous_alerts) * 100
            if change_percent > 20:
                trend = "increasing"
            elif change_percent < -20:
                trend = "decreasing"
            else:
                trend = "stable"
        
        return {
            "current_period_alerts": current_alerts,
            "previous_period_alerts": previous_alerts,
            "trend": trend,
            "change_percent": round(change_percent, 1)
        }
    
    async def calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health metrics."""
        # Get system-related metrics
        system_metrics = {
            name: metric for name, metric in self.current_metrics.items()
            if metric.category == "systems"
        }
        
        if not system_metrics:
            return {"status": "unknown", "score": 0.0}
        
        # Calculate health score (0-100)
        total_score = 0
        metric_count = 0
        
        for metric in system_metrics.values():
            if metric.status == "normal":
                total_score += 100
            elif metric.status == "warning":
                total_score += 60
            elif metric.status == "critical":
                total_score += 20
            metric_count += 1
        
        health_score = total_score / metric_count if metric_count > 0 else 100
        
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "status": status,
            "score": round(health_score, 1),
            "metric_count": metric_count
        }
    
    def get_dashboard_html(self) -> str:
        """Generate HTML for the security dashboard."""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>GAN Cyber Range Security Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #1a1a1a;
                    color: #ffffff;
                }
                .dashboard-header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                .dashboard-header h1 {
                    color: #00d4ff;
                    margin: 0;
                    font-size: 2.5em;
                }
                .status-overview {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .status-card {
                    background: #2a2a2a;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                    border-left: 4px solid #00d4ff;
                }
                .status-card.critical {
                    border-left-color: #ff4757;
                }
                .status-card.warning {
                    border-left-color: #ffa502;
                }
                .status-card h3 {
                    margin: 0 0 10px 0;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    color: #888;
                }
                .status-card .value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #00d4ff;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .metric-card {
                    background: #2a2a2a;
                    border-radius: 8px;
                    padding: 20px;
                }
                .metric-card h4 {
                    margin: 0 0 15px 0;
                    color: #00d4ff;
                    text-transform: uppercase;
                    font-size: 0.9em;
                }
                .alerts-section {
                    background: #2a2a2a;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 30px;
                }
                .alerts-section h3 {
                    margin: 0 0 20px 0;
                    color: #00d4ff;
                }
                .alert-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 4px;
                    background: #1a1a1a;
                }
                .alert-item.critical {
                    border-left: 4px solid #ff4757;
                }
                .alert-item.high {
                    border-left: 4px solid #ffa502;
                }
                .alert-item.medium {
                    border-left: 4px solid #3742fa;
                }
                .connection-status {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 10px 15px;
                    border-radius: 20px;
                    background: #2ed573;
                    color: white;
                    font-size: 0.8em;
                    font-weight: bold;
                }
                .connection-status.disconnected {
                    background: #ff4757;
                }
            </style>
        </head>
        <body>
            <div class="connection-status" id="connectionStatus">Connected</div>
            
            <div class="dashboard-header">
                <h1>🛡️ Security Operations Center</h1>
                <p>Real-time Security Monitoring Dashboard</p>
            </div>
            
            <div class="status-overview" id="statusOverview">
                <!-- Status cards will be populated here -->
            </div>
            
            <div class="metrics-grid" id="metricsGrid">
                <!-- Metric cards will be populated here -->
            </div>
            
            <div class="alerts-section">
                <h3>Recent Security Alerts</h3>
                <div id="alertsList">
                    <!-- Alerts will be populated here -->
                </div>
            </div>
            
            <script>
                class SecurityDashboard {
                    constructor() {
                        this.ws = null;
                        this.reconnectDelay = 1000;
                        this.maxReconnectDelay = 30000;
                        this.connect();
                    }
                    
                    connect() {
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                        
                        this.ws.onopen = () => {
                            console.log('Connected to dashboard');
                            this.updateConnectionStatus(true);
                            this.reconnectDelay = 1000;
                        };
                        
                        this.ws.onmessage = (event) => {
                            const data = JSON.parse(event.data);
                            this.handleMessage(data);
                        };
                        
                        this.ws.onclose = () => {
                            console.log('Disconnected from dashboard');
                            this.updateConnectionStatus(false);
                            this.scheduleReconnect();
                        };
                        
                        this.ws.onerror = (error) => {
                            console.error('WebSocket error:', error);
                        };
                    }
                    
                    scheduleReconnect() {
                        setTimeout(() => {
                            this.connect();
                        }, this.reconnectDelay);
                        
                        this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
                    }
                    
                    updateConnectionStatus(connected) {
                        const status = document.getElementById('connectionStatus');
                        if (connected) {
                            status.textContent = 'Connected';
                            status.className = 'connection-status';
                        } else {
                            status.textContent = 'Disconnected';
                            status.className = 'connection-status disconnected';
                        }
                    }
                    
                    handleMessage(data) {
                        switch (data.type) {
                            case 'dashboard_summary':
                                this.updateDashboard(data);
                                break;
                            case 'metric_update':
                                this.updateMetric(data.data);
                                break;
                            case 'new_alert':
                                this.addAlert(data.data);
                                break;
                            case 'threat_indicator_update':
                                this.updateThreatIndicator(data.data);
                                break;
                        }
                    }
                    
                    updateDashboard(data) {
                        if (data.overview) {
                            this.updateStatusOverview(data.overview);
                        }
                        if (data.metrics) {
                            this.updateMetrics(data.metrics);
                        }
                        if (data.recent_alerts) {
                            this.updateAlerts(data.recent_alerts);
                        }
                    }
                    
                    updateStatusOverview(overview) {
                        const container = document.getElementById('statusOverview');
                        container.innerHTML = `
                            <div class="status-card ${overview.security_status}">
                                <h3>Security Status</h3>
                                <div class="value">${overview.security_status.toUpperCase()}</div>
                            </div>
                            <div class="status-card ${overview.critical_alerts > 0 ? 'critical' : ''}">
                                <h3>Critical Alerts</h3>
                                <div class="value">${overview.critical_alerts}</div>
                            </div>
                            <div class="status-card ${overview.warning_metrics > 0 ? 'warning' : ''}">
                                <h3>Warning Metrics</h3>
                                <div class="value">${overview.warning_metrics}</div>
                            </div>
                            <div class="status-card">
                                <h3>Threat Indicators</h3>
                                <div class="value">${overview.active_indicators}</div>
                            </div>
                        `;
                    }
                    
                    updateMetrics(metrics) {
                        const container = document.getElementById('metricsGrid');
                        let html = '';
                        
                        for (const [category, categoryMetrics] of Object.entries(metrics)) {
                            html += `<div class="metric-card">
                                <h4>${category.toUpperCase()} METRICS</h4>`;
                            
                            for (const [name, metric] of Object.entries(categoryMetrics)) {
                                const statusClass = metric.status === 'critical' ? 'critical' : 
                                                  metric.status === 'warning' ? 'warning' : '';
                                html += `<div class="metric-item ${statusClass}">
                                    <span>${name}: ${metric.value} ${metric.unit}</span>
                                    <span class="status">${metric.status}</span>
                                </div>`;
                            }
                            
                            html += '</div>';
                        }
                        
                        container.innerHTML = html;
                    }
                    
                    updateAlerts(alerts) {
                        const container = document.getElementById('alertsList');
                        let html = '';
                        
                        for (const alert of alerts) {
                            const time = new Date(alert.timestamp).toLocaleTimeString();
                            html += `<div class="alert-item ${alert.severity}">
                                <span>${alert.title}</span>
                                <span>${time}</span>
                            </div>`;
                        }
                        
                        container.innerHTML = html || '<p>No recent alerts</p>';
                    }
                    
                    updateMetric(metric) {
                        // Update individual metric in real-time
                        console.log('Metric update:', metric);
                    }
                    
                    addAlert(alert) {
                        // Add new alert to the list
                        console.log('New alert:', alert);
                        // You could add visual/audio notifications here
                    }
                    
                    updateThreatIndicator(indicator) {
                        // Update threat indicator
                        console.log('Threat indicator update:', indicator);
                    }
                }
                
                // Initialize dashboard when page loads
                document.addEventListener('DOMContentLoaded', () => {
                    new SecurityDashboard();
                });
            </script>
        </body>
        </html>
        '''
    
    async def start_metric_simulation(self) -> None:
        """Start simulated metrics for demonstration purposes."""
        while True:
            try:
                import random
                
                # Simulate various security metrics
                await self.update_metric(
                    "threat_detection_rate",
                    random.uniform(0, 30),
                    {"source": "siem"}
                )
                
                await self.update_metric(
                    "incident_response_time",
                    random.uniform(5, 120),
                    {"team": "soc"}
                )
                
                await self.update_metric(
                    "network_anomalies",
                    random.uniform(0, 20),
                    {"network_segment": "dmz"}
                )
                
                await self.update_metric(
                    "system_compromise_risk",
                    random.uniform(0, 1),
                    {"risk_model": "ml_based"}
                )
                
                await self.update_metric(
                    "failed_auth_attempts",
                    random.uniform(0, 150),
                    {"auth_system": "ldap"}
                )
                
                await self.update_metric(
                    "patch_compliance",
                    random.uniform(60, 100),
                    {"system_type": "servers"}
                )
                
                # Occasionally generate alerts
                if random.random() < 0.1:  # 10% chance
                    severities = ["low", "medium", "high", "critical"]
                    await self.add_security_alert(
                        f"alert_{datetime.now().strftime('%H%M%S')}",
                        f"Simulated security event detected",
                        random.choice(severities),
                        "automated_detection",
                        [f"server-{random.randint(1, 10)}"]
                    )
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metric simulation: {e}")
                await asyncio.sleep(5)