# 📈 Metrics Collection & Monitoring

## Prometheus Metrics Framework

### Application Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps

# Business Metrics
attack_attempts = Counter('cyber_range_attack_attempts_total', 
                         'Total number of attack attempts', ['attack_type', 'target'])
successful_attacks = Counter('cyber_range_successful_attacks_total',
                           'Total number of successful attacks', ['attack_type', 'target'])
detection_time = Histogram('cyber_range_detection_time_seconds',
                          'Time to detect an attack', ['attack_type'])
patch_deployment_time = Histogram('cyber_range_patch_deployment_seconds',
                                 'Time to deploy a patch', ['vulnerability_type'])

# Performance Metrics
request_duration = Histogram('http_request_duration_seconds',
                           'HTTP request duration', ['method', 'endpoint'])
request_count = Counter('http_requests_total',
                       'Total HTTP requests', ['method', 'endpoint', 'status'])

# Infrastructure Metrics
active_connections = Gauge('active_database_connections', 'Active database connections')
queue_size = Gauge('task_queue_size', 'Number of tasks in queue')
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')

# Security Metrics
failed_auth_attempts = Counter('failed_authentication_attempts_total',
                              'Failed authentication attempts', ['ip_address'])
suspicious_activities = Counter('suspicious_activities_total',
                               'Suspicious activities detected', ['activity_type'])

def track_request_metrics(func):
    """Decorator to track HTTP request metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            status = '200'
            return result
        except Exception as e:
            status = '500'
            raise
        finally:
            duration = time.time() - start_time
            request_duration.observe(duration)
            request_count.labels(
                method='GET',  # Extract from request
                endpoint='/api/v1',  # Extract from request
                status=status
            ).inc()
    return wrapper
```

### Cyber Range Specific Metrics

```python
class CyberRangeMetrics:
    """Specialized metrics for cyber range operations"""
    
    def __init__(self):
        # Attack simulation metrics
        self.attack_success_rate = Gauge(
            'attack_success_rate_percentage',
            'Percentage of successful attacks'
        )
        self.mean_time_to_compromise = Histogram(
            'mean_time_to_compromise_seconds',
            'Average time to successful compromise'
        )
        self.mean_time_to_detection = Histogram(
            'mean_time_to_detection_seconds',
            'Average time to detect an attack'
        )
        self.mean_time_to_remediation = Histogram(
            'mean_time_to_remediation_seconds',
            'Average time to remediate a vulnerability'
        )
        
        # Agent performance metrics
        self.red_team_effectiveness = Gauge(
            'red_team_effectiveness_score',
            'Red team effectiveness score (0-100)'
        )
        self.blue_team_effectiveness = Gauge(
            'blue_team_effectiveness_score',
            'Blue team effectiveness score (0-100)'
        )
        
        # Environment metrics
        self.vulnerable_services_count = Gauge(
            'vulnerable_services_active_count',
            'Number of active vulnerable services'
        )
        self.patched_vulnerabilities = Counter(
            'patched_vulnerabilities_total',
            'Total vulnerabilities patched'
        )
        
    def record_attack_attempt(self, attack_type: str, target: str):
        """Record an attack attempt"""
        attack_attempts.labels(attack_type=attack_type, target=target).inc()
        
    def record_successful_attack(self, attack_type: str, target: str, time_to_compromise: float):
        """Record a successful attack"""
        successful_attacks.labels(attack_type=attack_type, target=target).inc()
        self.mean_time_to_compromise.observe(time_to_compromise)
        
    def record_detection(self, attack_type: str, detection_time_seconds: float):
        """Record attack detection"""
        detection_time.labels(attack_type=attack_type).observe(detection_time_seconds)
        
    def record_patch_deployment(self, vuln_type: str, deployment_time: float):
        """Record patch deployment"""
        patch_deployment_time.labels(vulnerability_type=vuln_type).observe(deployment_time)
        
    def update_success_rate(self, success_rate: float):
        """Update overall attack success rate"""
        self.attack_success_rate.set(success_rate)
        
    def update_team_effectiveness(self, red_score: float, blue_score: float):
        """Update team effectiveness scores"""
        self.red_team_effectiveness.set(red_score)
        self.blue_team_effectiveness.set(blue_score)
```

### Custom Metrics Collection

```python
import asyncio
from typing import Dict, Any

class MetricsCollector:
    """Advanced metrics collection and aggregation"""
    
    def __init__(self):
        self.cyber_range_metrics = CyberRangeMetrics()
        self.collection_interval = 30  # seconds
        
    async def start_collection(self):
        """Start continuous metrics collection"""
        while True:
            try:
                await self.collect_business_metrics()
                await self.collect_performance_metrics()
                await self.collect_security_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)  # Shorter retry interval
                
    async def collect_business_metrics(self):
        """Collect business-specific metrics"""
        # Calculate attack success rate
        total_attacks = await self.get_total_attacks()
        successful_attacks = await self.get_successful_attacks()
        success_rate = (successful_attacks / total_attacks * 100) if total_attacks > 0 else 0
        
        self.cyber_range_metrics.update_success_rate(success_rate)
        
        # Update active services count
        active_services = await self.get_active_vulnerable_services()
        self.cyber_range_metrics.vulnerable_services_count.set(len(active_services))
        
    async def collect_performance_metrics(self):
        """Collect system performance metrics"""
        # Database connections
        db_connections = await self.get_active_db_connections()
        active_connections.set(db_connections)
        
        # Queue sizes
        task_queue = await self.get_task_queue_size()
        queue_size.set(task_queue)
        
        # Memory usage
        memory_bytes = await self.get_memory_usage()
        memory_usage.set(memory_bytes)
        
    async def collect_security_metrics(self):
        """Collect security-related metrics"""
        # This would integrate with your security monitoring
        pass
        
    async def get_total_attacks(self) -> int:
        """Get total number of attacks from database"""
        # Implementation depends on your data store
        return 0
        
    async def get_successful_attacks(self) -> int:
        """Get number of successful attacks"""
        return 0
        
    async def get_active_vulnerable_services(self) -> list:
        """Get list of active vulnerable services"""
        return []
        
    async def get_active_db_connections(self) -> int:
        """Get number of active database connections"""
        return 0
        
    async def get_task_queue_size(self) -> int:
        """Get current task queue size"""
        return 0
        
    async def get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        import psutil
        return psutil.virtual_memory().used
```

## Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'gan-cyber-range'

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'gan-cyber-range-app'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
    scrape_timeout: 4s
    
  - job_name: 'gan-cyber-range-business'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics/business'
    scrape_interval: 10s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
```

## Alert Rules

```yaml
# monitoring/prometheus/alert_rules.yml
groups:
  - name: cyber_range_alerts
    rules:
      - alert: HighAttackSuccessRate
        expr: attack_success_rate_percentage > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High attack success rate detected"
          description: "Attack success rate is {{ $value }}% which is above threshold"
          
      - alert: SlowDetectionTime
        expr: rate(mean_time_to_detection_seconds[5m]) > 300
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Slow attack detection"
          description: "Average detection time is {{ $value }} seconds"
          
      - alert: ServiceDown
        expr: up{job="gan-cyber-range-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "GAN Cyber Range service has been down for more than 1 minute"
          
      - alert: HighMemoryUsage
        expr: memory_usage_bytes / (1024 * 1024 * 1024) > 3.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"
```

## Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "GAN Cyber Range Metrics",
    "description": "Comprehensive metrics dashboard for cyber range operations",
    "tags": ["cyber-range", "security", "monitoring"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Attack Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "attack_success_rate_percentage",
            "legendFormat": "Success Rate %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 50},
                {"color": "red", "value": 80}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Attack Attempts Over Time",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cyber_range_attack_attempts_total[5m])",
            "legendFormat": "{{attack_type}} on {{target}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "Detection & Remediation Times",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(mean_time_to_detection_seconds[5m])",
            "legendFormat": "Detection Time"
          },
          {
            "expr": "rate(mean_time_to_remediation_seconds[5m])",
            "legendFormat": "Remediation Time"
          }
        ]
      }
    ]
  }
}
```