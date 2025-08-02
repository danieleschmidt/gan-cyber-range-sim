# 📝 Structured Logging Configuration

## Python Logging Setup

### Structured JSON Logging

```python
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import traceback
import uuid

class StructuredLogger:
    """Structured JSON logger for cyber range operations"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove default handlers
        self.logger.handlers.clear()
        
        # Add structured handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        
        # Add file handler for persistent logs
        file_handler = logging.FileHandler('/var/log/gan-cyber-range/app.json')
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
    def info(self, message: str, **kwargs):
        self._log('INFO', message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self._log('WARNING', message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self._log('ERROR', message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        self._log('CRITICAL', message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        self._log('DEBUG', message, **kwargs)
        
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with context"""
        extra = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'service': 'gan-cyber-range',
            'trace_id': kwargs.get('trace_id', str(uuid.uuid4())),
            **kwargs
        }
        
        if level == 'ERROR' or level == 'CRITICAL':
            extra['stack_trace'] = traceback.format_exc()
            
        self.logger.log(getattr(logging, level), message, extra=extra)

class StructuredFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'trace_id'):
            log_entry['trace_id'] = record.trace_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'attack_type'):
            log_entry['attack_type'] = record.attack_type
        if hasattr(record, 'target'):
            log_entry['target'] = record.target
        if hasattr(record, 'stack_trace'):
            log_entry['stack_trace'] = record.stack_trace
            
        return json.dumps(log_entry)
```

### Cyber Range Specific Logging

```python
class CyberRangeLogger:
    """Specialized logger for cyber range events"""
    
    def __init__(self):
        self.security_logger = StructuredLogger('security')
        self.attack_logger = StructuredLogger('attacks')
        self.defense_logger = StructuredLogger('defense')
        self.performance_logger = StructuredLogger('performance')
        
    def log_attack_attempt(self, attack_type: str, target: str, agent_id: str, **context):
        """Log an attack attempt"""
        self.attack_logger.info(
            f"Attack attempt: {attack_type} on {target}",
            event_type='attack_attempt',
            attack_type=attack_type,
            target=target,
            agent_id=agent_id,
            **context
        )
        
    def log_successful_attack(self, attack_type: str, target: str, agent_id: str, 
                            time_to_compromise: float, **context):
        """Log a successful attack"""
        self.attack_logger.warning(
            f"Successful attack: {attack_type} compromised {target}",
            event_type='attack_success',
            attack_type=attack_type,
            target=target,
            agent_id=agent_id,
            time_to_compromise=time_to_compromise,
            **context
        )
        
    def log_attack_detection(self, attack_type: str, target: str, detection_method: str,
                           detection_time: float, **context):
        """Log attack detection"""
        self.defense_logger.info(
            f"Attack detected: {attack_type} on {target} via {detection_method}",
            event_type='attack_detection',
            attack_type=attack_type,
            target=target,
            detection_method=detection_method,
            detection_time=detection_time,
            **context
        )
        
    def log_patch_deployment(self, vulnerability_id: str, patch_id: str, 
                           deployment_time: float, **context):
        """Log patch deployment"""
        self.defense_logger.info(
            f"Patch deployed: {patch_id} for {vulnerability_id}",
            event_type='patch_deployment',
            vulnerability_id=vulnerability_id,
            patch_id=patch_id,
            deployment_time=deployment_time,
            **context
        )
        
    def log_security_incident(self, incident_type: str, severity: str, 
                            description: str, **context):
        """Log security incidents"""
        self.security_logger.critical(
            f"Security incident: {incident_type} - {description}",
            event_type='security_incident',
            incident_type=incident_type,
            severity=severity,
            description=description,
            **context
        )
        
    def log_performance_metric(self, metric_name: str, value: float, unit: str, **context):
        """Log performance metrics"""
        self.performance_logger.debug(
            f"Performance metric: {metric_name} = {value} {unit}",
            event_type='performance_metric',
            metric_name=metric_name,
            value=value,
            unit=unit,
            **context
        )
```

### FastAPI Logging Middleware

```python
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import uuid

class LoggingMiddleware(BaseHTTPMiddleware):
    """HTTP request/response logging middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = StructuredLogger('http')
        
    async def dispatch(self, request: Request, call_next):
        # Generate trace ID for request correlation
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        
        start_time = time.time()
        
        # Log incoming request
        self.logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            event_type='http_request',
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
            client_ip=request.client.host,
            user_agent=request.headers.get('user-agent'),
            trace_id=trace_id
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log response
            self.logger.info(
                f"Response: {response.status_code} for {request.method} {request.url.path}",
                event_type='http_response',
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_seconds=duration,
                trace_id=trace_id
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            self.logger.error(
                f"Request error: {str(e)} for {request.method} {request.url.path}",
                event_type='http_error',
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_seconds=duration,
                trace_id=trace_id
            )
            raise
```

## Log Aggregation Configuration

### Fluentd Configuration

```ruby
# fluent.conf
<source>
  @type tail
  path /var/log/gan-cyber-range/*.json
  pos_file /var/log/fluentd/gan-cyber-range.log.pos
  tag gan.cyber.range
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%LZ
</source>

<filter gan.cyber.range>
  @type record_transformer
  <record>
    hostname "#{Socket.gethostname}"
    environment "#{ENV['ENVIRONMENT'] || 'development'}"
  </record>
</filter>

# Security events - high priority
<match gan.cyber.range>
  @type copy
  <store>
    @type elasticsearch
    host elasticsearch
    port 9200
    index_name cyber-range-logs
    type_name _doc
    include_tag_key true
    tag_key @log_name
    flush_interval 5s
    <buffer>
      @type file
      path /var/log/fluentd/buffer/elasticsearch
      flush_mode interval
      flush_interval 5s
      chunk_limit_size 2MB
      total_limit_size 512MB
      retry_max_interval 30
      retry_forever true
    </buffer>
  </store>
  
  # Also send security events to alerting system
  <store>
    @type http
    endpoint http://alertmanager:9093/api/v1/alerts
    http_method post
    <format>
      @type json
    </format>
    <buffer>
      flush_interval 1s
    </buffer>
  </store>
</match>
```

### Elasticsearch Index Template

```json
{
  "template": {
    "index_patterns": ["cyber-range-logs-*"],
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 1,
      "index.lifecycle.name": "cyber-range-policy",
      "index.refresh_interval": "5s"
    },
    "mappings": {
      "properties": {
        "timestamp": {
          "type": "date",
          "format": "strict_date_optional_time"
        },
        "level": {
          "type": "keyword"
        },
        "event_type": {
          "type": "keyword"
        },
        "attack_type": {
          "type": "keyword"
        },
        "target": {
          "type": "keyword"
        },
        "agent_id": {
          "type": "keyword"
        },
        "trace_id": {
          "type": "keyword"
        },
        "message": {
          "type": "text",
          "analyzer": "standard"
        },
        "duration_seconds": {
          "type": "float"
        },
        "client_ip": {
          "type": "ip"
        }
      }
    }
  }
}
```

## Log Analysis Queries

### Common Kibana Queries

```json
{
  "security_incidents": {
    "query": {
      "bool": {
        "must": [
          {"term": {"event_type": "security_incident"}},
          {"range": {"timestamp": {"gte": "now-1h"}}}
        ]
      }
    }
  },
  "successful_attacks": {
    "query": {
      "bool": {
        "must": [
          {"term": {"event_type": "attack_success"}},
          {"range": {"timestamp": {"gte": "now-24h"}}}
        ]
      }
    },
    "aggs": {
      "by_attack_type": {
        "terms": {"field": "attack_type"}
      }
    }
  },
  "slow_requests": {
    "query": {
      "bool": {
        "must": [
          {"term": {"event_type": "http_response"}},
          {"range": {"duration_seconds": {"gte": 5}}}
        ]
      }
    }
  }
}
```

## Log Retention Policy

### ILM Policy for Elasticsearch

```json
{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": {
            "max_size": "5GB",
            "max_age": "1d"
          }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "allocate": {
            "number_of_replicas": 0
          }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "allocate": {
            "number_of_replicas": 0
          }
        }
      },
      "delete": {
        "min_age": "90d"
      }
    }
  }
}
```

## Log Monitoring Alerts

### LogQL Queries for Loki

```yaml
# Prometheus rules for log-based alerts
groups:
  - name: cyber_range_log_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          rate({service="gan-cyber-range"} |= "ERROR" [5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in logs"
          
      - alert: SecurityIncidentDetected
        expr: |
          count_over_time({service="gan-cyber-range"} |~ "security_incident" [1m]) > 0
        for: 0s
        labels:
          severity: critical
        annotations:
          summary: "Security incident detected in logs"
          
      - alert: SuccessfulAttackSpike
        expr: |
          rate({service="gan-cyber-range"} |~ "attack_success" [5m]) > 0.5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Spike in successful attacks detected"
```