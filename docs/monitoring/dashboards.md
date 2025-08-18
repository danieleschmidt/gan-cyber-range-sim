# Dashboard Configuration

This document provides comprehensive dashboard configurations for monitoring the GAN Cyber Range Simulator.

## Overview

Our monitoring dashboards provide real-time visibility into:
- System health and performance
- Security metrics and incidents
- Business metrics and AI agent performance
- Infrastructure utilization

## Grafana Dashboard JSON Configurations

### Main System Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "GAN Cyber Range - System Overview",
    "tags": ["gan-cyber-range", "system", "overview"],
    "timezone": "UTC",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "System Health Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"gan-cyber-range\"}",
            "legendFormat": "Service Status"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            },
            "mappings": [
              {"options": {"0": {"text": "DOWN"}}, "type": "value"},
              {"options": {"1": {"text": "UP"}}, "type": "value"}
            ]
          }
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"gan-cyber-range\"}[5m])",
            "legendFormat": "{{method}} {{handler}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ]
      },
      {
        "id": 3,
        "title": "Response Time Distribution",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.90, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "90th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 4,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{container=\"cyber-range\"} / 1024 / 1024",
            "legendFormat": "Memory Usage (MB)"
          },
          {
            "expr": "container_spec_memory_limit_bytes{container=\"cyber-range\"} / 1024 / 1024",
            "legendFormat": "Memory Limit (MB)"
          }
        ]
      },
      {
        "id": 5,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total{container=\"cyber-range\"}[5m]) * 100",
            "legendFormat": "CPU Usage %"
          }
        ]
      },
      {
        "id": 6,
        "title": "Active Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "http_connections_active",
            "legendFormat": "Active Connections"
          }
        ]
      }
    ]
  }
}
```

### Security Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "GAN Cyber Range - Security Monitoring",
    "tags": ["gan-cyber-range", "security"],
    "timezone": "UTC",
    "refresh": "15s",
    "panels": [
      {
        "id": 1,
        "title": "Authentication Events",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(auth_attempts_total[5m])",
            "legendFormat": "Total Attempts"
          },
          {
            "expr": "rate(auth_failed_attempts_total[5m])",
            "legendFormat": "Failed Attempts"
          },
          {
            "expr": "rate(auth_successful_attempts_total[5m])",
            "legendFormat": "Successful Attempts"
          }
        ]
      },
      {
        "id": 2,
        "title": "Security Events by Type",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (type) (security_events_total)",
            "legendFormat": "{{type}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "Container Security Violations",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(container_security_violations_total[1h])",
            "legendFormat": "Violations (Last Hour)"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 5}
              ]
            }
          }
        }
      },
      {
        "id": 4,
        "title": "Network Policy Violations",
        "type": "table",
        "targets": [
          {
            "expr": "topk(10, sum by (source_ip, destination_ip, protocol) (network_violations_total))",
            "format": "table",
            "instant": true
          }
        ]
      },
      {
        "id": 5,
        "title": "Failed Login Sources",
        "type": "worldmap",
        "targets": [
          {
            "expr": "sum by (country) (auth_failed_attempts_by_country_total)",
            "legendFormat": "{{country}}"
          }
        ]
      }
    ]
  }
}
```

### AI Agent Performance Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "GAN Cyber Range - AI Agent Performance",
    "tags": ["gan-cyber-range", "agents", "ai"],
    "timezone": "UTC",
    "refresh": "30s",
    "panels": [
      {
        "id": 1,
        "title": "Active Agents",
        "type": "stat",
        "targets": [
          {
            "expr": "active_agents_count",
            "legendFormat": "Total Active"
          },
          {
            "expr": "active_agents_count{type=\"red_team\"}",
            "legendFormat": "Red Team"
          },
          {
            "expr": "active_agents_count{type=\"blue_team\"}",
            "legendFormat": "Blue Team"
          }
        ]
      },
      {
        "id": 2,
        "title": "Agent Decision Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(agent_decision_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(agent_decision_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Agent Success Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(agent_actions_successful_total[10m]) / rate(agent_actions_total[10m]) * 100",
            "legendFormat": "Success Rate %"
          }
        ]
      },
      {
        "id": 4,
        "title": "LLM API Calls",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(llm_api_calls_total[5m])",
            "legendFormat": "{{provider}}"
          }
        ]
      },
      {
        "id": 5,
        "title": "Agent Memory Usage",
        "type": "heatmap",
        "targets": [
          {
            "expr": "agent_memory_usage_bytes",
            "legendFormat": "{{agent_id}}"
          }
        ]
      },
      {
        "id": 6,
        "title": "Top Performing Agents",
        "type": "table",
        "targets": [
          {
            "expr": "topk(10, sum by (agent_id, agent_type) (agent_successful_actions_total))",
            "format": "table",
            "instant": true
          }
        ]
      }
    ]
  }
}
```

### Business Metrics Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "GAN Cyber Range - Business Metrics",
    "tags": ["gan-cyber-range", "business", "kpi"],
    "timezone": "UTC",
    "refresh": "1m",
    "panels": [
      {
        "id": 1,
        "title": "Active Simulations",
        "type": "stat",
        "targets": [
          {
            "expr": "active_simulations_count",
            "legendFormat": "Running"
          }
        ]
      },
      {
        "id": 2,
        "title": "Simulation Success Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "rate(simulations_successful_total[1h]) / rate(simulations_started_total[1h]) * 100",
            "legendFormat": "Success Rate %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 60},
                {"color": "green", "value": 80}
              ]
            }
          }
        }
      },
      {
        "id": 3,
        "title": "Attack Techniques by Category",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (technique_category) (attack_techniques_used_total)",
            "legendFormat": "{{technique_category}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Mean Time to Detection (MTTD)",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(attack_detection_time_seconds)",
            "legendFormat": "MTTD (seconds)"
          }
        ]
      },
      {
        "id": 5,
        "title": "User Activity",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(user_sessions_total[10m])",
            "legendFormat": "New Sessions"
          },
          {
            "expr": "active_user_sessions",
            "legendFormat": "Active Sessions"
          }
        ]
      },
      {
        "id": 6,
        "title": "Scenario Completion Times",
        "type": "histogram",
        "targets": [
          {
            "expr": "scenario_completion_time_seconds_bucket",
            "legendFormat": "{{le}}"
          }
        ]
      }
    ]
  }
}
```

## Custom Grafana Plugins

### Installation Script

```bash
#!/bin/bash
# Install custom Grafana plugins for GAN Cyber Range

grafana-cli plugins install grafana-worldmap-panel
grafana-cli plugins install grafana-piechart-panel
grafana-cli plugins install natel-discrete-panel
grafana-cli plugins install vonage-status-panel
grafana-cli plugins install agenty-flowcharting-panel

# Restart Grafana to load plugins
systemctl restart grafana-server
```

## Dashboard Provisioning

### Datasource Configuration

```yaml
# datasources.yml
apiVersion: 1

datasources:
- name: Prometheus
  type: prometheus
  url: http://prometheus:9090
  access: proxy
  isDefault: true
  
- name: Loki
  type: loki
  url: http://loki:3100
  access: proxy
  
- name: Elasticsearch
  type: elasticsearch
  url: http://elasticsearch:9200
  access: proxy
  database: "gan-cyber-range-*"
  interval: "Daily"
  timeField: "@timestamp"
```

### Dashboard Provisioning

```yaml
# dashboards.yml
apiVersion: 1

providers:
- name: 'GAN Cyber Range Dashboards'
  orgId: 1
  folder: 'GAN Cyber Range'
  type: file
  disableDeletion: false
  editable: true
  updateIntervalSeconds: 300
  options:
    path: /var/lib/grafana/dashboards/gan-cyber-range
```

## Alert Integration in Dashboards

### Alert Panels

```json
{
  "id": 10,
  "title": "Active Alerts",
  "type": "table",
  "targets": [
    {
      "expr": "ALERTS{job=\"gan-cyber-range\"}",
      "format": "table",
      "instant": true
    }
  ],
  "transformations": [
    {
      "id": "organize",
      "options": {
        "excludeByName": {
          "Time": true,
          "__name__": true,
          "job": true
        }
      }
    }
  ]
}
```

## Dashboard Templates and Variables

### Environment Variable

```json
{
  "templating": {
    "list": [
      {
        "name": "environment",
        "type": "custom",
        "options": [
          {"text": "Production", "value": "prod"},
          {"text": "Staging", "value": "staging"},
          {"text": "Development", "value": "dev"}
        ],
        "current": {"text": "Production", "value": "prod"}
      }
    ]
  }
}
```

### Instance Variable

```json
{
  "name": "instance",
  "type": "query",
  "query": "label_values(up{job=\"gan-cyber-range\"}, instance)",
  "refresh": 1,
  "includeAll": true,
  "multi": true
}
```

## Advanced Dashboard Features

### Drill-down Links

```json
{
  "links": [
    {
      "title": "View Logs",
      "url": "/d/logs/gan-cyber-range-logs?orgId=1&var-instance=$instance&from=$__from&to=$__to",
      "type": "dashboard"
    },
    {
      "title": "Alert Manager",
      "url": "http://alertmanager:9093/#/alerts?filter={job=\"gan-cyber-range\"}",
      "type": "absolute",
      "targetBlank": true
    }
  ]
}
```

### Annotations

```json
{
  "annotations": {
    "list": [
      {
        "name": "Deployments",
        "datasource": "Prometheus",
        "expr": "changes(process_start_time_seconds{job=\"gan-cyber-range\"}[5m]) > 0",
        "titleFormat": "Deployment",
        "textFormat": "New version deployed on {{instance}}",
        "iconColor": "green"
      },
      {
        "name": "Incidents",
        "datasource": "Prometheus", 
        "expr": "ALERTS{severity=\"critical\"}",
        "titleFormat": "{{alertname}}",
        "textFormat": "{{summary}}",
        "iconColor": "red"
      }
    ]
  }
}
```

## Mobile-Responsive Dashboards

### Layout Configuration

```json
{
  "gridPos": {
    "h": 8,
    "w": 12,
    "x": 0,
    "y": 0
  },
  "responsive": true,
  "breakpoints": {
    "xs": {"gridPos": {"w": 24, "h": 6}},
    "sm": {"gridPos": {"w": 12, "h": 8}},
    "md": {"gridPos": {"w": 12, "h": 8}},
    "lg": {"gridPos": {"w": 12, "h": 8}}
  }
}
```

## Dashboard Automation

### Automated Dashboard Updates

```bash
#!/bin/bash
# Update dashboards from Git repository

DASHBOARD_DIR="/var/lib/grafana/dashboards/gan-cyber-range"
REPO_URL="https://github.com/your-org/gan-cyber-range-dashboards.git"

cd "$DASHBOARD_DIR"
git pull origin main

# Reload Grafana configuration
curl -X POST http://admin:admin@grafana:3000/api/admin/provisioning/dashboards/reload
```

### Dashboard Export/Import

```bash
# Export dashboard
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
     "http://grafana:3000/api/dashboards/uid/system-overview" | \
     jq '.dashboard' > system-overview.json

# Import dashboard
curl -X POST \
     -H "Authorization: Bearer $GRAFANA_API_KEY" \
     -H "Content-Type: application/json" \
     -d @system-overview.json \
     "http://grafana:3000/api/dashboards/db"
```

## Performance Optimization

### Query Optimization

1. **Use Recording Rules**
   ```yaml
   # recording_rules.yml
   groups:
   - name: gan_cyber_range_rules
     rules:
     - record: gan_cyber_range:request_rate
       expr: sum(rate(http_requests_total{job="gan-cyber-range"}[5m]))
     
     - record: gan_cyber_range:error_rate  
       expr: sum(rate(http_requests_total{job="gan-cyber-range",status=~"5.."}[5m])) / sum(rate(http_requests_total{job="gan-cyber-range"}[5m]))
   ```

2. **Dashboard Performance**
   - Use appropriate time ranges
   - Limit number of series
   - Cache query results
   - Use template variables effectively

## Dashboard Maintenance

### Regular Tasks

1. **Weekly Reviews**
   - Check dashboard performance
   - Review panel relevance
   - Update queries for accuracy
   - Verify alert thresholds

2. **Monthly Updates**
   - Add new business metrics
   - Remove obsolete panels
   - Update dashboard documentation
   - Review user feedback

3. **Quarterly Audits**
   - Dashboard usage analytics
   - Performance optimization
   - Security review
   - Compliance validation

---

For dashboard configuration support or custom visualization requests, contact our monitoring team at monitoring@gan-cyber-range.org or join our [Discord Community](https://discord.gg/gan-cyber-range).