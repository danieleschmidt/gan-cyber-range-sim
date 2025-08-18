# Alerting Configuration

This document describes the comprehensive alerting strategy for the GAN Cyber Range Simulator.

## Overview

The alerting system provides proactive notifications for:
- System health issues
- Security incidents
- Performance degradation
- Business metric anomalies

## Alert Categories

### Critical Alerts (P1)
Immediate response required - system is down or compromised

### High Priority Alerts (P2)
Service degradation affecting users

### Medium Priority Alerts (P3)
Potential issues requiring investigation

### Low Priority Alerts (P4)
Information and trending alerts

## Prometheus Alert Rules

### System Health Alerts

```yaml
# alerts/system-health.yml
groups:
- name: system-health
  rules:
  - alert: ApplicationDown
    expr: up{job="gan-cyber-range"} == 0
    for: 30s
    labels:
      severity: critical
      category: system
    annotations:
      summary: "GAN Cyber Range application is down"
      description: "The GAN Cyber Range application has been down for more than 30 seconds"
      runbook_url: "https://docs.gan-cyber-range.org/runbooks/application-down"

  - alert: HighMemoryUsage
    expr: (container_memory_usage_bytes{container="cyber-range"} / container_spec_memory_limit_bytes) * 100 > 90
    for: 2m
    labels:
      severity: warning
      category: resource
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }}% of the limit"

  - alert: HighCPUUsage
    expr: rate(container_cpu_usage_seconds_total{container="cyber-range"}[5m]) * 100 > 80
    for: 5m
    labels:
      severity: warning
      category: resource
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }}% averaged over 5 minutes"

  - alert: DiskSpaceLow
    expr: (1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100 > 85
    for: 1m
    labels:
      severity: warning
      category: storage
    annotations:
      summary: "Disk space is running low"
      description: "Disk usage is {{ $value }}% on {{ $labels.mountpoint }}"
```

### Security Alerts

```yaml
# alerts/security.yml
groups:
- name: security
  rules:
  - alert: HighFailedAuthAttempts
    expr: increase(auth_failed_attempts_total[5m]) > 10
    for: 1m
    labels:
      severity: high
      category: security
    annotations:
      summary: "High number of failed authentication attempts"
      description: "{{ $value }} failed authentication attempts in the last 5 minutes"
      runbook_url: "https://docs.gan-cyber-range.org/runbooks/failed-auth"

  - alert: SuspiciousActivity
    expr: increase(security_events_total{type="suspicious"}[10m]) > 5
    for: 30s
    labels:
      severity: high
      category: security
    annotations:
      summary: "Suspicious activity detected"
      description: "{{ $value }} suspicious security events in the last 10 minutes"

  - alert: ContainerEscapeAttempt
    expr: increase(container_escape_attempts_total[1m]) > 0
    for: 0s
    labels:
      severity: critical
      category: security
    annotations:
      summary: "Container escape attempt detected"
      description: "Container escape attempt detected on {{ $labels.instance }}"
      action_required: "IMMEDIATE INVESTIGATION REQUIRED"

  - alert: UnauthorizedNetworkAccess
    expr: increase(network_violations_total[5m]) > 0
    for: 30s
    labels:
      severity: high
      category: security
    annotations:
      summary: "Unauthorized network access detected"
      description: "{{ $value }} network policy violations in the last 5 minutes"
```

### Performance Alerts

```yaml
# alerts/performance.yml
groups:
- name: performance
  rules:
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2.0
    for: 2m
    labels:
      severity: warning
      category: performance
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }}s"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
    for: 2m
    labels:
      severity: warning
      category: performance
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }}"

  - alert: DatabaseConnectionPoolExhausted
    expr: database_connections_active / database_connections_max > 0.9
    for: 1m
    labels:
      severity: high
      category: database
    annotations:
      summary: "Database connection pool near exhaustion"
      description: "Using {{ $value | humanizePercentage }} of available connections"

  - alert: AgentResponseTimeout
    expr: increase(agent_timeout_total[5m]) > 5
    for: 30s
    labels:
      severity: warning
      category: agent
    annotations:
      summary: "AI agents experiencing timeouts"
      description: "{{ $value }} agent timeouts in the last 5 minutes"
```

### Business Logic Alerts

```yaml
# alerts/business.yml
groups:
- name: business
  rules:
  - alert: LowSimulationSuccessRate
    expr: rate(simulations_successful_total[1h]) / rate(simulations_started_total[1h]) < 0.8
    for: 5m
    labels:
      severity: warning
      category: business
    annotations:
      summary: "Low simulation success rate"
      description: "Simulation success rate is {{ $value | humanizePercentage }} over the last hour"

  - alert: NoActiveSimulations
    expr: active_simulations_count == 0
    for: 10m
    labels:
      severity: warning
      category: business
    annotations:
      summary: "No active simulations"
      description: "No simulations have been running for 10 minutes"

  - alert: HighAgentFailureRate
    expr: rate(agent_failures_total[15m]) / rate(agent_actions_total[15m]) > 0.1
    for: 2m
    labels:
      severity: warning
      category: agent
    annotations:
      summary: "High agent failure rate"
      description: "Agent failure rate is {{ $value | humanizePercentage }}"
```

## Notification Channels

### Slack Integration

```yaml
# alertmanager.yml
global:
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    group_wait: 10s
  - match:
      severity: high
    receiver: 'high-priority-alerts'
  - match:
      category: security
    receiver: 'security-alerts'

receivers:
- name: 'default'
  slack_configs:
  - channel: '#gan-cyber-range-alerts'
    title: 'GAN Cyber Range Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'critical-alerts'
  slack_configs:
  - channel: '#gan-cyber-range-critical'
    title: '🚨 CRITICAL ALERT 🚨'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
    send_resolved: true

- name: 'security-alerts'
  slack_configs:
  - channel: '#gan-cyber-range-security'
    title: '🔒 Security Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
  email_configs:
  - to: 'security@gan-cyber-range.org'
    subject: 'Security Alert: {{ .GroupLabels.alertname }}'
    body: |
      Alert Details:
      {{ range .Alerts }}
      - {{ .Annotations.summary }}
      - {{ .Annotations.description }}
      {{ end }}

- name: 'high-priority-alerts'
  slack_configs:
  - channel: '#gan-cyber-range-alerts'
    title: '⚠️ High Priority Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

### Email Notifications

```yaml
# Email configuration
email_configs:
- to: 'ops@gan-cyber-range.org'
  from: 'alerts@gan-cyber-range.org'
  smarthost: 'smtp.gmail.com:587'
  auth_username: 'alerts@gan-cyber-range.org'
  auth_password: 'app-password'
  subject: 'GAN Cyber Range Alert: {{ .GroupLabels.alertname }}'
  headers:
    Priority: 'high'
  body: |
    Alert: {{ .GroupLabels.alertname }}
    Severity: {{ .CommonLabels.severity }}
    
    {{ range .Alerts }}
    Summary: {{ .Annotations.summary }}
    Description: {{ .Annotations.description }}
    Started: {{ .StartsAt }}
    {{ if .Annotations.runbook_url }}
    Runbook: {{ .Annotations.runbook_url }}
    {{ end }}
    {{ end }}
```

### PagerDuty Integration

```yaml
pagerduty_configs:
- routing_key: 'YOUR_PAGERDUTY_ROUTING_KEY'
  description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
  details:
    severity: '{{ .CommonLabels.severity }}'
    category: '{{ .CommonLabels.category }}'
    description: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    runbook_url: '{{ range .Alerts }}{{ .Annotations.runbook_url }}{{ end }}'
```

## Alert Testing

### Simulating Alerts

```bash
# Test application down alert
kubectl scale deployment gan-cyber-range --replicas=0

# Test high memory usage
kubectl exec -it <pod-name> -- stress --vm 1 --vm-bytes 1G --timeout 10s

# Test high error rate
curl -X POST http://localhost:8080/api/v1/test/generate-errors?count=100

# Test security alert
curl -X POST http://localhost:8080/auth/login -d '{"username":"invalid","password":"invalid"}' -H "Content-Type: application/json"
```

### Alert Validation

```bash
# Check alert rules
promtool check rules alerts/*.yml

# Test alert queries
curl 'http://prometheus:9090/api/v1/query?query=up{job="gan-cyber-range"}'

# Check alertmanager configuration
amtool config check alertmanager.yml
```

## Runbook Templates

### System Down Runbook

```markdown
# System Down Response

## Immediate Actions
1. Check application pods: `kubectl get pods -l app=gan-cyber-range`
2. Check pod logs: `kubectl logs -l app=gan-cyber-range --tail=100`
3. Check service health: `curl http://service-endpoint/health`

## Investigation Steps
1. Check resource utilization
2. Review recent deployments
3. Check dependencies (database, Redis)
4. Review error logs

## Recovery Actions
1. Restart pods if necessary
2. Scale up if resource issue
3. Rollback deployment if needed
4. Engage on-call engineer if persistent
```

### Security Incident Runbook

```markdown
# Security Incident Response

## Immediate Actions
1. Document the alert details
2. Check for ongoing attacks
3. Review security logs
4. Isolate affected systems if necessary

## Investigation Steps
1. Correlate with other security events
2. Check authentication logs
3. Review network traffic
4. Analyze affected containers

## Response Actions
1. Block suspicious IPs if identified
2. Reset compromised credentials
3. Increase monitoring sensitivity
4. Document findings
5. Update security policies if needed
```

## Monitoring Best Practices

### Alert Hygiene

1. **Avoid Alert Fatigue**
   - Set appropriate thresholds
   - Use proper alert grouping
   - Implement alert suppression
   - Regular review and tuning

2. **Clear Documentation**
   - Meaningful alert names
   - Descriptive annotations
   - Link to runbooks
   - Include context in messages

3. **Testing and Validation**
   - Regular alert testing
   - Validate notification channels
   - Test runbook procedures
   - Measure response times

### Alert Lifecycle

1. **Alert Creation**
   - Define clear criteria
   - Set appropriate severity
   - Create actionable alerts
   - Include recovery conditions

2. **Alert Response**
   - Acknowledge receipt
   - Follow runbook procedures
   - Document actions taken
   - Communicate with team

3. **Alert Resolution**
   - Verify root cause
   - Implement fixes
   - Update documentation
   - Conduct post-incident review

## Integration with Incident Management

### ServiceNow Integration

```yaml
# ServiceNow webhook receiver
webhook_configs:
- url: 'https://your-instance.service-now.com/api/now/table/incident'
  http_config:
    basic_auth:
      username: 'api_user'
      password: 'api_password'
  send_resolved: true
```

### JIRA Integration

```yaml
# JIRA ticket creation
webhook_configs:
- url: 'https://your-domain.atlassian.net/rest/api/2/issue'
  http_config:
    basic_auth:
      username: 'api_user'
      password: 'api_token'
  send_resolved: false
```

## Compliance and Audit

### Alert Retention

- Store alert history for compliance requirements
- Archive resolved alerts for trend analysis
- Maintain alert configuration version control
- Document alert policy changes

### Reporting

- Generate monthly alert summaries
- Track mean time to detection (MTTD)
- Monitor mean time to resolution (MTTR)
- Report on alert accuracy and effectiveness

---

For alert configuration questions or incident response support, contact the DevOps team at devops@gan-cyber-range.org or join our [Discord Community](https://discord.gg/gan-cyber-range).