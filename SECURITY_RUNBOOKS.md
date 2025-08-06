# 🛡️ Security Operations Runbooks

## Overview

This document provides operational runbooks for managing the GAN Cyber Range defensive security systems. These runbooks ensure consistent, effective response to security incidents and operational scenarios.

## 🚨 Incident Response Runbooks

### 1. Critical Malware Detection

**Trigger:** Malware detected with confidence > 0.8
**SLA:** Response within 5 minutes
**Escalation:** P1 incident

#### Immediate Actions (0-5 minutes)
```bash
# 1. Isolate affected systems immediately
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: malware-isolation-$(date +%s)
spec:
  podSelector:
    matchLabels:
      infected: "true"
  policyTypes:
  - Ingress
  - Egress
EOF

# 2. Collect memory dump
kubectl exec $AFFECTED_POD -- dd if=/dev/mem of=/tmp/memory-dump-$(date +%s).raw

# 3. Preserve logs
kubectl logs $AFFECTED_POD > malware-incident-$(date +%s).log
```

#### Investigation Phase (5-30 minutes)
1. **Analyze malware sample**
   - Run through automated analysis sandbox
   - Check against threat intelligence feeds
   - Identify persistence mechanisms

2. **Scope assessment**
   - Check for lateral movement indicators
   - Review network connections from infected host
   - Search for similar IoCs across environment

3. **Impact analysis**
   - Assess data access by compromised system
   - Review privileged operations performed
   - Check for credential compromise

#### Recovery Phase (30+ minutes)
1. **System remediation**
   - Rebuild infected systems from clean images
   - Apply latest security patches
   - Implement additional monitoring

2. **Security hardening**
   - Update detection rules based on IoCs
   - Deploy additional honeypots if applicable
   - Review and strengthen access controls

### 2. Data Exfiltration Detection

**Trigger:** Unusual data transfer > 100MB outside normal hours
**SLA:** Response within 15 minutes
**Escalation:** P1 if sensitive data involved

#### Immediate Actions (0-15 minutes)
```python
# 1. Block egress traffic from source
from gan_cyber_range.security.isolation import NetworkIsolation

isolation = NetworkIsolation()
await isolation.adaptive_threat_response({
    "type": "data_exfiltration",
    "source_ip": source_ip,
    "confidence": 0.9
}, response_level="aggressive")

# 2. Identify data accessed
data_access_logs = query_data_access_logs(
    start_time=incident_time - timedelta(hours=2),
    end_time=incident_time,
    source_ip=source_ip
)
```

#### Investigation Phase
1. **Data classification assessment**
   - Identify types of data accessed
   - Assess sensitivity and compliance implications
   - Review data access permissions

2. **Threat actor profiling**
   - Analyze attack patterns and TTPs
   - Check for known threat group signatures
   - Assess sophistication level

3. **Legal and compliance notification**
   - Notify legal team if PII/PCI data involved
   - Prepare breach notification documentation
   - Coordinate with compliance team

### 3. Advanced Persistent Threat (APT) Detection

**Trigger:** Correlation of multiple sophisticated techniques
**SLA:** Response within 30 minutes
**Escalation:** P1 with executive notification

#### Immediate Actions
1. **Activate incident response team**
   - Notify CISO and senior security staff
   - Establish dedicated communication channel
   - Begin evidence preservation

2. **Enhanced monitoring deployment**
```python
# Deploy additional monitoring
from gan_cyber_range.agents.blue_team import BlueTeamAgent

blue_team = BlueTeamAgent()
enhanced_monitoring = await blue_team.deploy_adaptive_honeypot(
    "apt_monitoring", 
    "critical_network_segments"
)
```

3. **External coordination**
   - Contact threat intelligence providers
   - Coordinate with law enforcement if needed
   - Engage external incident response support

## 🔍 Operational Runbooks

### Daily Security Operations

#### Morning Security Briefing (9:00 AM)
```bash
#!/bin/bash
# Daily security status check

echo "=== Daily Security Status Report ==="
echo "Date: $(date)"
echo

# Check overnight alerts
echo "Overnight Critical Alerts:"
python3 -c "
from gan_cyber_range.security.siem import SIEMEngine
siem = SIEMEngine()
summary = siem.get_alert_summary(12)  # Last 12 hours
print(f'Total alerts: {summary[\"total_alerts\"]}')
print(f'Critical: {summary[\"severity_breakdown\"].get(\"critical\", 0)}')
print(f'High: {summary[\"severity_breakdown\"].get(\"high\", 0)}')
"

# Check system health
echo -e "\nSystem Health Status:"
python3 -c "
from gan_cyber_range.monitoring.health_check import HealthChecker
checker = HealthChecker()
summary = checker.get_health_summary()
print(f'Overall status: {summary[\"overall_status\"]}')
print(f'Total checks: {summary[\"total_checks\"]}')
"

# Check incident status
echo -e "\nActive Incidents:"
python3 -c "
from gan_cyber_range.security.incident_response import IncidentResponseOrchestrator
orchestrator = IncidentResponseOrchestrator()
summary = orchestrator.get_incident_summary()
print(f'Active incidents: {summary[\"active_incidents\"]}')
for incident in summary[\"recent_incidents\"][:5]:
    print(f'  - {incident[\"title\"]} ({incident[\"severity\"]})')
"
```

#### Threat Intelligence Update (Weekly)
```python
#!/usr/bin/env python3
"""Weekly threat intelligence update process."""

from gan_cyber_range.agents.blue_team import BlueTeamAgent
from gan_cyber_range.security.siem import SIEMEngine

async def weekly_threat_intel_update():
    """Update threat intelligence feeds and detection rules."""
    
    blue_team = BlueTeamAgent()
    siem = SIEMEngine()
    
    print("Starting weekly threat intelligence update...")
    
    # 1. Update threat indicators
    new_indicators = await fetch_threat_intelligence_feeds()
    for indicator in new_indicators:
        blue_team.threat_indicators.add(indicator)
    
    print(f"Added {len(new_indicators)} new threat indicators")
    
    # 2. Update detection rules
    new_rules = generate_detection_rules_from_intel(new_indicators)
    for rule in new_rules:
        siem.add_detection_rule(rule)
    
    print(f"Added {len(new_rules)} new detection rules")
    
    # 3. Test detection effectiveness
    test_results = await test_detection_rules(new_rules)
    print(f"Detection rule testing: {test_results}")
    
    # 4. Generate weekly threat report
    report = generate_weekly_threat_report()
    print("Weekly threat report generated and distributed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(weekly_threat_intel_update())
```

### Performance Monitoring

#### System Performance Check
```python
#!/usr/bin/env python3
"""Monitor defensive system performance."""

import time
import psutil
from gan_cyber_range.monitoring.security_dashboard import SecurityDashboard

async def performance_check():
    """Check system performance and resource utilization."""
    
    dashboard = SecurityDashboard()
    
    # CPU and Memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    await dashboard.update_metric("cpu_usage", cpu_percent)
    await dashboard.update_metric("memory_usage", memory.percent)
    
    # Event processing rate
    event_rate = measure_event_processing_rate()
    await dashboard.update_metric("event_processing_rate", event_rate)
    
    # Alert processing time
    alert_processing_time = measure_alert_processing_time()
    await dashboard.update_metric("alert_processing_time", alert_processing_time)
    
    print(f"Performance check complete:")
    print(f"  CPU: {cpu_percent}%")
    print(f"  Memory: {memory.percent}%")
    print(f"  Event rate: {event_rate}/sec")
    print(f"  Alert processing: {alert_processing_time}ms")

def measure_event_processing_rate():
    """Measure current event processing rate."""
    # Implementation specific to your environment
    return 1000.0  # events per second

def measure_alert_processing_time():
    """Measure alert processing time."""
    # Implementation specific to your environment
    return 50.0  # milliseconds
```

## 🔧 Maintenance Runbooks

### Model Retraining (Monthly)

```python
#!/usr/bin/env python3
"""Monthly ML model retraining process."""

from gan_cyber_range.security.ml_threat_detection import MLThreatDetector, AutomatedMLPipeline

async def monthly_model_retraining():
    """Retrain ML models with latest data."""
    
    detector = MLThreatDetector()
    pipeline = AutomatedMLPipeline(detector)
    
    print("Starting monthly ML model retraining...")
    
    # 1. Collect training data from last month
    training_events = collect_training_data(days=30)
    print(f"Collected {len(training_events)} training events")
    
    # 2. Retrain models
    results = await detector.train_models(training_events)
    
    # 3. Validate model performance
    validation_results = validate_model_performance(detector)
    
    # 4. Deploy new models if performance improved
    if should_deploy_models(validation_results):
        await detector.save_models("models/production")
        print("New models deployed to production")
    else:
        print("Models not deployed - performance did not improve")
    
    # 5. Update performance metrics
    await pipeline.monitor_performance()
    
    print("Monthly model retraining complete")

def collect_training_data(days=30):
    """Collect training data from specified number of days."""
    # Implementation specific to your data storage
    return []

def validate_model_performance(detector):
    """Validate model performance against test dataset."""
    # Implementation specific to your validation approach
    return {"precision": 0.85, "recall": 0.80}

def should_deploy_models(results):
    """Determine if new models should be deployed."""
    return results["precision"] > 0.80 and results["recall"] > 0.75
```

### Security Baseline Update (Quarterly)

```bash
#!/bin/bash
# Quarterly security baseline update

echo "Starting quarterly security baseline update..."

# 1. Update security policies
echo "Updating security policies..."
kubectl apply -f security/policies/

# 2. Update network segmentation rules
echo "Updating network segmentation..."
python3 scripts/update_network_policies.py

# 3. Review and update access controls
echo "Reviewing access controls..."
python3 scripts/access_control_review.py

# 4. Update security configurations
echo "Updating security configurations..."
kubectl patch configmap security-config --patch "$(cat security/config-updates.yaml)"

# 5. Test all defensive systems
echo "Testing defensive systems..."
python3 validate_security_system.py

echo "Quarterly security baseline update complete"
```

## 📊 Monitoring and Alerting

### Dashboard Health Check
```python
#!/usr/bin/env python3
"""Monitor security dashboard health."""

from gan_cyber_range.monitoring.security_dashboard import SecurityDashboard
import asyncio

async def dashboard_health_check():
    """Check dashboard health and connectivity."""
    
    dashboard = SecurityDashboard()
    
    # Check WebSocket connections
    active_connections = len(dashboard.connected_clients)
    print(f"Active dashboard connections: {active_connections}")
    
    # Check metric updates
    last_update_times = {}
    for metric_name in dashboard.current_metrics:
        metric = dashboard.current_metrics[metric_name]
        last_update_times[metric_name] = metric.timestamp
    
    # Alert if metrics are stale (> 5 minutes)
    from datetime import datetime, timedelta
    stale_threshold = datetime.now() - timedelta(minutes=5)
    
    stale_metrics = [
        name for name, timestamp in last_update_times.items()
        if timestamp < stale_threshold
    ]
    
    if stale_metrics:
        print(f"WARNING: Stale metrics detected: {stale_metrics}")
        # Send alert to operations team
        await send_operations_alert(f"Stale dashboard metrics: {stale_metrics}")
    
    print("Dashboard health check complete")

async def send_operations_alert(message):
    """Send alert to operations team."""
    # Implementation specific to your alerting system
    print(f"ALERT: {message}")
```

### Automated Health Checks

```bash
#!/bin/bash
# Automated health check script (run every 15 minutes)

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="/var/log/security-health-check.log"

echo "[$TIMESTAMP] Starting automated health check" >> $LOG_FILE

# 1. Check critical services
SERVICES=("siem-engine" "blue-team-agent" "incident-response" "security-dashboard")
for service in "${SERVICES[@]}"; do
    if systemctl is-active --quiet $service; then
        echo "[$TIMESTAMP] ✓ $service is running" >> $LOG_FILE
    else
        echo "[$TIMESTAMP] ✗ $service is not running" >> $LOG_FILE
        # Send critical alert
        curl -X POST https://alerts.company.com/webhook \
             -H "Content-Type: application/json" \
             -d "{\"alert\": \"Critical service $service is down\", \"severity\": \"critical\"}"
    fi
done

# 2. Check resource usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')

echo "[$TIMESTAMP] CPU Usage: ${CPU_USAGE}%" >> $LOG_FILE
echo "[$TIMESTAMP] Memory Usage: ${MEMORY_USAGE}%" >> $LOG_FILE

# Alert if resource usage is high
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "[$TIMESTAMP] HIGH CPU USAGE ALERT: ${CPU_USAGE}%" >> $LOG_FILE
fi

if (( $(echo "$MEMORY_USAGE > 85" | bc -l) )); then
    echo "[$TIMESTAMP] HIGH MEMORY USAGE ALERT: ${MEMORY_USAGE}%" >> $LOG_FILE
fi

echo "[$TIMESTAMP] Automated health check complete" >> $LOG_FILE
```

## 🎯 Escalation Procedures

### P1 Incident Escalation
1. **Immediate (< 5 minutes)**
   - Notify on-call security engineer
   - Create incident channel in Slack/Teams
   - Begin automated response procedures

2. **Within 15 minutes**
   - Notify security manager
   - Engage incident response team
   - Brief executive team if business impact

3. **Within 30 minutes**
   - External notification if required
   - Law enforcement contact if criminal activity
   - Customer communication if data breach

### P2 Incident Escalation
1. **Within 1 hour**
   - Notify security team during business hours
   - Create incident ticket
   - Begin investigation

2. **Within 4 hours**
   - Notify security manager
   - Provide initial assessment
   - Plan remediation steps

## 📝 Documentation and Reporting

### Incident Documentation Template
```markdown
# Security Incident Report

**Incident ID:** INC-YYYY-MMDD-XXX
**Date/Time:** 
**Severity:** P1/P2/P3/P4
**Status:** Open/In Progress/Resolved/Closed

## Summary
Brief description of the incident and its impact.

## Timeline
- **Detection:** When and how was the incident detected?
- **Response:** What immediate actions were taken?
- **Investigation:** Key investigation findings
- **Resolution:** How was the incident resolved?

## Technical Details
- **Attack Vector:** 
- **Affected Systems:** 
- **IoCs Identified:** 
- **MITRE ATT&CK TTPs:** 

## Impact Assessment
- **Business Impact:** 
- **Data Involved:** 
- **System Downtime:** 
- **Financial Impact:** 

## Response Actions Taken
1. Immediate containment actions
2. Investigation steps
3. Eradication measures
4. Recovery procedures

## Lessons Learned
- What went well?
- What could be improved?
- Action items for improvement

## Follow-up Actions
- [ ] Security control improvements
- [ ] Process updates
- [ ] Training needs
- [ ] Technology enhancements
```

### Weekly Security Report Template
```python
#!/usr/bin/env python3
"""Generate weekly security report."""

from datetime import datetime, timedelta
from gan_cyber_range.security.siem import SIEMEngine
from gan_cyber_range.security.incident_response import IncidentResponseOrchestrator

def generate_weekly_report():
    """Generate comprehensive weekly security report."""
    
    report_date = datetime.now()
    week_start = report_date - timedelta(days=7)
    
    # Initialize components
    siem = SIEMEngine()
    incident_response = IncidentResponseOrchestrator()
    
    # Collect metrics
    alert_summary = siem.get_alert_summary(168)  # 7 days
    incident_summary = incident_response.get_incident_summary()
    
    # Generate report
    report = f"""
# Weekly Security Report
**Week Ending:** {report_date.strftime('%Y-%m-%d')}

## Executive Summary
This week we detected {alert_summary['total_alerts']} security alerts and managed 
{incident_summary['total_incidents_24h']} incidents. The security posture remains 
strong with no critical breaches.

## Key Metrics
- **Total Alerts:** {alert_summary['total_alerts']}
- **Critical Alerts:** {alert_summary['severity_breakdown'].get('critical', 0)}
- **High Priority Alerts:** {alert_summary['severity_breakdown'].get('high', 0)}
- **Active Incidents:** {incident_summary['active_incidents']}
- **Mean Response Time:** {incident_summary['metrics']['avg_response_time_minutes']:.1f} minutes

## Threat Landscape
- **Top Threat Types:** Based on alert analysis
- **Attack Trends:** Observed patterns and techniques
- **Geographic Distribution:** Source IP analysis

## Defensive Effectiveness
- **Detection Rate:** Percentage of attacks detected
- **False Positive Rate:** Current false positive metrics
- **Response Time:** Average time to respond to incidents

## Recommendations
1. Areas for improvement based on weekly analysis
2. Security control enhancements needed
3. Process optimization opportunities

## Upcoming Activities
- Scheduled security updates
- Planned security assessments
- Training and awareness activities
"""
    
    # Save and distribute report
    with open(f"reports/weekly-security-{report_date.strftime('%Y-%m-%d')}.md", "w") as f:
        f.write(report)
    
    print("Weekly security report generated successfully")

if __name__ == "__main__":
    generate_weekly_report()
```

---

*These runbooks provide standardized procedures for operating the GAN Cyber Range defensive security systems effectively and consistently.*