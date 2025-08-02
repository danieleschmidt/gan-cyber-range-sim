# 📚 Operational Runbooks

## Overview

This directory contains operational runbooks for common scenarios and incidents in the GAN Cyber Range Simulator.

## Available Runbooks

### Incident Response
- [Security Incident Response](./security-incident-response.md)
- [Performance Degradation](./performance-degradation.md)
- [Service Outage](./service-outage.md)
- [Data Breach Response](./data-breach-response.md)

### Maintenance Procedures
- [Deployment Procedures](./deployment-procedures.md)
- [Backup and Recovery](./backup-recovery.md)
- [System Updates](./system-updates.md)
- [Certificate Rotation](./certificate-rotation.md)

### Monitoring and Alerting
- [Alert Response Procedures](./alert-response.md)
- [Log Analysis](./log-analysis.md)
- [Metrics Investigation](./metrics-investigation.md)

### Environment Management
- [Environment Setup](./environment-setup.md)
- [Configuration Management](./configuration-management.md)
- [Resource Scaling](./resource-scaling.md)

## Runbook Template

Each runbook follows this standard structure:

```markdown
# [Runbook Title]

## Overview
Brief description of the scenario

## Prerequisites
- Required access/permissions
- Tools needed
- Contact information

## Immediate Actions
- Step-by-step immediate response
- Safety measures
- Escalation criteria

## Investigation Steps
- Diagnostic procedures
- Data collection
- Root cause analysis

## Resolution Steps
- Detailed remediation steps
- Verification procedures
- Rollback plans

## Post-Incident
- Documentation requirements
- Lessons learned
- Preventive measures

## Contacts
- Escalation paths
- Subject matter experts
- External contacts
```

## Emergency Contacts

- **Security Team**: security@gan-cyber-range.org
- **Infrastructure Team**: infrastructure@gan-cyber-range.org
- **On-Call Engineer**: oncall@gan-cyber-range.org
- **Management Escalation**: management@gan-cyber-range.org

## Quick Reference

### Common Commands
```bash
# Check service status
kubectl get pods -n cyber-range

# View recent logs
kubectl logs -f deployment/gan-cyber-range -n cyber-range --tail=100

# Check metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

### Dashboard Links
- [Grafana Dashboard](http://grafana.cyber-range.local/dashboard)
- [Prometheus Metrics](http://prometheus.cyber-range.local:9090)
- [Kibana Logs](http://kibana.cyber-range.local:5601)
- [Alert Manager](http://alertmanager.cyber-range.local:9093)