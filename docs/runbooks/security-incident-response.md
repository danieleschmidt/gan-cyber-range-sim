# Security Incident Response Runbook

## Overview

This runbook provides step-by-step procedures for responding to security incidents in the GAN Cyber Range Simulator platform.

## Incident Classification

### Severity Levels

**P1 - Critical**
- Active data breach or unauthorized access
- Container escape or privilege escalation
- Complete compromise of system integrity
- Response time: **Immediate (within 15 minutes)**

**P2 - High**
- Suspicious activity indicating potential breach
- Failed authentication patterns suggesting brute force
- Unexpected network traffic or data exfiltration attempts
- Response time: **30 minutes**

**P3 - Medium**
- Policy violations or configuration issues
- Minor security tool alerts
- Unusual but not immediately threatening activity
- Response time: **2 hours**

**P4 - Low**
- Informational security events
- Routine security maintenance
- Documentation of security improvements needed
- Response time: **Next business day**

## Prerequisites

### Required Access
- Administrative access to Kubernetes cluster
- Grafana/Prometheus monitoring dashboards
- Log aggregation system (ELK stack)
- Security scanning tools access
- Communication channels (Slack, email, phone)

### Required Tools
- kubectl CLI
- Docker CLI
- Network analysis tools (tcpdump, wireshark)
- Forensics tools
- Incident response documentation templates

### Emergency Contacts
- **Security Lead**: security-lead@gan-cyber-range.org / +1-XXX-XXX-XXXX
- **Platform Lead**: platform-lead@gan-cyber-range.org / +1-XXX-XXX-XXXX
- **Legal Counsel**: legal@gan-cyber-range.org
- **External CERT**: cert-contact@external-org.gov

## Immediate Actions (First 15 Minutes)

### Step 1: Alert Acknowledgment
```bash
# Acknowledge the alert in monitoring system
curl -X POST "http://alertmanager:9093/api/v1/alerts/silence" \
  -H "Content-Type: application/json" \
  -d '{
    "matchers": [{"name": "alertname", "value": "SecurityIncident"}],
    "startsAt": "'$(date -Iseconds)'",
    "endsAt": "'$(date -d '+1 hour' -Iseconds)'",
    "comment": "Security incident acknowledged by [YOUR_NAME]"
  }'
```

### Step 2: Initial Assessment
```bash
# Check system status
kubectl get pods -A | grep -E "(Pending|Error|CrashLoop)"

# Check recent security events
kubectl logs -l app=gan-cyber-range --since=1h | grep -E "(SECURITY|ERROR|WARN)"

# Check active connections
kubectl exec -it deployment/gan-cyber-range -- netstat -antup
```

### Step 3: Immediate Containment (if P1/P2)
```bash
# If active breach suspected - isolate affected pods
kubectl label pods <affected-pod> quarantine=true
kubectl patch deployment gan-cyber-range -p '{"spec":{"template":{"spec":{"nodeSelector":{"quarantine":"true"}}}}}'

# Block suspicious IP addresses (if identified)
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-suspicious-ip
spec:
  podSelector:
    matchLabels:
      app: gan-cyber-range
  policyTypes:
  - Ingress
  ingress:
  - from: []
    except:
    - ipBlock:
        cidr: <suspicious-ip>/32
EOF
```

### Step 4: Evidence Preservation
```bash
# Create incident directory
INCIDENT_ID="SEC-$(date +%Y%m%d-%H%M%S)"
mkdir -p "/var/log/security-incidents/$INCIDENT_ID"
cd "/var/log/security-incidents/$INCIDENT_ID"

# Capture initial state
kubectl get all -A > initial-cluster-state.txt
kubectl describe pods -l app=gan-cyber-range > pod-descriptions.txt
date > incident-timeline.txt
echo "Incident $INCIDENT_ID initiated by $(whoami)" >> incident-timeline.txt
```

### Step 5: Notification
```bash
# Send initial alert
cat > initial-alert.md << EOF
# Security Incident Alert - $INCIDENT_ID

**Severity**: [P1/P2/P3/P4]
**Detected At**: $(date)
**Detected By**: $(whoami)
**Initial Assessment**: [Brief description]
**Status**: Investigation in progress
**Estimated Impact**: [Users affected/Data at risk/Service availability]

## Immediate Actions Taken
- Alert acknowledged
- Initial containment measures implemented
- Evidence preservation started
- Stakeholders notified

## Next Steps
- Detailed investigation
- Root cause analysis
- Recovery planning

**Incident Commander**: $(whoami)
**Next Update**: $(date -d '+30 minutes')
EOF

# Send to security team
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"'"$(cat initial-alert.md)"'"}' \
  $SLACK_SECURITY_WEBHOOK
```

## Investigation Steps (15-60 Minutes)

### Step 6: Detailed Log Analysis
```bash
# Export relevant logs
kubectl logs deployment/gan-cyber-range --since=2h > application-logs.txt
kubectl logs -l app=gan-cyber-range --all-containers=true --since=2h > all-container-logs.txt

# Search for indicators of compromise
grep -E "(failed|error|unauthorized|suspicious|attack)" application-logs.txt > security-events.txt
grep -E "(\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b)" application-logs.txt | sort | uniq -c | sort -nr > ip-frequency.txt

# Check authentication logs
grep -i "auth" application-logs.txt | tail -100 > auth-events.txt
```

### Step 7: Network Analysis
```bash
# Capture network connections
kubectl exec -it deployment/gan-cyber-range -- ss -tulpn > network-connections.txt

# Check iptables rules
kubectl exec -it deployment/gan-cyber-range -- iptables -L -n > iptables-rules.txt

# If packet capture is available
kubectl exec -it deployment/gan-cyber-range -- tcpdump -c 1000 -w /tmp/traffic-capture.pcap
kubectl cp deployment/gan-cyber-range:/tmp/traffic-capture.pcap ./traffic-capture.pcap
```

### Step 8: Container and Image Analysis
```bash
# Check running processes
kubectl exec -it deployment/gan-cyber-range -- ps auxf > running-processes.txt

# Check file system changes
kubectl exec -it deployment/gan-cyber-range -- find / -type f -newer /etc/passwd 2>/dev/null > modified-files.txt

# Scan container image for vulnerabilities
docker pull $(kubectl get deployment gan-cyber-range -o jsonpath='{.spec.template.spec.containers[0].image}')
trivy image $(kubectl get deployment gan-cyber-range -o jsonpath='{.spec.template.spec.containers[0].image}') > image-vulnerability-scan.txt
```

### Step 9: Database and Data Integrity Check
```bash
# Check for unauthorized data access
kubectl exec -it deployment/gan-cyber-range -- python -c "
import os
import psycopg2
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()

# Check recent login attempts
cur.execute('SELECT * FROM auth_logs WHERE created_at > NOW() - INTERVAL \\'2 hours\\' ORDER BY created_at DESC LIMIT 100;')
with open('recent-auth-attempts.txt', 'w') as f:
    for row in cur.fetchall():
        f.write(str(row) + '\n')

# Check data modification patterns
cur.execute('SELECT table_name, COUNT(*) FROM information_schema.tables WHERE table_schema = \\'public\\' GROUP BY table_name;')
with open('table-row-counts.txt', 'w') as f:
    for row in cur.fetchall():
        f.write(f'{row[0]}: {row[1]} rows\n')
"
```

## Analysis and Response (1-4 Hours)

### Step 10: Threat Intelligence Correlation
```bash
# Check IP addresses against threat intelligence
for ip in $(cat ip-frequency.txt | awk '{print $2}' | head -20); do
  curl -s "https://api.virustotal.com/api/v3/ip_addresses/$ip" \
    -H "X-Apikey: $VIRUSTOTAL_API_KEY" | jq '.data.attributes.reputation' >> ip-reputation.txt
done

# Check file hashes if suspicious files found
for hash in $(cat suspicious-file-hashes.txt); do
  curl -s "https://api.virustotal.com/api/v3/files/$hash" \
    -H "X-Apikey: $VIRUSTOTAL_API_KEY" | jq '.data.attributes' >> file-analysis.txt
done
```

### Step 11: Impact Assessment
```bash
# Create impact assessment document
cat > impact-assessment.md << EOF
# Impact Assessment - $INCIDENT_ID

## Affected Systems
- [ ] Web Application
- [ ] Database
- [ ] Container Runtime
- [ ] Network Infrastructure
- [ ] User Authentication System

## Data at Risk
- [ ] User credentials
- [ ] Simulation data
- [ ] System configurations
- [ ] Log data
- [ ] Backup data

## Business Impact
- Users affected: [Number]
- Services disrupted: [List]
- Data potentially compromised: [Description]
- Estimated downtime: [Duration]
- Financial impact: [Estimate if known]

## Compliance Implications
- [ ] GDPR notification required
- [ ] SOX compliance impact
- [ ] Customer notification required
- [ ] Regulatory reporting needed

## Evidence Summary
- Log files: $(ls -la *.txt | wc -l) files collected
- Network captures: $(ls -la *.pcap | wc -l) files
- System snapshots: Available
- Forensic images: [Status]
EOF
```

### Step 12: Recovery Planning
```bash
# Document recovery steps
cat > recovery-plan.md << EOF
# Recovery Plan - $INCIDENT_ID

## Immediate Recovery Actions
1. [ ] Stop malicious processes
2. [ ] Remove malicious files
3. [ ] Reset compromised credentials
4. [ ] Apply security patches
5. [ ] Update firewall rules

## System Restoration
1. [ ] Restore from clean backup if necessary
2. [ ] Rebuild affected containers
3. [ ] Update configuration
4. [ ] Implement additional security controls
5. [ ] Test system functionality

## Monitoring Enhancement
1. [ ] Add new detection rules
2. [ ] Increase logging verbosity
3. [ ] Deploy additional monitoring
4. [ ] Update alert thresholds

## Communication Plan
1. [ ] Internal stakeholder updates
2. [ ] Customer notifications
3. [ ] Regulatory notifications
4. [ ] Public communications (if needed)
EOF
```

## Resolution Steps (2-8 Hours)

### Step 13: Execute Recovery Plan
```bash
# Stop malicious processes (if identified)
kubectl exec -it deployment/gan-cyber-range -- pkill -f "malicious-process"

# Remove malicious files
kubectl exec -it deployment/gan-cyber-range -- rm -f /tmp/malicious-file

# Reset compromised credentials
kubectl create secret generic gan-cyber-range-secrets \
  --from-literal=database-password="$(openssl rand -base64 32)" \
  --from-literal=api-key="$(openssl rand -base64 32)" \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart affected services
kubectl rollout restart deployment/gan-cyber-range
kubectl rollout status deployment/gan-cyber-range
```

### Step 14: System Hardening
```bash
# Apply additional security policies
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: enhanced-security-policy
spec:
  podSelector:
    matchLabels:
      app: gan-cyber-range
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: allowed-namespace
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
EOF

# Update pod security standards
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gan-cyber-range-secured
  labels:
    app: gan-cyber-range
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    fsGroup: 1001
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
EOF
```

### Step 15: Validation and Testing
```bash
# Verify system integrity
kubectl exec -it deployment/gan-cyber-range -- /app/scripts/health-check.sh > post-recovery-health.txt

# Test application functionality
curl -f http://localhost:8080/health
curl -f http://localhost:8080/api/v1/status

# Run security scan
trivy k8s --report summary cluster > post-recovery-security-scan.txt
```

## Post-Incident Activities (Day 1-7)

### Step 16: Documentation and Reporting
```bash
# Create comprehensive incident report
cat > incident-report-$INCIDENT_ID.md << EOF
# Security Incident Report - $INCIDENT_ID

## Executive Summary
[High-level summary for management]

## Incident Details
- **Detection Time**: $(grep "Incident.*initiated" incident-timeline.txt)
- **Resolution Time**: $(date)
- **Duration**: [Calculate duration]
- **Severity**: [Final assessment]

## Timeline of Events
$(cat incident-timeline.txt)

## Root Cause Analysis
[Detailed analysis of how the incident occurred]

## Impact Assessment
$(cat impact-assessment.md)

## Response Actions Taken
[List all actions taken during incident response]

## Lessons Learned
[What went well, what could be improved]

## Recommendations
[Specific actions to prevent similar incidents]

## Evidence
[List of all evidence collected and preserved]
EOF
```

### Step 17: Follow-up Actions
```bash
# Create tracking tickets for improvements
cat > follow-up-actions.md << EOF
# Follow-up Actions - $INCIDENT_ID

## Security Improvements
- [ ] Implement additional monitoring rules
- [ ] Enhance authentication mechanisms
- [ ] Update security policies
- [ ] Conduct security training

## Process Improvements
- [ ] Update incident response procedures
- [ ] Improve alerting thresholds
- [ ] Enhance documentation
- [ ] Schedule tabletop exercises

## Technical Improvements
- [ ] Apply security patches
- [ ] Implement additional controls
- [ ] Upgrade vulnerable components
- [ ] Enhance logging and monitoring
EOF
```

### Step 18: Knowledge Transfer
```bash
# Update runbooks based on lessons learned
echo "## Lessons from $INCIDENT_ID" >> security-incident-response.md
echo "- [Specific lesson 1]" >> security-incident-response.md
echo "- [Specific lesson 2]" >> security-incident-response.md

# Schedule team briefing
echo "Team briefing scheduled for $(date -d '+3 days') to discuss incident $INCIDENT_ID" >> team-communications.txt
```

## Escalation Procedures

### When to Escalate
- Unable to contain incident within 1 hour
- Evidence of data breach or unauthorized access
- Incident affects critical business operations
- Legal or regulatory implications identified
- External threat actor involvement suspected

### Escalation Contacts
1. **Platform Lead**: platform-lead@gan-cyber-range.org
2. **CISO**: ciso@gan-cyber-range.org
3. **Legal Counsel**: legal@gan-cyber-range.org
4. **External IR Firm**: emergency@incident-response-firm.com
5. **Law Enforcement**: (if criminal activity suspected)

## Communication Templates

### Internal Notification
```
Subject: Security Incident $INCIDENT_ID - [SEVERITY] - Initial Alert

A security incident has been detected and is being investigated.

Incident ID: $INCIDENT_ID
Severity: [P1/P2/P3/P4]
Detected: $(date)
Status: Under investigation

Initial assessment indicates [brief description].

Immediate actions taken:
- System monitoring enhanced
- Containment measures implemented
- Investigation in progress

Next update will be provided in 1 hour or sooner if significant developments occur.

For questions, contact: [Incident Commander]
```

### Customer Notification (if required)
```
Subject: Security Notice - GAN Cyber Range Platform

Dear Valued Customers,

We are writing to inform you of a security incident that occurred on our platform on [DATE]. 

We detected [brief description of incident] and immediately began our incident response procedures. Our security team, working with external experts, has contained the issue and is conducting a thorough investigation.

Impact to your data: [Description of impact]
Actions we're taking: [Description of response]
Actions you should take: [If any]

We take the security of your data seriously and are committed to transparency. We will provide updates as our investigation progresses.

If you have questions or concerns, please contact us at security@gan-cyber-range.org.

Sincerely,
[Name], Chief Security Officer
```

## Compliance Requirements

### Regulatory Notifications
- **GDPR**: 72 hours to regulators, prompt to data subjects
- **SOX**: Immediate notification if financial systems affected
- **HIPAA**: 60 days if health information involved
- **State Laws**: Variable requirements by jurisdiction

### Documentation Requirements
- Detailed incident timeline
- Evidence preservation
- Response actions taken
- Impact assessment
- Lessons learned
- Remediation actions

---

**Last Updated**: [Date]
**Version**: 1.0
**Owner**: Security Team
**Review Schedule**: Quarterly

For questions or updates to this runbook, contact security@gan-cyber-range.org.