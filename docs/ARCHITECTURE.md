# Architecture Overview

## System Design Philosophy

The GAN Cyber Range Simulator follows a microservices architecture designed for security, scalability, and research reproducibility. The system implements adversarial training patterns where red and blue team agents evolve through competitive learning.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GAN Cyber Range Simulator               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│  │  Red Team   │◄────┤   Orchestr- │────►│  Blue Team  │  │
│  │   Agents    │     │   ator      │     │   Agents    │  │
│  └─────────────┘     └─────────────┘     └─────────────┘  │
│         │                   │                   │         │
│         ▼                   ▼                   ▼         │
│  ┌─────────────────────────────────────────────────────┐  │
│  │            Kubernetes Environment               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │ Vulnerable  │  │   Network   │  │ Monitoring  │  │  │
│  │  │  Services   │  │   Topology  │  │   Stack     │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                            │                              │
│  ┌─────────────────────────────────────────────────────┐  │
│  │               Metrics & Analytics               │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent Framework

#### Red Team Agents (Attackers)
- **LLM-Powered Planning**: Uses GPT-4/Claude for attack strategy generation
- **Tool Integration**: Interfaces with security tools (nmap, metasploit)
- **Learning System**: Adapts tactics based on success/failure feedback
- **Skill Progression**: Evolves from basic to advanced techniques

#### Blue Team Agents (Defenders)
- **Proactive Defense**: Continuous monitoring and threat hunting
- **Automated Response**: Real-time patch deployment and mitigation
- **Threat Intelligence**: Pattern recognition and IOC generation
- **Recovery Operations**: System restoration and hardening

### 2. Kubernetes-Native Environment

#### Pod Architecture
```yaml
# Example vulnerable service deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vulnerable-webapp
  namespace: cyber-range
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vulnerable-webapp
  template:
    metadata:
      labels:
        app: vulnerable-webapp
        vulnerability-level: "medium"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: webapp
        image: gan-cyber-range/vuln-webapp:latest
        resources:
          limits:
            cpu: 500m
            memory: 512Mi
        env:
        - name: VULNERABILITY_CONFIG
          valueFrom:
            configMapKeyRef:
              name: vuln-config
              key: webapp.yaml
```

#### Network Policies
- **Micro-segmentation**: Each service isolated by default
- **Controlled Communication**: Explicit allow rules only
- **Traffic Monitoring**: All network flows logged and analyzed
- **Attack Simulation**: Realistic network topologies

### 3. Vulnerability Simulation Engine

#### Dynamic Vulnerability Injection
```python
class VulnerabilityEngine:
    """Dynamically injects vulnerabilities for training"""
    
    def __init__(self, difficulty_level: str):
        self.difficulty = difficulty_level
        self.active_vulns = []
    
    async def inject_vulnerability(self, target: str, vuln_type: str):
        """Safely inject vulnerability for training"""
        vuln_config = self.generate_vuln_config(vuln_type)
        await self.deploy_vulnerable_instance(target, vuln_config)
        
    def generate_vuln_config(self, vuln_type: str) -> dict:
        """Generate vulnerability configuration"""
        return {
            'type': vuln_type,
            'difficulty': self.difficulty,
            'detection_signatures': self.get_signatures(vuln_type),
            'mitigation_hints': self.get_mitigation_hints(vuln_type)
        }
```

#### Supported Vulnerability Classes
- **Web Application**: SQLi, XSS, CSRF, XXE, SSRF
- **Infrastructure**: Misconfigurations, weak credentials
- **Container**: Privilege escalation, resource exhaustion
- **Network**: Man-in-the-middle, DNS spoofing
- **Supply Chain**: Dependency vulnerabilities, malicious packages

### 4. Training and Learning Framework

#### Reinforcement Learning Integration
```python
class AdversarialTraining:
    """GAN-style training for security agents"""
    
    def __init__(self):
        self.red_team_policy = PPOPolicy()
        self.blue_team_policy = SACPolicy()
        self.environment = CyberRangeEnv()
    
    async def train_episode(self):
        """Single training episode"""
        state = await self.environment.reset()
        
        while not self.environment.done:
            # Red team action
            red_action = await self.red_team_policy.select_action(state)
            
            # Environment response
            state, reward, done = await self.environment.step(red_action)
            
            # Blue team response
            blue_action = await self.blue_team_policy.select_action(state)
            
            # Update policies
            await self.update_policies(red_action, blue_action, reward)
```

#### Reward Mechanisms
- **Red Team Rewards**: Successful exploitation, stealth, persistence
- **Blue Team Rewards**: Quick detection, effective mitigation, minimal disruption
- **Balanced Competition**: Ensures neither side dominates completely

### 5. Monitoring and Observability

#### Metrics Collection
```yaml
# Prometheus configuration
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'cyber-range-agents'
    static_configs:
      - targets: ['red-team:8080', 'blue-team:8080']
    metrics_path: /metrics
    
  - job_name: 'vulnerable-services'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ['cyber-range']
```

#### Real-time Dashboard
- **Attack Progress**: Live visualization of attack chains
- **Defense Effectiveness**: Detection rates and response times
- **System Health**: Resource utilization and performance
- **Learning Metrics**: Agent improvement over time

## Security Architecture

### Isolation Boundaries

#### Network Isolation
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cyber-range-isolation
  namespace: cyber-range
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: cyber-range
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: cyber-range
```

#### Resource Limits
- **CPU**: Maximum 4 cores per agent
- **Memory**: 8GB limit per agent
- **Storage**: 20GB per persistent volume
- **Network**: Rate limiting on external traffic

### Data Protection
- **Encryption**: All data encrypted at rest and in transit
- **Secret Management**: Kubernetes secrets for sensitive data
- **Access Control**: RBAC with minimal privileges
- **Audit Logging**: Comprehensive activity tracking

## Scalability Considerations

### Horizontal Scaling
- **Agent Parallelization**: Multiple agents per team
- **Environment Replication**: Concurrent simulation environments
- **Load Balancing**: Traffic distribution across instances
- **Auto-scaling**: Resource adjustment based on demand

### Performance Optimization
- **Caching**: Redis for frequently accessed data
- **Database**: PostgreSQL with read replicas
- **Message Queuing**: RabbitMQ for async communication
- **CDN**: Static asset delivery optimization

## Integration Points

### External Security Tools
- **SIEM Integration**: Splunk, Elastic Stack
- **Vulnerability Scanners**: Nessus, OpenVAS
- **Threat Intelligence**: MISP, ThreatConnect
- **Incident Response**: PagerDuty, ServiceNow

### API Architecture
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="GAN Cyber Range API")

class SimulationRequest(BaseModel):
    scenario: str
    duration_minutes: int
    red_team_config: dict
    blue_team_config: dict

@app.post("/simulations/start")
async def start_simulation(request: SimulationRequest):
    """Start a new cyber range simulation"""
    simulation_id = await create_simulation(request)
    return {"simulation_id": simulation_id, "status": "started"}
```

## Deployment Models

### Development Environment
- **Local Kubernetes**: k3s or minikube
- **Resource Requirements**: 8GB RAM, 4 CPU cores
- **Storage**: 50GB available space
- **Network**: Isolated development network

### Production Environment
- **Cloud Kubernetes**: EKS, GKE, or AKS
- **High Availability**: Multi-zone deployment
- **Backup Strategy**: Automated daily backups
- **Disaster Recovery**: Cross-region replication

## Research Integration

### Reproducibility Framework
- **Scenario Versioning**: Git-based scenario management
- **Environment Snapshots**: Container image versioning
- **Experiment Tracking**: MLflow integration
- **Result Archival**: Long-term data storage

### Publication Support
- **Metrics Export**: Research-ready data formats
- **Visualization**: Publication-quality charts
- **Statistical Analysis**: R/Python integration
- **Citation Generation**: Automated bibliography

This architecture enables scalable, secure, and reproducible cybersecurity research while maintaining isolation and safety for all participants.