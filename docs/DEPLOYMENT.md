# Deployment Guide

This guide covers deployment strategies for the GAN Cyber Range Simulator in various environments.

## Security Considerations

**⚠️ CRITICAL**: This tool contains security research components. Ensure proper isolation and access controls in all deployment environments.

## Prerequisites

- Kubernetes cluster (v1.29+)
- Docker container runtime
- Helm 3.14+
- kubectl configured with appropriate access

## Development Deployment

### Local Development

```bash
# Setup development environment
./scripts/setup-dev.sh

# Start local services
docker-compose up -d

# Run the application
python -m gan_cyber_range.api
```

### Kubernetes Development

```bash
# Create namespace
kubectl create namespace cyber-range-dev

# Apply development manifests
kubectl apply -f deployments/dev/ -n cyber-range-dev

# Port forward for local access
kubectl port-forward service/cyber-range-api 8000:8000 -n cyber-range-dev
```

## Production Deployment

### Infrastructure Requirements

- **CPU**: 4 cores minimum per node
- **Memory**: 8GB RAM minimum per node
- **Storage**: 100GB persistent storage
- **Network**: Isolated network segment

### Security Hardening

```yaml
# Example security policy
apiVersion: v1
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
```

### Helm Deployment

```bash
# Add repository
helm repo add cyber-range https://charts.example.com/cyber-range

# Install with production values
helm install cyber-range cyber-range/gan-cyber-range \
  --namespace cyber-range \
  --create-namespace \
  --values values-production.yaml
```

## Monitoring and Observability

### Metrics Collection

- Prometheus for metrics
- Grafana for visualization
- AlertManager for notifications

### Health Checks

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Logging

- Structured JSON logging
- ELK stack integration
- Log retention policies
- Security event auditing

## Disaster Recovery

### Backup Strategy

- Daily configuration backups
- Model checkpoint preservation
- Database snapshots
- Infrastructure as Code versioning

### Recovery Procedures

1. **Data Recovery**: Restore from latest backup
2. **Service Recovery**: Redeploy from known good state
3. **Network Recovery**: Validate isolation boundaries
4. **Verification**: Run security validation tests

## Scaling and Performance

### Horizontal Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cyber-range-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cyber-range-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Performance Optimization

- Container resource limits
- Database connection pooling
- Caching strategies
- Load balancing configuration

## Security Operations

### Access Control

- RBAC implementation
- Service account management
- Network segmentation
- Secret management

### Vulnerability Management

- Regular security scans
- Dependency updates
- Container image updates
- Compliance monitoring

## Troubleshooting

### Common Issues

**Pod Crashes**
```bash
kubectl logs -f deployment/cyber-range-api -n cyber-range
kubectl describe pod <pod-name> -n cyber-range
```

**Network Connectivity**
```bash
kubectl exec -it <pod-name> -n cyber-range -- nc -zv service-name 8000
```

**Resource Constraints**
```bash
kubectl top pods -n cyber-range
kubectl describe nodes
```

### Debug Commands

```bash
# Check service status
kubectl get all -n cyber-range

# View logs
kubectl logs -l app=cyber-range -n cyber-range --tail=100

# Debug networking
kubectl exec -it debug-pod -n cyber-range -- /bin/bash
```

## References

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Monitoring Guide](https://prometheus.io/docs/)