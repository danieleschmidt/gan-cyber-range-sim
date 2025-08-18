# Build & Deployment Guide

This guide covers building, containerizing, and deploying the GAN Cyber Range Simulator.

## Overview

The project uses a multi-stage Docker build approach with security hardening and comprehensive SBOM generation.

## Build Targets

### Available Docker Build Targets

| Target | Purpose | Use Case |
|--------|---------|----------|
| `development` | Full development environment | Local development, debugging |
| `testing` | Runs tests during build | CI/CD validation |
| `security-scan` | Security scanning tools | Security validation |
| `production` | Minimal production image | Production deployment |
| `sbom` | SBOM generation | Compliance reporting |

### Build Commands

```bash
# Development build
docker build --target development -t gan-cyber-range:dev .

# Production build
docker build --target production -t gan-cyber-range:prod .

# Testing build (includes test execution)
docker build --target testing -t gan-cyber-range:test .

# Security scanning build
docker build --target security-scan -t gan-cyber-range:security .

# SBOM generation build
docker build --target sbom -t gan-cyber-range:sbom .
```

## Build Configuration

### Multi-Stage Build Architecture

```dockerfile
base → development → production
     ↘ testing
     ↘ security-scan
     ↘ sbom
```

### Security Features

1. **Non-root user**: All processes run as `cyberrange` user (UID 1001)
2. **Minimal attack surface**: Production image includes only necessary packages
3. **Read-only filesystem**: Application files are read-only in production
4. **Health checks**: Built-in health monitoring endpoints
5. **Security scanning**: Integrated Bandit, Safety, and Semgrep scanning

## Makefile Targets

### Development

```bash
# Install dependencies
make install-dev

# Run tests
make test
make test-coverage

# Code quality
make lint
make format

# Security checks
make security
```

### Docker Operations

```bash
# Build all variants
make docker-build-all

# Build specific target
make docker-build TARGET=production

# Run container
make docker-run

# Push to registry
make docker-push
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
make k8s-deploy

# Deploy with custom namespace
make k8s-deploy NAMESPACE=custom-namespace

# Clean up deployment
make k8s-clean
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Deploy

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build test image
      run: docker build --target testing -t gan-cyber-range:test .
    
    - name: Run security scans
      run: docker build --target security-scan -t gan-cyber-range:security .
    
    - name: Generate SBOM
      run: |
        docker build --target sbom -t gan-cyber-range:sbom .
        docker run --rm gan-cyber-range:sbom cat /tmp/sbom.json > sbom.json
    
    - name: Upload SBOM artifact
      uses: actions/upload-artifact@v3
      with:
        name: sbom
        path: sbom.json

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Build production image
      run: docker build --target production -t gan-cyber-range:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
        docker push gan-cyber-range:${{ github.sha }}
```

## Deployment Environments

### Local Development

```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f

# Execute commands in container
docker-compose exec app bash

# Stop environment
docker-compose down
```

### Production Deployment

#### Container Registry

```bash
# Tag for registry
docker tag gan-cyber-range:prod your-registry.com/gan-cyber-range:v1.0.0

# Push to registry
docker push your-registry.com/gan-cyber-range:v1.0.0

# Pull and run
docker pull your-registry.com/gan-cyber-range:v1.0.0
docker run -d --name cyber-range \
  -p 8080:8000 \
  -e ENVIRONMENT=production \
  your-registry.com/gan-cyber-range:v1.0.0
```

#### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gan-cyber-range
  namespace: cybersec-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gan-cyber-range
  template:
    metadata:
      labels:
        app: gan-cyber-range
    spec:
      serviceAccountName: cyber-range-sa
      securityContext:
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
      containers:
      - name: cyber-range
        image: your-registry.com/gan-cyber-range:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: cyber-range-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
---
apiVersion: v1
kind: Service
metadata:
  name: gan-cyber-range-service
spec:
  selector:
    app: gan-cyber-range
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

## Security Considerations

### Container Security

1. **Base Image Security**
   - Uses official Python slim images
   - Regular security updates
   - Minimal package installation

2. **Runtime Security**
   - Non-root execution
   - Read-only filesystem where possible
   - Capability dropping
   - Resource limits

3. **Network Security**
   - Network policies for isolation
   - Ingress/egress restrictions
   - TLS termination at load balancer

### Build Security

1. **Supply Chain Security**
   - SBOM generation for all dependencies
   - Vulnerability scanning in CI/CD
   - Signed container images
   - Reproducible builds

2. **Secrets Management**
   - No secrets in container images
   - Environment variable injection
   - Kubernetes secrets integration
   - Vault integration support

## Monitoring and Observability

### Health Checks

```bash
# Application health
curl http://localhost:8080/health

# Detailed health check
curl http://localhost:8080/health/detailed

# Readiness check
curl http://localhost:8080/ready
```

### Metrics Endpoints

```bash
# Prometheus metrics
curl http://localhost:8080/metrics

# Application metrics
curl http://localhost:8080/api/v1/metrics/application

# System metrics
curl http://localhost:8080/api/v1/metrics/system
```

### Logging

```yaml
# Fluentd configuration for log collection
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/gan-cyber-range-*.log
      pos_file /var/log/fluentd-gan-cyber-range.log.pos
      tag kubernetes.gan-cyber-range
      format json
    </source>
    
    <match kubernetes.gan-cyber-range>
      @type elasticsearch
      host elasticsearch.monitoring.svc.cluster.local
      port 9200
      index_name gan-cyber-range
    </match>
```

## Performance Optimization

### Build Optimization

1. **Layer Caching**
   ```dockerfile
   # Install dependencies before copying source code
   COPY requirements.txt pyproject.toml ./
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   ```

2. **Multi-stage Benefits**
   - Smaller production images
   - Parallel build stages
   - Build-time testing
   - Development tool separation

3. **Build Context Optimization**
   ```dockerignore
   # Comprehensive .dockerignore
   .git
   tests/
   docs/
   __pycache__/
   *.pyc
   .pytest_cache/
   ```

### Runtime Optimization

1. **Resource Limits**
   ```yaml
   resources:
     requests:
       memory: "512Mi"
       cpu: "500m"
     limits:
       memory: "2Gi"
       cpu: "2000m"
   ```

2. **Horizontal Pod Autoscaling**
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: gan-cyber-range-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: gan-cyber-range
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

## Troubleshooting

### Common Build Issues

1. **Dependency Installation Failures**
   ```bash
   # Check pip logs
   docker build --progress=plain --no-cache .
   
   # Debug in interactive mode
   docker run -it --entrypoint bash python:3.13-slim
   ```

2. **Permission Errors**
   ```bash
   # Check user/group setup
   docker run --rm gan-cyber-range:dev id
   
   # Check file permissions
   docker run --rm gan-cyber-range:dev ls -la /app
   ```

3. **Health Check Failures**
   ```bash
   # Test health endpoint directly
   docker run -p 8080:8000 gan-cyber-range:dev
   curl http://localhost:8080/health
   
   # Check application logs
   docker logs <container_id>
   ```

### Runtime Issues

1. **Database Connection Problems**
   ```bash
   # Check environment variables
   kubectl describe pod <pod-name>
   
   # Test database connectivity
   kubectl exec -it <pod-name> -- python -c "import os; print(os.getenv('DATABASE_URL'))"
   ```

2. **Memory Issues**
   ```bash
   # Check resource usage
   kubectl top pods
   
   # Increase memory limits
   kubectl patch deployment gan-cyber-range -p '{"spec":{"template":{"spec":{"containers":[{"name":"cyber-range","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
   ```

3. **Network Connectivity**
   ```bash
   # Test service connectivity
   kubectl port-forward service/gan-cyber-range-service 8080:80
   
   # Check network policies
   kubectl describe networkpolicy
   ```

## Compliance and Auditing

### SBOM Generation

```bash
# Generate comprehensive SBOM
./scripts/generate-sbom.sh

# Validate SBOM format
cyclonedx-cli validate --input-file sbom/comprehensive-sbom.json
```

### Security Scanning

```bash
# Container vulnerability scanning
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image gan-cyber-range:prod

# License compliance check
docker run --rm -v $(pwd):/src \
  licensefinder/license_finder
```

### Audit Trail

All builds and deployments should maintain:
- Build artifacts with checksums
- SBOM files for compliance
- Security scan results
- Deployment configuration versions
- Access logs and audit trails

---

For additional deployment support or questions, see our [Troubleshooting Guide](./TROUBLESHOOTING.md) or join our [Discord Community](https://discord.gg/gan-cyber-range).