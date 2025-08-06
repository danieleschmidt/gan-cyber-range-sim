# GAN Cyber Range Simulator - Complete Deployment Guide

## 🎯 Production-Ready Deployment

This guide provides comprehensive instructions for deploying the GAN Cyber Range Simulator in production environments with enterprise-grade reliability, security, and performance.

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / RHEL 8+ / Kubernetes 1.29+
- **Python**: 3.10+  
- **Memory**: Minimum 4GB RAM, Recommended 8GB+
- **Storage**: Minimum 20GB SSD
- **Network**: Internet access for LLM APIs

### Required Services
- **Kubernetes Cluster** (for container orchestration)
- **PostgreSQL 13+** (for persistent data)
- **Redis 6+** (for caching and sessions)
- **Monitoring Stack** (Prometheus + Grafana)

## 🚀 Quick Start Deployment

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
cd quantum-inspired-task-planner

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Configure required variables
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DATABASE_URL="postgresql://user:pass@localhost:5432/ganrange"
export REDIS_URL="redis://localhost:6379"
```

### 3. Database Setup
```bash
# Initialize database
python -m gan_cyber_range.cli db-init

# Run migrations
python -m gan_cyber_range.cli db-migrate
```

### 4. Start Services
```bash
# Start API server
uvicorn gan_cyber_range.api.server:app --host 0.0.0.0 --port 8000

# Start monitoring
python -m gan_cyber_range.cli monitor --start
```

## 🏢 Enterprise Deployment

### Docker Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  gan-cyber-range:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/ganrange
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: ganrange
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gan-cyber-range
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
      containers:
      - name: gan-cyber-range
        image: gan-cyber-range:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: gan-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## 🔒 Security Configuration

### 1. API Security
```python
# Security headers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["your-domain.com", "*.your-domain.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 2. Network Security
```yaml
# Network policies for Kubernetes
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cyber-range-netpol
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
```

### 3. Secrets Management
```bash
# Create Kubernetes secrets
kubectl create secret generic gan-secrets \
  --from-literal=openai-api-key="your-key" \
  --from-literal=anthropic-api-key="your-key" \
  --from-literal=database-url="your-db-url"
```

## 📊 Monitoring and Observability

### 1. Health Checks
```python
# Custom health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "0.1.0",
        "services": {
            "database": await check_database_health(),
            "redis": await check_redis_health(),
            "llm_services": await check_llm_health()
        }
    }
```

### 2. Metrics Collection
```yaml
# Prometheus configuration
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'gan-cyber-range'
  static_configs:
  - targets: ['gan-cyber-range:8000']
  metrics_path: '/metrics'
```

### 3. Logging Configuration
```python
# Structured logging
import structlog

logger = structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
```

## ⚡ Performance Optimization

### 1. Caching Strategy
```python
# Redis caching configuration
CACHE_CONFIG = {
    "default": {
        "backend": "redis",
        "url": os.getenv("REDIS_URL"),
        "serializer": "pickle",
        "ttl": 3600,  # 1 hour default
    },
    "llm_responses": {
        "ttl": 86400,  # 24 hours for LLM responses
    },
    "agent_analysis": {
        "ttl": 1800,   # 30 minutes for analysis
    }
}
```

### 2. Database Optimization
```sql
-- Performance indexes
CREATE INDEX CONCURRENTLY idx_simulations_status 
ON simulations (status, created_at);

CREATE INDEX CONCURRENTLY idx_agent_actions_simulation 
ON agent_actions (simulation_id, timestamp);

CREATE INDEX CONCURRENTLY idx_security_events_timestamp 
ON security_events (timestamp DESC, severity);
```

### 3. Connection Pooling
```python
# Database connection pool
from sqlalchemy.pool import QueuePool

engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## 🌍 Global Deployment Architecture

### Multi-Region Setup
```yaml
# Global load balancer configuration
apiVersion: v1
kind: Service
metadata:
  name: gan-cyber-range-global
  annotations:
    cloud.google.com/global-lb: "true"
spec:
  type: LoadBalancer
  selector:
    app: gan-cyber-range
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
```

### Regional Configurations
- **US-East**: Primary region (Virginia)
- **US-West**: Secondary region (Oregon)  
- **EU-West**: European users (Ireland)
- **Asia-Pacific**: Asian users (Singapore)

## 🔧 Configuration Management

### Environment Variables
```bash
# Production environment
export ENVIRONMENT="production"
export LOG_LEVEL="INFO"
export DEBUG="false"
export MAX_WORKERS="4"
export MAX_SIMULATIONS="100"
export REQUEST_TIMEOUT="300"
export LLM_TIMEOUT="60"
export CIRCUIT_BREAKER_THRESHOLD="5"
export RETRY_MAX_ATTEMPTS="3"
```

### Feature Flags
```python
# Feature toggles
FEATURE_FLAGS = {
    "advanced_analytics": True,
    "real_time_monitoring": True,
    "auto_scaling": True,
    "experimental_agents": False,
    "multi_tenant": True
}
```

## 📈 Scaling Guidelines

### Horizontal Scaling
- **API Servers**: 3-10 replicas depending on load
- **Worker Processes**: 5-20 workers for simulation processing
- **Database**: Read replicas in each region
- **Cache**: Redis cluster with 3-6 nodes

### Vertical Scaling
- **CPU**: 2-8 cores per API server
- **Memory**: 4-16GB per API server
- **Storage**: SSD with 1000+ IOPS

### Auto-scaling Configuration
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
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🧪 Testing and Validation

### Pre-deployment Testing
```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests  
python -m pytest tests/integration/ -v

# Performance tests
python -m pytest tests/performance/ -v

# Security tests
python -m pytest tests/security/ -v
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8000/health

# Using Artillery
artillery quick --count 100 --num 10 http://localhost:8000/api/v1/simulations
```

### Smoke Tests
```bash
# Post-deployment validation
curl -f http://localhost:8000/health
curl -f http://localhost:8000/metrics
curl -X POST http://localhost:8000/api/v1/simulations -d @test-simulation.json
```

## 🚨 Disaster Recovery

### Backup Strategy
- **Database**: Daily automated backups with 30-day retention
- **Configuration**: Version controlled in Git
- **Secrets**: Encrypted backup in secure storage
- **Application State**: Regular snapshots

### Recovery Procedures
1. **Database Recovery**: Restore from latest backup
2. **Service Recovery**: Redeploy from version control
3. **Configuration Recovery**: Apply from infrastructure as code
4. **Validation**: Run smoke tests and health checks

## 📞 Support and Maintenance

### Monitoring Alerts
- High memory usage (>90%)
- High CPU usage (>80%)
- Database connection errors
- LLM API failures
- Simulation failures (>5% error rate)

### Maintenance Windows
- **Planned**: Sundays 2-4 AM UTC
- **Emergency**: As needed with 15-minute notice
- **Updates**: Monthly security patches

### Support Contacts
- **Primary**: ops-team@company.com  
- **Secondary**: dev-team@company.com
- **Emergency**: +1-555-CYBER-OPS

---

## ✅ Deployment Checklist

- [ ] Environment configured
- [ ] Dependencies installed
- [ ] Database initialized
- [ ] Secrets configured
- [ ] Health checks passing
- [ ] Monitoring enabled
- [ ] Load balancer configured
- [ ] SSL certificates installed
- [ ] Backup strategy implemented
- [ ] Documentation updated

## 🎉 Post-Deployment Success Criteria

1. **Availability**: 99.9% uptime SLA
2. **Performance**: <200ms API response time
3. **Scalability**: Handle 1000+ concurrent simulations
4. **Security**: Pass all security scans
5. **Monitoring**: Full observability stack operational

---

*For additional support or questions, please refer to the documentation or contact the development team.*