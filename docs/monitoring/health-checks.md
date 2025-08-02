# 🏥 Health Check Configuration

## Application Health Endpoints

### FastAPI Health Check

```python
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
import asyncio
import aioredis
import asyncpg

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    )

@app.get("/health/ready")
async def readiness_check():
    """Readiness check with dependencies"""
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "external_services": await check_external_services()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def check_database():
    """Check PostgreSQL connection"""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False

async def check_redis():
    """Check Redis connection"""
    try:
        redis = aioredis.from_url(REDIS_URL)
        await redis.ping()
        await redis.close()
        return True
    except Exception:
        return False

async def check_external_services():
    """Check external API dependencies"""
    try:
        # Check OpenAI API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                timeout=5.0
            )
            return response.status_code == 200
    except Exception:
        return False
```

## Kubernetes Health Check Configuration

```yaml
# deployments/k8s/health-check.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gan-cyber-range
spec:
  containers:
  - name: app
    image: gan-cyber-range:latest
    ports:
    - containerPort: 8000
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 5
      timeoutSeconds: 5
      failureThreshold: 3
    startupProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 5
      timeoutSeconds: 5
      failureThreshold: 30
```

## Resource Monitoring

### CPU and Memory Limits

```yaml
resources:
  limits:
    cpu: "2"
    memory: "4Gi"
  requests:
    cpu: "1"
    memory: "2Gi"
```

### Custom Health Metrics

```python
import psutil
from prometheus_client import Gauge

# Custom health metrics
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('app_memory_usage_percent', 'Memory usage percentage')
disk_usage = Gauge('app_disk_usage_percent', 'Disk usage percentage')

@app.get("/metrics/health")
async def health_metrics():
    """Expose detailed health metrics"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    
    cpu_usage.set(cpu_percent)
    memory_usage.set(memory_percent)
    disk_usage.set(disk_percent)
    
    return {
        "cpu_usage_percent": cpu_percent,
        "memory_usage_percent": memory_percent,
        "disk_usage_percent": disk_percent,
        "healthy": cpu_percent < 80 and memory_percent < 85 and disk_percent < 90
    }
```

## Monitoring Integration

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'gan-cyber-range'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'gan-cyber-range-health'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/health'
    scrape_interval: 10s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "GAN Cyber Range Health",
    "panels": [
      {
        "title": "Service Health Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"gan-cyber-range\"}",
            "legendFormat": "Service Up"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "http_request_duration_seconds{job=\"gan-cyber-range\"}",
            "legendFormat": "Response Time"
          }
        ]
      }
    ]
  }
}
```