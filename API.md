# GAN Cyber Range Simulator - API Reference

## 🚀 REST API Overview

The GAN Cyber Range Simulator provides a comprehensive REST API for programmatic access to all simulation capabilities.

**Base URL**: `https://api.gan-cyber-range.org/v1`  
**Authentication**: JWT Bearer tokens  
**Content-Type**: `application/json`  
**Rate Limits**: 1000 requests/hour per user

## 🔐 Authentication

### Login

```http
POST /auth/login
Content-Type: application/json

{
  "username": "researcher@example.com",
  "password": "secure_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "username": "researcher@example.com",
    "role": "researcher",
    "permissions": ["create_simulation", "view_simulation"]
  }
}
```

### API Key Authentication

```http
GET /api/v1/simulations
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 🎯 Simulation Management

### Create Simulation

```http
POST /api/v1/simulations
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Advanced APT Simulation",
  "description": "Multi-stage attack simulation with nation-state tactics",
  "config": {
    "duration_hours": 2.0,
    "realtime_factor": 60,
    "difficulty": "hard",
    "network_topology": "multi-tier",
    "services": [
      {
        "name": "webapp",
        "type": "web_application",
        "vulnerabilities": ["CVE-2023-1234", "CVE-2023-5678"],
        "replicas": 3
      },
      {
        "name": "database",
        "type": "database",
        "vulnerabilities": ["CVE-2023-9999"],
        "replicas": 1
      }
    ],
    "agents": {
      "red_team": {
        "model": "gpt-4",
        "skill_level": "expert",
        "count": 2
      },
      "blue_team": {
        "model": "claude-3",
        "skill_level": "expert",
        "defense_strategy": "proactive",
        "count": 2
      }
    }
  }
}
```

**Response**:
```json
{
  "simulation_id": "sim_01HZ8Q2K3M4N5P6R7S8T9V0W1X",
  "status": "created",
  "created_at": "2025-08-04T13:00:00Z",
  "estimated_completion": "2025-08-04T15:00:00Z",
  "dashboard_url": "https://dashboard.gan-cyber-range.org/simulations/sim_01HZ8Q2K3M4N5P6R7S8T9V0W1X"
}
```

### Start Simulation

```http
POST /api/v1/simulations/{simulation_id}/start
Authorization: Bearer <token>

{
  "delay_seconds": 0,
  "notification_webhook": "https://your-app.com/webhooks/simulation"
}
```

### Get Simulation Status

```http
GET /api/v1/simulations/{simulation_id}
Authorization: Bearer <token>
```

**Response**:
```json
{
  "simulation_id": "sim_01HZ8Q2K3M4N5P6R7S8T9V0W1X",
  "status": "running",
  "progress": {
    "current_round": 45,
    "total_rounds": 120,
    "completion_percentage": 37.5
  },
  "metrics": {
    "attacks_attempted": 123,
    "attacks_successful": 67,
    "services_compromised": 2,
    "patches_deployed": 34,
    "defense_effectiveness": 0.654
  },
  "agents": [
    {
      "name": "red_team_1",
      "type": "attacker",
      "status": "active",
      "actions_count": 67,
      "success_rate": 0.612
    },
    {
      "name": "blue_team_1", 
      "type": "defender",
      "status": "active",
      "actions_count": 89,
      "success_rate": 0.743
    }
  ],
  "services": [
    {
      "name": "webapp",
      "status": "compromised",
      "compromise_time": "2025-08-04T13:23:45Z",
      "patches_applied": 3
    },
    {
      "name": "database",
      "status": "running",
      "patches_applied": 1
    }
  ]
}
```

### Stop Simulation

```http
POST /api/v1/simulations/{simulation_id}/stop
Authorization: Bearer <token>

{
  "reason": "Manual termination",
  "save_results": true
}
```

### Get Simulation Results

```http
GET /api/v1/simulations/{simulation_id}/results
Authorization: Bearer <token>
```

**Response**:
```json
{
  "simulation_id": "sim_01HZ8Q2K3M4N5P6R7S8T9V0W1X",
  "duration": "PT1H23M45S",
  "summary": {
    "total_attacks": 156,
    "successful_attacks": 89,
    "services_compromised": 3,
    "patches_deployed": 45,
    "compromise_rate": 0.571,
    "defense_effectiveness": 0.692,
    "avg_detection_time": "PT4M32S",
    "avg_remediation_time": "PT12M18S"
  },
  "attack_timeline": [
    {
      "timestamp": "2025-08-04T13:05:23Z",
      "agent": "red_team_1",
      "action": "port_scan",
      "target": "webapp",
      "success": true,
      "impact": "reconnaissance"
    },
    {
      "timestamp": "2025-08-04T13:07:45Z",
      "agent": "blue_team_1",
      "action": "threat_detection",
      "target": "network_traffic",
      "success": true,
      "impact": "early_warning"
    }
  ],
  "final_state": {
    "services_status": {
      "webapp": "compromised",
      "database": "patched", 
      "api_gateway": "running"
    },
    "network_topology": {
      "isolated_segments": 2,
      "active_honeypots": 3
    }
  },
  "recommendations": [
    "Implement WAF rules to prevent SQL injection",
    "Deploy automated patching for critical vulnerabilities",
    "Enhance network segmentation between web and database tiers"
  ]
}
```

## 👥 Agent Management

### List Available Agents

```http
GET /api/v1/agents
Authorization: Bearer <token>
```

**Response**:
```json
{
  "agents": [
    {
      "id": "red_team_advanced",
      "name": "Advanced Red Team",
      "type": "attacker",
      "model": "gpt-4",
      "skill_levels": ["beginner", "intermediate", "advanced", "expert"],
      "capabilities": [
        "reconnaissance",
        "vulnerability_exploitation", 
        "privilege_escalation",
        "persistence",
        "data_exfiltration"
      ]
    },
    {
      "id": "blue_team_proactive",
      "name": "Proactive Blue Team",
      "type": "defender",
      "model": "claude-3",
      "strategies": ["reactive", "proactive", "predictive"],
      "capabilities": [
        "threat_detection",
        "incident_response",
        "automated_patching",
        "honeypot_deployment"
      ]
    }
  ]
}
```

### Create Custom Agent

```http
POST /api/v1/agents
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Custom APT Simulator",
  "type": "attacker",
  "model": "gpt-4",
  "config": {
    "skill_level": "expert",
    "tactics": ["reconnaissance", "lateral_movement", "data_exfiltration"],
    "tools": ["custom_exploits", "living_off_the_land"],
    "persistence_methods": ["registry_modification", "scheduled_tasks"],
    "learning_enabled": true,
    "max_actions_per_round": 8
  }
}
```

### Get Agent Performance

```http
GET /api/v1/agents/{agent_id}/performance
Authorization: Bearer <token>
```

**Response**:
```json
{
  "agent_id": "red_team_advanced",
  "performance_metrics": {
    "total_simulations": 1245,
    "avg_success_rate": 0.673,
    "avg_actions_per_simulation": 23.4,
    "specialties": [
      {
        "category": "web_exploitation",
        "success_rate": 0.834
      },
      {
        "category": "privilege_escalation", 
        "success_rate": 0.712
      }
    ]
  },
  "learning_stats": {
    "patterns_learned": 2847,
    "success_rate_improvement": 0.123,
    "adaptation_speed": "fast"
  }
}
```

## 🏗️ Environment Management

### List Service Templates

```http
GET /api/v1/environments/services
Authorization: Bearer <token>
```

**Response**:
```json
{
  "services": [
    {
      "id": "vulnerable_webapp",
      "name": "Vulnerable Web Application",
      "type": "web_application",
      "description": "OWASP Top 10 vulnerable web app",
      "vulnerabilities": [
        {
          "cve_id": "CVE-2023-1234",
          "severity": "critical",
          "description": "SQL Injection in login form"
        }
      ],
      "resource_requirements": {
        "cpu": "100m",
        "memory": "256Mi",
        "storage": "1Gi"
      }
    }
  ]
}
```

### Deploy Custom Service

```http
POST /api/v1/environments/services
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "custom-api-service",
  "type": "api_server",
  "image": "your-registry/custom-api:v1.0",
  "vulnerabilities": [
    {
      "type": "authentication_bypass",
      "severity": "high",
      "exploitable": true,
      "patch_available": false
    }
  ],
  "environment": {
    "DATABASE_URL": "postgresql://user:pass@db:5432/api_db",
    "DEBUG": "true"
  },
  "resources": {
    "cpu": "500m",
    "memory": "512Mi",
    "replicas": 2
  }
}
```

### Get Environment Status

```http
GET /api/v1/environments/{environment_id}/status
Authorization: Bearer <token>
```

**Response**:
```json
{
  "environment_id": "env_01HZ8Q2K3M4N5P6R7S8T9V0W1X",
  "status": "healthy",
  "services": [
    {
      "name": "webapp",
      "status": "running",
      "replicas": {
        "desired": 3,
        "ready": 3,
        "available": 3
      },
      "health_checks": {
        "liveness": "passing",
        "readiness": "passing"
      },
      "resource_usage": {
        "cpu_percent": 23.5,
        "memory_percent": 45.2
      }
    }
  ],
  "network": {
    "policies_active": 15,
    "isolated_segments": 3,
    "traffic_encrypted": true
  }
}
```

## 📊 Metrics & Monitoring

### Get Real-time Metrics

```http
GET /api/v1/metrics/live
Authorization: Bearer <token>
```

**Response**:
```json
{
  "timestamp": "2025-08-04T13:30:00Z",
  "system_metrics": {
    "active_simulations": 23,
    "total_agents": 46,
    "cpu_usage_percent": 67.3,
    "memory_usage_percent": 54.2,
    "network_throughput_mbps": 123.4
  },
  "performance_metrics": {
    "avg_response_time_ms": 145.6,
    "requests_per_second": 342.1,
    "cache_hit_rate": 0.876,
    "error_rate": 0.002
  },
  "security_metrics": {
    "failed_authentications": 12,
    "policy_violations": 0,
    "active_threats_detected": 3
  }
}
```

### Get Historical Data

```http
GET /api/v1/metrics/history?metric=simulation_performance&from=2025-08-01&to=2025-08-04
Authorization: Bearer <token>
```

**Response**:
```json
{
  "metric": "simulation_performance",
  "time_range": {
    "from": "2025-08-01T00:00:00Z",
    "to": "2025-08-04T23:59:59Z"
  },
  "data_points": [
    {
      "timestamp": "2025-08-01T00:00:00Z",
      "value": 0.834,
      "metadata": {
        "simulations_count": 45,
        "avg_duration_minutes": 78.3
      }
    }
  ],
  "aggregations": {
    "avg": 0.823,
    "min": 0.692,
    "max": 0.891,
    "trend": "improving"
  }
}
```

### Export Metrics

```http
POST /api/v1/metrics/export
Authorization: Bearer <token>
Content-Type: application/json

{
  "format": "csv",
  "metrics": ["simulation_results", "agent_performance", "security_events"],
  "time_range": {
    "from": "2025-08-01T00:00:00Z",
    "to": "2025-08-04T23:59:59Z"
  },
  "filters": {
    "simulation_type": ["red_team_exercise", "blue_team_training"],
    "difficulty": ["medium", "hard"]
  }
}
```

**Response**:
```json
{
  "export_id": "exp_01HZ8Q2K3M4N5P6R7S8T9V0W1X",
  "status": "processing",
  "estimated_completion": "2025-08-04T13:35:00Z",
  "download_url": null
}
```

## 🔒 Security & Administration

### List Users

```http
GET /api/v1/admin/users
Authorization: Bearer <admin_token>
```

**Response**:
```json
{
  "users": [
    {
      "id": "user_01HZ8Q2K3M4N5P6R7S8T9V0W1X",
      "username": "researcher@example.com",
      "role": "researcher", 
      "active": true,
      "last_login": "2025-08-04T12:30:00Z",
      "simulations_created": 15,
      "permissions": ["create_simulation", "view_simulation"]
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 1,
    "total_pages": 1
  }
}
```

### Create User

```http
POST /api/v1/admin/users
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "username": "newuser@example.com",
  "email": "newuser@example.com",
  "role": "student",
  "password": "secure_password_123",
  "permissions": ["view_simulation"]
}
```

### Get Audit Log

```http
GET /api/v1/admin/audit?from=2025-08-01&to=2025-08-04&user_id=user_123
Authorization: Bearer <admin_token>
```

**Response**:
```json
{
  "audit_entries": [
    {
      "id": "audit_01HZ8Q2K3M4N5P6R7S8T9V0W1X",
      "timestamp": "2025-08-04T13:00:00Z",
      "event_type": "simulation_created",
      "user_id": "user_123",
      "resource_type": "simulation",
      "resource_id": "sim_456",
      "details": {
        "simulation_name": "Advanced APT Test",
        "duration_hours": 2.0,
        "agents_count": 4
      },
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0..."
    }
  ]
}
```

## 🔗 Webhooks

### Register Webhook

```http
POST /api/v1/webhooks
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/simulation-events",
  "events": [
    "simulation.started",
    "simulation.completed", 
    "agent.action.executed",
    "service.compromised"
  ],
  "secret": "your_webhook_secret",
  "active": true
}
```

### Webhook Payload Example

```json
{
  "event": "simulation.completed",
  "timestamp": "2025-08-04T15:00:00Z",
  "data": {
    "simulation_id": "sim_01HZ8Q2K3M4N5P6R7S8T9V0W1X",
    "duration": "PT2H0M0S",
    "status": "completed",
    "results": {
      "compromise_rate": 0.667,
      "defense_effectiveness": 0.743
    }
  },
  "webhook_id": "wh_01HZ8Q2K3M4N5P6R7S8T9V0W1X"
}
```

## 🐍 Python SDK

### Installation

```bash
pip install gan-cyber-range-sdk
```

### Basic Usage

```python
from gan_cyber_range import GanCyberRangeClient

# Initialize client
client = GanCyberRangeClient(
    api_key="your_api_key",
    base_url="https://api.gan-cyber-range.org/v1"
)

# Create and start simulation
simulation = await client.simulations.create({
    "name": "Python SDK Test",
    "config": {
        "duration_hours": 1.0,
        "services": ["webapp", "database"],
        "agents": {
            "red_team": {"model": "gpt-4", "skill_level": "advanced"},
            "blue_team": {"model": "claude-3", "skill_level": "advanced"}
        }
    }
})

# Start simulation
await client.simulations.start(simulation.id)

# Monitor progress
async for status in client.simulations.watch(simulation.id):
    print(f"Progress: {status.progress.completion_percentage}%")
    if status.status == "completed":
        break

# Get results
results = await client.simulations.get_results(simulation.id)
print(f"Compromise rate: {results.summary.compromise_rate:.2%}")
```

## 📋 Error Codes

| Code | Message | Description |
|------|---------|-------------|
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource does not exist |
| 409 | Conflict | Resource already exists or conflicts |
| 422 | Unprocessable Entity | Valid JSON but invalid data |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | System maintenance or overload |

### Error Response Format

```json
{
  "error": {
    "code": "SIMULATION_NOT_FOUND",
    "message": "Simulation with ID 'sim_invalid' not found",
    "details": {
      "simulation_id": "sim_invalid",
      "suggestion": "Check the simulation ID and try again"
    },
    "request_id": "req_01HZ8Q2K3M4N5P6R7S8T9V0W1X"
  }
}
```

## 🚦 Rate Limits

| Endpoint Category | Rate Limit | Window |
|------------------|------------|---------|
| Authentication | 10 requests | 1 minute |
| Simulation Management | 100 requests | 1 hour |
| Metrics & Monitoring | 500 requests | 1 hour |
| Admin Operations | 50 requests | 1 hour |
| Webhooks | 1000 requests | 1 hour |

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1625097600
```

---

*Generated with [Claude Code](https://claude.ai/code) - Complete API reference for GAN Cyber Range Simulator*