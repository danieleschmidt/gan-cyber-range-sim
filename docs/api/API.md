# GAN Cyber Range API Documentation

Generated on: 2025-08-08 16:24:10


        # GAN Cyber Range API

        A comprehensive API for managing adversarial cybersecurity simulations using AI agents.

        ## Features

        - **Simulation Management**: Create, monitor, and control cyber range simulations
        - **Agent Configuration**: Configure red team (attacker) and blue team (defender) AI agents
        - **Real-time Monitoring**: Track simulation progress and metrics
        - **Security**: Enterprise-grade authentication and authorization
        - **Performance**: Optimized for high-throughput simulations

        ## Authentication

        This API uses JWT tokens for authentication. Include your token in the Authorization header:

        ```
        Authorization: Bearer <your-jwt-token>
        ```

        ## Rate Limiting

        API calls are rate limited to prevent abuse:
        - 1000 requests per hour for authenticated users
        - 100 requests per hour for unauthenticated users

        ## Response Format

        All responses follow a consistent format:

        ```json
        {
            "status": "success|error",
            "data": {},
            "message": "Human readable message",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        ```

        ## Error Handling

        Standard HTTP status codes are used:
        - 200: Success
        - 201: Created
        - 400: Bad Request
        - 401: Unauthorized
        - 404: Not Found
        - 422: Validation Error
        - 500: Internal Server Error
        

## Base URLs

- `http://localhost:8000` - Development server
- `https://api.gan-cyber-range.org` - Production server

## Endpoints

### Health

#### GET /health

**Summary:** Health Check

Check service health.

---

### Monitoring

#### GET /metrics

**Summary:** Get Metrics

Get system metrics.

---

### Simulations

#### GET /api/v1/simulations

**Summary:** List Simulations

List all simulations.

---

#### POST /api/v1/simulations

**Summary:** Create Simulation

Create and start a new cyber range simulation.

---

#### GET /api/v1/simulations/{simulation_id}/status

**Summary:** Get Simulation Status

Get the current status of a simulation.

---

#### GET /api/v1/simulations/{simulation_id}/results

**Summary:** Get Simulation Results

Get the results of a completed simulation.

---

#### POST /api/v1/simulations/{simulation_id}/stop

**Summary:** Stop Simulation

Stop a running simulation.

---

## Data Models

### AgentAction

Represents an action taken by an agent.

---

### BlueTeamConfig

Configuration for blue team agent.

---

### HTTPValidationError

---

### HealthCheck

Health check response.

---

### MetricsResponse

Metrics response model.

---

### RedTeamConfig

Configuration for red team agent.

---

### SimulationConfig

Configuration for a simulation.

---

### SimulationRequest

Request to start a new simulation.

**Example:**

```json
{
  "simulation_config": {
    "vulnerable_services": [
      "webapp",
      "database",
      "api-gateway"
    ],
    "network_topology": "multi-tier",
    "difficulty": "medium",
    "duration_hours": 2.0,
    "realtime_factor": 60
  },
  "red_team_config": {
    "name": "AdvancedRedTeam",
    "llm_model": "gpt-4",
    "skill_level": "advanced",
    "tools": [
      "nmap",
      "metasploit",
      "custom_exploits"
    ]
  },
  "blue_team_config": {
    "name": "ProactiveBlueTeam",
    "llm_model": "claude-3",
    "skill_level": "advanced",
    "defense_strategy": "proactive"
  }
}
```

---

### SimulationResponse

Response from starting a simulation.

**Example:**

```json
{
  "status": "started",
  "simulation_id": "sim_abc123def456",
  "simulation_url": "/api/v1/simulations/sim_abc123def456",
  "estimated_duration_minutes": 120,
  "created_at": "2024-01-01T12:00:00Z"
}
```

---

### SimulationResults

Results from a completed simulation.

---

### SimulationStatus

Current status of a simulation.

**Example:**

```json
{
  "simulation_id": "sim_abc123def456",
  "status": "running",
  "is_running": true,
  "start_time": "2024-01-01T12:00:00Z",
  "elapsed_time_seconds": 1800,
  "progress": {
    "red_team_actions": 15,
    "blue_team_responses": 12,
    "services_compromised": 2,
    "services_defended": 3,
    "current_phase": "exploitation"
  },
  "metrics": {
    "compromise_rate": 0.4,
    "detection_rate": 0.8,
    "mean_time_to_detection": 45.0,
    "mean_time_to_response": 30.0
  }
}
```

---

### ValidationError

---

