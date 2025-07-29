# Development Guide

## Development Environment Setup

### Prerequisites

- **Python 3.10+** - Latest stable version recommended
- **Docker 24.0+** - For containerization
- **Kubernetes 1.29+** - Local (k3s/minikube) or cloud cluster
- **Git** - Version control
- **Make** - Build automation (optional)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/gan-cyber-range-sim.git
cd gan-cyber-range-sim

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development tools
pip install -r requirements.txt[dev]

# Setup pre-commit hooks
pre-commit install

# Verify installation
python -m pytest tests/ -v
```

### Local Kubernetes Setup

#### Option 1: k3s (Recommended for Linux)
```bash
# Install k3s
curl -sfL https://get.k3s.io | sh -

# Configure kubectl
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config
export KUBECONFIG=~/.kube/config

# Verify installation
kubectl get nodes
```

#### Option 2: minikube (Cross-platform)
```bash
# Install minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start minikube
minikube start --cpus=4 --memory=8192

# Enable required addons
minikube addons enable metrics-server
minikube addons enable ingress
```

### Development Workflow

#### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/new-vulnerability-type

# Make changes and test
python -m pytest tests/test_new_feature.py

# Run security checks
bandit -r src/
safety check

# Code formatting
black src/ tests/
isort src/ tests/

# Pre-commit checks
pre-commit run --all-files
```

#### 2. Testing Strategy

##### Unit Tests
```bash
# Run unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=gan_cyber_range --cov-report=html
```

##### Integration Tests
```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/ -v

# Cleanup
docker-compose -f docker-compose.test.yml down
```

##### Security Tests
```bash
# Run security-focused tests
pytest tests/security/ -v -m security

# Vulnerability simulation tests
pytest tests/scenarios/ -v -m vulnerability
```

#### 3. Local Development Server

```bash
# Start the development server
uvicorn gan_cyber_range.api:app --reload --host 0.0.0.0 --port 8000

# Access the API documentation
open http://localhost:8000/docs
```

## Project Structure

```
gan-cyber-range-sim/
├── src/
│   └── gan_cyber_range/
│       ├── __init__.py
│       ├── agents/           # Red/Blue team agent implementations
│       ├── api/             # FastAPI REST API
│       ├── cli/             # Command-line interface
│       ├── environment/     # Kubernetes environment management
│       ├── scenarios/       # Pre-built attack scenarios
│       ├── training/        # ML training and evaluation
│       └── utils/           # Shared utilities
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── security/           # Security-specific tests
│   └── scenarios/          # Scenario validation tests
├── docs/                   # Documentation
├── deployments/            # Kubernetes manifests
├── scripts/               # Utility scripts
└── config/               # Configuration files
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Type hints and docstrings
from typing import Dict, List, Optional
import asyncio

class RedTeamAgent:
    """Red team agent for attack simulation."""
    
    def __init__(self, llm_model: str, skill_level: str) -> None:
        """Initialize red team agent.
        
        Args:
            llm_model: The LLM model to use (e.g., 'gpt-4')
            skill_level: Agent skill level ('basic', 'intermediate', 'advanced')
        """
        self.llm_model = llm_model
        self.skill_level = skill_level
        self._attack_history: List[Dict] = []
    
    async def plan_attack(self, target_info: Dict) -> Dict:
        """Plan attack strategy based on target information.
        
        Args:
            target_info: Information about the target system
            
        Returns:
            Attack plan with stages and techniques
            
        Raises:
            ValueError: If target_info is invalid
        """
        if not target_info:
            raise ValueError("Target information cannot be empty")
            
        # Implementation here
        return {"stages": [], "techniques": []}
```

### Security Coding Practices

#### Input Validation
```python
from pydantic import BaseModel, validator
from typing import Literal

class AttackConfig(BaseModel):
    target: str
    technique: Literal["sqli", "xss", "xxe", "ssrf"]
    intensity: int
    
    @validator('target')
    def validate_target(cls, v):
        if not v or len(v) > 100:
            raise ValueError('Invalid target specification')
        return v
    
    @validator('intensity')
    def validate_intensity(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Intensity must be between 1 and 10')
        return v
```

#### Secure Configuration
```python
import os
from pathlib import Path
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with security defaults."""
    
    debug: bool = False
    secret_key: str
    database_url: str
    redis_url: str = "redis://localhost:6379"
    
    # Security settings
    cors_origins: List[str] = ["http://localhost:3000"]
    api_rate_limit: int = 100
    session_timeout: int = 3600
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Usage
settings = Settings()
```

### Testing Best Practices

#### Test Structure
```python
import pytest
from unittest.mock import patch, AsyncMock
from gan_cyber_range.agents import RedTeamAgent

class TestRedTeamAgent:
    """Test suite for RedTeamAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create test agent instance."""
        return RedTeamAgent(llm_model="gpt-4", skill_level="intermediate")
    
    @pytest.mark.asyncio
    async def test_plan_attack_success(self, agent):
        """Test successful attack planning."""
        target_info = {"ip": "192.168.1.100", "services": ["http", "ssh"]}
        
        with patch.object(agent, '_generate_plan') as mock_generate:
            mock_generate.return_value = {"stages": ["recon", "exploit"]}
            
            result = await agent.plan_attack(target_info)
            
            assert "stages" in result
            assert len(result["stages"]) > 0
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_plan_attack_invalid_input(self, agent):
        """Test attack planning with invalid input."""
        with pytest.raises(ValueError, match="Target information cannot be empty"):
            await agent.plan_attack({})
```

#### Security Test Examples
```python
import pytest
from gan_cyber_range.api import app
from fastapi.testclient import TestClient

class TestSecurityEndpoints:
    """Security-focused API tests."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_sql_injection_protection(self):
        """Test API protection against SQL injection."""
        malicious_payload = "'; DROP TABLE users; --"
        
        response = self.client.post("/simulations", json={
            "name": malicious_payload,
            "scenario": "basic"
        })
        
        # Should sanitize input and not cause internal error
        assert response.status_code in [400, 422]  # Bad request or validation error
        assert "DROP TABLE" not in response.text
    
    def test_rate_limiting(self):
        """Test API rate limiting."""
        # Simulate rapid requests
        responses = []
        for _ in range(150):  # Exceed rate limit
            response = self.client.get("/health")
            responses.append(response.status_code)
        
        # Should get rate limited
        assert 429 in responses  # Too Many Requests
```

## Configuration Management

### Environment Variables
```bash
# .env file for development
DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=postgresql://user:pass@localhost/ganrange
REDIS_URL=redis://localhost:6379

# LLM API Keys (for testing only - use secrets in production)
OPENAI_API_KEY=sk-test-key-here
ANTHROPIC_API_KEY=sk-ant-test-key-here

# Kubernetes configuration
KUBECONFIG=/path/to/kubeconfig
NAMESPACE=cyber-range-dev
```

### Configuration Files
```yaml
# config/development.yaml
api:
  host: "0.0.0.0"
  port: 8000
  debug: true
  
agents:
  red_team:
    model: "gpt-4"
    max_concurrent: 3
    timeout_seconds: 300
  
  blue_team:
    model: "claude-3"
    response_time_ms: 1000
    auto_patch: true

kubernetes:
  namespace: "cyber-range-dev"
  resource_limits:
    cpu: "2"
    memory: "4Gi"
  
logging:
  level: "DEBUG"
  format: "json"
  file: "logs/development.log"
```

## Build and Deployment

### Local Build
```bash
# Build container images
docker build -t gan-cyber-range:dev .

# Run with docker-compose
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Development Scripts
```bash
# scripts/dev-setup.sh
#!/bin/bash
set -e

echo "Setting up development environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Setup pre-commit
pre-commit install

# Create local config
cp config/development.yaml.example config/development.yaml

echo "Development environment ready!"
```

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

## Debugging and Troubleshooting

### Common Issues

#### 1. Kubernetes Connection Issues
```bash
# Check cluster connection
kubectl cluster-info

# Verify namespace
kubectl get namespaces

# Check resource limits
kubectl describe quota -n cyber-range
```

#### 2. Agent Communication Problems
```bash
# Check agent logs
kubectl logs -f deployment/red-team-agent -n cyber-range

# Verify service discovery
kubectl get svc -n cyber-range

# Test internal connectivity
kubectl exec -it red-team-agent-pod -- curl http://blue-team-service:8080/health
```

#### 3. Performance Issues
```bash
# Monitor resource usage
kubectl top pods -n cyber-range

# Check for resource constraints
kubectl describe pod <pod-name> -n cyber-range

# Review metrics
curl http://localhost:9090/metrics
```

### Debugging Tools

#### 1. Interactive Debugging
```python
# Use debugger in code
import pdb; pdb.set_trace()

# Or with IPython
import IPython; IPython.embed()
```

#### 2. Logging Configuration
```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

## Performance Optimization

### Profiling
```bash
# Profile Python code
python -m cProfile -o profile.stats main.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"

# Memory profiling
python -m memory_profiler main.py
```

### Optimization Guidelines
- Use async/await for I/O operations
- Implement connection pooling for databases
- Cache frequently accessed data
- Use batch operations where possible
- Profile before optimizing

## Security Development Practices

### Threat Modeling
1. **Identify Assets**: What needs protection?
2. **Identify Threats**: What could go wrong?
3. **Identify Vulnerabilities**: How could threats be realized?
4. **Identify Countermeasures**: How to mitigate risks?

### Security Review Checklist
- [ ] Input validation implemented
- [ ] Output encoding applied
- [ ] Authentication/authorization verified
- [ ] Secrets management secured
- [ ] Error handling appropriate
- [ ] Logging security events
- [ ] Rate limiting configured
- [ ] Dependencies updated

This development guide ensures consistent, secure, and maintainable code while supporting the research mission of the project.