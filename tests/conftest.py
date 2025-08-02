"""Pytest configuration and shared fixtures for GAN Cyber Range tests."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, Generator
import logging
import os


# Disable noisy loggers during testing
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("kubernetes").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_kubernetes_client():
    """Mock Kubernetes client for testing without cluster dependency."""
    mock_client = MagicMock()
    
    # Mock common Kubernetes operations
    mock_client.create_namespace = AsyncMock()
    mock_client.delete_namespace = AsyncMock()
    mock_client.create_deployment = AsyncMock()
    mock_client.delete_deployment = AsyncMock()
    mock_client.get_pods = AsyncMock(return_value=[])
    mock_client.get_services = AsyncMock(return_value=[])
    
    return mock_client


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without API calls."""
    mock_client = MagicMock()
    mock_client.generate = AsyncMock(return_value="Mock LLM response")
    mock_client.chat = AsyncMock(return_value="Mock chat response")
    mock_client.stream = AsyncMock()
    
    # Mock different response types
    mock_client.attack_strategy = AsyncMock(return_value={
        "techniques": ["reconnaissance", "exploitation"],
        "tools": ["nmap", "metasploit"],
        "timeline": "1-2 hours"
    })
    
    mock_client.defense_strategy = AsyncMock(return_value={
        "monitors": ["network_traffic", "system_logs"],
        "responses": ["isolate", "patch"],
        "escalation": "automatic"
    })
    
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="Mock OpenAI response"))]
    ))
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text="Mock Anthropic response")]
    ))
    return mock_client


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def isolated_environment():
    """Mock isolated environment for security testing."""
    return {
        "network_isolation": True,
        "resource_limits": {"cpu": "2", "memory": "4Gi"},
        "security_policies": ["network_policy", "pod_security"],
        "namespace": "test-cyber-range",
        "container_runtime": "containerd",
        "security_context": {
            "runAsNonRoot": True,
            "readOnlyRootFilesystem": True,
            "allowPrivilegeEscalation": False
        }
    }


@pytest.fixture
def sample_vulnerabilities():
    """Sample vulnerability data for testing."""
    return [
        {
            "id": "CVE-2024-0001",
            "type": "sql_injection",
            "severity": "high",
            "affected_service": "webapp",
            "exploit_available": True
        },
        {
            "id": "CVE-2024-0002", 
            "type": "xss",
            "severity": "medium",
            "affected_service": "frontend",
            "exploit_available": False
        }
    ]


@pytest.fixture
def sample_attack_scenarios():
    """Sample attack scenarios for testing."""
    return [
        {
            "name": "Basic Web Attack",
            "techniques": ["reconnaissance", "exploitation"],
            "target_services": ["webapp"],
            "expected_duration": 300,
            "difficulty": "beginner"
        },
        {
            "name": "APT Simulation",
            "techniques": ["spear_phishing", "lateral_movement", "persistence"],
            "target_services": ["email", "workstation", "server"],
            "expected_duration": 3600,
            "difficulty": "advanced"
        }
    ]


@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/15",
        "log_level": "DEBUG",
        "kubernetes_namespace": "test-cyber-range",
        "simulation_timeout": 60,
        "max_concurrent_simulations": 2
    }


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.exists = AsyncMock(return_value=False)
    mock_redis.expire = AsyncMock(return_value=True)
    return mock_redis


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.fetch_all = AsyncMock(return_value=[])
    mock_db.fetch_one = AsyncMock(return_value=None)
    mock_db.transaction = AsyncMock()
    return mock_db


@pytest.fixture
def security_test_environment():
    """Enhanced security testing environment with strict controls."""
    return {
        "network_policies": {
            "default_deny": True,
            "allowed_ingress": [],
            "allowed_egress": ["dns", "internal"]
        },
        "pod_security": {
            "enforce": "restricted",
            "audit": "restricted", 
            "warn": "restricted"
        },
        "rbac": {
            "service_account": "cyber-range-test",
            "permissions": ["get", "list", "watch"]
        },
        "resource_quotas": {
            "cpu": "4",
            "memory": "8Gi",
            "pods": "10",
            "services": "5"
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for all tests."""
    # Set test environment variables
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    
    yield
    
    # Cleanup after test
    test_env_vars = ["ENVIRONMENT", "LOG_LEVEL", "DATABASE_URL"]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def performance_benchmark():
    """Fixture for performance benchmarking."""
    def _benchmark_function(func, *args, **kwargs):
        import time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    return _benchmark_function


# Pytest collection hooks for better test organization
def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test location."""
    for item in items:
        # Add markers based on test path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            
        # Add slow marker for tests that take longer than expected
        if "slow" in item.name or "test_long" in item.name:
            item.add_marker(pytest.mark.slow)


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test for quick validation"
    )
    config.addinivalue_line(
        "markers", "regression: mark test as regression test"
    )
    config.addinivalue_line(
        "markers", "agent: mark test as AI agent functionality test"
    )
    config.addinivalue_line(
        "markers", "kubernetes: mark test as requiring Kubernetes cluster"
    )
    config.addinivalue_line(
        "markers", "network: mark test as involving network operations"
    )