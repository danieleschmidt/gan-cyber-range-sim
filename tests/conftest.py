"""Pytest configuration and shared fixtures for GAN Cyber Range tests."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_kubernetes_client():
    """Mock Kubernetes client for testing without cluster dependency."""
    return MagicMock()


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without API calls."""
    mock_client = MagicMock()
    mock_client.generate.return_value = "Mock LLM response"
    return mock_client


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def isolated_environment():
    """Mock isolated environment for security testing."""
    return {
        "network_isolation": True,
        "resource_limits": {"cpu": "2", "memory": "4Gi"},
        "security_policies": ["network_policy", "pod_security"]
    }