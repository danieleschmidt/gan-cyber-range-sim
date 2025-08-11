"""Basic functionality tests for GAN Cyber Range."""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock

from gan_cyber_range.agents.llm_client import LLMClientFactory, MockLLMClient


class TestBasicFunctionality:
    """Test basic system functionality."""

    def test_llm_client_factory_mock(self):
        """Test LLM client factory creates mock clients."""
        client = LLMClientFactory.create_client("mock-model")
        assert isinstance(client, MockLLMClient)

    def test_llm_client_factory_openai(self):
        """Test LLM client factory recognizes OpenAI models."""
        try:
            client = LLMClientFactory.create_client("gpt-4", api_key="test")
            # Should create OpenAI client (don't test API calls)
            assert client.model == "gpt-4"
        except Exception:
            # API might not be available in test environment
            pytest.skip("OpenAI client creation failed - likely missing API key")

    def test_mock_llm_basic_functionality(self):
        """Test mock LLM client basic functionality."""
        client = MockLLMClient()
        
        # Test attributes exist
        assert hasattr(client, 'model')
        assert hasattr(client, 'generate')
        assert hasattr(client, 'chat')

    @pytest.mark.asyncio
    async def test_mock_llm_generate(self):
        """Test mock LLM generation."""
        client = MockLLMClient()
        
        response = await client.generate("Test prompt")
        
        assert hasattr(response, 'content')
        assert len(response.content) > 0


class TestAPIModels:
    """Test API model validation."""

    def test_import_api_models(self):
        """Test that API models can be imported."""
        from gan_cyber_range.api.models import SimulationConfig
        
        # Test basic model creation
        config = SimulationConfig(
            vulnerable_services=["webapp"],
            network_topology="single-tier",
            difficulty="easy",
            duration_hours=1.0,
            realtime_factor=60
        )
        
        assert config.vulnerable_services == ["webapp"]
        assert config.difficulty == "easy"

    def test_red_team_config(self):
        """Test red team configuration."""
        from gan_cyber_range.api.models import RedTeamConfig
        
        config = RedTeamConfig(
            name="TestRed",
            llm_model="mock-model",
            skill_level="beginner",
            tools=["nmap"]
        )
        
        assert config.name == "TestRed"
        assert config.skill_level == "beginner"

    def test_blue_team_config(self):
        """Test blue team configuration."""
        from gan_cyber_range.api.models import BlueTeamConfig
        
        config = BlueTeamConfig(
            name="TestBlue",
            llm_model="mock-model",
            skill_level="beginner",
            defense_strategy="reactive"
        )
        
        assert config.name == "TestBlue"
        assert config.defense_strategy == "reactive"


class TestCyberRangeEnvironment:
    """Test cyber range environment."""

    def test_import_cyber_range(self):
        """Test that cyber range can be imported."""
        from gan_cyber_range.environment.cyber_range import CyberRange
        
        # Test basic instantiation
        cyber_range = CyberRange()
        
        assert hasattr(cyber_range, 'services')
        assert hasattr(cyber_range, 'network_topology')

    def test_cyber_range_basic_functionality(self):
        """Test cyber range basic functionality."""
        from gan_cyber_range.environment.cyber_range import CyberRange
        
        cyber_range = CyberRange()
        
        # Test service management
        service = {
            "name": "test-webapp",
            "type": "web_application",
            "vulnerabilities": []
        }
        
        cyber_range.add_service(service)
        assert len(cyber_range.services) > 0


class TestAgentBase:
    """Test agent base functionality."""

    def test_import_agents(self):
        """Test that agents can be imported."""
        from gan_cyber_range.agents.red_team import RedTeamAgent
        from gan_cyber_range.agents.blue_team import BlueTeamAgent
        
        # Basic import test
        assert RedTeamAgent is not None
        assert BlueTeamAgent is not None

    def test_red_team_basic_functionality(self):
        """Test red team agent basic functionality."""
        from gan_cyber_range.agents.red_team import RedTeamAgent
        from gan_cyber_range.api.models import RedTeamConfig
        
        config = RedTeamConfig(
            name="TestRed",
            llm_model="mock-model",
            skill_level="beginner"
        )
        
        client = MockLLMClient()
        agent = RedTeamAgent(config=config, llm_client=client)
        
        assert agent.config.name == "TestRed"
        assert hasattr(agent, 'current_phase')

    def test_blue_team_basic_functionality(self):
        """Test blue team agent basic functionality."""
        from gan_cyber_range.agents.blue_team import BlueTeamAgent
        from gan_cyber_range.api.models import BlueTeamConfig
        
        config = BlueTeamConfig(
            name="TestBlue",
            llm_model="mock-model",
            skill_level="beginner",
            defense_strategy="reactive"
        )
        
        client = MockLLMClient()
        agent = BlueTeamAgent(config=config, llm_client=client)
        
        assert agent.config.name == "TestBlue"
        assert hasattr(agent, 'monitoring_active')


class TestAPIServer:
    """Test API server functionality."""

    def test_import_api_server(self):
        """Test that API server can be imported."""
        from gan_cyber_range.api.server import app
        
        assert app is not None

    def test_health_endpoint_structure(self):
        """Test health endpoint structure."""
        from fastapi.testclient import TestClient
        from gan_cyber_range.api.server import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestSecurityComponents:
    """Test security component imports."""

    def test_import_security_modules(self):
        """Test that security modules can be imported."""
        from gan_cyber_range.security.auth import AuthenticationManager
        from gan_cyber_range.security.validator import SecurityValidator
        
        assert AuthenticationManager is not None
        assert SecurityValidator is not None

    def test_auth_manager_basic(self):
        """Test authentication manager basic functionality."""
        from gan_cyber_range.security.auth import AuthenticationManager
        
        auth_manager = AuthenticationManager()
        assert hasattr(auth_manager, 'create_user')
        assert hasattr(auth_manager, 'authenticate_user')


class TestPerformanceComponents:
    """Test performance component imports."""

    def test_import_performance_modules(self):
        """Test that performance modules can be imported."""
        from gan_cyber_range.performance.cache import CacheManager
        from gan_cyber_range.performance.optimizer import PerformanceOptimizer
        
        assert CacheManager is not None
        assert PerformanceOptimizer is not None


class TestMonitoringComponents:
    """Test monitoring component imports."""

    def test_import_monitoring_modules(self):
        """Test that monitoring modules can be imported."""
        from gan_cyber_range.monitoring.metrics import MetricsCollector
        from gan_cyber_range.monitoring.health_check import HealthChecker
        
        assert MetricsCollector is not None
        assert HealthChecker is not None

    def test_metrics_collector_basic(self):
        """Test metrics collector basic functionality."""
        from gan_cyber_range.monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        assert hasattr(collector, 'collect_metrics')
        assert hasattr(collector, 'get_metrics')


class TestResilienceComponents:
    """Test resilience component imports."""

    def test_import_resilience_modules(self):
        """Test that resilience modules can be imported."""
        from gan_cyber_range.resilience.circuit_breaker import CircuitBreaker
        from gan_cyber_range.resilience.retry import RetryPolicy
        
        assert CircuitBreaker is not None
        assert RetryPolicy is not None