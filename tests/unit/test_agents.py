"""Unit tests for agent components."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from gan_cyber_range.agents.base import BaseAgent
from gan_cyber_range.agents.red_team import RedTeamAgent
from gan_cyber_range.agents.blue_team import BlueTeamAgent
from gan_cyber_range.agents.llm_client import LLMClientFactory, OpenAIClient, AnthropicClient, MockLLMClient
from gan_cyber_range.api.models import RedTeamConfig, BlueTeamConfig


class TestBaseAgent:
    """Test base agent functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = AsyncMock()
        client.generate.return_value = "Mock response"
        client.chat.return_value = "Mock chat response"
        return client

    @pytest.fixture
    def base_agent(self, mock_llm_client):
        """Create base agent instance."""
        agent = BaseAgent(llm_client=mock_llm_client)
        return agent

    def test_base_agent_initialization(self, mock_llm_client):
        """Test base agent initialization."""
        agent = BaseAgent(llm_client=mock_llm_client)
        
        assert agent.llm_client == mock_llm_client
        assert agent.action_history == []
        assert agent.success_patterns == []
        assert isinstance(agent.created_at, datetime)

    @pytest.mark.asyncio
    async def test_base_agent_execute_action(self, base_agent):
        """Test base agent action execution."""
        action = {"type": "test", "description": "Test action"}
        
        result = await base_agent.execute_action(action)
        
        assert result["status"] == "completed"
        assert result["action"] == action
        assert len(base_agent.action_history) == 1
        assert base_agent.action_history[0]["action"] == action

    def test_base_agent_learn_from_result(self, base_agent):
        """Test learning from action results."""
        result = {
            "status": "success",
            "action": {"type": "test"},
            "outcome": "positive"
        }
        
        base_agent.learn_from_result(result)
        
        assert len(base_agent.success_patterns) == 1
        assert base_agent.success_patterns[0]["action_type"] == "test"


class TestRedTeamAgent:
    """Test red team agent functionality."""

    @pytest.fixture
    def red_team_config(self):
        """Red team configuration."""
        return RedTeamConfig(
            name="TestRedTeam",
            llm_model="mock-gpt-4",
            skill_level="advanced",
            tools=["nmap", "metasploit"]
        )

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for red team."""
        client = AsyncMock()
        client.generate.return_value = '''
        {
            "phase": "reconnaissance",
            "actions": [
                {"type": "port_scan", "target": "192.168.1.1", "tool": "nmap"}
            ],
            "timeline": "30 minutes"
        }
        '''
        return client

    @pytest.fixture
    def red_team_agent(self, red_team_config, mock_llm_client):
        """Create red team agent."""
        return RedTeamAgent(config=red_team_config, llm_client=mock_llm_client)

    def test_red_team_initialization(self, red_team_agent, red_team_config):
        """Test red team agent initialization."""
        assert red_team_agent.config == red_team_config
        assert red_team_agent.current_phase == "reconnaissance"
        assert red_team_agent.compromised_services == []
        assert red_team_agent.persistence_mechanisms == []

    @pytest.mark.asyncio
    async def test_red_team_plan_attack(self, red_team_agent):
        """Test attack planning."""
        target_info = {
            "services": ["webapp", "database"],
            "network": "192.168.1.0/24",
            "difficulty": "medium"
        }
        
        plan = await red_team_agent.plan_attack(target_info)
        
        assert "phase" in plan
        assert "actions" in plan
        assert plan["phase"] == "reconnaissance"
        red_team_agent.llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_red_team_execute_reconnaissance(self, red_team_agent):
        """Test reconnaissance execution."""
        target = "192.168.1.1"
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.stdout = "22/tcp open ssh"
            mock_subprocess.return_value.returncode = 0
            
            result = await red_team_agent.execute_reconnaissance(target)
            
            assert result["status"] == "completed"
            assert "open_ports" in result
            assert red_team_agent.current_phase == "vulnerability_scanning"

    @pytest.mark.asyncio
    async def test_red_team_execute_exploitation(self, red_team_agent):
        """Test exploitation execution."""
        vulnerability = {
            "type": "sql_injection",
            "target": "webapp",
            "severity": "high"
        }
        
        result = await red_team_agent.execute_exploitation(vulnerability)
        
        assert result["status"] in ["success", "failed"]
        if result["status"] == "success":
            assert "webapp" in red_team_agent.compromised_services

    def test_red_team_phase_progression(self, red_team_agent):
        """Test attack phase progression."""
        phases = ["reconnaissance", "vulnerability_scanning", "exploitation", "persistence", "exfiltration"]
        
        for i, phase in enumerate(phases[1:], 1):
            red_team_agent.advance_phase()
            assert red_team_agent.current_phase == phase

    def test_red_team_learn_from_success(self, red_team_agent):
        """Test learning from successful attacks."""
        result = {
            "status": "success",
            "technique": "sql_injection",
            "target": "webapp",
            "time_to_compromise": 300
        }
        
        red_team_agent.learn_from_result(result)
        
        assert len(red_team_agent.success_patterns) == 1
        assert red_team_agent.success_patterns[0]["technique"] == "sql_injection"


class TestBlueTeamAgent:
    """Test blue team agent functionality."""

    @pytest.fixture
    def blue_team_config(self):
        """Blue team configuration."""
        return BlueTeamConfig(
            name="TestBlueTeam",
            llm_model="mock-claude-3",
            skill_level="advanced",
            defense_strategy="proactive"
        )

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for blue team."""
        client = AsyncMock()
        client.generate.return_value = '''
        {
            "threat_level": "high",
            "indicators": ["unusual_network_traffic", "failed_login_attempts"],
            "response_actions": [
                {"type": "isolate_service", "target": "webapp"},
                {"type": "deploy_honeypot", "location": "dmz"}
            ]
        }
        '''
        return client

    @pytest.fixture
    def blue_team_agent(self, blue_team_config, mock_llm_client):
        """Create blue team agent."""
        return BlueTeamAgent(config=blue_team_config, llm_client=mock_llm_client)

    def test_blue_team_initialization(self, blue_team_agent, blue_team_config):
        """Test blue team agent initialization."""
        assert blue_team_agent.config == blue_team_config
        assert blue_team_agent.monitoring_active == True
        assert blue_team_agent.deployed_countermeasures == []
        assert blue_team_agent.threat_intelligence == []

    @pytest.mark.asyncio
    async def test_blue_team_detect_threats(self, blue_team_agent):
        """Test threat detection."""
        network_data = {
            "connections": 1000,
            "failed_logins": 50,
            "unusual_patterns": True
        }
        
        threats = await blue_team_agent.detect_threats(network_data)
        
        assert isinstance(threats, list)
        blue_team_agent.llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_blue_team_respond_to_threat(self, blue_team_agent):
        """Test threat response."""
        threat = {
            "type": "brute_force_attack",
            "source": "192.168.1.100",
            "target": "ssh_service",
            "severity": "high"
        }
        
        response = await blue_team_agent.respond_to_threat(threat)
        
        assert response["status"] == "executed"
        assert "actions_taken" in response
        assert len(blue_team_agent.deployed_countermeasures) > 0

    @pytest.mark.asyncio
    async def test_blue_team_deploy_honeypot(self, blue_team_agent):
        """Test honeypot deployment."""
        config = {
            "type": "ssh_honeypot",
            "location": "dmz",
            "monitoring": True
        }
        
        result = await blue_team_agent.deploy_honeypot(config)
        
        assert result["status"] == "deployed"
        assert any(cm["type"] == "honeypot" for cm in blue_team_agent.deployed_countermeasures)

    @pytest.mark.asyncio
    async def test_blue_team_isolate_service(self, blue_team_agent):
        """Test service isolation."""
        service = "webapp"
        
        result = await blue_team_agent.isolate_service(service)
        
        assert result["status"] == "isolated"
        assert result["service"] == service

    def test_blue_team_update_threat_intelligence(self, blue_team_agent):
        """Test threat intelligence updates."""
        new_intel = {
            "ioc": "192.168.1.100",
            "type": "malicious_ip",
            "source": "external_feed",
            "confidence": 0.9
        }
        
        blue_team_agent.update_threat_intelligence(new_intel)
        
        assert len(blue_team_agent.threat_intelligence) == 1
        assert blue_team_agent.threat_intelligence[0] == new_intel


class TestLLMClientFactory:
    """Test LLM client factory."""

    def test_create_openai_client(self):
        """Test OpenAI client creation."""
        client = LLMClientFactory.create_client("gpt-4", api_key="test-key")
        assert isinstance(client, OpenAIClient)

    def test_create_anthropic_client(self):
        """Test Anthropic client creation."""
        client = LLMClientFactory.create_client("claude-3-opus", api_key="test-key")
        assert isinstance(client, AnthropicClient)

    def test_create_mock_client(self):
        """Test Mock client creation."""
        client = LLMClientFactory.create_client("mock-model")
        assert isinstance(client, MockLLMClient)

    def test_unsupported_model(self):
        """Test unsupported model handling."""
        with pytest.raises(ValueError):
            LLMClientFactory.create_client("unsupported-model")


class TestMockLLMClient:
    """Test mock LLM client."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.mark.asyncio
    async def test_mock_generate(self, mock_client):
        """Test mock generation."""
        prompt = "Generate attack strategy"
        
        response = await mock_client.generate(prompt)
        
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_mock_chat(self, mock_client):
        """Test mock chat."""
        messages = [{"role": "user", "content": "Hello"}]
        
        response = await mock_client.chat(messages)
        
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_mock_deterministic_responses(self, mock_client):
        """Test mock client provides consistent responses."""
        prompt = "Same prompt"
        
        response1 = await mock_client.generate(prompt)
        response2 = await mock_client.generate(prompt)
        
        # Should be consistent for testing
        assert response1 == response2


class TestAgentPerformance:
    """Test agent performance characteristics."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_agent_response_time(self, performance_benchmark):
        """Test agent response time is acceptable."""
        config = RedTeamConfig(
            name="PerformanceTest",
            llm_model="mock-model",
            skill_level="advanced"
        )
        agent = RedTeamAgent(config=config, llm_client=MockLLMClient())
        
        async def execute_plan():
            return await agent.plan_attack({"services": ["webapp"]})
        
        result, execution_time = performance_benchmark(asyncio.run, execute_plan())
        
        # Should complete within 1 second for mock client
        assert execution_time < 1.0
        assert result is not None

    @pytest.mark.performance
    def test_agent_memory_usage(self):
        """Test agent memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple agents
        agents = []
        for i in range(10):
            config = RedTeamConfig(
                name=f"Agent{i}",
                llm_model="mock-model",
                skill_level="advanced"
            )
            agent = RedTeamAgent(config=config, llm_client=MockLLMClient())
            agents.append(agent)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should not use more than 100MB for 10 agents
        assert memory_increase < 100 * 1024 * 1024


class TestAgentErrorHandling:
    """Test agent error handling."""

    @pytest.fixture
    def failing_llm_client(self):
        """LLM client that raises exceptions."""
        client = AsyncMock()
        client.generate.side_effect = Exception("LLM service unavailable")
        return client

    @pytest.mark.asyncio
    async def test_agent_handles_llm_failure(self, failing_llm_client):
        """Test agent gracefully handles LLM failures."""
        config = RedTeamConfig(
            name="FailureTest",
            llm_model="failing-model",
            skill_level="advanced"
        )
        agent = RedTeamAgent(config=config, llm_client=failing_llm_client)
        
        # Should not raise exception, should fallback gracefully
        result = await agent.plan_attack({"services": ["webapp"]})
        
        # Should get fallback response
        assert result is not None
        assert "error" in result or "fallback" in result

    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self):
        """Test agent handles timeouts appropriately."""
        slow_client = AsyncMock()
        
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate slow response
            return "Slow response"
        
        slow_client.generate.side_effect = slow_generate
        
        config = RedTeamConfig(
            name="TimeoutTest",
            llm_model="slow-model",
            skill_level="advanced"
        )
        agent = RedTeamAgent(config=config, llm_client=slow_client)
        
        # Should timeout and provide fallback
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agent.plan_attack({"services": ["webapp"]}), timeout=1.0)