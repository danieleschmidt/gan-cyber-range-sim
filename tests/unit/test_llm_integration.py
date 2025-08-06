"""Tests for LLM integration layer."""

import pytest
import json
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from gan_cyber_range.agents.llm_client import (
    LLMRequest, LLMResponse, OpenAIClient, AnthropicClient, 
    MockLLMClient, LLMClientFactory, AgentLLMIntegration
)


@pytest.fixture
def sample_llm_request():
    """Sample LLM request for testing."""
    return LLMRequest(
        prompt="Analyze the following security environment...",
        context={"agent_type": "red_team", "skill_level": "advanced"},
        max_tokens=1000,
        temperature=0.7
    )


@pytest.fixture
def sample_environment_state():
    """Sample environment state for testing."""
    return {
        "services": [
            {
                "name": "webapp",
                "type": "web_application",
                "ip": "10.0.1.10",
                "ports": [80, 443],
                "status": "running",
                "vulnerabilities": [
                    {"cve_id": "CVE-2023-12345", "severity": "high", "exploitable": True}
                ]
            }
        ],
        "security_events": [
            {"type": "sql_injection_attempt", "severity": "high", "timestamp": "2024-01-01T12:00:00Z"}
        ],
        "network_logs": []
    }


class TestLLMRequest:
    """Test LLM request model."""
    
    def test_llm_request_creation(self, sample_llm_request):
        """Test LLM request creation."""
        assert sample_llm_request.prompt == "Analyze the following security environment..."
        assert sample_llm_request.context["agent_type"] == "red_team"
        assert sample_llm_request.max_tokens == 1000
        assert sample_llm_request.temperature == 0.7


class TestLLMResponse:
    """Test LLM response model."""
    
    def test_llm_response_creation(self):
        """Test LLM response creation."""
        response = LLMResponse(
            content='{"analysis": "test"}',
            tokens_used=150,
            latency_ms=200.5,
            model="test-model",
            timestamp=datetime.now()
        )
        
        assert response.content == '{"analysis": "test"}'
        assert response.tokens_used == 150
        assert response.latency_ms == 200.5


class TestMockLLMClient:
    """Test mock LLM client."""
    
    @pytest.mark.asyncio
    async def test_mock_client_red_team_response(self, sample_llm_request):
        """Test mock client red team response."""
        sample_llm_request.context["agent_type"] = "red_team"
        
        client = MockLLMClient()
        response = await client.generate(sample_llm_request)
        
        assert response.model == "mock-model"
        assert response.tokens_used > 0
        assert response.parsed_json is not None
        assert "attack_opportunities" in response.parsed_json
        assert "recommended_actions" in response.parsed_json
    
    @pytest.mark.asyncio
    async def test_mock_client_blue_team_response(self, sample_llm_request):
        """Test mock client blue team response."""
        sample_llm_request.context["agent_type"] = "blue_team"
        
        client = MockLLMClient()
        response = await client.generate(sample_llm_request)
        
        assert response.parsed_json is not None
        assert "detected_threats" in response.parsed_json
        assert "defensive_actions" in response.parsed_json


class TestOpenAIClient:
    """Test OpenAI client."""
    
    @pytest.mark.asyncio
    async def test_openai_client_success(self, sample_llm_request):
        """Test successful OpenAI API call."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"analysis": "successful"}'
        mock_response.usage.total_tokens = 200
        
        client = OpenAIClient("gpt-4", "test-key")
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            response = await client.generate(sample_llm_request)
            
            assert response.content == '{"analysis": "successful"}'
            assert response.tokens_used == 200
            assert response.model == "gpt-4"
            assert response.parsed_json == {"analysis": "successful"}
    
    @pytest.mark.asyncio
    async def test_openai_client_error_handling(self, sample_llm_request):
        """Test OpenAI client error handling."""
        client = OpenAIClient("gpt-4", "test-key")
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            response = await client.generate(sample_llm_request)
            
            assert "Error:" in response.content
            assert response.tokens_used == 0
    
    def test_openai_system_prompt_red_team(self):
        """Test system prompt generation for red team."""
        client = OpenAIClient("gpt-4", "test-api-key")
        context = {"agent_type": "red_team", "skill_level": "advanced"}
        
        prompt = client._build_system_prompt(context)
        
        assert "red team agent" in prompt.lower()
        assert "attack" in prompt.lower()
        assert "exploit" in prompt.lower()
    
    def test_openai_system_prompt_blue_team(self):
        """Test system prompt generation for blue team."""
        client = OpenAIClient("gpt-4", "test-api-key")
        context = {"agent_type": "blue_team", "skill_level": "advanced"}
        
        prompt = client._build_system_prompt(context)
        
        assert "blue team agent" in prompt.lower()
        assert "defend" in prompt.lower()
        assert "threat" in prompt.lower()


class TestAnthropicClient:
    """Test Anthropic client."""
    
    @pytest.mark.asyncio
    async def test_anthropic_client_success(self, sample_llm_request):
        """Test successful Anthropic API call."""
        # Skip if anthropic not available
        try:
            import anthropic
        except ImportError:
            pytest.skip("Anthropic package not available")
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '{"analysis": "successful"}'
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 100
        
        client = AnthropicClient("claude-3-sonnet-20240229", "test-key")
        
        with patch.object(client.client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            response = await client.generate(sample_llm_request)
            
            assert response.content == '{"analysis": "successful"}'
            assert response.tokens_used == 200
            assert response.parsed_json == {"analysis": "successful"}
    
    @pytest.mark.asyncio
    async def test_anthropic_client_import_error(self):
        """Test Anthropic client with missing package."""
        with patch.dict('sys.modules', {'anthropic': None}):
            with pytest.raises(ImportError):
                AnthropicClient()


class TestLLMClientFactory:
    """Test LLM client factory."""
    
    def test_factory_creates_openai_client(self):
        """Test factory creates OpenAI client for GPT models."""
        client = LLMClientFactory.create_client("gpt-4")
        assert isinstance(client, OpenAIClient)
        
        client = LLMClientFactory.create_client("openai-test")
        assert isinstance(client, OpenAIClient)
    
    def test_factory_creates_anthropic_client(self):
        """Test factory creates Anthropic client for Claude models."""
        client = LLMClientFactory.create_client("claude-3-sonnet-20240229")
        assert isinstance(client, AnthropicClient)
        
        client = LLMClientFactory.create_client("anthropic-test")
        assert isinstance(client, AnthropicClient)
    
    def test_factory_creates_mock_client_for_unknown(self):
        """Test factory creates mock client for unknown models."""
        client = LLMClientFactory.create_client("unknown-model")
        assert isinstance(client, MockLLMClient)
        
        client = LLMClientFactory.create_client("mock-test")
        assert isinstance(client, MockLLMClient)


class TestAgentLLMIntegration:
    """Test agent LLM integration."""
    
    @pytest.fixture
    def llm_integration(self):
        """Create LLM integration instance."""
        return AgentLLMIntegration("mock-model")
    
    @pytest.mark.asyncio
    async def test_analyze_environment_red_team(self, llm_integration, sample_environment_state):
        """Test environment analysis for red team."""
        agent_context = {"skill_level": "advanced", "tools": ["nmap"]}
        
        result = await llm_integration.analyze_environment(
            "red_team", sample_environment_state, agent_context
        )
        
        assert isinstance(result, dict)
        assert "attack_opportunities" in result
        assert "recommended_actions" in result
        assert "situation_analysis" in result
    
    @pytest.mark.asyncio
    async def test_analyze_environment_blue_team(self, llm_integration, sample_environment_state):
        """Test environment analysis for blue team."""
        agent_context = {"skill_level": "advanced", "defense_strategy": "proactive"}
        
        result = await llm_integration.analyze_environment(
            "blue_team", sample_environment_state, agent_context
        )
        
        assert isinstance(result, dict)
        assert "detected_threats" in result
        assert "defensive_actions" in result
        assert "threat_assessment" in result
    
    @pytest.mark.asyncio
    async def test_plan_actions_red_team(self, llm_integration):
        """Test action planning for red team."""
        analysis = {
            "situation_analysis": "Test analysis",
            "attack_opportunities": [{"target": "webapp", "vulnerability": "sql_injection"}]
        }
        agent_context = {"skill_level": "advanced"}
        
        result = await llm_integration.plan_actions("red_team", analysis, agent_context)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all("action" in action for action in result)
    
    @pytest.mark.asyncio
    async def test_plan_actions_blue_team(self, llm_integration):
        """Test action planning for blue team."""
        analysis = {
            "threat_assessment": "Test analysis",
            "detected_threats": [{"threat": "sql_injection", "severity": "high"}]
        }
        agent_context = {"defense_strategy": "proactive"}
        
        result = await llm_integration.plan_actions("blue_team", analysis, agent_context)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all("action" in action for action in result)
    
    def test_build_analysis_prompt(self, llm_integration, sample_environment_state):
        """Test analysis prompt building."""
        prompt = llm_integration._build_analysis_prompt("red_team", sample_environment_state)
        
        assert "SERVICES:" in prompt
        assert "SECURITY EVENTS:" in prompt
        assert "webapp" in prompt
        assert "sql_injection_attempt" in prompt
    
    def test_build_planning_prompt(self, llm_integration):
        """Test planning prompt building."""
        analysis = {"test": "data"}
        prompt = llm_integration._build_planning_prompt("red_team", analysis)
        
        assert "ANALYSIS RESULTS:" in prompt
        assert "tactical action plan" in prompt
        assert '"test": "data"' in prompt
    
    def test_fallback_analysis_red_team(self, llm_integration, sample_environment_state):
        """Test fallback analysis for red team."""
        result = llm_integration._fallback_analysis("red_team", sample_environment_state)
        
        assert "situation_analysis" in result
        assert "attack_opportunities" in result
        assert "recommended_actions" in result
    
    def test_fallback_analysis_blue_team(self, llm_integration, sample_environment_state):
        """Test fallback analysis for blue team."""
        result = llm_integration._fallback_analysis("blue_team", sample_environment_state)
        
        assert "threat_assessment" in result
        assert "detected_threats" in result
        assert "defensive_actions" in result
    
    def test_fallback_actions_red_team(self, llm_integration):
        """Test fallback action generation for red team."""
        analysis = {"test": "data"}
        result = llm_integration._fallback_actions("red_team", analysis)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["action"] == "reconnaissance"
    
    def test_fallback_actions_blue_team(self, llm_integration):
        """Test fallback action generation for blue team."""
        analysis = {"test": "data"}
        result = llm_integration._fallback_actions("blue_team", analysis)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["action"] == "monitoring"


class TestLLMClientRateLimit:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting works correctly."""
        import time
        
        client = MockLLMClient()
        client._rate_limit_delay = 0.1  # 100ms delay
        
        start_time = time.time()
        
        # Make two requests
        await client._rate_limit()
        await client._rate_limit()
        
        elapsed = time.time() - start_time
        
        # Should take at least the rate limit delay
        assert elapsed >= 0.1


class TestJSONParsing:
    """Test JSON parsing functionality."""
    
    def test_parse_clean_json(self):
        """Test parsing clean JSON response."""
        client = MockLLMClient()
        content = '{"test": "value"}'
        
        result = client._parse_json_response(content)
        
        assert result == {"test": "value"}
    
    def test_parse_json_with_markdown(self):
        """Test parsing JSON with markdown formatting."""
        client = MockLLMClient()
        content = '```json\n{"test": "value"}\n```'
        
        result = client._parse_json_response(content)
        
        assert result == {"test": "value"}
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        client = MockLLMClient()
        content = '{"test": invalid}'
        
        result = client._parse_json_response(content)
        
        assert result is None