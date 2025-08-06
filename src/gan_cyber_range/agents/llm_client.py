"""LLM client for agent intelligence integration."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

import openai
try:
    import anthropic
except ImportError:
    anthropic = None

from ..resilience.circuit_breaker import with_circuit_breaker, CircuitBreakerConfig
from ..resilience.retry import with_retry, RetryPolicy, LLM_RETRY_POLICY


@dataclass
class LLMRequest:
    """Represents a request to an LLM."""
    prompt: str
    context: Dict[str, Any]
    max_tokens: int = 2000
    temperature: float = 0.7
    response_format: str = "json"


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""
    content: str
    tokens_used: int
    latency_ms: float
    model: str
    timestamp: datetime
    parsed_json: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._rate_limit_delay = 1.0  # seconds between requests
        self._last_request_time = 0.0
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM."""
        pass
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            # Clean up common LLM response artifacts
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON response: {content[:200]}...")
            return None


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client for agent intelligence."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(model, api_key)
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    @with_circuit_breaker(
        "openai_api", 
        CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
    )
    @with_retry(LLM_RETRY_POLICY)
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API with resilience patterns."""
        await self._rate_limit()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": self._build_system_prompt(request.context)
                },
                {
                    "role": "user", 
                    "content": request.prompt
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            llm_response = LLMResponse(
                content=content,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model=self.model,
                timestamp=datetime.now()
            )
            
            if request.response_format == "json":
                llm_response.parsed_json = self._parse_json_response(content)
            
            return llm_response
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise  # Let circuit breaker and retry handle the error
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with context."""
        agent_type = context.get("agent_type", "security_agent")
        
        base_prompt = f"""You are an expert {agent_type} in a cybersecurity simulation environment. 
        
Your role is to analyze security situations and provide actionable intelligence in JSON format.

Current context:
- Agent Type: {agent_type}
- Skill Level: {context.get('skill_level', 'intermediate')}
- Available Tools: {context.get('tools', [])}
- Current Round: {context.get('round', 1)}

Always respond with valid JSON containing your analysis and recommended actions.
Focus on realistic, practical cybersecurity approaches.
Consider the current threat landscape and defensive capabilities.
"""
        
        if agent_type == "red_team":
            base_prompt += """
As a red team agent, your goal is to:
1. Identify vulnerabilities and attack vectors
2. Plan sophisticated but realistic attacks
3. Adapt based on defensive measures
4. Progress through attack phases methodically
5. Learn from previous successes and failures

Response format:
{
    "analysis": "your analysis of the current situation",
    "targets": [{"name": "target", "priority": 0.8, "approach": "method"}],
    "actions": [{"type": "action_type", "target": "target_name", "reasoning": "why"}],
    "confidence": 0.7
}
"""
        else:  # blue_team
            base_prompt += """
As a blue team agent, your goal is to:
1. Detect and analyze security threats
2. Respond rapidly to incidents
3. Deploy proactive defensive measures
4. Learn from attack patterns
5. Protect critical assets effectively

Response format:
{
    "analysis": "your analysis of the threat landscape",
    "threats": [{"id": "threat_id", "severity": "high", "confidence": 0.8}],
    "actions": [{"type": "defensive_action", "target": "system", "urgency": 0.9}],
    "recommendations": ["immediate actions to take"]
}
"""
        
        return base_prompt


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client for agent intelligence."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        super().__init__(model, api_key)
        if not anthropic:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API."""
        await self._rate_limit()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            system_prompt = self._build_system_prompt(request.context)
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ]
            )
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            llm_response = LLMResponse(
                content=content,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model=self.model,
                timestamp=datetime.now()
            )
            
            if request.response_format == "json":
                llm_response.parsed_json = self._parse_json_response(content)
            
            return llm_response
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                tokens_used=0,
                latency_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                model=self.model,
                timestamp=datetime.now()
            )
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with context for Claude."""
        agent_type = context.get("agent_type", "security_agent")
        
        base_prompt = f"""You are an expert cybersecurity {agent_type} operating in a realistic simulation environment.

Your capabilities:
- Deep cybersecurity knowledge across offensive and defensive domains
- Understanding of real-world attack techniques and defensive measures
- Ability to adapt strategies based on environmental changes
- Knowledge of compliance, threat intelligence, and incident response

Current context:
- Agent Type: {agent_type}
- Skill Level: {context.get('skill_level', 'intermediate')}
- Available Tools: {context.get('tools', [])}
- Current Round: {context.get('round', 1)}
- Simulation Phase: {context.get('phase', 'active')}

Respond with valid JSON containing your professional analysis and recommendations.
Base decisions on current cybersecurity best practices and realistic capabilities.
"""
        
        if agent_type == "red_team":
            base_prompt += """
As a red team security professional, focus on:
- Realistic attack vectors and exploitation techniques
- Proper attack chain progression (recon → exploit → persist → exfiltrate)
- Evasion techniques to avoid detection
- Adapting to blue team countermeasures
- Professional ethical hacking methodologies

JSON Response format:
{
    "situation_analysis": "current environment assessment",
    "attack_opportunities": [{"target": "system", "vulnerability": "type", "success_probability": 0.7}],
    "recommended_actions": [{"action": "technique", "target": "system", "reasoning": "tactical rationale"}],
    "risk_assessment": {"detection_risk": 0.3, "success_probability": 0.8},
    "next_phase": "recommended next attack phase"
}
"""
        else:  # blue_team
            base_prompt += """
As a blue team security professional, focus on:
- Threat detection and analysis techniques
- Incident response and containment strategies
- Proactive defense and threat hunting
- Security monitoring and alerting
- Risk-based security decision making

JSON Response format:
{
    "threat_assessment": "current threat landscape analysis",
    "detected_threats": [{"threat": "type", "severity": "level", "confidence": 0.8}],
    "defensive_actions": [{"action": "countermeasure", "target": "system", "priority": 0.9}],
    "security_posture": {"overall_risk": 0.4, "critical_systems_protected": true},
    "recommendations": ["strategic defensive improvements"]
}
"""
        
        return base_prompt


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing and fallback scenarios."""
    
    def __init__(self, model: str = "mock-model"):
        super().__init__(model)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate mock response for testing."""
        await asyncio.sleep(0.1)  # Simulate latency
        
        agent_type = request.context.get("agent_type", "security_agent")
        
        if agent_type == "red_team":
            mock_content = json.dumps({
                "situation_analysis": "Environment contains vulnerable services ready for exploitation",
                "attack_opportunities": [
                    {"target": "webapp", "vulnerability": "sql_injection", "success_probability": 0.7},
                    {"target": "database", "vulnerability": "weak_credentials", "success_probability": 0.6}
                ],
                "recommended_actions": [
                    {"action": "vulnerability_scan", "target": "webapp", "reasoning": "Identify specific SQL injection points"},
                    {"action": "credential_brute_force", "target": "database", "reasoning": "Exploit weak authentication"}
                ],
                "risk_assessment": {"detection_risk": 0.3, "success_probability": 0.65},
                "next_phase": "exploitation"
            })
        else:  # blue_team
            mock_content = json.dumps({
                "threat_assessment": "Multiple active threats detected requiring immediate response",
                "detected_threats": [
                    {"threat": "sql_injection_attempt", "severity": "high", "confidence": 0.8},
                    {"threat": "brute_force_attack", "severity": "medium", "confidence": 0.6}
                ],
                "defensive_actions": [
                    {"action": "patch_deployment", "target": "webapp", "priority": 0.9},
                    {"action": "access_control_hardening", "target": "database", "priority": 0.7}
                ],
                "security_posture": {"overall_risk": 0.4, "critical_systems_protected": True},
                "recommendations": ["Deploy web application firewall", "Implement multi-factor authentication"]
            })
        
        response = LLMResponse(
            content=mock_content,
            tokens_used=150,
            latency_ms=100,
            model=self.model,
            timestamp=datetime.now()
        )
        
        if request.response_format == "json":
            response.parsed_json = self._parse_json_response(mock_content)
        
        return response


class LLMClientFactory:
    """Factory for creating appropriate LLM clients."""
    
    @staticmethod
    def create_client(model: str, api_key: Optional[str] = None) -> BaseLLMClient:
        """Create appropriate LLM client based on model name."""
        if model.startswith("gpt-") or model.startswith("openai"):
            return OpenAIClient(model, api_key)
        elif model.startswith("claude-") or model.startswith("anthropic"):
            return AnthropicClient(model, api_key)
        elif model.startswith("mock"):
            return MockLLMClient(model)
        else:
            # Default to mock for unknown models
            logging.warning(f"Unknown model {model}, using mock client")
            return MockLLMClient(model)


class AgentLLMIntegration:
    """Integration layer between agents and LLM clients."""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.client = LLMClientFactory.create_client(model, api_key)
        self.model = model
        self.logger = logging.getLogger(f"AgentLLMIntegration.{model}")
        
    async def analyze_environment(
        self, 
        agent_type: str, 
        environment_state: Dict[str, Any], 
        agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get LLM analysis of environment state."""
        
        context = {
            "agent_type": agent_type,
            **agent_context
        }
        
        prompt = self._build_analysis_prompt(agent_type, environment_state)
        
        request = LLMRequest(
            prompt=prompt,
            context=context,
            max_tokens=1500,
            temperature=0.7,
            response_format="json"
        )
        
        response = await self.client.generate(request)
        
        if response.parsed_json:
            return response.parsed_json
        else:
            # Fallback to basic analysis if JSON parsing fails
            self.logger.warning("LLM response parsing failed, using fallback analysis")
            return self._fallback_analysis(agent_type, environment_state)
    
    async def plan_actions(
        self, 
        agent_type: str,
        analysis: Dict[str, Any], 
        agent_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get LLM-generated action plan."""
        
        context = {
            "agent_type": agent_type,
            **agent_context
        }
        
        prompt = self._build_planning_prompt(agent_type, analysis)
        
        request = LLMRequest(
            prompt=prompt,
            context=context,
            max_tokens=1200,
            temperature=0.6,
            response_format="json"
        )
        
        response = await self.client.generate(request)
        
        if response.parsed_json and "actions" in response.parsed_json:
            return response.parsed_json["actions"]
        else:
            self.logger.warning("LLM action planning failed, using fallback actions")
            return self._fallback_actions(agent_type, analysis)
    
    def _build_analysis_prompt(self, agent_type: str, environment_state: Dict[str, Any]) -> str:
        """Build prompt for environment analysis."""
        services = environment_state.get("services", [])
        security_events = environment_state.get("security_events", [])
        network_logs = environment_state.get("network_logs", [])
        
        prompt = f"""Analyze the current cybersecurity environment:

SERVICES:
{json.dumps(services, indent=2)}

RECENT SECURITY EVENTS:
{json.dumps(security_events[-5:], indent=2)}

RECENT NETWORK ACTIVITY:
{json.dumps(network_logs[-10:], indent=2)}

Provide a comprehensive analysis considering:
1. Current threat landscape
2. Vulnerable attack surfaces
3. Active defensive measures
4. Risk assessment
5. Tactical opportunities

Respond with valid JSON containing your analysis and recommendations."""
        
        return prompt
    
    def _build_planning_prompt(self, agent_type: str, analysis: Dict[str, Any]) -> str:
        """Build prompt for action planning."""
        prompt = f"""Based on your analysis, create a tactical action plan:

ANALYSIS RESULTS:
{json.dumps(analysis, indent=2)}

Create a prioritized list of actions to take this round. Consider:
1. Immediate tactical objectives
2. Risk vs reward for each action
3. Resource constraints and capabilities
4. Likely success probability
5. Strategic positioning for next round

Respond with valid JSON containing specific, actionable steps."""
        
        return prompt
    
    def _fallback_analysis(self, agent_type: str, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback analysis if LLM fails."""
        services = environment_state.get("services", [])
        
        if agent_type == "red_team":
            return {
                "situation_analysis": "Fallback analysis: Services available for testing",
                "attack_opportunities": [
                    {"target": service.get("name", "unknown"), "vulnerability": "generic", "success_probability": 0.5}
                    for service in services[:3]
                ],
                "recommended_actions": [
                    {"action": "reconnaissance", "target": "network", "reasoning": "Gather intelligence"}
                ],
                "risk_assessment": {"detection_risk": 0.5, "success_probability": 0.5}
            }
        else:
            return {
                "threat_assessment": "Fallback analysis: Monitoring for threats",
                "detected_threats": [
                    {"threat": "unknown_activity", "severity": "medium", "confidence": 0.5}
                ],
                "defensive_actions": [
                    {"action": "monitoring", "target": "network", "priority": 0.7}
                ],
                "security_posture": {"overall_risk": 0.5, "critical_systems_protected": True}
            }
    
    def _fallback_actions(self, agent_type: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Provide fallback actions if LLM planning fails."""
        if agent_type == "red_team":
            return [
                {"action": "reconnaissance", "target": "network", "reasoning": "Fallback recon action"}
            ]
        else:
            return [
                {"action": "monitoring", "target": "infrastructure", "reasoning": "Fallback monitoring action"}
            ]