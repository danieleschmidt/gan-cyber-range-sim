"""Red team (attacker) agent implementation."""

import asyncio
import random
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import BaseAgent, AgentAction
from .llm_client import AgentLLMIntegration


class RedTeamAgent(BaseAgent):
    """Red team agent that performs offensive security actions."""
    
    def __init__(
        self,
        name: str = "RedTeam",
        llm_model: str = "gpt-4",
        skill_level: str = "advanced",
        tools: List[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, llm_model, skill_level, **kwargs)
        self.tools = tools or ["nmap", "metasploit", "custom_exploits"]
        self.llm_integration = AgentLLMIntegration(llm_model, api_key)
        self.attack_patterns = [
            "reconnaissance",
            "vulnerability_scan",
            "exploit_attempt",
            "privilege_escalation",
            "persistence",
            "data_exfiltration",
            "lateral_movement"
        ]
    
    async def analyze_environment(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environment for attack opportunities using LLM intelligence."""
        try:
            # Get LLM-powered analysis
            agent_context = {
                "skill_level": self.skill_level,
                "tools": self.tools,
                "round": self._round_counter,
                "previous_successes": len(self.memory.successes),
                "attack_phase": self._determine_attack_phase()
            }
            
            llm_analysis = await self.llm_integration.analyze_environment(
                "red_team", environment_state, agent_context
            )
            
            # Combine LLM analysis with traditional heuristics
            enhanced_analysis = self._enhance_analysis_with_heuristics(
                llm_analysis, environment_state
            )
            
            return enhanced_analysis
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed, falling back to heuristics: {e}")
            return self._fallback_analysis(environment_state)
    
    async def plan_actions(self, analysis: Dict[str, Any]) -> List[AgentAction]:
        """Plan attack actions using LLM intelligence."""
        try:
            # Get LLM-powered action plan
            agent_context = {
                "skill_level": self.skill_level,
                "tools": self.tools,
                "round": self._round_counter,
                "max_actions": self.max_actions_per_round,
                "previous_actions": [a.type for a in self.memory.actions[-5:]]
            }
            
            llm_actions = await self.llm_integration.plan_actions(
                "red_team", analysis, agent_context
            )
            
            # Convert LLM actions to AgentAction objects
            actions = self._convert_llm_actions_to_agent_actions(llm_actions, analysis)
            
            # Ensure we don't exceed max actions per round
            return actions[:self.max_actions_per_round]
            
        except Exception as e:
            self.logger.warning(f"LLM action planning failed, falling back to heuristics: {e}")
            return self._fallback_action_planning(analysis)
    
    async def execute_action(self, action: AgentAction) -> AgentAction:
        """Execute an attack action."""
        self.logger.info(f"Executing {action.type} against {action.target}")
        
        # Simulate action execution with realistic delays
        execution_time = random.uniform(1, 5)  # 1-5 seconds
        await asyncio.sleep(execution_time)
        
        # Determine success based on action type and skill level
        success_probability = self._calculate_success_probability(action)
        action.success = random.random() < success_probability
        
        # Add execution metadata
        action.metadata.update({
            "execution_time": execution_time,
            "success_probability": success_probability,
            "timestamp": datetime.now().isoformat()
        })
        
        if action.success:
            self.logger.info(f"Successfully executed {action.type} against {action.target}")
            # Update payload with results
            action.payload.update(self._generate_success_results(action))
        else:
            self.logger.warning(f"Failed to execute {action.type} against {action.target}")
            action.payload["failure_reason"] = self._generate_failure_reason(action)
        
        return action
    
    def _assess_vulnerability(self, service: Dict[str, Any]) -> float:
        """Assess vulnerability score of a service."""
        base_score = 0.3  # Base vulnerability
        
        # Factor in known vulnerabilities
        if "vulnerabilities" in service:
            base_score += len(service["vulnerabilities"]) * 0.2
        
        # Factor in service type
        high_risk_services = ["webapp", "database", "api"]
        if service.get("type") in high_risk_services:
            base_score += 0.3
        
        # Factor in patches
        if service.get("last_patched"):
            # Older patches increase vulnerability
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _calculate_priority(self, service: Dict[str, Any], vulnerability_score: float) -> float:
        """Calculate attack priority for a service."""
        priority = vulnerability_score * 0.7
        
        # High-value targets get priority boost
        if service.get("business_critical", False):
            priority += 0.3
        
        # Consider defensive measures
        if service.get("security_hardened", False):
            priority -= 0.2
        
        return max(0, min(priority, 1.0))
    
    def _determine_attack_phase(self) -> str:
        """Determine current attack phase based on memory."""
        if not self.memory.successes:
            return "reconnaissance"
        
        # Progress through phases based on successful actions
        successful_types = {a.type for a in self.memory.successes}
        
        if "data_exfiltration" in successful_types:
            return "persistence"
        elif "exploit_attempt" in successful_types:
            return "privilege_escalation"
        elif "vulnerability_scan" in successful_types:
            return "exploit_attempt"
        elif "reconnaissance" in successful_types:
            return "vulnerability_scan"
        else:
            return "reconnaissance"
    
    def _select_action_for_phase(self, phase: str, target: Dict[str, Any]) -> str:
        """Select appropriate action for current attack phase."""
        phase_actions = {
            "reconnaissance": ["port_scan", "service_enumeration", "banner_grabbing"],
            "vulnerability_scan": ["vulnerability_scan", "weak_credential_check"],
            "exploit_attempt": ["buffer_overflow", "sql_injection", "xss_attack"],
            "privilege_escalation": ["sudo_exploit", "kernel_exploit", "service_exploit"],
            "persistence": ["backdoor_install", "user_creation", "service_persistence"],
            "data_exfiltration": ["database_dump", "file_exfiltration", "network_sniffing"],
            "lateral_movement": ["network_pivot", "credential_reuse", "service_hop"]
        }
        
        available_actions = phase_actions.get(phase, ["reconnaissance"])
        return random.choice(available_actions)
    
    def _calculate_success_probability(self, action: AgentAction) -> float:
        """Calculate probability of action success."""
        base_probability = {"beginner": 0.3, "intermediate": 0.5, "advanced": 0.7}.get(
            self.skill_level, 0.5
        )
        
        # Adjust based on vulnerability score
        vuln_score = action.payload.get("vulnerability_score", 0.5)
        probability = base_probability + (vuln_score * 0.3)
        
        # Learning bonus - agents get better over time
        if self.memory.successes:
            learning_bonus = min(len(self.memory.successes) * 0.05, 0.2)
            probability += learning_bonus
        
        return min(probability, 0.9)  # Cap at 90% success rate
    
    def _generate_success_results(self, action: AgentAction) -> Dict[str, Any]:
        """Generate realistic results for successful actions."""
        results = {
            "status": "success",
            "data_collected": [],
            "access_gained": []
        }
        
        if action.type in ["port_scan", "service_enumeration"]:
            results["data_collected"] = ["open_ports", "service_versions", "os_fingerprint"]
        elif action.type == "vulnerability_scan":
            results["data_collected"] = ["cve_list", "exploit_paths", "risk_assessment"]
        elif action.type in ["sql_injection", "xss_attack"]:
            results["access_gained"] = ["database_access", "session_hijack"]
            results["data_collected"] = ["user_credentials", "sensitive_data"]
        
        return results
    
    def _generate_failure_reason(self, action: AgentAction) -> str:
        """Generate realistic failure reasons."""
        reasons = [
            "Target service is patched",
            "Firewall blocked connection",
            "Intrusion detection system triggered",
            "Authentication required",
            "Service is offline",
            "Rate limiting in effect",
            "Honeypot detected"
        ]
        return random.choice(reasons)
    
    def _enhance_analysis_with_heuristics(self, llm_analysis: Dict[str, Any], environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance LLM analysis with traditional heuristics."""
        services = environment_state.get("services", [])
        
        # Add traditional vulnerability scoring
        enhanced_opportunities = []
        for opportunity in llm_analysis.get("attack_opportunities", []):
            target_name = opportunity.get("target")
            
            # Find the service and add heuristic data
            for service in services:
                if service.get("name") == target_name:
                    vuln_score = self._assess_vulnerability(service)
                    priority = self._calculate_priority(service, vuln_score)
                    
                    opportunity.update({
                        "heuristic_vulnerability_score": vuln_score,
                        "heuristic_priority": priority,
                        "service_data": service
                    })
                    break
            
            enhanced_opportunities.append(opportunity)
        
        # Update the analysis
        llm_analysis["attack_opportunities"] = enhanced_opportunities
        llm_analysis["traditional_analysis"] = self._fallback_analysis(environment_state)
        
        return llm_analysis
    
    def _fallback_analysis(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis using traditional heuristics."""
        services = environment_state.get("services", [])
        network_topology = environment_state.get("network_topology", {})
        security_status = environment_state.get("security_status", {})
        
        # Identify potential targets
        targets = []
        for service in services:
            if service.get("status") == "running":
                vulnerability_score = self._assess_vulnerability(service)
                targets.append({
                    "service": service.get("name"),
                    "ip": service.get("ip"),
                    "ports": service.get("open_ports", []),
                    "vulnerability_score": vulnerability_score,
                    "priority": self._calculate_priority(service, vulnerability_score)
                })
        
        # Sort targets by priority
        targets.sort(key=lambda x: x["priority"], reverse=True)
        
        return {
            "targets": targets,
            "network_map": network_topology,
            "defensive_measures": security_status.get("active_defenses", []),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _convert_llm_actions_to_agent_actions(self, llm_actions: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[AgentAction]:
        """Convert LLM action recommendations to AgentAction objects."""
        actions = []
        
        for llm_action in llm_actions:
            action_type = llm_action.get("action", "reconnaissance")
            target = llm_action.get("target", "unknown")
            reasoning = llm_action.get("reasoning", "LLM recommended action")
            
            # Find target details from analysis
            target_data = self._find_target_data(target, analysis)
            
            action = AgentAction(
                type=action_type,
                target=target,
                payload={
                    "llm_reasoning": reasoning,
                    "target_data": target_data,
                    "attack_phase": self._determine_attack_phase(),
                    "tools_available": self.tools,
                    "confidence": llm_action.get("confidence", 0.5)
                },
                metadata={
                    "agent_name": self.name,
                    "skill_level": self.skill_level,
                    "llm_generated": True,
                    "llm_action_data": llm_action
                }
            )
            actions.append(action)
        
        return actions
    
    def _find_target_data(self, target_name: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Find target data from analysis."""
        # Look in attack opportunities
        for opportunity in analysis.get("attack_opportunities", []):
            if opportunity.get("target") == target_name:
                return opportunity
        
        # Look in traditional targets
        for target in analysis.get("targets", []):
            if target.get("service") == target_name:
                return target
        
        return {"target": target_name, "data_available": False}
    
    def _fallback_action_planning(self, analysis: Dict[str, Any]) -> List[AgentAction]:
        """Fallback action planning using traditional heuristics."""
        targets = analysis.get("targets", [])
        if not targets:
            return []
        
        actions = []
        current_phase = self._determine_attack_phase()
        
        # Select top targets based on skill level
        max_targets = {"beginner": 1, "intermediate": 2, "advanced": 3}.get(self.skill_level, 2)
        selected_targets = targets[:max_targets]
        
        for target in selected_targets:
            action_type = self._select_action_for_phase(current_phase, target)
            
            action = AgentAction(
                type=action_type,
                target=target["service"],
                payload={
                    "target_ip": target["ip"],
                    "target_ports": target["ports"],
                    "vulnerability_score": target["vulnerability_score"],
                    "attack_phase": current_phase,
                    "tools_available": self.tools
                },
                metadata={
                    "agent_name": self.name,
                    "skill_level": self.skill_level,
                    "priority": target["priority"],
                    "fallback_planning": True
                }
            )
            actions.append(action)
        
        return actions