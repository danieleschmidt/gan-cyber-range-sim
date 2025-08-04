"""Red team (attacker) agent implementation."""

import asyncio
import random
from typing import Any, Dict, List
from datetime import datetime

from .base import BaseAgent, AgentAction


class RedTeamAgent(BaseAgent):
    """Red team agent that performs offensive security actions."""
    
    def __init__(
        self,
        name: str = "RedTeam",
        llm_model: str = "gpt-4",
        skill_level: str = "advanced",
        tools: List[str] = None,
        **kwargs
    ):
        super().__init__(name, llm_model, skill_level, **kwargs)
        self.tools = tools or ["nmap", "metasploit", "custom_exploits"]
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
        """Analyze environment for attack opportunities."""
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
    
    async def plan_actions(self, analysis: Dict[str, Any]) -> List[AgentAction]:
        """Plan attack actions based on analysis."""
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
                    "priority": target["priority"]
                }
            )
            actions.append(action)
        
        return actions
    
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