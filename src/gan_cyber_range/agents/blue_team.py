"""Blue team (defender) agent implementation."""

import asyncio
import random
from typing import Any, Dict, List
from datetime import datetime, timedelta

from .base import BaseAgent, AgentAction


class BlueTeamAgent(BaseAgent):
    """Blue team agent that performs defensive security actions."""
    
    def __init__(
        self,
        name: str = "BlueTeam",
        llm_model: str = "claude-3",
        skill_level: str = "advanced",
        defense_strategy: str = "proactive",
        tools: List[str] = None,
        **kwargs
    ):
        super().__init__(name, llm_model, skill_level, **kwargs)
        self.defense_strategy = defense_strategy
        self.tools = tools or ["ids", "auto_patcher", "honeypots", "firewall"]
        self.defensive_actions = [
            "threat_detection",
            "vulnerability_patching",
            "access_control",
            "network_monitoring",
            "incident_response",
            "system_hardening",
            "honeypot_deployment"
        ]
        self.threat_intelligence = []
        self.active_incidents = []
    
    async def analyze_environment(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environment for security threats and vulnerabilities."""
        services = environment_state.get("services", [])
        network_logs = environment_state.get("network_logs", [])
        security_events = environment_state.get("security_events", [])
        attack_indicators = environment_state.get("attack_indicators", [])
        
        # Analyze threats
        active_threats = self._detect_active_threats(security_events, attack_indicators)
        
        # Assess vulnerabilities
        vulnerabilities = self._assess_system_vulnerabilities(services)
        
        # Analyze attack patterns
        attack_patterns = self._analyze_attack_patterns(network_logs, security_events)
        
        # Prioritize defensive actions
        priorities = self._calculate_defense_priorities(active_threats, vulnerabilities)
        
        return {
            "active_threats": active_threats,
            "vulnerabilities": vulnerabilities,
            "attack_patterns": attack_patterns,
            "defense_priorities": priorities,
            "system_health": self._assess_system_health(services),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def plan_actions(self, analysis: Dict[str, Any]) -> List[AgentAction]:
        """Plan defensive actions based on threat analysis."""
        active_threats = analysis.get("active_threats", [])
        vulnerabilities = analysis.get("vulnerabilities", [])
        priorities = analysis.get("defense_priorities", [])
        
        actions = []
        
        # Respond to active threats first
        for threat in active_threats[:3]:  # Handle top 3 threats
            response_action = self._plan_threat_response(threat)
            if response_action:
                actions.append(response_action)
        
        # Address vulnerabilities based on strategy
        if self.defense_strategy == "proactive":
            vuln_actions = self._plan_vulnerability_mitigation(vulnerabilities)
            actions.extend(vuln_actions[:2])  # Top 2 vulnerabilities
        
        # Deploy preventive measures
        if len(actions) < self.max_actions_per_round:
            preventive_action = self._plan_preventive_action(analysis)
            if preventive_action:
                actions.append(preventive_action)
        
        return actions
    
    async def execute_action(self, action: AgentAction) -> AgentAction:
        """Execute a defensive action."""
        self.logger.info(f"Executing {action.type} for {action.target}")
        
        # Simulate action execution
        execution_time = random.uniform(0.5, 3.0)  # Defenders are typically faster
        await asyncio.sleep(execution_time)
        
        # Determine success based on action type and resources
        success_probability = self._calculate_defense_success_probability(action)
        action.success = random.random() < success_probability
        
        # Add execution metadata
        action.metadata.update({
            "execution_time": execution_time,
            "success_probability": success_probability,
            "timestamp": datetime.now().isoformat(),
            "defense_strategy": self.defense_strategy
        })
        
        if action.success:
            self.logger.info(f"Successfully executed {action.type} for {action.target}")
            action.payload.update(self._generate_defense_results(action))
            
            # Update threat intelligence
            self._update_threat_intelligence(action)
        else:
            self.logger.warning(f"Failed to execute {action.type} for {action.target}")
            action.payload["failure_reason"] = self._generate_defense_failure_reason(action)
        
        return action
    
    def _detect_active_threats(self, security_events: List[Dict], indicators: List[Dict]) -> List[Dict]:
        """Detect active threats from security events and indicators."""
        threats = []
        
        # Analyze security events for threat patterns
        for event in security_events[-10:]:  # Recent events
            if event.get("severity", "low") in ["high", "critical"]:
                threat = {
                    "id": f"threat_{len(threats) + 1}",
                    "type": event.get("type", "unknown"),
                    "severity": event.get("severity", "medium"),
                    "source": event.get("source_ip", "unknown"),
                    "target": event.get("target", "unknown"),
                    "confidence": self._calculate_threat_confidence(event),
                    "first_seen": event.get("timestamp", datetime.now().isoformat())
                }
                threats.append(threat)
        
        # Correlate indicators of compromise
        for indicator in indicators:
            if indicator.get("confidence", 0) > 0.7:
                threat = {
                    "id": f"ioc_{len(threats) + 1}",
                    "type": "indicator_match",
                    "severity": "high",
                    "indicator": indicator.get("value"),
                    "ioc_type": indicator.get("type"),
                    "confidence": indicator.get("confidence"),
                    "first_seen": datetime.now().isoformat()
                }
                threats.append(threat)
        
        return threats
    
    def _assess_system_vulnerabilities(self, services: List[Dict]) -> List[Dict]:
        """Assess vulnerabilities in system services."""
        vulnerabilities = []
        
        for service in services:
            service_vulns = service.get("vulnerabilities", [])
            for vuln in service_vulns:
                vulnerability = {
                    "id": vuln.get("cve_id", f"vuln_{len(vulnerabilities) + 1}"),
                    "service": service.get("name"),
                    "severity": vuln.get("severity", "medium"),
                    "cvss_score": vuln.get("cvss_score", 5.0),
                    "exploitable": vuln.get("exploitable", False),
                    "patch_available": vuln.get("patch_available", True),
                    "discovered": vuln.get("discovered", datetime.now().isoformat())
                }
                vulnerabilities.append(vulnerability)
        
        # Sort by severity and exploitability
        vulnerabilities.sort(
            key=lambda x: (x["cvss_score"], x["exploitable"]), 
            reverse=True
        )
        
        return vulnerabilities
    
    def _analyze_attack_patterns(self, network_logs: List[Dict], security_events: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in attacks for better defense."""
        patterns = {
            "common_sources": {},
            "attack_types": {},
            "time_patterns": {},
            "target_patterns": {}
        }
        
        # Analyze source IPs
        for log in network_logs[-100:]:  # Recent logs
            source = log.get("source_ip", "unknown")
            patterns["common_sources"][source] = patterns["common_sources"].get(source, 0) + 1
        
        # Analyze attack types
        for event in security_events[-50:]:
            attack_type = event.get("type", "unknown")
            patterns["attack_types"][attack_type] = patterns["attack_types"].get(attack_type, 0) + 1
        
        return patterns
    
    def _calculate_defense_priorities(self, threats: List[Dict], vulnerabilities: List[Dict]) -> List[Dict]:
        """Calculate defense action priorities."""
        priorities = []
        
        # Prioritize active threats
        for threat in threats:
            priority = {
                "type": "threat_response",
                "target": threat["id"],
                "urgency": self._calculate_urgency(threat),
                "impact": self._calculate_impact(threat)
            }
            priorities.append(priority)
        
        # Prioritize critical vulnerabilities
        for vuln in vulnerabilities[:5]:  # Top 5 vulns
            if vuln["cvss_score"] >= 7.0:  # High/Critical
                priority = {
                    "type": "vulnerability_patch",
                    "target": vuln["id"],
                    "urgency": vuln["cvss_score"] / 10.0,
                    "impact": 0.8 if vuln["exploitable"] else 0.5
                }
                priorities.append(priority)
        
        # Sort by combined urgency and impact
        priorities.sort(key=lambda x: x["urgency"] * x["impact"], reverse=True)
        
        return priorities
    
    def _plan_threat_response(self, threat: Dict[str, Any]) -> AgentAction:
        """Plan response to an active threat."""
        response_type = self._select_response_type(threat)
        
        return AgentAction(
            type=response_type,
            target=threat.get("target", threat["id"]),
            payload={
                "threat_id": threat["id"],
                "threat_type": threat["type"],
                "severity": threat["severity"],
                "source": threat.get("source", "unknown"),
                "confidence": threat.get("confidence", 0.5),
                "response_strategy": self.defense_strategy
            },
            metadata={
                "agent_name": self.name,
                "threat_data": threat
            }
        )
    
    def _plan_vulnerability_mitigation(self, vulnerabilities: List[Dict]) -> List[AgentAction]:
        """Plan actions to mitigate vulnerabilities."""
        actions = []
        
        for vuln in vulnerabilities[:3]:  # Top 3 vulnerabilities
            if vuln.get("patch_available", True):
                action = AgentAction(
                    type="patch_deployment",
                    target=vuln["service"],
                    payload={
                        "vulnerability_id": vuln["id"],
                        "cvss_score": vuln["cvss_score"],
                        "severity": vuln["severity"],
                        "patch_available": vuln["patch_available"]
                    },
                    metadata={
                        "agent_name": self.name,
                        "vulnerability_data": vuln
                    }
                )
                actions.append(action)
        
        return actions
    
    def _plan_preventive_action(self, analysis: Dict[str, Any]) -> AgentAction:
        """Plan preventive security action."""
        preventive_actions = [
            "honeypot_deployment",
            "security_monitoring",
            "access_control_review",
            "network_segmentation",
            "threat_hunting"
        ]
        
        action_type = random.choice(preventive_actions)
        
        return AgentAction(
            type=action_type,
            target="infrastructure",
            payload={
                "action_scope": "system_wide",
                "preventive": True,
                "strategy": self.defense_strategy
            },
            metadata={
                "agent_name": self.name,
                "analysis_data": analysis
            }
        )
    
    def _select_response_type(self, threat: Dict[str, Any]) -> str:
        """Select appropriate response type for threat."""
        threat_type = threat.get("type", "unknown")
        severity = threat.get("severity", "medium")
        
        if severity == "critical":
            return "incident_isolation"
        elif threat_type in ["malware", "backdoor"]:
            return "malware_removal"
        elif threat_type in ["brute_force", "credential_attack"]:
            return "access_control_hardening"
        elif threat_type in ["network_scan", "reconnaissance"]:
            return "network_monitoring"
        else:
            return "security_alert"
    
    def _calculate_defense_success_probability(self, action: AgentAction) -> float:
        """Calculate probability of defensive action success."""
        base_probability = {
            "beginner": 0.6, 
            "intermediate": 0.75, 
            "advanced": 0.85
        }.get(self.skill_level, 0.75)
        
        # Adjust based on action type
        action_difficulty = {
            "patch_deployment": 0.9,
            "incident_isolation": 0.8,
            "honeypot_deployment": 0.85,
            "malware_removal": 0.7,
            "network_monitoring": 0.95
        }.get(action.type, 0.8)
        
        probability = base_probability * action_difficulty
        
        # Experience bonus
        successful_similar = len([
            a for a in self.memory.successes 
            if a.type == action.type
        ])
        experience_bonus = min(successful_similar * 0.02, 0.1)
        probability += experience_bonus
        
        return min(probability, 0.95)  # Cap at 95% success rate
    
    def _calculate_threat_confidence(self, event: Dict[str, Any]) -> float:
        """Calculate confidence level for threat detection."""
        base_confidence = 0.5
        
        # Higher confidence for known attack patterns
        if event.get("type") in ["sql_injection", "xss_attack", "buffer_overflow"]:
            base_confidence += 0.3
        
        # Higher confidence for multiple indicators
        if event.get("indicators_count", 0) > 3:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _calculate_urgency(self, threat: Dict[str, Any]) -> float:
        """Calculate urgency score for threat."""
        urgency = 0.5
        
        if threat.get("severity") == "critical":
            urgency = 1.0
        elif threat.get("severity") == "high":
            urgency = 0.8
        elif threat.get("severity") == "medium":
            urgency = 0.5
        else:
            urgency = 0.3
        
        # Increase urgency for high-confidence threats
        confidence = threat.get("confidence", 0.5)
        urgency = urgency * (0.5 + confidence * 0.5)
        
        return urgency
    
    def _calculate_impact(self, threat: Dict[str, Any]) -> float:
        """Calculate potential impact of threat."""
        # Base impact on threat type
        impact_map = {
            "malware": 0.9,
            "data_exfiltration": 1.0,
            "privilege_escalation": 0.8,
            "lateral_movement": 0.7,
            "reconnaissance": 0.3
        }
        
        return impact_map.get(threat.get("type", "unknown"), 0.5)
    
    def _generate_defense_results(self, action: AgentAction) -> Dict[str, Any]:
        """Generate realistic results for successful defensive actions."""
        results = {
            "status": "success",
            "actions_taken": [],
            "systems_protected": []
        }
        
        if action.type == "patch_deployment":
            results["actions_taken"] = ["vulnerability_patched", "system_restarted"]
            results["systems_protected"] = [action.target]
        elif action.type == "incident_isolation":
            results["actions_taken"] = ["network_isolated", "processes_terminated"]
            results["threat_contained"] = True
        elif action.type == "honeypot_deployment":
            results["actions_taken"] = ["honeypot_created", "monitoring_enabled"]
            results["deception_active"] = True
        
        return results
    
    def _generate_defense_failure_reason(self, action: AgentAction) -> str:
        """Generate realistic failure reasons for defensive actions."""
        reasons = [
            "Insufficient privileges",
            "System resources unavailable",
            "Network connectivity issues",
            "Service dependencies not met",
            "Configuration conflicts",
            "Patch incompatibility",
            "System maintenance window required"
        ]
        return random.choice(reasons)
    
    def _update_threat_intelligence(self, action: AgentAction) -> None:
        """Update threat intelligence based on successful actions."""
        if action.success and action.type in ["incident_isolation", "malware_removal"]:
            threat_info = {
                "timestamp": datetime.now().isoformat(),
                "action_type": action.type,
                "target": action.target,
                "indicators": action.payload.get("indicators", []),
                "mitigation": action.payload.get("actions_taken", [])
            }
            self.threat_intelligence.append(threat_info)
            
            # Keep only recent intelligence (last 100 items)
            self.threat_intelligence = self.threat_intelligence[-100:]
    
    def _assess_system_health(self, services: List[Dict]) -> Dict[str, Any]:
        """Assess overall system health."""
        total_services = len(services)
        healthy_services = len([s for s in services if s.get("status") == "running"])
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
            "critical_services_up": True,  # Simplified
            "last_assessment": datetime.now().isoformat()
        }