"""Blue team (defender) agent implementation."""

import asyncio
import random
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from .base import BaseAgent, AgentAction
from .llm_client import AgentLLMIntegration


class BlueTeamAgent(BaseAgent):
    """Blue team agent that performs defensive security actions."""
    
    def __init__(
        self,
        name: str = "BlueTeam",
        llm_model: str = "claude-3",
        skill_level: str = "advanced",
        defense_strategy: str = "proactive",
        tools: List[str] = None,
        threat_intelligence_feeds: List[str] = None,
        auto_response_enabled: bool = True,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, llm_model, skill_level, **kwargs)
        self.defense_strategy = defense_strategy
        self.llm_integration = AgentLLMIntegration(llm_model, api_key)
        self.tools = tools or ["ids", "auto_patcher", "honeypots", "firewall", "siem", "edr"]
        self.threat_intelligence_feeds = threat_intelligence_feeds or [
            "mitre_attack", "cti_feeds", "osint", "internal_intel"
        ]
        self.auto_response_enabled = auto_response_enabled
        self.defensive_actions = [
            "threat_detection",
            "vulnerability_patching",
            "access_control",
            "network_monitoring",
            "incident_response",
            "system_hardening",
            "honeypot_deployment",
            "threat_hunting",
            "forensic_analysis",
            "containment",
            "eradication",
            "recovery"
        ]
        self.threat_intelligence = []
        self.active_incidents = []
        self.incident_history = []
        self.threat_indicators = set()
        self.quarantine_zones = []
        self.security_baselines = {}
        self.detection_rules = []
        self.response_playbooks = self._initialize_playbooks()
    
    async def analyze_environment(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environment for security threats using LLM intelligence."""
        try:
            # Get LLM-powered analysis
            agent_context = {
                "skill_level": self.skill_level,
                "tools": self.tools,
                "round": self._round_counter,
                "defense_strategy": self.defense_strategy,
                "threat_intel_feeds": self.threat_intelligence_feeds,
                "previous_actions": len(self.memory.actions),
                "auto_response": self.auto_response_enabled
            }
            
            llm_analysis = await self.llm_integration.analyze_environment(
                "blue_team", environment_state, agent_context
            )
            
            # Enhance with traditional security analysis
            enhanced_analysis = self._enhance_analysis_with_traditional_methods(
                llm_analysis, environment_state
            )
            
            return enhanced_analysis
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed, falling back to traditional methods: {e}")
            return self._fallback_analysis(environment_state)
    
    async def plan_actions(self, analysis: Dict[str, Any]) -> List[AgentAction]:
        """Plan defensive actions using LLM intelligence."""
        try:
            # Get LLM-powered action plan
            agent_context = {
                "skill_level": self.skill_level,
                "tools": self.tools,
                "round": self._round_counter,
                "max_actions": self.max_actions_per_round,
                "defense_strategy": self.defense_strategy,
                "recent_successes": len(self.memory.successes),
                "active_incidents": len(self.active_incidents)
            }
            
            llm_actions = await self.llm_integration.plan_actions(
                "blue_team", analysis, agent_context
            )
            
            # Convert LLM actions to AgentAction objects
            actions = self._convert_llm_actions_to_agent_actions(llm_actions, analysis)
            
            # Ensure we don't exceed max actions per round
            return actions[:self.max_actions_per_round]
            
        except Exception as e:
            self.logger.warning(f"LLM action planning failed, falling back to traditional planning: {e}")
            return self._fallback_action_planning(analysis)
    
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
        
        # Correlate with threat intelligence feeds
        for threat in threats:
            threat['intelligence_matches'] = self._correlate_with_threat_intel(threat)
            threat['risk_score'] = self._calculate_risk_score(threat)
        
        # Sort by risk score
        threats.sort(key=lambda x: x.get('risk_score', 0.5), reverse=True)
        
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
    
    def _initialize_playbooks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize automated response playbooks."""
        return {
            "malware_detected": {
                "severity": "high",
                "steps": [
                    {"action": "isolate_host", "priority": 1},
                    {"action": "collect_artifacts", "priority": 2},
                    {"action": "scan_network", "priority": 3},
                    {"action": "update_signatures", "priority": 4}
                ]
            },
            "brute_force_attack": {
                "severity": "medium",
                "steps": [
                    {"action": "block_source_ip", "priority": 1},
                    {"action": "lock_account", "priority": 2},
                    {"action": "enhance_monitoring", "priority": 3}
                ]
            },
            "data_exfiltration": {
                "severity": "critical",
                "steps": [
                    {"action": "block_egress", "priority": 1},
                    {"action": "isolate_affected_systems", "priority": 2},
                    {"action": "activate_incident_response", "priority": 3},
                    {"action": "notify_stakeholders", "priority": 4}
                ]
            },
            "privilege_escalation": {
                "severity": "high",
                "steps": [
                    {"action": "terminate_suspicious_processes", "priority": 1},
                    {"action": "review_access_logs", "priority": 2},
                    {"action": "patch_vulnerabilities", "priority": 3},
                    {"action": "strengthen_access_controls", "priority": 4}
                ]
            }
        }
    
    def _correlate_with_threat_intel(self, threat: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Correlate threat with intelligence feeds."""
        matches = []
        
        threat_indicators = [
            threat.get("source"),
            threat.get("indicator"),
            threat.get("type")
        ]
        
        for indicator in threat_indicators:
            if indicator and indicator in self.threat_indicators:
                matches.append({
                    "indicator": indicator,
                    "source": "internal_intel",
                    "confidence": 0.8
                })
        
        # Simulate external threat intelligence correlation
        if threat.get("type") in ["malware", "c2_communication", "data_exfiltration"]:
            matches.append({
                "indicator": threat.get("type"),
                "source": "mitre_attack",
                "confidence": 0.9,
                "technique_id": self._map_to_mitre_technique(threat.get("type"))
            })
        
        return matches
    
    def _map_to_mitre_technique(self, threat_type: str) -> str:
        """Map threat type to MITRE ATT&CK technique."""
        technique_mapping = {
            "malware": "T1059",  # Command and Scripting Interpreter
            "c2_communication": "T1071",  # Application Layer Protocol
            "data_exfiltration": "T1041",  # Exfiltration Over C2 Channel
            "privilege_escalation": "T1068",  # Exploitation for Privilege Escalation
            "lateral_movement": "T1021",  # Remote Services
            "persistence": "T1053"  # Scheduled Task/Job
        }
        return technique_mapping.get(threat_type, "T1000")
    
    def _calculate_risk_score(self, threat: Dict[str, Any]) -> float:
        """Calculate comprehensive risk score for threat."""
        base_score = 0.5
        
        # Severity contribution
        severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }
        base_score += severity_weights.get(threat.get("severity", "medium"), 0.5) * 0.3
        
        # Confidence contribution
        confidence = threat.get("confidence", 0.5)
        base_score += confidence * 0.2
        
        # Intelligence matches contribution
        intel_matches = threat.get("intelligence_matches", [])
        if intel_matches:
            avg_intel_confidence = sum(m.get("confidence", 0) for m in intel_matches) / len(intel_matches)
            base_score += avg_intel_confidence * 0.2
        
        # Asset value contribution (if targeting critical assets)
        if threat.get("target") in ["database", "domain_controller", "api_gateway"]:
            base_score += 0.3
        
        return min(base_score, 1.0)
    
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
    
    async def execute_automated_response(self, threat: Dict[str, Any]) -> List[AgentAction]:
        """Execute automated response based on threat type."""
        if not self.auto_response_enabled:
            return []
        
        threat_type = threat.get("type", "unknown")
        playbook_key = self._map_threat_to_playbook(threat_type)
        
        if playbook_key not in self.response_playbooks:
            return []
        
        playbook = self.response_playbooks[playbook_key]
        actions = []
        
        for step in sorted(playbook["steps"], key=lambda x: x["priority"]):
            action = AgentAction(
                type=step["action"],
                target=threat.get("target", "unknown"),
                payload={
                    "threat_id": threat.get("id"),
                    "automated": True,
                    "playbook": playbook_key,
                    "step_priority": step["priority"]
                },
                metadata={
                    "agent_name": self.name,
                    "response_type": "automated",
                    "threat_data": threat
                }
            )
            actions.append(action)
        
        return actions
    
    def _map_threat_to_playbook(self, threat_type: str) -> str:
        """Map threat type to response playbook."""
        mapping = {
            "malware": "malware_detected",
            "brute_force": "brute_force_attack",
            "credential_attack": "brute_force_attack",
            "data_exfiltration": "data_exfiltration",
            "privilege_escalation": "privilege_escalation",
            "lateral_movement": "privilege_escalation"
        }
        return mapping.get(threat_type, "malware_detected")
    
    async def conduct_threat_hunt(self, hunt_hypothesis: str) -> Dict[str, Any]:
        """Conduct proactive threat hunting."""
        hunt_results = {
            "hypothesis": hunt_hypothesis,
            "start_time": datetime.now().isoformat(),
            "findings": [],
            "indicators_discovered": [],
            "confidence_score": 0.0
        }
        
        # Simulate threat hunting activities
        hunting_techniques = [
            "log_analysis",
            "network_traffic_analysis", 
            "endpoint_analysis",
            "behavioral_analysis"
        ]
        
        for technique in hunting_techniques:
            findings = await self._execute_hunt_technique(technique, hunt_hypothesis)
            hunt_results["findings"].extend(findings)
        
        # Extract new indicators of compromise
        for finding in hunt_results["findings"]:
            if finding.get("indicators"):
                hunt_results["indicators_discovered"].extend(finding["indicators"])
                self.threat_indicators.update(finding["indicators"])
        
        # Calculate overall confidence
        if hunt_results["findings"]:
            avg_confidence = sum(f.get("confidence", 0) for f in hunt_results["findings"]) / len(hunt_results["findings"])
            hunt_results["confidence_score"] = avg_confidence
        
        hunt_results["end_time"] = datetime.now().isoformat()
        return hunt_results
    
    async def _execute_hunt_technique(self, technique: str, hypothesis: str) -> List[Dict[str, Any]]:
        """Execute specific threat hunting technique."""
        findings = []
        
        # Simulate different hunting techniques
        if technique == "log_analysis":
            findings.append({
                "technique": "log_analysis",
                "description": f"Analyzed logs for patterns related to: {hypothesis}",
                "confidence": 0.7,
                "indicators": [f"suspicious_pattern_{hash(hypothesis) % 1000}"],
                "evidence": "Multiple failed authentication attempts from single IP"
            })
        
        elif technique == "network_traffic_analysis":
            findings.append({
                "technique": "network_traffic_analysis",
                "description": f"Network traffic analysis for: {hypothesis}",
                "confidence": 0.8,
                "indicators": [f"network_indicator_{hash(hypothesis) % 1000}"],
                "evidence": "Unusual outbound connections to suspicious domains"
            })
        
        elif technique == "endpoint_analysis":
            findings.append({
                "technique": "endpoint_analysis",
                "description": f"Endpoint forensics for: {hypothesis}",
                "confidence": 0.6,
                "indicators": [f"endpoint_artifact_{hash(hypothesis) % 1000}"],
                "evidence": "Suspicious process execution patterns detected"
            })
        
        elif technique == "behavioral_analysis":
            findings.append({
                "technique": "behavioral_analysis",
                "description": f"User behavior analysis for: {hypothesis}",
                "confidence": 0.9,
                "indicators": [f"behavior_anomaly_{hash(hypothesis) % 1000}"],
                "evidence": "Anomalous user access patterns identified"
            })
        
        # Simulate async processing time
        await asyncio.sleep(0.1)
        return findings
    
    async def create_incident(self, threat: Dict[str, Any], severity: str = "medium") -> str:
        """Create security incident from threat."""
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{len(self.active_incidents) + 1:03d}"
        
        incident = {
            "id": incident_id,
            "title": f"Security Incident: {threat.get('type', 'Unknown threat')}",
            "severity": severity,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "threat_data": threat,
            "assigned_analyst": self.name,
            "timeline": [{
                "timestamp": datetime.now().isoformat(),
                "action": "incident_created",
                "details": f"Incident created from threat {threat.get('id')}"
            }],
            "containment_actions": [],
            "eradication_actions": [],
            "recovery_actions": []
        }
        
        self.active_incidents.append(incident)
        self.logger.info(f"Created security incident: {incident_id}")
        
        return incident_id
    
    async def update_incident(self, incident_id: str, action: str, details: str) -> bool:
        """Update incident with new action/information."""
        for incident in self.active_incidents:
            if incident["id"] == incident_id:
                incident["timeline"].append({
                    "timestamp": datetime.now().isoformat(),
                    "action": action,
                    "details": details
                })
                
                # Update appropriate action list
                if action.startswith("containment"):
                    incident["containment_actions"].append(details)
                elif action.startswith("eradication"):
                    incident["eradication_actions"].append(details)
                elif action.startswith("recovery"):
                    incident["recovery_actions"].append(details)
                
                return True
        
        return False
    
    async def deploy_adaptive_honeypot(self, threat_type: str, location: str) -> Dict[str, Any]:
        """Deploy adaptive honeypot based on observed threats."""
        honeypot_config = {
            "id": f"honeypot_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "adaptive",
            "threat_mimicry": threat_type,
            "location": location,
            "deployed_at": datetime.now().isoformat(),
            "interactions": [],
            "intelligence_gathered": []
        }
        
        # Configure honeypot based on threat type
        if threat_type == "web_attack":
            honeypot_config.update({
                "services": ["http", "https"],
                "vulnerabilities": ["sql_injection", "xss", "path_traversal"],
                "port": 8080
            })
        elif threat_type == "lateral_movement":
            honeypot_config.update({
                "services": ["ssh", "rdp", "smb"],
                "credentials": ["weak_passwords", "default_accounts"],
                "port": 22
            })
        elif threat_type == "data_exfiltration":
            honeypot_config.update({
                "services": ["ftp", "database"],
                "fake_data": ["customer_records", "financial_data"],
                "port": 21
            })
        
        self.logger.info(f"Deployed adaptive honeypot: {honeypot_config['id']}")
        return honeypot_config
    
    def _enhance_analysis_with_traditional_methods(self, llm_analysis: Dict[str, Any], environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance LLM analysis with traditional security methods."""
        services = environment_state.get("services", [])
        network_logs = environment_state.get("network_logs", [])
        security_events = environment_state.get("security_events", [])
        attack_indicators = environment_state.get("attack_indicators", [])
        
        # Add traditional threat detection
        traditional_threats = self._detect_active_threats(security_events, attack_indicators)
        traditional_vulns = self._assess_system_vulnerabilities(services)
        
        # Merge with LLM analysis
        llm_threats = llm_analysis.get("detected_threats", [])
        all_threats = traditional_threats + [
            {"id": f"llm_{i}", **threat} for i, threat in enumerate(llm_threats)
        ]
        
        # Enhanced analysis
        enhanced_analysis = {
            **llm_analysis,
            "traditional_threats": traditional_threats,
            "traditional_vulnerabilities": traditional_vulns,
            "all_threats": all_threats,
            "system_health": self._assess_system_health(services),
            "attack_patterns": self._analyze_attack_patterns(network_logs, security_events),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return enhanced_analysis
    
    def _fallback_analysis(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis using traditional security methods."""
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
    
    def _convert_llm_actions_to_agent_actions(self, llm_actions: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[AgentAction]:
        """Convert LLM action recommendations to AgentAction objects."""
        actions = []
        
        for llm_action in llm_actions:
            action_type = llm_action.get("action", "monitoring")
            target = llm_action.get("target", "infrastructure")
            priority = llm_action.get("priority", 0.5)
            
            # Find threat or vulnerability data from analysis
            action_data = self._find_action_context(action_type, target, analysis)
            
            action = AgentAction(
                type=action_type,
                target=target,
                payload={
                    "llm_reasoning": llm_action.get("reasoning", "LLM recommended action"),
                    "priority": priority,
                    "action_data": action_data,
                    "defense_strategy": self.defense_strategy,
                    "urgency": llm_action.get("urgency", 0.5)
                },
                metadata={
                    "agent_name": self.name,
                    "skill_level": self.skill_level,
                    "llm_generated": True,
                    "defense_strategy": self.defense_strategy,
                    "llm_action_data": llm_action
                }
            )
            actions.append(action)
        
        return actions
    
    def _find_action_context(self, action_type: str, target: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Find context data for an action from analysis."""
        # Look for threat data
        for threat in analysis.get("all_threats", []):
            if threat.get("target") == target or threat.get("id") == target:
                return {"type": "threat_response", "threat_data": threat}
        
        # Look for vulnerability data
        for vuln in analysis.get("vulnerabilities", []):
            if vuln.get("service") == target or vuln.get("id") == target:
                return {"type": "vulnerability_mitigation", "vuln_data": vuln}
        
        return {"type": "general_action", "target": target}
    
    def _fallback_action_planning(self, analysis: Dict[str, Any]) -> List[AgentAction]:
        """Fallback action planning using traditional methods."""
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