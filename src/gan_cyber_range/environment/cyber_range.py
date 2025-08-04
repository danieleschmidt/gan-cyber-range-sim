"""Core cyber range environment implementation."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..agents.base import BaseAgent
from ..agents.red_team import RedTeamAgent
from ..agents.blue_team import BlueTeamAgent


@dataclass
class Service:
    """Represents a service in the cyber range."""
    name: str
    type: str
    ip: str
    ports: List[int] = field(default_factory=list)
    status: str = "running"
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    business_critical: bool = False
    security_hardened: bool = False
    last_patched: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type,
            "ip": self.ip,
            "open_ports": self.ports,
            "status": self.status,
            "vulnerabilities": self.vulnerabilities,
            "business_critical": self.business_critical,
            "security_hardened": self.security_hardened,
            "last_patched": self.last_patched
        }


@dataclass
class SimulationResults:
    """Results from a cyber range simulation."""
    simulation_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    red_team_actions: List[Dict[str, Any]] = field(default_factory=list)
    blue_team_actions: List[Dict[str, Any]] = field(default_factory=list)
    services_compromised: int = 0
    patches_deployed: int = 0
    attacks_blocked: int = 0
    total_attacks: int = 0
    
    @property
    def duration(self) -> timedelta:
        """Get simulation duration."""
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def compromise_rate(self) -> float:
        """Calculate compromise rate."""
        if self.total_attacks == 0:
            return 0.0
        return self.services_compromised / self.total_attacks
    
    @property
    def defense_effectiveness(self) -> float:
        """Calculate defense effectiveness."""
        if self.total_attacks == 0:
            return 1.0
        return self.attacks_blocked / self.total_attacks
    
    @property
    def avg_detection_time(self) -> str:
        """Calculate average detection time."""
        # Simplified - would be calculated from actual timing data
        return "2.3 minutes"


class CyberRange:
    """Main cyber range environment for adversarial simulation."""
    
    def __init__(
        self,
        vulnerable_services: List[str] = None,
        network_topology: str = "multi-tier",
        difficulty: str = "medium",
        isolation_enabled: bool = True
    ):
        self.simulation_id = str(uuid.uuid4())
        self.network_topology = network_topology
        self.difficulty = difficulty
        self.isolation_enabled = isolation_enabled
        
        # Initialize services
        self.services = self._initialize_services(vulnerable_services or ["webapp", "database", "api-gateway"])
        
        # Simulation state
        self.is_running = False
        self.current_round = 0
        self.simulation_results = SimulationResults(
            simulation_id=self.simulation_id,
            start_time=datetime.now()
        )
        
        # Environment state tracking
        self.network_logs: List[Dict[str, Any]] = []
        self.security_events: List[Dict[str, Any]] = []
        self.attack_indicators: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"CyberRange.{self.simulation_id[:8]}")
        
    def _initialize_services(self, service_names: List[str]) -> List[Service]:
        """Initialize vulnerable services in the range."""
        services = []
        base_ip = "10.0.1"
        
        service_configs = {
            "webapp": {
                "type": "web_application",
                "ports": [80, 443, 8080],
                "vulnerabilities": [
                    {"cve_id": "CVE-2023-12345", "severity": "high", "cvss_score": 8.5, "exploitable": True},
                    {"cve_id": "CVE-2023-12346", "severity": "medium", "cvss_score": 6.2, "exploitable": False}
                ],
                "business_critical": True
            },
            "database": {
                "type": "database",
                "ports": [3306, 5432, 1433],
                "vulnerabilities": [
                    {"cve_id": "CVE-2023-12347", "severity": "critical", "cvss_score": 9.8, "exploitable": True}
                ],
                "business_critical": True,
                "security_hardened": True
            },
            "api-gateway": {
                "type": "api",
                "ports": [8000, 8443],
                "vulnerabilities": [
                    {"cve_id": "CVE-2023-12348", "severity": "medium", "cvss_score": 5.4, "exploitable": False}
                ],
                "business_critical": False
            },
            "fileserver": {
                "type": "file_server",
                "ports": [21, 22, 445],
                "vulnerabilities": [
                    {"cve_id": "CVE-2023-12349", "severity": "high", "cvss_score": 7.8, "exploitable": True}
                ],
                "business_critical": False
            },
            "mailserver": {
                "type": "mail_server",
                "ports": [25, 587, 993],
                "vulnerabilities": [],
                "business_critical": True,
                "security_hardened": True
            }
        }
        
        for i, service_name in enumerate(service_names):
            config = service_configs.get(service_name, {})
            service = Service(
                name=service_name,
                type=config.get("type", "unknown"),
                ip=f"{base_ip}.{10 + i}",
                ports=config.get("ports", []),
                vulnerabilities=config.get("vulnerabilities", []),
                business_critical=config.get("business_critical", False),
                security_hardened=config.get("security_hardened", False),
                last_patched=None
            )
            services.append(service)
        
        return services
    
    def get_environment_state(self) -> Dict[str, Any]:
        """Get current environment state for agents."""
        return {
            "services": [service.to_dict() for service in self.services],
            "network_topology": {
                "type": self.network_topology,
                "segments": ["dmz", "internal", "management"]
            },
            "security_status": {
                "active_defenses": ["firewall", "ids", "antivirus"],
                "monitoring_enabled": True
            },
            "network_logs": self.network_logs[-50:],  # Recent logs
            "security_events": self.security_events[-20:],  # Recent events
            "attack_indicators": self.attack_indicators,
            "simulation_metadata": {
                "round": self.current_round,
                "difficulty": self.difficulty,
                "services_count": len(self.services)
            }
        }
    
    async def simulate(
        self,
        red_team: RedTeamAgent,
        blue_team: BlueTeamAgent,
        duration_hours: float = 1.0,
        realtime_factor: int = 60
    ) -> SimulationResults:
        """Run adversarial simulation between red and blue teams."""
        self.logger.info(f"Starting simulation {self.simulation_id}")
        self.is_running = True
        
        # Activate agents
        red_team.activate()
        blue_team.activate()
        
        # Calculate simulation parameters
        total_rounds = int(duration_hours * 60)  # 1 round per minute
        round_duration = 60 / realtime_factor  # Accelerated time
        
        try:
            for round_num in range(1, total_rounds + 1):
                self.current_round = round_num
                self.logger.info(f"Starting round {round_num}/{total_rounds}")
                
                # Get current environment state
                env_state = self.get_environment_state()
                
                # Execute agents concurrently
                red_actions_task = asyncio.create_task(red_team.act(env_state))
                blue_actions_task = asyncio.create_task(blue_team.act(env_state))
                
                # Wait for both agents to complete their actions
                red_actions, blue_actions = await asyncio.gather(
                    red_actions_task, blue_actions_task
                )
                
                # Process actions and update environment
                await self._process_red_team_actions(red_actions)
                await self._process_blue_team_actions(blue_actions)
                
                # Update simulation results
                self._update_simulation_results(red_actions, blue_actions)
                
                # Log round summary
                self._log_round_summary(round_num, red_actions, blue_actions)
                
                # Wait for next round
                await asyncio.sleep(round_duration)
                
                # Check for early termination conditions
                if self._should_terminate_simulation():
                    self.logger.info("Early termination condition met")
                    break
        
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise
        
        finally:
            self.is_running = False
            red_team.deactivate()
            blue_team.deactivate()
            self.simulation_results.end_time = datetime.now()
        
        self.logger.info(f"Simulation completed: {self.simulation_results.duration}")
        return self.simulation_results
    
    async def _process_red_team_actions(self, actions: List[Any]) -> None:
        """Process red team actions and update environment."""
        for action in actions:
            if action.success:
                # Simulate network activity
                self._generate_network_log(action, "red_team")
                
                # Create security events for successful attacks
                if action.type in ["exploit_attempt", "privilege_escalation"]:
                    self._generate_security_event(action, "attack_detected")
                
                # Update service state if compromised
                if action.type in ["sql_injection", "buffer_overflow", "xss_attack"]:
                    self._compromise_service(action.target)
    
    async def _process_blue_team_actions(self, actions: List[Any]) -> None:
        """Process blue team actions and update environment."""
        for action in actions:
            if action.success:
                # Generate network activity
                self._generate_network_log(action, "blue_team")
                
                # Apply defensive measures
                if action.type == "patch_deployment":
                    self._apply_patch(action.target, action.payload.get("vulnerability_id"))
                elif action.type == "incident_isolation":
                    self._isolate_service(action.target)
                elif action.type == "honeypot_deployment":
                    self._deploy_honeypot()
    
    def _generate_network_log(self, action: Any, team: str) -> None:
        """Generate network log entry for action."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "source_ip": "192.168.1.100" if team == "red_team" else "10.0.1.5",
            "dest_ip": self._get_service_ip(action.target),
            "action_type": action.type,
            "team": team,
            "success": action.success,
            "port": 80,  # Simplified
            "protocol": "TCP"
        }
        self.network_logs.append(log_entry)
        
        # Keep logs limited
        if len(self.network_logs) > 1000:
            self.network_logs = self.network_logs[-500:]
    
    def _generate_security_event(self, action: Any, event_type: str) -> None:
        """Generate security event for monitoring."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": action.type,
            "severity": "high" if action.success else "medium",
            "source_ip": "192.168.1.100",  # Attacker IP
            "target": action.target,
            "event_type": event_type,
            "indicators_count": 3,  # Simplified
            "confidence": 0.8 if action.success else 0.5
        }
        self.security_events.append(event)
        
        # Keep events limited
        if len(self.security_events) > 200:
            self.security_events = self.security_events[-100:]
    
    def _compromise_service(self, service_name: str) -> None:
        """Mark a service as compromised."""
        for service in self.services:
            if service.name == service_name:
                service.status = "compromised"
                self.simulation_results.services_compromised += 1
                self.logger.warning(f"Service {service_name} compromised")
                break
    
    def _apply_patch(self, service_name: str, vulnerability_id: str) -> None:
        """Apply patch to a service."""
        for service in self.services:
            if service.name == service_name:
                # Remove the patched vulnerability
                service.vulnerabilities = [
                    v for v in service.vulnerabilities 
                    if v.get("cve_id") != vulnerability_id
                ]
                service.last_patched = datetime.now().isoformat()
                self.simulation_results.patches_deployed += 1
                self.logger.info(f"Patch applied to {service_name} for {vulnerability_id}")
                break
    
    def _isolate_service(self, service_name: str) -> None:
        """Isolate a compromised service."""
        for service in self.services:
            if service.name == service_name:
                service.status = "isolated"
                self.logger.info(f"Service {service_name} isolated")
                break
    
    def _deploy_honeypot(self) -> None:
        """Deploy a honeypot service."""
        honeypot = Service(
            name=f"honeypot_{len(self.services) + 1}",
            type="honeypot",
            ip=f"10.0.1.{100 + len(self.services)}",
            ports=[80, 22, 443],
            vulnerabilities=[],
            business_critical=False
        )
        self.services.append(honeypot)
        self.logger.info("Honeypot deployed")
    
    def _get_service_ip(self, service_name: str) -> str:
        """Get IP address for a service."""
        for service in self.services:
            if service.name == service_name:
                return service.ip
        return "10.0.1.1"  # Default
    
    def _update_simulation_results(self, red_actions: List[Any], blue_actions: List[Any]) -> None:
        """Update simulation results with action data."""
        # Count total attacks
        self.simulation_results.total_attacks += len(red_actions)
        
        # Count blocked attacks (simplified)
        successful_blue_actions = len([a for a in blue_actions if a.success])
        self.simulation_results.attacks_blocked += successful_blue_actions
        
        # Store action data
        for action in red_actions:
            self.simulation_results.red_team_actions.append({
                "round": self.current_round,
                "type": action.type,
                "target": action.target,
                "success": action.success,
                "timestamp": action.timestamp.isoformat() if hasattr(action.timestamp, 'isoformat') else str(action.timestamp)
            })
        
        for action in blue_actions:
            self.simulation_results.blue_team_actions.append({
                "round": self.current_round,
                "type": action.type,
                "target": action.target,
                "success": action.success,
                "timestamp": action.timestamp.isoformat() if hasattr(action.timestamp, 'isoformat') else str(action.timestamp)
            })
    
    def _log_round_summary(self, round_num: int, red_actions: List[Any], blue_actions: List[Any]) -> None:
        """Log summary of round activities."""
        red_successes = len([a for a in red_actions if a.success])
        blue_successes = len([a for a in blue_actions if a.success])
        
        self.logger.info(
            f"Round {round_num} complete - "
            f"Red: {red_successes}/{len(red_actions)} successful, "
            f"Blue: {blue_successes}/{len(blue_actions)} successful"
        )
    
    def _should_terminate_simulation(self) -> bool:
        """Check if simulation should terminate early."""
        # Terminate if all services are compromised
        compromised_services = len([s for s in self.services if s.status == "compromised"])
        if compromised_services >= len(self.services) * 0.8:  # 80% compromised
            return True
        
        # Terminate if no activity for several rounds
        if self.current_round > 10:
            recent_actions = (
                len(self.simulation_results.red_team_actions[-5:]) +
                len(self.simulation_results.blue_team_actions[-5:])
            )
            if recent_actions == 0:
                return True
        
        return False
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        return {
            "simulation_id": self.simulation_id,
            "is_running": self.is_running,
            "current_round": self.current_round,
            "services_total": len(self.services),
            "services_compromised": len([s for s in self.services if s.status == "compromised"]),
            "services_isolated": len([s for s in self.services if s.status == "isolated"]),
            "total_attacks": self.simulation_results.total_attacks,
            "attacks_blocked": self.simulation_results.attacks_blocked,
            "patches_deployed": self.simulation_results.patches_deployed,
            "duration": str(self.simulation_results.duration)
        }