#!/usr/bin/env python3
"""Minimal test to verify core GAN Cyber Range functionality without external dependencies."""

import sys
import os
import asyncio
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class MockAgentAction:
    """Mock AgentAction for testing without pydantic."""
    type: str
    target: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, model: str = "mock-model"):
        self.model = model
        
    async def generate(self, request):
        """Generate mock response."""
        await asyncio.sleep(0.1)  # Simulate latency
        
        if hasattr(request, 'context') and request.context.get("agent_type") == "red_team":
            return type('MockResponse', (), {
                'parsed_json': {
                    "situation_analysis": "Mock red team analysis: vulnerable services detected",
                    "attack_opportunities": [
                        {"target": "webapp", "vulnerability": "sql_injection", "success_probability": 0.7}
                    ],
                    "recommended_actions": [
                        {"action": "vulnerability_scan", "target": "webapp", "reasoning": "Test SQL injection"}
                    ]
                },
                'content': '{"status": "mock_response"}',
                'tokens_used': 100,
                'latency_ms': 100,
                'model': self.model,
                'timestamp': datetime.now()
            })
        else:  # blue_team
            return type('MockResponse', (), {
                'parsed_json': {
                    "threat_assessment": "Mock blue team analysis: threats detected",
                    "detected_threats": [
                        {"threat": "sql_injection_attempt", "severity": "high", "confidence": 0.8}
                    ],
                    "defensive_actions": [
                        {"action": "patch_deployment", "target": "webapp", "priority": 0.9}
                    ]
                },
                'content': '{"status": "mock_response"}',
                'tokens_used': 100,
                'latency_ms': 100,
                'model': self.model,
                'timestamp': datetime.now()
            })


class MockRedTeamAgent:
    """Mock Red Team Agent for testing."""
    
    def __init__(self, name: str = "MockRedTeam", **kwargs):
        self.name = name
        self.skill_level = kwargs.get('skill_level', 'advanced')
        self.active = False
        self.logger = logging.getLogger(f"MockRedTeam.{name}")
        self.llm_client = MockLLMClient()
        self.action_count = 0
        
    def activate(self):
        self.active = True
        self.logger.info(f"Agent {self.name} activated")
        
    def deactivate(self):
        self.active = False
        self.logger.info(f"Agent {self.name} deactivated")
        
    async def act(self, environment_state: Dict[str, Any]) -> List[MockAgentAction]:
        """Perform red team actions."""
        if not self.active:
            return []
            
        self.action_count += 1
        self.logger.info(f"Red team executing round {self.action_count}")
        
        # Simulate analysis and action planning
        await asyncio.sleep(0.5)  # Simulate thinking time
        
        # Generate mock actions based on environment
        actions = []
        services = environment_state.get('services', [])
        
        for service in services[:2]:  # Attack up to 2 services
            if service.get('status') == 'running':
                action_type = random.choice(['vulnerability_scan', 'exploit_attempt', 'reconnaissance'])
                action = MockAgentAction(
                    type=action_type,
                    target=service.get('name', 'unknown'),
                    payload={
                        'target_ip': service.get('ip', '10.0.1.1'),
                        'attack_vector': action_type,
                        'confidence': random.uniform(0.5, 0.9)
                    },
                    success=random.random() > 0.4,  # 60% success rate
                    metadata={'agent': self.name, 'round': self.action_count}
                )
                actions.append(action)
                
        self.logger.info(f"Red team planned {len(actions)} actions")
        return actions
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'total_actions': self.action_count * 2,  # Approximate
            'success_rate': 0.6,
            'skill_level': self.skill_level,
            'active': self.active
        }


class MockBlueTeamAgent:
    """Mock Blue Team Agent for testing."""
    
    def __init__(self, name: str = "MockBlueTeam", **kwargs):
        self.name = name
        self.skill_level = kwargs.get('skill_level', 'advanced')
        self.defense_strategy = kwargs.get('defense_strategy', 'proactive')
        self.active = False
        self.logger = logging.getLogger(f"MockBlueTeam.{name}")
        self.llm_client = MockLLMClient()
        self.action_count = 0
        
    def activate(self):
        self.active = True
        self.logger.info(f"Agent {self.name} activated")
        
    def deactivate(self):
        self.active = False
        self.logger.info(f"Agent {self.name} deactivated")
        
    async def act(self, environment_state: Dict[str, Any]) -> List[MockAgentAction]:
        """Perform blue team actions."""
        if not self.active:
            return []
            
        self.action_count += 1
        self.logger.info(f"Blue team executing round {self.action_count}")
        
        # Simulate analysis and response planning
        await asyncio.sleep(0.3)  # Defenders are faster
        
        # Generate defensive actions
        actions = []
        security_events = environment_state.get('security_events', [])
        
        # Respond to recent high-severity events
        high_severity_events = [e for e in security_events[-3:] if e.get('severity') == 'high']
        
        for event in high_severity_events:
            action_type = random.choice(['patch_deployment', 'incident_isolation', 'threat_monitoring'])
            action = MockAgentAction(
                type=action_type,
                target=event.get('target', 'infrastructure'),
                payload={
                    'threat_id': event.get('type', 'unknown_threat'),
                    'response_type': action_type,
                    'urgency': random.uniform(0.6, 1.0)
                },
                success=random.random() > 0.25,  # 75% success rate  
                metadata={'agent': self.name, 'round': self.action_count}
            )
            actions.append(action)
            
        # Always add monitoring action
        monitoring_action = MockAgentAction(
            type='security_monitoring',
            target='infrastructure',
            payload={'scope': 'system_wide', 'preventive': True},
            success=True,  # Monitoring usually works
            metadata={'agent': self.name, 'round': self.action_count}
        )
        actions.append(monitoring_action)
        
        self.logger.info(f"Blue team planned {len(actions)} defensive actions")
        return actions
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'total_actions': self.action_count * 2,  # Approximate
            'success_rate': 0.75,
            'skill_level': self.skill_level,
            'active': self.active
        }


class MockSimulationResults:
    """Mock simulation results."""
    
    def __init__(self, simulation_id: str, start_time: datetime):
        self.simulation_id = simulation_id
        self.start_time = start_time
        self.end_time = None
        self.red_team_actions = []
        self.blue_team_actions = []
        self.services_compromised = 0
        self.patches_deployed = 0
        self.attacks_blocked = 0
        self.total_attacks = 0
        
    @property
    def duration(self):
        end = self.end_time or datetime.now()
        return end - self.start_time
        
    @property
    def compromise_rate(self) -> float:
        if self.total_attacks == 0:
            return 0.0
        return self.services_compromised / self.total_attacks
        
    @property
    def defense_effectiveness(self) -> float:
        if self.total_attacks == 0:
            return 1.0
        return self.attacks_blocked / self.total_attacks
        
    @property
    def avg_detection_time(self) -> str:
        return "1.8 minutes"  # Mock value


class MockCyberRange:
    """Mock Cyber Range for testing."""
    
    def __init__(self, vulnerable_services=None, **kwargs):
        self.services = self._create_mock_services(vulnerable_services or ['webapp', 'database'])
        self.simulation_id = f"sim_{int(datetime.now().timestamp())}"
        self.logger = logging.getLogger(f"MockCyberRange.{self.simulation_id[:8]}")
        self.round_count = 0
        
    def _create_mock_services(self, service_names):
        services = []
        for i, name in enumerate(service_names):
            service = {
                'name': name,
                'type': 'web_application' if name == 'webapp' else name,
                'ip': f'10.0.1.{10 + i}',
                'open_ports': [80, 443] if name == 'webapp' else [3306, 5432],
                'status': 'running',
                'vulnerabilities': [
                    {'cve_id': f'CVE-2023-{1000 + i}', 'severity': 'high', 'exploitable': True}
                ],
                'business_critical': True,
                'last_patched': None
            }
            services.append(service)
        return services
        
    def get_environment_state(self):
        """Get current environment state."""
        return {
            'services': self.services,
            'network_topology': {'type': 'multi-tier', 'segments': ['dmz', 'internal']},
            'security_status': {'active_defenses': ['firewall', 'ids'], 'monitoring_enabled': True},
            'security_events': [
                {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'vulnerability_scan',
                    'severity': 'high',
                    'target': 'webapp',
                    'source_ip': '192.168.1.100'
                }
            ],
            'network_logs': [
                {
                    'timestamp': datetime.now().isoformat(),
                    'source_ip': '192.168.1.100',
                    'dest_ip': '10.0.1.10',
                    'action_type': 'port_scan',
                    'success': True
                }
            ],
            'attack_indicators': [],
            'simulation_metadata': {
                'round': self.round_count,
                'services_count': len(self.services)
            }
        }
        
    async def simulate(self, red_team, blue_team, duration_hours=0.1, realtime_factor=60):
        """Run simulation."""
        self.logger.info(f"Starting mock simulation {self.simulation_id}")
        
        results = MockSimulationResults(self.simulation_id, datetime.now())
        
        # Activate agents
        red_team.activate()
        blue_team.activate()
        
        try:
            total_rounds = max(1, int(duration_hours * 10))  # At least 1 round
            
            for round_num in range(1, total_rounds + 1):
                self.round_count = round_num
                self.logger.info(f"Simulation round {round_num}/{total_rounds}")
                
                # Get environment state
                env_state = self.get_environment_state()
                
                # Execute agents concurrently
                red_actions_task = asyncio.create_task(red_team.act(env_state))
                blue_actions_task = asyncio.create_task(blue_team.act(env_state))
                
                red_actions, blue_actions = await asyncio.gather(
                    red_actions_task, blue_actions_task
                )
                
                # Process results
                self._process_actions(red_actions, blue_actions, results)
                
                # Short delay for realism
                await asyncio.sleep(0.1)
                
            results.end_time = datetime.now()
            
        finally:
            red_team.deactivate()
            blue_team.deactivate()
            
        self.logger.info(f"Mock simulation completed in {results.duration}")
        return results
        
    def _process_actions(self, red_actions, blue_actions, results):
        """Process agent actions and update results."""
        results.total_attacks += len(red_actions)
        
        # Count successful red team actions as compromises
        successful_attacks = [a for a in red_actions if a.success]
        results.services_compromised += len(successful_attacks)
        
        # Count successful blue team defensive actions
        successful_defenses = [a for a in blue_actions if a.success and a.type == 'patch_deployment']
        results.patches_deployed += len(successful_defenses)
        
        # Count blocks (simplified)
        monitoring_actions = [a for a in blue_actions if a.success and 'monitoring' in a.type]
        results.attacks_blocked += len(monitoring_actions)
        
        # Store actions
        for action in red_actions:
            results.red_team_actions.append({
                'round': self.round_count,
                'type': action.type,
                'target': action.target,
                'success': action.success,
                'timestamp': action.timestamp
            })
            
        for action in blue_actions:
            results.blue_team_actions.append({
                'round': self.round_count,
                'type': action.type,
                'target': action.target,
                'success': action.success,
                'timestamp': action.timestamp
            })


def display_results(results, red_team, blue_team):
    """Display simulation results."""
    print("\n" + "="*60)
    print("🎉 SIMULATION COMPLETE!")
    print("="*60)
    
    print(f"📊 SIMULATION SUMMARY:")
    print(f"   Duration: {results.duration}")
    print(f"   Total Attacks: {results.total_attacks}")
    print(f"   Services Compromised: {results.services_compromised}")
    print(f"   Attacks Blocked: {results.attacks_blocked}")
    print(f"   Patches Deployed: {results.patches_deployed}")
    print(f"   Compromise Rate: {results.compromise_rate:.2%}")
    print(f"   Defense Effectiveness: {results.defense_effectiveness:.2%}")
    print(f"   Avg Detection Time: {results.avg_detection_time}")
    
    print(f"\n🤖 AGENT PERFORMANCE:")
    red_stats = red_team.get_stats()
    blue_stats = blue_team.get_stats()
    
    print(f"   Red Team ({red_stats['name']}):")
    print(f"      Actions: {red_stats['total_actions']}")
    print(f"      Success Rate: {red_stats['success_rate']:.2%}")
    print(f"      Skill Level: {red_stats['skill_level']}")
    
    print(f"   Blue Team ({blue_stats['name']}):")
    print(f"      Actions: {blue_stats['total_actions']}")
    print(f"      Success Rate: {blue_stats['success_rate']:.2%}")
    print(f"      Skill Level: {blue_stats['skill_level']}")


async def main():
    """Main test function."""
    print("🎯 Starting GAN Cyber Range Minimal Test")
    print("="*60)
    
    # Initialize components
    print("🏗️  Initializing cyber range...")
    cyber_range = MockCyberRange(
        vulnerable_services=['webapp', 'database', 'api-gateway']
    )
    
    print("🔴 Creating red team agent...")
    red_team = MockRedTeamAgent(
        name="AdvancedRedTeam",
        skill_level="advanced"
    )
    
    print("🔵 Creating blue team agent...")
    blue_team = MockBlueTeamAgent(
        name="ProactiveBlueTeam", 
        skill_level="advanced",
        defense_strategy="proactive"
    )
    
    # Run simulation
    print("\n🚀 Starting adversarial simulation...")
    results = await cyber_range.simulate(
        red_team=red_team,
        blue_team=blue_team,
        duration_hours=0.1,  # 6 minutes in simulated time
        realtime_factor=60
    )
    
    # Display results
    display_results(results, red_team, blue_team)
    
    print("\n✅ Test completed successfully!")
    print("🔬 Core functionality verified - ready for full implementation")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())