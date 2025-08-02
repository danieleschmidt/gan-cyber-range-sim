"""Factory Boy factories for creating test data."""

import factory
from datetime import datetime, timedelta
from typing import Dict, Any, List


class VulnerabilityFactory(factory.Factory):
    """Factory for creating vulnerability test data."""
    
    class Meta:
        model = dict
    
    id = factory.Sequence(lambda n: f"CVE-2024-{n:04d}")
    type = factory.Faker('random_element', elements=[
        'sql_injection', 'xss', 'xxe', 'ssrf', 'idor', 'buffer_overflow',
        'privilege_escalation', 'directory_traversal', 'command_injection'
    ])
    severity = factory.Faker('random_element', elements=['low', 'medium', 'high', 'critical'])
    cvss_score = factory.Faker('pyfloat', min_value=0.0, max_value=10.0, right_digits=1)
    affected_service = factory.Faker('random_element', elements=[
        'webapp', 'api', 'database', 'authentication', 'file_server'
    ])
    exploit_available = factory.Faker('boolean', chance_of_getting_true=30)
    patch_available = factory.Faker('boolean', chance_of_getting_true=70)
    description = factory.Faker('text', max_nb_chars=200)
    references = factory.LazyFunction(
        lambda: [f"https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-{factory.Faker('random_int', min=1, max=9999).evaluate(None, None, {'locale': None}):04d}"]
    )


class AttackScenarioFactory(factory.Factory):
    """Factory for creating attack scenario test data."""
    
    class Meta:
        model = dict
    
    name = factory.Faker('catch_phrase')
    description = factory.Faker('text', max_nb_chars=300)
    difficulty = factory.Faker('random_element', elements=['beginner', 'intermediate', 'advanced', 'expert'])
    estimated_duration = factory.Faker('random_int', min=300, max=7200)  # 5 minutes to 2 hours
    techniques = factory.LazyFunction(
        lambda: factory.Faker('random_sample', elements=[
            'reconnaissance', 'initial_access', 'execution', 'persistence',
            'privilege_escalation', 'defense_evasion', 'credential_access',
            'discovery', 'lateral_movement', 'collection', 'exfiltration',
            'command_and_control', 'impact'
        ], length=factory.Faker('random_int', min=2, max=6).evaluate(None, None, {'locale': None})).evaluate(None, None, {'locale': None})
    )
    target_services = factory.LazyFunction(
        lambda: factory.Faker('random_sample', elements=[
            'webapp', 'api', 'database', 'file_server', 'email',
            'dns', 'dhcp', 'authentication', 'monitoring'
        ], length=factory.Faker('random_int', min=1, max=4).evaluate(None, None, {'locale': None})).evaluate(None, None, {'locale': None})
    )
    success_criteria = factory.LazyFunction(
        lambda: [
            'obtain_initial_access',
            'escalate_privileges',
            'access_sensitive_data',
            'maintain_persistence'
        ]
    )


class AgentConfigFactory(factory.Factory):
    """Factory for creating agent configuration test data."""
    
    class Meta:
        model = dict
    
    agent_id = factory.Sequence(lambda n: f"agent-{n}")
    agent_type = factory.Faker('random_element', elements=['red_team', 'blue_team'])
    llm_model = factory.Faker('random_element', elements=[
        'gpt-4', 'gpt-3.5-turbo', 'claude-3-sonnet', 'claude-3-haiku'
    ])
    skill_level = factory.Faker('random_element', elements=['novice', 'intermediate', 'expert'])
    personality_traits = factory.LazyFunction(
        lambda: factory.Faker('random_sample', elements=[
            'aggressive', 'cautious', 'methodical', 'creative',
            'persistent', 'adaptive', 'analytical', 'collaborative'
        ], length=factory.Faker('random_int', min=2, max=4).evaluate(None, None, {'locale': None})).evaluate(None, None, {'locale': None})
    )
    tools = factory.LazyFunction(
        lambda: factory.Faker('random_sample', elements=[
            'nmap', 'metasploit', 'burp_suite', 'wireshark', 'nikto',
            'sqlmap', 'john_the_ripper', 'hashcat', 'gobuster',
            'custom_exploits', 'social_engineering_toolkit'
        ], length=factory.Faker('random_int', min=3, max=7).evaluate(None, None, {'locale': None})).evaluate(None, None, {'locale': None})
    )
    max_memory_size = factory.Faker('random_int', min=500, max=2000)
    temperature = factory.Faker('pyfloat', min_value=0.1, max_value=1.0, right_digits=2)


class NetworkTopologyFactory(factory.Factory):
    """Factory for creating network topology test data."""
    
    class Meta:
        model = dict
    
    topology_id = factory.Sequence(lambda n: f"topology-{n}")
    name = factory.Faker('company')
    topology_type = factory.Faker('random_element', elements=[
        'flat', 'segmented', 'multi_tier', 'dmz', 'zero_trust'
    ])
    subnets = factory.LazyFunction(
        lambda: [
            {
                'cidr': f'192.168.{i}.0/24',
                'name': f'subnet-{i}',
                'security_zone': factory.Faker('random_element', elements=[
                    'public', 'dmz', 'internal', 'restricted', 'management'
                ]).evaluate(None, None, {'locale': None})
            }
            for i in range(1, factory.Faker('random_int', min=2, max=6).evaluate(None, None, {'locale': None}) + 1)
        ]
    )
    services = factory.LazyFunction(
        lambda: [
            {
                'name': f'service-{i}',
                'type': factory.Faker('random_element', elements=[
                    'web_server', 'database', 'api', 'file_server',
                    'dns', 'email', 'authentication', 'monitoring'
                ]).evaluate(None, None, {'locale': None}),
                'port': factory.Faker('random_int', min=80, max=9999).evaluate(None, None, {'locale': None}),
                'protocol': factory.Faker('random_element', elements=['tcp', 'udp']).evaluate(None, None, {'locale': None})
            }
            for i in range(1, factory.Faker('random_int', min=5, max=15).evaluate(None, None, {'locale': None}) + 1)
        ]
    )


class SimulationResultFactory(factory.Factory):
    """Factory for creating simulation result test data."""
    
    class Meta:
        model = dict
    
    simulation_id = factory.Sequence(lambda n: f"sim-{n}")
    scenario_name = factory.Faker('catch_phrase')
    start_time = factory.Faker('date_time_between', start_date='-1h', end_date='now')
    end_time = factory.LazyAttribute(
        lambda obj: obj.start_time + timedelta(
            seconds=factory.Faker('random_int', min=300, max=3600).evaluate(None, None, {'locale': None})
        )
    )
    red_team_agent = factory.SubFactory(AgentConfigFactory, agent_type='red_team')
    blue_team_agent = factory.SubFactory(AgentConfigFactory, agent_type='blue_team')
    network_topology = factory.SubFactory(NetworkTopologyFactory)
    
    # Results metrics
    attacks_attempted = factory.Faker('random_int', min=5, max=50)
    attacks_successful = factory.LazyAttribute(
        lambda obj: factory.Faker('random_int', min=0, max=obj.attacks_attempted).evaluate(None, None, {'locale': None})
    )
    vulnerabilities_exploited = factory.LazyFunction(
        lambda: [VulnerabilityFactory() for _ in range(factory.Faker('random_int', min=1, max=5).evaluate(None, None, {'locale': None}))]
    )
    patches_deployed = factory.Faker('random_int', min=0, max=10)
    mean_time_to_detection = factory.Faker('random_int', min=30, max=300)  # seconds
    mean_time_to_response = factory.Faker('random_int', min=60, max=600)   # seconds
    
    # Calculated fields
    compromise_rate = factory.LazyAttribute(
        lambda obj: obj.attacks_successful / obj.attacks_attempted if obj.attacks_attempted > 0 else 0.0
    )
    simulation_score = factory.LazyAttribute(
        lambda obj: {
            'red_team': factory.Faker('pyfloat', min_value=0.0, max_value=100.0, right_digits=1).evaluate(None, None, {'locale': None}),
            'blue_team': factory.Faker('pyfloat', min_value=0.0, max_value=100.0, right_digits=1).evaluate(None, None, {'locale': None})
        }
    )


class ContainerSecurityProfileFactory(factory.Factory):
    """Factory for creating container security profile test data."""
    
    class Meta:
        model = dict
    
    profile_name = factory.Sequence(lambda n: f"security-profile-{n}")
    security_context = factory.LazyFunction(
        lambda: {
            'runAsNonRoot': True,
            'runAsUser': factory.Faker('random_int', min=1000, max=65535).evaluate(None, None, {'locale': None}),
            'runAsGroup': factory.Faker('random_int', min=1000, max=65535).evaluate(None, None, {'locale': None}),
            'readOnlyRootFilesystem': True,
            'allowPrivilegeEscalation': False,
            'capabilities': {
                'drop': ['ALL'],
                'add': []
            }
        }
    )
    resource_limits = factory.LazyFunction(
        lambda: {
            'cpu': f"{factory.Faker('pyfloat', min_value=0.1, max_value=4.0, right_digits=1).evaluate(None, None, {'locale': None})}",
            'memory': f"{factory.Faker('random_int', min=128, max=4096).evaluate(None, None, {'locale': None})}Mi",
            'ephemeral-storage': f"{factory.Faker('random_int', min=1, max=20).evaluate(None, None, {'locale': None})}Gi"
        }
    )
    network_policy = factory.LazyFunction(
        lambda: {
            'ingress': [
                {
                    'from': [{'podSelector': {'matchLabels': {'role': 'frontend'}}}],
                    'ports': [{'protocol': 'TCP', 'port': 8080}]
                }
            ],
            'egress': [
                {
                    'to': [{'podSelector': {'matchLabels': {'role': 'database'}}}],
                    'ports': [{'protocol': 'TCP', 'port': 5432}]
                }
            ]
        }
    )


# Utility functions for test data creation
def create_vulnerability_dataset(count: int = 10) -> List[Dict[str, Any]]:
    """Create a dataset of vulnerabilities for testing."""
    return [VulnerabilityFactory() for _ in range(count)]


def create_attack_scenarios(count: int = 5) -> List[Dict[str, Any]]:
    """Create a set of attack scenarios for testing."""
    return [AttackScenarioFactory() for _ in range(count)]


def create_simulation_results(count: int = 3) -> List[Dict[str, Any]]:
    """Create simulation results for testing."""
    return [SimulationResultFactory() for _ in range(count)]


def create_red_blue_agent_pair() -> Dict[str, Dict[str, Any]]:
    """Create a matched pair of red and blue team agents."""
    return {
        'red_team': AgentConfigFactory(agent_type='red_team'),
        'blue_team': AgentConfigFactory(agent_type='blue_team')
    }