"""Cyber range specific test fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, List


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_vulnerable_service():
    """Mock vulnerable service for testing."""
    service = MagicMock()
    service.name = "test-webapp"
    service.vulnerabilities = ["sql_injection", "xss"]
    service.status = "running"
    service.exploit = AsyncMock()
    service.patch = AsyncMock()
    return service


@pytest.fixture
def mock_red_team_agent():
    """Mock red team agent for testing."""
    agent = MagicMock()
    agent.name = "TestRedAgent"
    agent.llm_model = "gpt-4"
    agent.plan_attack = AsyncMock(return_value={
        "reconnaissance": ["nmap_scan", "web_crawl"],
        "exploitation": ["sql_injection", "privilege_escalation"],
        "persistence": ["backdoor_install"],
        "exfiltration": ["data_theft"]
    })
    agent.execute_stage = AsyncMock(return_value={
        "success": True,
        "output": "Mock attack execution result",
        "artifacts": ["exploit_log.txt", "stolen_data.json"]
    })
    return agent


@pytest.fixture
def mock_blue_team_agent():
    """Mock blue team agent for testing."""
    agent = MagicMock()
    agent.name = "TestBlueAgent"
    agent.llm_model = "claude-3-sonnet"
    agent.detect_threats = AsyncMock(return_value=[
        {
            "type": "suspicious_network_activity",
            "severity": "high",
            "source_ip": "192.168.1.100",
            "timestamp": "2025-08-01T10:00:00Z"
        }
    ])
    agent.respond_to_threat = AsyncMock(return_value={
        "action": "isolate_host",
        "success": True,
        "details": "Host successfully isolated"
    })
    return agent


@pytest.fixture
def mock_cyber_range_environment():
    """Mock cyber range environment for testing."""
    env = MagicMock()
    env.services = ["webapp", "database", "api-gateway"]
    env.network_topology = "multi-tier"
    env.isolation_enabled = True
    env.deploy_service = AsyncMock()
    env.destroy_service = AsyncMock()
    env.get_service_status = MagicMock(return_value="running")
    return env


@pytest.fixture
def sample_attack_scenario():
    """Sample attack scenario configuration."""
    return {
        "name": "web_application_attack",
        "description": "Multi-stage web application attack scenario",
        "duration_minutes": 30,
        "objectives": [
            "Gain initial access through web vulnerabilities",
            "Escalate privileges on the target system",
            "Exfiltrate sensitive data"
        ],
        "target_services": ["webapp", "database"],
        "attack_vectors": ["sql_injection", "command_injection"],
        "success_criteria": {
            "initial_access": True,
            "privilege_escalation": True,
            "data_exfiltration": True
        }
    }


@pytest.fixture
def sample_defense_scenario():
    """Sample defense scenario configuration."""
    return {
        "name": "incident_response",
        "description": "Automated incident response scenario",
        "monitoring_enabled": True,
        "response_actions": [
            "threat_detection",
            "alert_generation",
            "automated_containment",
            "forensic_analysis"
        ],
        "success_metrics": {
            "detection_time": "< 5 minutes",
            "response_time": "< 10 minutes",
            "false_positive_rate": "< 5%"
        }
    }


@pytest.fixture
def mock_kubernetes_resources():
    """Mock Kubernetes resources for testing."""
    return {
        "deployments": [
            {
                "name": "vulnerable-webapp",
                "namespace": "cyber-range",
                "replicas": 1,
                "status": "Running"
            }
        ],
        "services": [
            {
                "name": "webapp-service",
                "namespace": "cyber-range",
                "type": "ClusterIP",
                "ports": [80, 443]
            }
        ],
        "network_policies": [
            {
                "name": "deny-all",
                "namespace": "cyber-range",
                "policy_types": ["Ingress", "Egress"]
            }
        ]
    }


@pytest.fixture
def mock_security_tools():
    """Mock security tools for testing."""
    return {
        "nmap": MagicMock(
            scan=AsyncMock(return_value={
                "hosts": ["192.168.1.100"],
                "open_ports": [22, 80, 443],
                "services": ["ssh", "http", "https"]
            })
        ),
        "metasploit": MagicMock(
            execute_exploit=AsyncMock(return_value={
                "success": True,
                "session_id": "meterpreter_1",
                "target": "192.168.1.100"
            })
        ),
        "nikto": MagicMock(
            scan_web=AsyncMock(return_value={
                "vulnerabilities": ["SQL Injection", "XSS"],
                "severity": "High"
            })
        )
    }


@pytest.fixture
def simulation_metrics():
    """Sample simulation metrics for testing."""
    return {
        "attacks_attempted": 15,
        "attacks_successful": 8,
        "patches_deployed": 12,
        "services_compromised": 3,
        "mean_time_to_detection": 4.2,
        "mean_time_to_remediation": 8.7,
        "false_positive_rate": 0.03,
        "simulation_duration": 1800,  # 30 minutes
        "performance_metrics": {
            "cpu_usage": 0.65,
            "memory_usage": 0.78,
            "network_traffic": "1.2 GB"
        }
    }


@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses for different scenarios."""
    return {
        "attack_planning": """
        Based on the target analysis, I recommend a multi-stage attack:
        1. Reconnaissance: Perform port scanning and service enumeration
        2. Initial Access: Exploit SQL injection vulnerability in login form
        3. Privilege Escalation: Use sudo misconfiguration to gain root
        4. Persistence: Install backdoor service
        5. Data Exfiltration: Compress and exfiltrate sensitive files
        """,
        "defense_strategy": """
        Detected potential intrusion. Recommended response:
        1. Isolate affected host from network
        2. Collect forensic evidence
        3. Apply security patches to vulnerable service
        4. Monitor for additional indicators of compromise
        5. Update security rules to prevent similar attacks
        """,
        "vulnerability_analysis": """
        Vulnerability Assessment Results:
        - SQL Injection in login form (CVSS: 9.8)
        - Cross-Site Scripting in search function (CVSS: 6.1)
        - Weak password policy (CVSS: 5.3)
        Recommended patches: Update to latest version, implement input validation
        """
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark data for testing."""
    return {
        "agent_response_time": {
            "p50": 1.2,
            "p95": 3.4,
            "p99": 8.1,
            "max": 15.0
        },
        "simulation_throughput": {
            "scenarios_per_hour": 12,
            "concurrent_agents": 8,
            "resource_utilization": 0.67
        },
        "scalability_limits": {
            "max_concurrent_simulations": 50,
            "max_agents_per_simulation": 20,
            "memory_per_simulation": "2GB"
        }
    }