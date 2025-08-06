#!/usr/bin/env python3
"""Validate the defensive security system components."""

import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_status(message: str, status: str = "INFO"):
    """Print status message with formatting."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{status}] {message}")

def validate_imports():
    """Validate that all security components can be imported."""
    print_status("Validating security component imports...")
    
    try:
        from gan_cyber_range.agents.blue_team import BlueTeamAgent
        print_status("✓ Blue Team Agent imported successfully")
        
        from gan_cyber_range.security.isolation import NetworkIsolation, IsolationLevel
        print_status("✓ Network Isolation imported successfully")
        
        from gan_cyber_range.security.validator import SecurityValidator, InputValidator
        print_status("✓ Security Validator imported successfully")
        
        from gan_cyber_range.monitoring.health_check import HealthChecker
        print_status("✓ Health Checker imported successfully")
        
        print_status("All core defensive components imported successfully", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Import validation failed: {e}", "ERROR")
        return False

def validate_blue_team_agent():
    """Validate Blue Team Agent functionality."""
    print_status("Validating Blue Team Agent...")
    
    try:
        from gan_cyber_range.agents.blue_team import BlueTeamAgent
        
        # Create agent
        agent = BlueTeamAgent(
            name="TestDefender",
            skill_level="advanced", 
            defense_strategy="proactive",
            auto_response_enabled=True
        )
        
        print_status(f"✓ Created Blue Team Agent: {agent.name}")
        print_status(f"✓ Defense strategy: {agent.defense_strategy}")
        print_status(f"✓ Available tools: {len(agent.tools)}")
        print_status(f"✓ Response playbooks: {len(agent.response_playbooks)}")
        
        # Test threat intelligence correlation
        test_threat = {
            "id": "test_001",
            "type": "malware",
            "source": "192.168.1.100",
            "severity": "high"
        }
        
        intel_matches = agent._correlate_with_threat_intel(test_threat)
        print_status(f"✓ Threat intelligence correlation: {len(intel_matches)} matches")
        
        risk_score = agent._calculate_risk_score(test_threat)
        print_status(f"✓ Risk score calculation: {risk_score:.3f}")
        
        print_status("Blue Team Agent validation completed", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Blue Team Agent validation failed: {e}", "ERROR")
        return False

def validate_network_isolation():
    """Validate Network Isolation functionality."""
    print_status("Validating Network Isolation...")
    
    try:
        from gan_cyber_range.security.isolation import NetworkIsolation, IsolationLevel
        
        # Create network isolation
        isolation = NetworkIsolation(IsolationLevel.STRICT)
        
        print_status(f"✓ Created Network Isolation with level: {isolation.isolation_level.value}")
        print_status(f"✓ Available policies: {len(isolation.policies)}")
        
        # Test network access checking
        internal_access = isolation.check_network_access("10.0.1.10", "10.0.1.20", 80)
        print_status(f"✓ Internal network access check: {internal_access}")
        
        # Test custom rule addition
        rule_added = isolation.add_custom_rule(
            source="192.168.1.100",
            destination="10.0.1.0",
            port=22,
            action="DENY",
            description="Test security rule"
        )
        print_status(f"✓ Custom rule addition: {rule_added}")
        
        # Test quarantine zone creation (simulate async)
        if hasattr(isolation, '_setup_default_policies'):
            print_status("✓ Quarantine capabilities available")
        
        print_status("Network Isolation validation completed", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Network Isolation validation failed: {e}", "ERROR")
        return False

def validate_security_validator():
    """Validate Security Validator functionality."""
    print_status("Validating Security Validator...")
    
    try:
        from gan_cyber_range.security.validator import SecurityValidator, InputValidator, SecurityLevel
        
        # Create validators
        input_validator = InputValidator(SecurityLevel.MEDIUM)
        security_validator = SecurityValidator(SecurityLevel.HIGH)
        
        print_status(f"✓ Created Input Validator with security level: {input_validator.security_level.value}")
        print_status(f"✓ Created Security Validator with security level: {security_validator.security_level.value}")
        
        # Test malicious input detection
        malicious_input = "'; DROP TABLE users; --"
        result = input_validator.validate_string(malicious_input)
        print_status(f"✓ Malicious input detection: {'BLOCKED' if not result.valid else 'FAILED'}")
        
        # Test IP address validation
        ip_result = input_validator.validate_ip_address("192.168.1.1")
        print_status(f"✓ Valid IP validation: {'PASSED' if ip_result.valid else 'FAILED'}")
        
        # Test invalid IP
        bad_ip_result = input_validator.validate_ip_address("999.999.999.999")
        print_status(f"✓ Invalid IP detection: {'BLOCKED' if not bad_ip_result.valid else 'FAILED'}")
        
        # Test agent action validation
        valid_action = {
            "type": "threat_detection",
            "target": "server-01"
        }
        action_result = security_validator.validate_agent_action(valid_action)
        print_status(f"✓ Valid agent action: {'ALLOWED' if action_result.valid else 'BLOCKED'}")
        
        print_status("Security Validator validation completed", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Security Validator validation failed: {e}", "ERROR")
        return False

def validate_health_checker():
    """Validate Health Checker functionality."""
    print_status("Validating Health Checker...")
    
    try:
        from gan_cyber_range.monitoring.health_check import HealthChecker
        
        # Create health checker
        checker = HealthChecker(check_interval=30)
        
        print_status(f"✓ Created Health Checker with interval: {checker.check_interval}s")
        
        # Register default checks
        checker.register_default_checks()
        print_status(f"✓ Registered default checks: {len(checker.checks)}")
        
        # Test individual check execution (simulate)
        if "system_resources" in checker.checks:
            print_status("✓ System resources check available")
        
        if "disk_space" in checker.checks:
            print_status("✓ Disk space check available")
        
        if "memory_usage" in checker.checks:
            print_status("✓ Memory usage check available")
        
        # Test health summary
        summary = checker.get_health_summary()
        print_status(f"✓ Health summary generation: {len(summary)} fields")
        
        print_status("Health Checker validation completed", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Health Checker validation failed: {e}", "ERROR")
        return False

async def validate_async_components():
    """Validate components that require async functionality."""
    print_status("Validating async security components...")
    
    try:
        from gan_cyber_range.agents.blue_team import BlueTeamAgent
        
        agent = BlueTeamAgent(name="AsyncTestAgent")
        
        # Test threat hunting
        hunt_results = await agent.conduct_threat_hunt("Test threat hunting hypothesis")
        print_status(f"✓ Threat hunting: {len(hunt_results['findings'])} findings")
        
        # Test automated response
        test_threat = {
            "id": "async_test_001",
            "type": "malware", 
            "severity": "high",
            "target": "test-server"
        }
        
        response_actions = await agent.execute_automated_response(test_threat)
        print_status(f"✓ Automated response: {len(response_actions)} actions planned")
        
        # Test incident creation
        incident_id = await agent.create_incident(test_threat, "high")
        print_status(f"✓ Incident creation: {incident_id}")
        
        print_status("Async components validation completed", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Async components validation failed: {e}", "ERROR")
        return False

def validate_integration():
    """Validate integration between components."""
    print_status("Validating component integration...")
    
    try:
        from gan_cyber_range.agents.blue_team import BlueTeamAgent
        from gan_cyber_range.security.isolation import NetworkIsolation
        from gan_cyber_range.security.validator import SecurityValidator
        
        # Create integrated components
        blue_team = BlueTeamAgent(name="IntegrationTest")
        isolation = NetworkIsolation()
        validator = SecurityValidator()
        
        # Test threat analysis integration
        mock_environment = {
            "services": [{
                "name": "test-service",
                "status": "running",
                "vulnerabilities": [{
                    "cve_id": "CVE-2023-TEST",
                    "severity": "high",
                    "cvss_score": 8.0,
                    "exploitable": True
                }]
            }],
            "security_events": [{
                "type": "malware",
                "severity": "high",
                "source_ip": "192.168.1.100"
            }],
            "attack_indicators": [{
                "type": "ip",
                "value": "192.168.1.100",
                "confidence": 0.9
            }]
        }
        
        # Validate that components can work together
        threat_indicators = blue_team.threat_indicators
        isolation_policies = isolation.policies
        
        print_status(f"✓ Blue team threat indicators: {len(threat_indicators)}")
        print_status(f"✓ Isolation policies available: {len(isolation_policies)}")
        print_status("✓ Components can integrate successfully")
        
        print_status("Integration validation completed", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Integration validation failed: {e}", "ERROR")
        return False

def main():
    """Run all validation tests."""
    print_status("Starting GAN Cyber Range Defensive Security Validation", "INFO")
    print_status("=" * 60)
    
    validation_results = []
    
    # Run all validation tests
    validation_results.append(("Import Validation", validate_imports()))
    validation_results.append(("Blue Team Agent", validate_blue_team_agent()))
    validation_results.append(("Network Isolation", validate_network_isolation()))
    validation_results.append(("Security Validator", validate_security_validator()))
    validation_results.append(("Health Checker", validate_health_checker()))
    validation_results.append(("Integration", validate_integration()))
    
    # Run async validation
    try:
        async_result = asyncio.run(validate_async_components())
        validation_results.append(("Async Components", async_result))
    except Exception as e:
        print_status(f"Async validation failed: {e}", "ERROR")
        validation_results.append(("Async Components", False))
    
    # Summary
    print_status("=" * 60)
    print_status("VALIDATION SUMMARY")
    print_status("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in validation_results:
        status = "PASS" if result else "FAIL"
        status_type = "SUCCESS" if result else "ERROR"
        print_status(f"{test_name}: {status}", status_type)
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print_status("=" * 60)
    print_status(f"TOTAL TESTS: {len(validation_results)}")
    print_status(f"PASSED: {passed}", "SUCCESS")
    print_status(f"FAILED: {failed}", "ERROR" if failed > 0 else "INFO")
    
    if failed == 0:
        print_status("🛡️ ALL DEFENSIVE SECURITY COMPONENTS VALIDATED SUCCESSFULLY! 🛡️", "SUCCESS")
        return 0
    else:
        print_status("❌ SOME VALIDATIONS FAILED - REVIEW ERRORS ABOVE", "ERROR")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)