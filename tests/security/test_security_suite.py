"""Comprehensive security testing suite for defensive capabilities."""

import asyncio
import pytest
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Import our security modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from gan_cyber_range.security.siem import SIEMEngine, SecurityEvent, EventType, AlertSeverity
from gan_cyber_range.security.incident_response import IncidentResponseOrchestrator, IncidentPriority
from gan_cyber_range.security.threat_detection import ThreatDetectionEngine, BehavioralAnalyzer
from gan_cyber_range.security.isolation import NetworkIsolation, IsolationLevel
from gan_cyber_range.security.validator import SecurityValidator, InputValidator, SecurityLevel
from gan_cyber_range.agents.blue_team import BlueTeamAgent
from gan_cyber_range.monitoring.health_check import HealthChecker


class TestSIEMEngine:
    """Test SIEM functionality."""
    
    @pytest.fixture
    def siem_engine(self):
        return SIEMEngine(max_events=1000, retention_hours=24)
    
    @pytest.fixture
    def sample_security_event(self):
        return SecurityEvent(
            id="test_event_001",
            timestamp=datetime.now(),
            event_type=EventType.AUTHENTICATION,
            source_ip="192.168.1.100",
            destination_ip="10.0.1.5",
            user="test_user",
            severity=AlertSeverity.MEDIUM,
            raw_data={"auth_result": "failed", "service": "ssh"},
            metadata={"attempt_count": 5}
        )
    
    @pytest.mark.asyncio
    async def test_event_ingestion(self, siem_engine, sample_security_event):
        """Test event ingestion and processing."""
        await siem_engine.ingest_event(sample_security_event)
        
        assert len(siem_engine.events) == 1
        assert siem_engine.event_stats["authentication"] == 1
        assert siem_engine.event_stats["total"] == 1
    
    @pytest.mark.asyncio
    async def test_detection_rule_evaluation(self, siem_engine):
        """Test detection rule evaluation."""
        # Create multiple failed auth events to trigger brute force detection
        for i in range(12):
            event = SecurityEvent(
                id=f"auth_fail_{i}",
                timestamp=datetime.now(),
                event_type=EventType.AUTHENTICATION,
                source_ip="192.168.1.100",
                metadata={"auth_result": "failed"}
            )
            await siem_engine.ingest_event(event)
        
        # Should have generated an alert
        assert len(siem_engine.alerts) > 0
        alert = siem_engine.alerts[0]
        assert "brute force" in alert.title.lower() or "authentication" in alert.title.lower()
        assert alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    def test_custom_detection_rule(self, siem_engine):
        """Test adding custom detection rules."""
        custom_rule = {
            "id": "test_custom_rule",
            "name": "Test Custom Detection",
            "description": "Test rule for validation",
            "event_type": EventType.NETWORK,
            "conditions": {"test_condition": True},
            "severity": AlertSeverity.MEDIUM,
            "mitre_techniques": ["T1000"]
        }
        
        result = siem_engine.add_detection_rule(custom_rule)
        assert result is True
        assert len(siem_engine.detection_rules) > 0
        
        # Test duplicate rule ID
        duplicate_result = siem_engine.add_detection_rule(custom_rule)
        assert duplicate_result is False
    
    def test_alert_summary(self, siem_engine):
        """Test alert summary generation."""
        summary = siem_engine.get_alert_summary(24)
        
        assert "total_alerts" in summary
        assert "severity_breakdown" in summary
        assert "alert_rate_per_hour" in summary
        assert summary["summary_period_hours"] == 24


class TestIncidentResponse:
    """Test incident response orchestration."""
    
    @pytest.fixture
    def incident_orchestrator(self):
        return IncidentResponseOrchestrator()
    
    @pytest.mark.asyncio
    async def test_incident_creation(self, incident_orchestrator):
        """Test security incident creation."""
        incident = await incident_orchestrator.create_incident(
            title="Test Security Incident",
            description="Test incident for validation",
            severity="high",
            detection_source="test_suite",
            detection_method="automated",
            confidence=0.9,
            affected_assets=["server-01", "database-01"],
            indicators_of_compromise=["ip:192.168.1.100", "file:/tmp/malware.exe"],
            mitre_techniques=["T1059", "T1055"]
        )
        
        assert incident.title == "Test Security Incident"
        assert incident.severity == "high"
        assert incident.priority == IncidentPriority.P2
        assert len(incident.affected_assets) == 2
        assert len(incident.indicators_of_compromise) == 2
        assert len(incident.mitre_techniques) == 2
        assert incident.id in incident_orchestrator.active_incidents
    
    @pytest.mark.asyncio
    async def test_automated_response_execution(self, incident_orchestrator):
        """Test automated response execution."""
        incident = await incident_orchestrator.create_incident(
            title="Malware Detection",
            description="Malware detected on endpoint",
            severity="critical",
            detection_source="edr",
            detection_method="signature",
            confidence=0.95,
            affected_assets=["workstation-05"],
            mitre_techniques=["T1059"]
        )
        
        # Wait a moment for automated response to execute
        await asyncio.sleep(0.1)
        
        assert len(incident.tasks) > 0
        # Should have automated tasks from malware_infection playbook
        automated_tasks = [task for task in incident.tasks if task.assigned_to == "automation"]
        assert len(automated_tasks) > 0
    
    def test_priority_calculation(self, incident_orchestrator):
        """Test incident priority calculation."""
        # Critical severity should result in P1
        p1_priority = incident_orchestrator._calculate_priority("critical", [], 0.9)
        assert p1_priority == IncidentPriority.P1
        
        # High severity with critical assets should upgrade priority
        p2_priority = incident_orchestrator._calculate_priority("medium", ["database_server"], 0.8)
        assert p2_priority == IncidentPriority.P2
    
    def test_incident_summary(self, incident_orchestrator):
        """Test incident summary generation."""
        summary = incident_orchestrator.get_incident_summary()
        
        assert "active_incidents" in summary
        assert "status_breakdown" in summary
        assert "metrics" in summary
        assert "recent_incidents" in summary


class TestThreatDetection:
    """Test behavioral threat detection."""
    
    @pytest.fixture
    def behavioral_analyzer(self):
        return BehavioralAnalyzer()
    
    @pytest.fixture
    def threat_detection_engine(self):
        return ThreatDetectionEngine()
    
    @pytest.fixture
    def suspicious_events(self):
        """Generate suspicious event patterns."""
        base_time = datetime.now()
        events = []
        
        # Credential stuffing pattern
        for i in range(15):
            events.append({
                "id": f"auth_fail_{i}",
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                "event_type": "authentication",
                "source_ip": "192.168.1.100",
                "service": f"service_{i % 3}",
                "auth_result": "failed"
            })
        
        # Data hoarding pattern
        for i in range(5):
            events.append({
                "id": f"data_access_{i}",
                "timestamp": (base_time + timedelta(minutes=30 + i)).isoformat(),
                "event_type": "file_access",
                "file_size": 50000000,  # 50MB
                "data_source": f"database_{i}",
                "timestamp": base_time.replace(hour=2).isoformat()  # Off hours
            })
        
        return events
    
    @pytest.mark.asyncio
    async def test_behavioral_pattern_detection(self, behavioral_analyzer, suspicious_events):
        """Test behavioral pattern detection."""
        detections = await behavioral_analyzer.analyze_event_sequence(suspicious_events)
        
        assert len(detections) > 0
        
        # Should detect credential stuffing and data hoarding
        detected_patterns = [d.name for d in detections]
        assert any("credential" in pattern.lower() for pattern in detected_patterns)
        assert any("data" in pattern.lower() for pattern in detected_patterns)
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, behavioral_analyzer):
        """Test statistical anomaly detection."""
        # Create events with volume anomaly
        events = []
        base_time = datetime.now()
        
        # Normal events
        for i in range(20):
            events.append({
                "id": f"normal_{i}",
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                "data_size": 1000000,  # 1MB
                "event_type": "network"
            })
        
        # Anomalous large transfer
        events.append({
            "id": "anomaly_1",
            "timestamp": (base_time + timedelta(minutes=21)).isoformat(),
            "data_size": 100000000,  # 100MB - much larger
            "event_type": "network"
        })
        
        detections = await behavioral_analyzer.analyze_event_sequence(events)
        
        # Should detect volume anomaly
        volume_detections = [d for d in detections if "volume" in d.name.lower()]
        assert len(volume_detections) > 0
    
    @pytest.mark.asyncio
    async def test_threat_intelligence_correlation(self, threat_detection_engine):
        """Test threat intelligence correlation."""
        events = [{
            "id": "malicious_conn_1",
            "timestamp": datetime.now().isoformat(),
            "source_ip": "192.168.1.100",  # Known malicious IP in test data
            "event_type": "network"
        }]
        
        detections = await threat_detection_engine.process_events(events)
        
        intel_detections = [d for d in detections if "intelligence" in d.name.lower()]
        assert len(intel_detections) > 0


class TestNetworkIsolation:
    """Test network isolation and containment."""
    
    @pytest.fixture
    def network_isolation(self):
        return NetworkIsolation(IsolationLevel.STRICT)
    
    def test_isolation_policy_application(self, network_isolation):
        """Test isolation policy application."""
        # Test applying strict policy
        result = asyncio.run(
            network_isolation.apply_isolation_policy("strict", "test-namespace")
        )
        assert result is True
        
        # Verify active rules were created
        assert len(network_isolation.active_rules) > 0
    
    def test_network_access_validation(self, network_isolation):
        """Test network access checking."""
        # Apply strict policy first
        asyncio.run(
            network_isolation.apply_isolation_policy("strict", "test-namespace")
        )
        
        # Test internal-to-internal access (should be allowed by some rules)
        internal_access = network_isolation.check_network_access(
            "10.0.1.10", "10.0.1.20", 80
        )
        
        # Test external access (should be denied by strict policy)
        external_access = network_isolation.check_network_access(
            "192.168.1.100", "8.8.8.8", 443
        )
        
        # With strict policy, external access should be denied
        assert external_access is False
    
    @pytest.mark.asyncio
    async def test_dynamic_quarantine(self, network_isolation):
        """Test dynamic quarantine zone creation."""
        zone_id = await network_isolation.create_dynamic_quarantine_zone(
            threat_type="malware",
            affected_hosts=["server-01", "server-02"],
            duration_hours=24
        )
        
        assert zone_id is not None
        assert zone_id.startswith("quarantine_")
        assert hasattr(network_isolation, 'quarantine_zones')
        assert len(network_isolation.quarantine_zones) == 1
    
    def test_custom_rule_management(self, network_isolation):
        """Test custom rule management."""
        # Add custom rule
        result = network_isolation.add_custom_rule(
            source="192.168.1.100",
            destination="10.0.1.0/24",
            port=22,
            action="DENY",
            description="Block SSH from suspicious IP"
        )
        assert result is True
        
        # Remove custom rule
        remove_result = network_isolation.remove_custom_rule("Block SSH from suspicious IP")
        assert remove_result is True


class TestSecurityValidation:
    """Test input validation and security checks."""
    
    @pytest.fixture
    def security_validator(self):
        return SecurityValidator(SecurityLevel.HIGH)
    
    @pytest.fixture
    def input_validator(self):
        return InputValidator(SecurityLevel.MEDIUM)
    
    def test_malicious_input_detection(self, input_validator):
        """Test malicious input detection."""
        # Test SQL injection patterns
        sql_injection = "'; DROP TABLE users; --"
        result = input_validator.validate_string(sql_injection)
        assert result.valid is False
        assert result.level.value in ["critical", "high"]
        
        # Test XSS patterns
        xss_payload = "<script>alert('xss')</script>"
        result = input_validator.validate_string(xss_payload)
        assert result.valid is False
        
        # Test command injection
        cmd_injection = "test; rm -rf /"
        result = input_validator.validate_string(cmd_injection)
        assert result.valid is False
    
    def test_ip_address_validation(self, input_validator):
        """Test IP address validation."""
        # Valid IP
        valid_ip = input_validator.validate_ip_address("192.168.1.1")
        assert valid_ip.valid is True
        
        # Invalid IP
        invalid_ip = input_validator.validate_ip_address("999.999.999.999")
        assert invalid_ip.valid is False
        
        # Malformed IP
        malformed_ip = input_validator.validate_ip_address("not.an.ip.address")
        assert malformed_ip.valid is False
    
    def test_agent_action_validation(self, security_validator):
        """Test agent action validation."""
        # Valid action
        valid_action = {
            "type": "threat_detection",
            "target": "server-01",
            "payload": {"scan_type": "full"}
        }
        result = security_validator.validate_agent_action(valid_action)
        assert result.valid is True
        
        # Invalid action type
        invalid_action = {
            "type": "malicious_action",
            "target": "server-01"
        }
        result = security_validator.validate_agent_action(invalid_action)
        assert result.valid is False
    
    def test_kubernetes_resource_validation(self, security_validator):
        """Test Kubernetes resource validation."""
        # Valid resource
        valid_resource = {
            "metadata": {"name": "test-pod"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "test-container",
                            "securityContext": {
                                "privileged": False,
                                "allowPrivilegeEscalation": False
                            }
                        }]
                    }
                }
            }
        }
        result = security_validator.validate_kubernetes_resource(valid_resource)
        assert result.valid is True
        
        # Privileged container (should be rejected)
        privileged_resource = {
            "metadata": {"name": "privileged-pod"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "privileged-container",
                            "securityContext": {"privileged": True}
                        }]
                    }
                }
            }
        }
        result = security_validator.validate_kubernetes_resource(privileged_resource)
        assert result.valid is False
        assert result.level.value == "critical"


class TestBlueTeamAgent:
    """Test Blue Team agent defensive capabilities."""
    
    @pytest.fixture
    def blue_team_agent(self):
        return BlueTeamAgent(
            name="TestDefender",
            skill_level="advanced",
            defense_strategy="proactive",
            auto_response_enabled=True
        )
    
    @pytest.fixture
    def threat_environment(self):
        """Mock threat environment for testing."""
        return {
            "services": [
                {
                    "name": "web-server",
                    "status": "running",
                    "vulnerabilities": [{
                        "cve_id": "CVE-2023-1234",
                        "severity": "high",
                        "cvss_score": 8.5,
                        "exploitable": True,
                        "patch_available": True
                    }]
                }
            ],
            "security_events": [
                {
                    "type": "sql_injection",
                    "severity": "high",
                    "source_ip": "192.168.1.100",
                    "target": "web-server",
                    "timestamp": datetime.now().isoformat(),
                    "indicators_count": 5
                }
            ],
            "attack_indicators": [
                {
                    "type": "ip",
                    "value": "192.168.1.100",
                    "confidence": 0.9
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_threat_analysis(self, blue_team_agent, threat_environment):
        """Test threat analysis capabilities."""
        analysis = await blue_team_agent.analyze_environment(threat_environment)
        
        assert "active_threats" in analysis
        assert "vulnerabilities" in analysis
        assert "defense_priorities" in analysis
        assert len(analysis["active_threats"]) > 0
        assert len(analysis["vulnerabilities"]) > 0
    
    @pytest.mark.asyncio
    async def test_action_planning(self, blue_team_agent, threat_environment):
        """Test defensive action planning."""
        analysis = await blue_team_agent.analyze_environment(threat_environment)
        actions = await blue_team_agent.plan_actions(analysis)
        
        assert len(actions) > 0
        
        # Should plan threat response actions
        threat_responses = [a for a in actions if "threat" in a.type or "incident" in a.type]
        assert len(threat_responses) > 0
    
    @pytest.mark.asyncio
    async def test_automated_response(self, blue_team_agent):
        """Test automated response capabilities."""
        threat = {
            "id": "threat_001",
            "type": "malware",
            "severity": "high",
            "target": "workstation-05",
            "confidence": 0.9
        }
        
        actions = await blue_team_agent.execute_automated_response(threat)
        assert len(actions) > 0
        
        # Should include isolation and evidence collection
        action_types = [a.type for a in actions]
        assert any("isolate" in action_type for action_type in action_types)
    
    @pytest.mark.asyncio
    async def test_threat_hunting(self, blue_team_agent):
        """Test proactive threat hunting."""
        hunt_results = await blue_team_agent.conduct_threat_hunt(
            "Suspicious PowerShell activity in environment"
        )
        
        assert "hypothesis" in hunt_results
        assert "findings" in hunt_results
        assert "confidence_score" in hunt_results
        assert len(hunt_results["findings"]) > 0
    
    @pytest.mark.asyncio
    async def test_incident_management(self, blue_team_agent):
        """Test incident creation and management."""
        threat = {
            "id": "threat_002",
            "type": "data_exfiltration",
            "severity": "critical"
        }
        
        incident_id = await blue_team_agent.create_incident(threat, "critical")
        assert incident_id.startswith("INC-")
        
        # Test incident updates
        update_success = await blue_team_agent.update_incident(
            incident_id, "containment_initiated", "Network isolation applied"
        )
        assert update_success is True


class TestHealthMonitoring:
    """Test system health monitoring."""
    
    @pytest.fixture
    def health_checker(self):
        checker = HealthChecker(check_interval=1)
        checker.register_default_checks()
        return checker
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self, health_checker):
        """Test health check execution."""
        # Run system resources check
        result = await health_checker.run_check("system_resources")
        
        assert result.name == "system_resources"
        assert result.status.value in ["healthy", "degraded", "unhealthy"]
        assert result.response_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_all_health_checks(self, health_checker):
        """Test running all health checks."""
        results = await health_checker.run_all_checks()
        
        assert len(results) > 0
        assert "system_resources" in results
        assert "disk_space" in results
        assert "memory_usage" in results
        
        for check_name, result in results.items():
            assert result.status.value in ["healthy", "degraded", "unhealthy"]
    
    def test_overall_health_status(self, health_checker):
        """Test overall health status calculation."""
        # Mock some results
        health_checker.results = {
            "check1": Mock(status=health_checker.HealthStatus.HEALTHY),
            "check2": Mock(status=health_checker.HealthStatus.DEGRADED)
        }
        
        overall_status = health_checker.get_overall_health()
        assert overall_status.value == "degraded"  # Should reflect worst status
    
    def test_health_summary(self, health_checker):
        """Test health summary generation."""
        summary = health_checker.get_health_summary()
        
        assert "overall_status" in summary
        assert "total_checks" in summary
        assert "checks" in summary


class TestIntegrationScenarios:
    """Integration tests for complete defensive scenarios."""
    
    @pytest.fixture
    def defensive_stack(self):
        """Create integrated defensive stack."""
        return {
            "siem": SIEMEngine(),
            "incident_response": IncidentResponseOrchestrator(),
            "threat_detection": ThreatDetectionEngine(),
            "network_isolation": NetworkIsolation(IsolationLevel.STRICT),
            "blue_team": BlueTeamAgent(auto_response_enabled=True)
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_threat_response(self, defensive_stack):
        """Test end-to-end threat detection and response."""
        siem = defensive_stack["siem"]
        incident_response = defensive_stack["incident_response"]
        blue_team = defensive_stack["blue_team"]
        
        # 1. Ingest malicious events
        malicious_event = SecurityEvent(
            id="malware_detection_001",
            timestamp=datetime.now(),
            event_type=EventType.MALWARE,
            source_ip="192.168.1.100",
            severity=AlertSeverity.CRITICAL,
            metadata={"malware_type": "trojan", "confidence": 0.95}
        )
        
        await siem.ingest_event(malicious_event)
        
        # 2. Should generate SIEM alert
        assert len(siem.alerts) > 0
        
        # 3. Create incident from alert
        siem_alert = siem.alerts[0]
        incident = await incident_response.create_incident(
            title=siem_alert.title,
            description=siem_alert.description,
            severity="critical",
            detection_source="siem",
            detection_method="signature",
            confidence=0.95,
            affected_assets=["workstation-05"]
        )
        
        # 4. Should trigger automated response
        assert len(incident.tasks) > 0
        
        # 5. Blue team should analyze and respond
        threat_environment = {
            "services": [],
            "security_events": [{"type": "malware", "severity": "critical"}],
            "attack_indicators": []
        }
        
        analysis = await blue_team.analyze_environment(threat_environment)
        actions = await blue_team.plan_actions(analysis)
        
        assert len(actions) > 0
        
        # Integration successful if we have:
        # - SIEM alert generated
        # - Incident created with automated tasks
        # - Blue team analysis and action planning
        assert len(siem.alerts) > 0
        assert len(incident.tasks) > 0
        assert len(actions) > 0
    
    @pytest.mark.asyncio
    async def test_multi_stage_attack_detection(self, defensive_stack):
        """Test detection of multi-stage attack progression."""
        siem = defensive_stack["siem"]
        threat_detection = defensive_stack["threat_detection"]
        
        # Simulate multi-stage attack events
        attack_events = [
            # Stage 1: Reconnaissance
            {
                "id": "recon_1",
                "timestamp": datetime.now().isoformat(),
                "event_type": "network",
                "source_ip": "192.168.1.100",
                "network_scan": True
            },
            # Stage 2: Initial compromise
            {
                "id": "compromise_1",
                "timestamp": (datetime.now() + timedelta(minutes=30)).isoformat(),
                "event_type": "authentication",
                "source_ip": "192.168.1.100",
                "auth_result": "success",
                "unusual_login": True
            },
            # Stage 3: Lateral movement
            {
                "id": "lateral_1",
                "timestamp": (datetime.now() + timedelta(hours=1)).isoformat(),
                "event_type": "process",
                "source_ip": "192.168.1.100",
                "remote_access": True,
                "privilege_change": True
            },
            # Stage 4: Data collection
            {
                "id": "collection_1",
                "timestamp": (datetime.now() + timedelta(hours=2)).isoformat(),
                "event_type": "file_access",
                "source_ip": "192.168.1.100",
                "file_size": 50000000,
                "unusual_access": True
            }
        ]
        
        # Process events through threat detection
        detections = await threat_detection.process_events(attack_events)
        
        # Should detect APT-style attack patterns
        apt_detections = [d for d in detections if "apt" in d.name.lower() or "advanced" in d.name.lower()]
        assert len(apt_detections) > 0 or len(detections) >= 2  # Multiple stage detection
    
    def test_performance_under_load(self, defensive_stack):
        """Test defensive system performance under load."""
        import time
        siem = defensive_stack["siem"]
        
        # Generate large number of events
        start_time = time.time()
        
        events = []
        for i in range(1000):
            event = SecurityEvent(
                id=f"load_test_{i}",
                timestamp=datetime.now(),
                event_type=EventType.NETWORK,
                source_ip=f"192.168.1.{i % 255}",
                severity=AlertSeverity.LOW
            )
            events.append(event)
        
        # Process events
        async def process_events():
            for event in events:
                await siem.ingest_event(event)
        
        asyncio.run(process_events())
        
        processing_time = time.time() - start_time
        
        # Should process 1000 events in reasonable time (< 10 seconds)
        assert processing_time < 10.0
        assert len(siem.events) == 1000
        assert siem.event_stats["total"] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])