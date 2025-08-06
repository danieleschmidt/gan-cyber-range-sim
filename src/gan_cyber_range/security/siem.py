"""Security Information and Event Management (SIEM) system."""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Security event types."""
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    FILE_ACCESS = "file_access"
    PROCESS = "process"
    MALWARE = "malware"
    ANOMALY = "anomaly"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    id: str
    timestamp: datetime
    event_type: EventType
    source_ip: str
    destination_ip: Optional[str] = None
    user: Optional[str] = None
    process: Optional[str] = None
    file_path: Optional[str] = None
    command: Optional[str] = None
    severity: AlertSeverity = AlertSeverity.LOW
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAlert:
    """Security alert generated from events."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    confidence: float
    created_at: datetime
    events: List[SecurityEvent]
    mitre_techniques: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SIEMEngine:
    """Security Information and Event Management engine."""
    
    def __init__(self, max_events: int = 100000, retention_hours: int = 168):
        self.max_events = max_events
        self.retention_hours = retention_hours
        self.events: deque = deque(maxlen=max_events)
        self.alerts: List[SecurityAlert] = []
        self.detection_rules: List[Dict[str, Any]] = []
        self.correlation_rules: List[Dict[str, Any]] = []
        self.threat_indicators: Set[str] = set()
        self.logger = logging.getLogger("SIEMEngine")
        
        # Event statistics
        self.event_stats = defaultdict(int)
        self.alert_stats = defaultdict(int)
        
        # Initialize default rules
        self._initialize_detection_rules()
        self._initialize_correlation_rules()
    
    def _initialize_detection_rules(self) -> None:
        """Initialize default detection rules."""
        self.detection_rules.extend([
            {
                "id": "failed_auth_brute_force",
                "name": "Multiple Failed Authentication Attempts",
                "description": "Detects potential brute force authentication attacks",
                "event_type": EventType.AUTHENTICATION,
                "conditions": {
                    "failed_auth_count": {"threshold": 10, "timeframe_minutes": 5},
                    "source_ip": "same"
                },
                "severity": AlertSeverity.HIGH,
                "mitre_techniques": ["T1110"]
            },
            {
                "id": "suspicious_network_traffic",
                "name": "Suspicious Network Traffic",
                "description": "Detects unusual network traffic patterns",
                "event_type": EventType.NETWORK,
                "conditions": {
                    "data_volume": {"threshold": 100000000, "timeframe_minutes": 10},  # 100MB
                    "unusual_ports": [4444, 5555, 6666, 7777, 8888]
                },
                "severity": AlertSeverity.MEDIUM,
                "mitre_techniques": ["T1041", "T1071"]
            },
            {
                "id": "malware_execution",
                "name": "Malware Execution Detected",
                "description": "Detects known malware signatures or behaviors",
                "event_type": EventType.MALWARE,
                "conditions": {
                    "malware_signature": "match"
                },
                "severity": AlertSeverity.CRITICAL,
                "mitre_techniques": ["T1059", "T1055"]
            },
            {
                "id": "privilege_escalation",
                "name": "Potential Privilege Escalation",
                "description": "Detects attempts to escalate privileges",
                "event_type": EventType.PROCESS,
                "conditions": {
                    "elevated_process": True,
                    "suspicious_commands": ["sudo", "su", "runas", "powershell -ep bypass"]
                },
                "severity": AlertSeverity.HIGH,
                "mitre_techniques": ["T1068", "T1134"]
            },
            {
                "id": "data_exfiltration",
                "name": "Potential Data Exfiltration",
                "description": "Detects potential data exfiltration attempts",
                "event_type": EventType.FILE_ACCESS,
                "conditions": {
                    "large_file_access": {"threshold": 50000000, "timeframe_minutes": 15},  # 50MB
                    "sensitive_files": ["/etc/passwd", "/etc/shadow", "*.key", "*.pem"]
                },
                "severity": AlertSeverity.CRITICAL,
                "mitre_techniques": ["T1041", "T1020"]
            }
        ])
    
    def _initialize_correlation_rules(self) -> None:
        """Initialize event correlation rules."""
        self.correlation_rules.extend([
            {
                "id": "apt_attack_chain",
                "name": "Advanced Persistent Threat Attack Chain",
                "description": "Correlates events indicating APT-style attack progression",
                "event_sequence": [
                    {"type": EventType.NETWORK, "pattern": "reconnaissance"},
                    {"type": EventType.AUTHENTICATION, "pattern": "credential_compromise"},
                    {"type": EventType.PROCESS, "pattern": "lateral_movement"},
                    {"type": EventType.FILE_ACCESS, "pattern": "data_collection"}
                ],
                "timeframe_hours": 24,
                "severity": AlertSeverity.CRITICAL,
                "confidence_threshold": 0.8
            },
            {
                "id": "insider_threat",
                "name": "Potential Insider Threat",
                "description": "Correlates events indicating insider threat activity",
                "event_sequence": [
                    {"type": EventType.AUTHENTICATION, "pattern": "after_hours_access"},
                    {"type": EventType.FILE_ACCESS, "pattern": "unusual_file_access"},
                    {"type": EventType.NETWORK, "pattern": "data_upload"}
                ],
                "timeframe_hours": 48,
                "severity": AlertSeverity.HIGH,
                "confidence_threshold": 0.7
            }
        ])
    
    async def ingest_event(self, event: SecurityEvent) -> None:
        """Ingest a security event for processing."""
        # Add to event store
        self.events.append(event)
        
        # Update statistics
        self.event_stats[event.event_type.value] += 1
        self.event_stats["total"] += 1
        
        # Process event through detection rules
        await self._process_detection_rules(event)
        
        # Process event through correlation rules
        await self._process_correlation_rules(event)
        
        # Update threat indicators
        self._update_threat_indicators(event)
        
        self.logger.debug(f"Ingested event: {event.id}")
    
    async def _process_detection_rules(self, event: SecurityEvent) -> None:
        """Process event against detection rules."""
        for rule in self.detection_rules:
            if rule["event_type"] != event.event_type:
                continue
            
            if await self._evaluate_detection_rule(rule, event):
                alert = await self._create_alert_from_rule(rule, [event])
                await self._generate_alert(alert)
    
    async def _evaluate_detection_rule(self, rule: Dict[str, Any], event: SecurityEvent) -> bool:
        """Evaluate if an event matches a detection rule."""
        conditions = rule.get("conditions", {})
        
        # Check failed authentication count
        if "failed_auth_count" in conditions:
            threshold_config = conditions["failed_auth_count"]
            threshold = threshold_config["threshold"]
            timeframe = threshold_config["timeframe_minutes"]
            
            # Count recent failed auth events from same source
            cutoff_time = datetime.now() - timedelta(minutes=timeframe)
            failed_count = sum(
                1 for e in self.events
                if (e.event_type == EventType.AUTHENTICATION and
                    e.timestamp > cutoff_time and
                    e.source_ip == event.source_ip and
                    e.metadata.get("auth_result") == "failed")
            )
            
            if failed_count >= threshold:
                return True
        
        # Check suspicious network traffic
        if "data_volume" in conditions or "unusual_ports" in conditions:
            if event.event_type == EventType.NETWORK:
                # Check data volume
                if "data_volume" in conditions:
                    volume_config = conditions["data_volume"]
                    if event.metadata.get("bytes_transferred", 0) > volume_config["threshold"]:
                        return True
                
                # Check unusual ports
                if "unusual_ports" in conditions:
                    port = event.metadata.get("destination_port")
                    if port in conditions["unusual_ports"]:
                        return True
        
        # Check malware signature
        if "malware_signature" in conditions:
            if event.event_type == EventType.MALWARE:
                return True
        
        # Check privilege escalation
        if "elevated_process" in conditions or "suspicious_commands" in conditions:
            if event.event_type == EventType.PROCESS:
                if conditions.get("elevated_process") and event.metadata.get("elevated"):
                    return True
                
                command = event.command or ""
                for suspicious_cmd in conditions.get("suspicious_commands", []):
                    if suspicious_cmd.lower() in command.lower():
                        return True
        
        # Check data exfiltration
        if "large_file_access" in conditions or "sensitive_files" in conditions:
            if event.event_type == EventType.FILE_ACCESS:
                # Check file size
                if "large_file_access" in conditions:
                    size_config = conditions["large_file_access"]
                    if event.metadata.get("file_size", 0) > size_config["threshold"]:
                        return True
                
                # Check sensitive file patterns
                if "sensitive_files" in conditions:
                    file_path = event.file_path or ""
                    for pattern in conditions["sensitive_files"]:
                        if pattern.replace("*", "") in file_path:
                            return True
        
        return False
    
    async def _process_correlation_rules(self, event: SecurityEvent) -> None:
        """Process event against correlation rules."""
        for rule in self.correlation_rules:
            if await self._evaluate_correlation_rule(rule, event):
                related_events = self._find_related_events(rule, event)
                alert = await self._create_alert_from_correlation(rule, related_events)
                await self._generate_alert(alert)
    
    async def _evaluate_correlation_rule(self, rule: Dict[str, Any], event: SecurityEvent) -> bool:
        """Evaluate if events match a correlation rule."""
        event_sequence = rule.get("event_sequence", [])
        timeframe_hours = rule.get("timeframe_hours", 24)
        confidence_threshold = rule.get("confidence_threshold", 0.5)
        
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        # Check if we have events matching the sequence
        sequence_matches = 0
        for sequence_item in event_sequence:
            required_type = sequence_item["type"]
            pattern = sequence_item["pattern"]
            
            # Look for matching events in the timeframe
            for stored_event in self.events:
                if (stored_event.timestamp > cutoff_time and
                    stored_event.event_type == required_type):
                    
                    # Simple pattern matching (could be enhanced)
                    if self._matches_pattern(stored_event, pattern):
                        sequence_matches += 1
                        break
        
        # Calculate confidence based on sequence completion
        confidence = sequence_matches / len(event_sequence)
        return confidence >= confidence_threshold
    
    def _matches_pattern(self, event: SecurityEvent, pattern: str) -> bool:
        """Check if event matches a correlation pattern."""
        # Simple pattern matching - could be enhanced with regex or ML
        if pattern == "reconnaissance":
            return "scan" in event.raw_data.get("activity", "").lower()
        elif pattern == "credential_compromise":
            return event.metadata.get("auth_result") == "success" and event.metadata.get("unusual_login", False)
        elif pattern == "lateral_movement":
            return "remote" in event.command.lower() if event.command else False
        elif pattern == "data_collection":
            return event.metadata.get("file_size", 0) > 1000000  # > 1MB
        elif pattern == "after_hours_access":
            hour = event.timestamp.hour
            return hour < 6 or hour > 22  # Outside business hours
        elif pattern == "unusual_file_access":
            return event.metadata.get("unusual_access", False)
        elif pattern == "data_upload":
            return event.metadata.get("direction") == "outbound"
        
        return False
    
    def _find_related_events(self, rule: Dict[str, Any], trigger_event: SecurityEvent) -> List[SecurityEvent]:
        """Find events related to correlation rule."""
        timeframe_hours = rule.get("timeframe_hours", 24)
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        related_events = [trigger_event]
        
        # Find other events in the correlation timeframe
        for event in self.events:
            if (event.timestamp > cutoff_time and
                event.id != trigger_event.id and
                (event.source_ip == trigger_event.source_ip or
                 event.user == trigger_event.user)):
                related_events.append(event)
        
        return related_events[:10]  # Limit to 10 related events
    
    async def _create_alert_from_rule(self, rule: Dict[str, Any], events: List[SecurityEvent]) -> SecurityAlert:
        """Create alert from detection rule match."""
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.alerts) + 1:03d}"
        
        return SecurityAlert(
            id=alert_id,
            title=rule["name"],
            description=rule["description"],
            severity=rule["severity"],
            confidence=0.9,  # High confidence for rule-based detection
            created_at=datetime.now(),
            events=events,
            mitre_techniques=rule.get("mitre_techniques", []),
            indicators=self._extract_indicators(events),
            recommended_actions=self._generate_recommendations(rule, events),
            metadata={
                "rule_id": rule["id"],
                "detection_type": "rule_based"
            }
        )
    
    async def _create_alert_from_correlation(self, rule: Dict[str, Any], events: List[SecurityEvent]) -> SecurityAlert:
        """Create alert from correlation rule match."""
        alert_id = f"CORR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.alerts) + 1:03d}"
        
        confidence = min(len(events) / len(rule.get("event_sequence", [])), 1.0)
        
        return SecurityAlert(
            id=alert_id,
            title=rule["name"],
            description=rule["description"],
            severity=rule["severity"],
            confidence=confidence,
            created_at=datetime.now(),
            events=events,
            mitre_techniques=self._extract_mitre_techniques(events),
            indicators=self._extract_indicators(events),
            recommended_actions=self._generate_recommendations(rule, events),
            metadata={
                "rule_id": rule["id"],
                "detection_type": "correlation_based",
                "sequence_completion": confidence
            }
        )
    
    async def _generate_alert(self, alert: SecurityAlert) -> None:
        """Generate and store a security alert."""
        self.alerts.append(alert)
        self.alert_stats[alert.severity.value] += 1
        self.alert_stats["total"] += 1
        
        self.logger.warning(
            f"SECURITY ALERT [{alert.severity.value.upper()}]: {alert.title} "
            f"(ID: {alert.id}, Confidence: {alert.confidence:.2f})"
        )
        
        # Could trigger external notifications here
        await self._notify_alert(alert)
    
    async def _notify_alert(self, alert: SecurityAlert) -> None:
        """Send alert notifications."""
        # Placeholder for external notification logic
        # Could integrate with Slack, email, SOAR platforms, etc.
        notification_data = {
            "alert_id": alert.id,
            "title": alert.title,
            "severity": alert.severity.value,
            "confidence": alert.confidence,
            "created_at": alert.created_at.isoformat(),
            "indicators": alert.indicators[:5],  # First 5 indicators
            "recommended_actions": alert.recommended_actions[:3]  # Top 3 actions
        }
        
        self.logger.info(f"Alert notification: {json.dumps(notification_data)}")
    
    def _extract_indicators(self, events: List[SecurityEvent]) -> List[str]:
        """Extract indicators of compromise from events."""
        indicators = set()
        
        for event in events:
            # IP addresses
            if event.source_ip:
                indicators.add(f"ip:{event.source_ip}")
            if event.destination_ip:
                indicators.add(f"ip:{event.destination_ip}")
            
            # File paths
            if event.file_path:
                indicators.add(f"file:{event.file_path}")
            
            # Commands
            if event.command:
                indicators.add(f"command:{event.command}")
            
            # Custom indicators from metadata
            for key, value in event.metadata.items():
                if key in ["hash", "domain", "url", "registry_key"]:
                    indicators.add(f"{key}:{value}")
        
        return list(indicators)[:20]  # Limit to 20 indicators
    
    def _extract_mitre_techniques(self, events: List[SecurityEvent]) -> List[str]:
        """Extract MITRE ATT&CK techniques from events."""
        techniques = set()
        
        # Map event types to common MITRE techniques
        technique_mapping = {
            EventType.AUTHENTICATION: ["T1110", "T1078"],
            EventType.NETWORK: ["T1071", "T1041"],
            EventType.FILE_ACCESS: ["T1005", "T1020"],
            EventType.PROCESS: ["T1059", "T1055"],
            EventType.MALWARE: ["T1059", "T1204"],
            EventType.ANOMALY: ["T1562", "T1070"]
        }
        
        for event in events:
            event_techniques = technique_mapping.get(event.event_type, [])
            techniques.update(event_techniques)
        
        return list(techniques)
    
    def _generate_recommendations(self, rule: Dict[str, Any], events: List[SecurityEvent]) -> List[str]:
        """Generate recommended actions for alerts."""
        recommendations = []
        
        rule_id = rule.get("id", "")
        severity = rule.get("severity", AlertSeverity.MEDIUM)
        
        # Generic recommendations based on severity
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            recommendations.extend([
                "Immediately investigate the affected systems",
                "Consider isolating compromised hosts from the network",
                "Review and collect forensic evidence"
            ])
        
        # Specific recommendations based on rule type
        if "brute_force" in rule_id:
            recommendations.extend([
                "Block source IP addresses",
                "Lock affected user accounts",
                "Review authentication logs for successful logins"
            ])
        elif "malware" in rule_id:
            recommendations.extend([
                "Run full antimalware scan on affected systems",
                "Check for persistence mechanisms",
                "Update antimalware signatures"
            ])
        elif "data_exfiltration" in rule_id:
            recommendations.extend([
                "Block outbound connections to suspicious destinations",
                "Review data access logs",
                "Assess potential data impact"
            ])
        elif "privilege_escalation" in rule_id:
            recommendations.extend([
                "Review user privileges and access controls",
                "Check for unauthorized administrative access",
                "Audit system configuration changes"
            ])
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _update_threat_indicators(self, event: SecurityEvent) -> None:
        """Update threat indicator database from events."""
        # Add source IP as potential threat indicator
        if event.source_ip and event.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self.threat_indicators.add(event.source_ip)
        
        # Add file hashes, domains, etc. from metadata
        for key, value in event.metadata.items():
            if key in ["file_hash", "domain", "url"] and isinstance(value, str):
                self.threat_indicators.add(value)
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alerts
            if alert.created_at > cutoff_time
        ]
        
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            "total_alerts": len(recent_alerts),
            "severity_breakdown": dict(severity_counts),
            "top_indicators": list(self.threat_indicators)[-10:],  # Last 10 indicators
            "alert_rate_per_hour": len(recent_alerts) / hours if hours > 0 else 0,
            "summary_period_hours": hours
        }
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event ingestion statistics."""
        return {
            "total_events": len(self.events),
            "event_types": dict(self.event_stats),
            "events_per_hour": len(self.events) / max(self.retention_hours, 1),
            "storage_utilization": len(self.events) / self.max_events
        }
    
    async def cleanup_old_data(self) -> None:
        """Clean up old events and alerts based on retention policy."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Clean up old alerts
        initial_alert_count = len(self.alerts)
        self.alerts = [
            alert for alert in self.alerts
            if alert.created_at > cutoff_time
        ]
        cleaned_alerts = initial_alert_count - len(self.alerts)
        
        # Note: Events are automatically cleaned up by deque maxlen
        
        if cleaned_alerts > 0:
            self.logger.info(f"Cleaned up {cleaned_alerts} old alerts")
    
    def add_detection_rule(self, rule: Dict[str, Any]) -> bool:
        """Add a custom detection rule."""
        try:
            # Validate required fields
            required_fields = ["id", "name", "description", "event_type", "conditions", "severity"]
            if not all(field in rule for field in required_fields):
                self.logger.error("Detection rule missing required fields")
                return False
            
            # Check for duplicate rule ID
            existing_ids = [r["id"] for r in self.detection_rules]
            if rule["id"] in existing_ids:
                self.logger.error(f"Detection rule ID already exists: {rule['id']}")
                return False
            
            self.detection_rules.append(rule)
            self.logger.info(f"Added detection rule: {rule['id']}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to add detection rule: {e}")
            return False
    
    def remove_detection_rule(self, rule_id: str) -> bool:
        """Remove a detection rule."""
        initial_count = len(self.detection_rules)
        self.detection_rules = [
            rule for rule in self.detection_rules
            if rule["id"] != rule_id
        ]
        
        removed = initial_count - len(self.detection_rules)
        if removed > 0:
            self.logger.info(f"Removed detection rule: {rule_id}")
            return True
        else:
            self.logger.warning(f"Detection rule not found: {rule_id}")
            return False