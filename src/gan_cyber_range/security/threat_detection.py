"""Advanced threat detection and behavioral analysis."""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque


class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(Enum):
    """Threat detection methods."""
    SIGNATURE = "signature"
    BEHAVIORAL = "behavioral"
    ANOMALY = "anomaly"
    MACHINE_LEARNING = "machine_learning"
    THREAT_INTELLIGENCE = "threat_intelligence"


@dataclass
class ThreatIndicator:
    """Indicator of compromise."""
    id: str
    type: str  # ip, domain, hash, url, etc.
    value: str
    confidence: float
    source: str
    first_seen: datetime
    last_seen: datetime
    context: Dict[str, Any]


@dataclass
class BehavioralPattern:
    """Behavioral pattern for analysis."""
    name: str
    description: str
    indicators: List[str]
    threshold: float
    timeframe_minutes: int
    severity: ThreatLevel


@dataclass 
class ThreatDetection:
    """Threat detection result."""
    id: str
    name: str
    description: str
    severity: ThreatLevel
    confidence: float
    method: DetectionMethod
    timestamp: datetime
    indicators: List[ThreatIndicator]
    affected_assets: List[str]
    mitre_techniques: List[str]
    kill_chain_stage: str
    recommended_actions: List[str]
    metadata: Dict[str, Any]


class BehavioralAnalyzer:
    """Behavioral analysis engine for threat detection."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.event_buffer = deque(maxlen=window_size)
        self.behavioral_patterns = []
        self.baseline_metrics = {}
        self.anomaly_thresholds = {}
        self.logger = logging.getLogger("BehavioralAnalyzer")
        
        self._initialize_patterns()
    
    def _initialize_patterns(self) -> None:
        """Initialize behavioral detection patterns."""
        self.behavioral_patterns = [
            BehavioralPattern(
                name="credential_stuffing",
                description="Multiple failed logins across different services",
                indicators=["failed_login", "multiple_services", "short_timeframe"],
                threshold=0.8,
                timeframe_minutes=10,
                severity=ThreatLevel.HIGH
            ),
            BehavioralPattern(
                name="data_hoarding",
                description="Unusual data access patterns indicating collection",
                indicators=["large_file_access", "multiple_databases", "off_hours"],
                threshold=0.7,
                timeframe_minutes=60,
                severity=ThreatLevel.HIGH
            ),
            BehavioralPattern(
                name="lateral_movement",
                description="Network traversal patterns indicating compromise",
                indicators=["remote_access", "privilege_escalation", "network_scan"],
                threshold=0.75,
                timeframe_minutes=30,
                severity=ThreatLevel.CRITICAL
            ),
            BehavioralPattern(
                name="exfiltration_prep",
                description="Data staging and compression before exfiltration",
                indicators=["file_compression", "staging_directory", "encryption"],
                threshold=0.8,
                timeframe_minutes=45,
                severity=ThreatLevel.CRITICAL
            ),
            BehavioralPattern(
                name="living_off_land",
                description="Abuse of legitimate tools for malicious purposes",
                indicators=["powershell_execution", "wmi_usage", "system_tools"],
                threshold=0.6,
                timeframe_minutes=20,
                severity=ThreatLevel.MEDIUM
            )
        ]
    
    async def analyze_event_sequence(self, events: List[Dict[str, Any]]) -> List[ThreatDetection]:
        """Analyze sequence of events for behavioral threats."""
        detections = []
        
        # Add events to buffer
        for event in events:
            self.event_buffer.append(event)
        
        # Analyze against behavioral patterns
        for pattern in self.behavioral_patterns:
            detection = await self._evaluate_behavioral_pattern(pattern)
            if detection:
                detections.append(detection)
        
        # Perform anomaly detection
        anomaly_detections = await self._detect_anomalies(events)
        detections.extend(anomaly_detections)
        
        return detections
    
    async def _evaluate_behavioral_pattern(self, pattern: BehavioralPattern) -> Optional[ThreatDetection]:
        """Evaluate events against a behavioral pattern."""
        cutoff_time = datetime.now() - timedelta(minutes=pattern.timeframe_minutes)
        recent_events = [
            event for event in self.event_buffer
            if datetime.fromisoformat(event.get("timestamp", datetime.now().isoformat())) > cutoff_time
        ]
        
        if len(recent_events) < 2:
            return None
        
        # Calculate pattern match score
        match_score = await self._calculate_pattern_score(pattern, recent_events)
        
        if match_score >= pattern.threshold:
            detection_id = f"BEHAV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Extract relevant indicators and assets
            indicators = self._extract_indicators_from_events(recent_events)
            affected_assets = self._extract_affected_assets(recent_events)
            
            return ThreatDetection(
                id=detection_id,
                name=pattern.name.replace("_", " ").title(),
                description=pattern.description,
                severity=pattern.severity,
                confidence=match_score,
                method=DetectionMethod.BEHAVIORAL,
                timestamp=datetime.now(),
                indicators=indicators,
                affected_assets=affected_assets,
                mitre_techniques=self._map_pattern_to_mitre(pattern.name),
                kill_chain_stage=self._map_pattern_to_kill_chain(pattern.name),
                recommended_actions=self._generate_pattern_recommendations(pattern),
                metadata={
                    "pattern_name": pattern.name,
                    "event_count": len(recent_events),
                    "timeframe_minutes": pattern.timeframe_minutes
                }
            )
        
        return None
    
    async def _calculate_pattern_score(self, pattern: BehavioralPattern, events: List[Dict[str, Any]]) -> float:
        """Calculate how well events match a behavioral pattern."""
        if not events:
            return 0.0
        
        pattern_indicators = pattern.indicators
        score_components = []
        
        # Analyze different aspects of the pattern
        if pattern.name == "credential_stuffing":
            failed_logins = sum(1 for e in events if e.get("auth_result") == "failed")
            unique_services = len(set(e.get("service") for e in events))
            score = min((failed_logins / 10.0) + (unique_services / 5.0), 1.0)
            score_components.append(score)
        
        elif pattern.name == "data_hoarding":
            large_accesses = sum(1 for e in events if e.get("file_size", 0) > 10000000)  # >10MB
            unique_sources = len(set(e.get("data_source") for e in events))
            off_hours = sum(1 for e in events if self._is_off_hours(e.get("timestamp")))
            score = (large_accesses / 5.0) + (unique_sources / 3.0) + (off_hours / len(events))
            score_components.append(min(score, 1.0))
        
        elif pattern.name == "lateral_movement":
            remote_accesses = sum(1 for e in events if e.get("remote_access", False))
            privilege_escalations = sum(1 for e in events if e.get("privilege_change", False))
            network_scans = sum(1 for e in events if e.get("network_scan", False))
            score = (remote_accesses + privilege_escalations + network_scans) / len(events)
            score_components.append(score)
        
        elif pattern.name == "exfiltration_prep":
            compressions = sum(1 for e in events if "compress" in e.get("action", "").lower())
            staging = sum(1 for e in events if "temp" in e.get("file_path", "").lower())
            encryptions = sum(1 for e in events if "encrypt" in e.get("action", "").lower())
            score = (compressions + staging + encryptions) / max(len(events), 1)
            score_components.append(score)
        
        elif pattern.name == "living_off_land":
            powershell_usage = sum(1 for e in events if "powershell" in e.get("process", "").lower())
            wmi_usage = sum(1 for e in events if "wmi" in e.get("process", "").lower())
            system_tools = sum(1 for e in events if any(tool in e.get("process", "").lower() 
                                                      for tool in ["cmd", "net", "sc", "reg"]))
            score = (powershell_usage + wmi_usage + system_tools) / max(len(events), 1)
            score_components.append(min(score, 1.0))
        
        # Return average of all score components
        return sum(score_components) / len(score_components) if score_components else 0.0
    
    def _is_off_hours(self, timestamp_str: str) -> bool:
        """Check if timestamp is outside business hours."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            hour = timestamp.hour
            return hour < 7 or hour > 19  # Before 7 AM or after 7 PM
        except:
            return False
    
    async def _detect_anomalies(self, events: List[Dict[str, Any]]) -> List[ThreatDetection]:
        """Detect statistical anomalies in event patterns."""
        detections = []
        
        if len(events) < 10:  # Need minimum events for statistical analysis
            return detections
        
        # Analyze event frequency anomalies
        frequency_anomaly = await self._detect_frequency_anomaly(events)
        if frequency_anomaly:
            detections.append(frequency_anomaly)
        
        # Analyze volume anomalies
        volume_anomaly = await self._detect_volume_anomaly(events)
        if volume_anomaly:
            detections.append(volume_anomaly)
        
        # Analyze timing anomalies
        timing_anomaly = await self._detect_timing_anomaly(events)
        if timing_anomaly:
            detections.append(timing_anomaly)
        
        return detections
    
    async def _detect_frequency_anomaly(self, events: List[Dict[str, Any]]) -> Optional[ThreatDetection]:
        """Detect anomalous event frequencies."""
        # Calculate event frequencies by type
        event_types = [e.get("event_type", "unknown") for e in events]
        type_counts = defaultdict(int)
        for event_type in event_types:
            type_counts[event_type] += 1
        
        # Simple anomaly detection - events occurring much more frequently than baseline
        baseline_frequency = len(events) / len(type_counts) if type_counts else 0
        
        for event_type, count in type_counts.items():
            if count > baseline_frequency * 5:  # 5x higher than baseline
                detection_id = f"FREQ-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                return ThreatDetection(
                    id=detection_id,
                    name="Event Frequency Anomaly",
                    description=f"Unusual frequency of {event_type} events detected",
                    severity=ThreatLevel.MEDIUM,
                    confidence=0.7,
                    method=DetectionMethod.ANOMALY,
                    timestamp=datetime.now(),
                    indicators=[],
                    affected_assets=[],
                    mitre_techniques=["T1562"],  # Impair Defenses
                    kill_chain_stage="Defense Evasion",
                    recommended_actions=[
                        "Investigate source of high-frequency events",
                        "Check for automated attack tools",
                        "Review event source configuration"
                    ],
                    metadata={
                        "anomalous_event_type": event_type,
                        "event_count": count,
                        "baseline_frequency": baseline_frequency
                    }
                )
        
        return None
    
    async def _detect_volume_anomaly(self, events: List[Dict[str, Any]]) -> Optional[ThreatDetection]:
        """Detect anomalous data volumes."""
        # Extract volume metrics from events
        volumes = [e.get("data_size", 0) for e in events if e.get("data_size", 0) > 0]
        
        if len(volumes) < 5:
            return None
        
        # Calculate statistical measures
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        max_volume = max(volumes)
        
        # Detect outliers using z-score
        if std_volume > 0:
            z_score = (max_volume - mean_volume) / std_volume
            if z_score > 3:  # More than 3 standard deviations
                detection_id = f"VOL-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                return ThreatDetection(
                    id=detection_id,
                    name="Data Volume Anomaly",
                    description="Unusually large data transfer detected",
                    severity=ThreatLevel.HIGH,
                    confidence=0.8,
                    method=DetectionMethod.ANOMALY,
                    timestamp=datetime.now(),
                    indicators=[],
                    affected_assets=[],
                    mitre_techniques=["T1041"],  # Exfiltration Over C2 Channel
                    kill_chain_stage="Exfiltration",
                    recommended_actions=[
                        "Investigate large data transfer",
                        "Check destination and purpose",
                        "Review data classification"
                    ],
                    metadata={
                        "max_volume_bytes": max_volume,
                        "mean_volume_bytes": mean_volume,
                        "z_score": z_score
                    }
                )
        
        return None
    
    async def _detect_timing_anomaly(self, events: List[Dict[str, Any]]) -> Optional[ThreatDetection]:
        """Detect timing-based anomalies."""
        # Extract timestamps and convert to hours
        timestamps = []
        for event in events:
            try:
                timestamp = datetime.fromisoformat(event.get("timestamp", "").replace('Z', '+00:00'))
                timestamps.append(timestamp.hour + timestamp.minute / 60.0)
            except:
                continue
        
        if len(timestamps) < 5:
            return None
        
        # Check for off-hours activity
        off_hours_count = sum(1 for hour in timestamps if hour < 6 or hour > 22)
        off_hours_ratio = off_hours_count / len(timestamps)
        
        if off_hours_ratio > 0.7:  # More than 70% off-hours activity
            detection_id = f"TIME-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            return ThreatDetection(
                id=detection_id,
                name="Off-Hours Activity Anomaly",
                description="Unusual activity detected outside business hours",
                severity=ThreatLevel.MEDIUM,
                confidence=0.6,
                method=DetectionMethod.ANOMALY,
                timestamp=datetime.now(),
                indicators=[],
                affected_assets=[],
                mitre_techniques=["T1530"],  # Data from Cloud Storage Object
                kill_chain_stage="Collection",
                recommended_actions=[
                    "Investigate off-hours activity",
                    "Check if activity is authorized",
                    "Review access controls"
                ],
                metadata={
                    "off_hours_ratio": off_hours_ratio,
                    "total_events": len(timestamps)
                }
            )
        
        return None
    
    def _extract_indicators_from_events(self, events: List[Dict[str, Any]]) -> List[ThreatIndicator]:
        """Extract threat indicators from events."""
        indicators = []
        
        for event in events:
            # Extract IP indicators
            source_ip = event.get("source_ip")
            if source_ip:
                indicator = ThreatIndicator(
                    id=f"ip_{hash(source_ip)}",
                    type="ip",
                    value=source_ip,
                    confidence=0.7,
                    source="behavioral_analysis",
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    context={"event_id": event.get("id")}
                )
                indicators.append(indicator)
            
            # Extract file indicators
            file_path = event.get("file_path")
            if file_path:
                indicator = ThreatIndicator(
                    id=f"file_{hash(file_path)}",
                    type="file",
                    value=file_path,
                    confidence=0.6,
                    source="behavioral_analysis",
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    context={"event_id": event.get("id")}
                )
                indicators.append(indicator)
            
            # Extract process indicators
            process = event.get("process")
            if process:
                indicator = ThreatIndicator(
                    id=f"process_{hash(process)}",
                    type="process",
                    value=process,
                    confidence=0.5,
                    source="behavioral_analysis",
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    context={"event_id": event.get("id")}
                )
                indicators.append(indicator)
        
        # Remove duplicates and limit
        unique_indicators = []
        seen_values = set()
        for indicator in indicators:
            if indicator.value not in seen_values:
                unique_indicators.append(indicator)
                seen_values.add(indicator.value)
        
        return unique_indicators[:10]  # Limit to 10 indicators
    
    def _extract_affected_assets(self, events: List[Dict[str, Any]]) -> List[str]:
        """Extract affected assets from events."""
        assets = set()
        
        for event in events:
            # Add hostnames
            hostname = event.get("hostname")
            if hostname:
                assets.add(hostname)
            
            # Add services
            service = event.get("service")
            if service:
                assets.add(service)
            
            # Add databases
            database = event.get("database")
            if database:
                assets.add(database)
        
        return list(assets)
    
    def _map_pattern_to_mitre(self, pattern_name: str) -> List[str]:
        """Map behavioral pattern to MITRE ATT&CK techniques."""
        mapping = {
            "credential_stuffing": ["T1110.003", "T1110.004"],
            "data_hoarding": ["T1005", "T1039", "T1025"],
            "lateral_movement": ["T1021", "T1210", "T1068"],
            "exfiltration_prep": ["T1560", "T1074", "T1002"],
            "living_off_land": ["T1059.001", "T1047", "T1218"]
        }
        return mapping.get(pattern_name, [])
    
    def _map_pattern_to_kill_chain(self, pattern_name: str) -> str:
        """Map behavioral pattern to cyber kill chain stage."""
        mapping = {
            "credential_stuffing": "Initial Access",
            "data_hoarding": "Collection",
            "lateral_movement": "Lateral Movement", 
            "exfiltration_prep": "Exfiltration",
            "living_off_land": "Defense Evasion"
        }
        return mapping.get(pattern_name, "Unknown")
    
    def _generate_pattern_recommendations(self, pattern: BehavioralPattern) -> List[str]:
        """Generate recommendations based on behavioral pattern."""
        base_recommendations = [
            "Investigate affected systems immediately",
            "Review security logs for related activity",
            "Consider isolating affected assets"
        ]
        
        pattern_specific = {
            "credential_stuffing": [
                "Lock affected accounts temporarily",
                "Implement account lockout policies",
                "Review authentication logs"
            ],
            "data_hoarding": [
                "Check data access permissions",
                "Review sensitive data locations",
                "Implement data loss prevention controls"
            ],
            "lateral_movement": [
                "Segment network to prevent spread",
                "Review privileged account usage",
                "Check for unauthorized access"
            ],
            "exfiltration_prep": [
                "Block outbound data transfers",
                "Review file compression activities",
                "Check staging directories"
            ],
            "living_off_land": [
                "Review PowerShell execution policies",
                "Monitor system tool usage",
                "Check for script-based attacks"
            ]
        }
        
        specific_recs = pattern_specific.get(pattern.name, [])
        return base_recommendations + specific_recs
    
    def update_baseline_metrics(self, events: List[Dict[str, Any]]) -> None:
        """Update baseline metrics for anomaly detection."""
        if not events:
            return
        
        # Update event frequency baseline
        event_types = [e.get("event_type", "unknown") for e in events]
        self.baseline_metrics["event_frequency"] = len(events) / len(set(event_types)) if event_types else 0
        
        # Update volume baseline
        volumes = [e.get("data_size", 0) for e in events if e.get("data_size", 0) > 0]
        if volumes:
            self.baseline_metrics["mean_volume"] = np.mean(volumes)
            self.baseline_metrics["std_volume"] = np.std(volumes)
        
        # Update timing baseline
        business_hours_count = 0
        for event in events:
            try:
                timestamp = datetime.fromisoformat(event.get("timestamp", "").replace('Z', '+00:00'))
                if 7 <= timestamp.hour <= 19:  # Business hours
                    business_hours_count += 1
            except:
                continue
        
        self.baseline_metrics["business_hours_ratio"] = business_hours_count / len(events) if events else 0
        
        self.logger.info("Updated baseline metrics for behavioral analysis")


class ThreatDetectionEngine:
    """Main threat detection engine coordinating all detection methods."""
    
    def __init__(self):
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.threat_indicators = {}
        self.detection_rules = []
        self.active_threats = []
        self.logger = logging.getLogger("ThreatDetectionEngine")
    
    async def process_events(self, events: List[Dict[str, Any]]) -> List[ThreatDetection]:
        """Process events through all detection methods."""
        all_detections = []
        
        # Behavioral analysis
        behavioral_detections = await self.behavioral_analyzer.analyze_event_sequence(events)
        all_detections.extend(behavioral_detections)
        
        # Signature-based detection
        signature_detections = await self._signature_detection(events)
        all_detections.extend(signature_detections)
        
        # Threat intelligence correlation
        intel_detections = await self._threat_intelligence_correlation(events)
        all_detections.extend(intel_detections)
        
        # Update active threats
        for detection in all_detections:
            self.active_threats.append(detection)
        
        # Keep only recent threats (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_threats = [
            threat for threat in self.active_threats
            if threat.timestamp > cutoff_time
        ]
        
        return all_detections
    
    async def _signature_detection(self, events: List[Dict[str, Any]]) -> List[ThreatDetection]:
        """Simple signature-based detection."""
        detections = []
        
        # Define some basic signatures
        signatures = {
            "powershell_encoded": {
                "pattern": "-EncodedCommand",
                "severity": ThreatLevel.HIGH,
                "description": "Encoded PowerShell command execution"
            },
            "suspicious_network": {
                "pattern": "443.*suspicious",
                "severity": ThreatLevel.MEDIUM,
                "description": "Connection to suspicious domain"
            }
        }
        
        for event in events:
            command = event.get("command", "")
            network_data = event.get("network_data", "")
            
            for sig_name, sig_data in signatures.items():
                if (sig_data["pattern"] in command or sig_data["pattern"] in network_data):
                    detection_id = f"SIG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{sig_name}"
                    detection = ThreatDetection(
                        id=detection_id,
                        name=f"Signature Match: {sig_name}",
                        description=sig_data["description"],
                        severity=sig_data["severity"],
                        confidence=0.9,
                        method=DetectionMethod.SIGNATURE,
                        timestamp=datetime.now(),
                        indicators=[],
                        affected_assets=[event.get("hostname", "unknown")],
                        mitre_techniques=["T1059"],
                        kill_chain_stage="Execution",
                        recommended_actions=[
                            "Investigate the matched signature",
                            "Check for additional indicators",
                            "Consider blocking related activity"
                        ],
                        metadata={"signature_name": sig_name, "matched_pattern": sig_data["pattern"]}
                    )
                    detections.append(detection)
        
        return detections
    
    async def _threat_intelligence_correlation(self, events: List[Dict[str, Any]]) -> List[ThreatDetection]:
        """Correlate events with threat intelligence."""
        detections = []
        
        # Simulate threat intelligence feeds
        threat_intel = {
            "malicious_ips": ["192.168.1.100", "10.0.0.50"],
            "suspicious_domains": ["evil.com", "malware.org"],
            "known_hashes": ["abc123", "def456"]
        }
        
        for event in events:
            source_ip = event.get("source_ip")
            domain = event.get("domain")
            file_hash = event.get("file_hash")
            
            # Check against malicious IPs
            if source_ip in threat_intel["malicious_ips"]:
                detection_id = f"INTEL-{datetime.now().strftime('%Y%m%d%H%M%S')}-IP"
                detection = ThreatDetection(
                    id=detection_id,
                    name="Threat Intelligence Match: Malicious IP",
                    description=f"Communication with known malicious IP: {source_ip}",
                    severity=ThreatLevel.HIGH,
                    confidence=0.95,
                    method=DetectionMethod.THREAT_INTELLIGENCE,
                    timestamp=datetime.now(),
                    indicators=[],
                    affected_assets=[event.get("hostname", "unknown")],
                    mitre_techniques=["T1071"],
                    kill_chain_stage="Command and Control",
                    recommended_actions=[
                        "Block communication with malicious IP",
                        "Investigate affected systems",
                        "Check for additional compromise"
                    ],
                    metadata={"malicious_ip": source_ip, "intel_source": "threat_feed"}
                )
                detections.append(detection)
        
        return detections
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of current threat landscape."""
        if not self.active_threats:
            return {
                "total_threats": 0,
                "severity_breakdown": {},
                "top_techniques": [],
                "recent_detections": []
            }
        
        severity_counts = defaultdict(int)
        techniques = defaultdict(int)
        
        for threat in self.active_threats:
            severity_counts[threat.severity.value] += 1
            for technique in threat.mitre_techniques:
                techniques[technique] += 1
        
        # Get top techniques
        top_techniques = sorted(techniques.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get recent detections
        recent_detections = sorted(self.active_threats, key=lambda x: x.timestamp, reverse=True)[:10]
        
        return {
            "total_threats": len(self.active_threats),
            "severity_breakdown": dict(severity_counts),
            "top_techniques": [{"technique": t[0], "count": t[1]} for t in top_techniques],
            "recent_detections": [
                {
                    "id": d.id,
                    "name": d.name,
                    "severity": d.severity.value,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in recent_detections
            ]
        }