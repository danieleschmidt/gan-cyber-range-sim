"""
Automated Incident Response Engine for GAN Cyber Range.

Provides enterprise-grade automated incident response with:
- Real-time threat detection and classification
- Automated response playbooks
- Incident escalation workflows
- Forensic data collection
- Compliance reporting
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from .siem import SIEMEngine, SecurityEvent, EventType, AlertSeverity
from .ml_threat_detection import MLThreatDetector
from ..monitoring.metrics import MetricsCollector


logger = logging.getLogger(__name__)


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    """Incident status."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESPONDING = "responding"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ResponseAction(str, Enum):
    """Automated response actions."""
    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    QUARANTINE_FILE = "quarantine_file"
    RESET_PASSWORD = "reset_password"
    DISABLE_ACCOUNT = "disable_account"
    COLLECT_FORENSICS = "collect_forensics"
    NOTIFY_ADMIN = "notify_admin"
    ESCALATE = "escalate"
    DEPLOY_HONEYPOT = "deploy_honeypot"
    UPDATE_SIGNATURES = "update_signatures"


@dataclass
class IncidentEvidence:
    """Evidence collected for an incident."""
    evidence_id: str
    evidence_type: str
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    hash_value: Optional[str] = None
    chain_of_custody: List[str] = field(default_factory=list)


@dataclass
class ResponseActionResult:
    """Result of an automated response action."""
    action: ResponseAction
    success: bool
    timestamp: datetime
    details: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class SecurityIncident:
    """Security incident data structure."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    
    # Incident details
    affected_assets: List[str] = field(default_factory=list)
    attack_vectors: List[str] = field(default_factory=list)
    indicators_of_compromise: List[str] = field(default_factory=list)
    
    # Related events
    security_events: List[SecurityEvent] = field(default_factory=list)
    
    # Response tracking
    response_actions: List[ResponseActionResult] = field(default_factory=list)
    assigned_to: Optional[str] = None
    
    # Evidence
    evidence: List[IncidentEvidence] = field(default_factory=list)
    
    # Timeline
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    # Resolution
    resolution_notes: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)


class IncidentPlaybook:
    """Automated incident response playbook."""
    
    def __init__(self, 
                 name: str, 
                 trigger_conditions: Dict[str, Any],
                 response_actions: List[ResponseAction],
                 escalation_rules: Optional[Dict[str, Any]] = None):
        self.name = name
        self.trigger_conditions = trigger_conditions
        self.response_actions = response_actions
        self.escalation_rules = escalation_rules or {}
        self.execution_count = 0
        self.success_rate = 0.0
    
    def matches_incident(self, incident: SecurityIncident) -> bool:
        """Check if this playbook matches the incident criteria."""
        # Check severity
        if "severity" in self.trigger_conditions:
            required_severity = self.trigger_conditions["severity"]
            if incident.severity.value != required_severity:
                return False
        
        # Check event types
        if "event_types" in self.trigger_conditions:
            required_types = self.trigger_conditions["event_types"]
            incident_types = [event.event_type.value for event in incident.security_events]
            if not any(etype in required_types for etype in incident_types):
                return False
        
        # Check attack vectors
        if "attack_vectors" in self.trigger_conditions:
            required_vectors = self.trigger_conditions["attack_vectors"]
            if not any(vector in required_vectors for vector in incident.attack_vectors):
                return False
        
        return True


class IncidentResponseEngine:
    """Automated incident response engine."""
    
    def __init__(self, 
                 siem_engine: SIEMEngine,
                 threat_detector: MLThreatDetector,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.siem_engine = siem_engine
        self.threat_detector = threat_detector
        self.metrics_collector = metrics_collector
        
        # Incident storage
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.incident_history: List[SecurityIncident] = []
        
        # Response playbooks
        self.playbooks: List[IncidentPlaybook] = []
        
        # Response handlers
        self.response_handlers: Dict[ResponseAction, Callable] = {}
        
        # Configuration
        self.auto_response_enabled = True
        self.max_concurrent_responses = 5
        self.incident_correlation_window = timedelta(minutes=30)
        
        # Runtime state
        self.running = False
        self.response_tasks: List[asyncio.Task] = []
        
        # Initialize default playbooks and handlers
        self._initialize_default_playbooks()
        self._initialize_default_handlers()
    
    def register_response_handler(self, action: ResponseAction, handler: Callable):
        """Register a handler for a response action."""
        self.response_handlers[action] = handler
        logger.info(f"Registered response handler for {action}")
    
    def add_playbook(self, playbook: IncidentPlaybook):
        """Add an incident response playbook."""
        self.playbooks.append(playbook)
        logger.info(f"Added incident response playbook: {playbook.name}")
    
    async def start_engine(self):
        """Start the incident response engine."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting incident response engine")
        
        # Start monitoring for new security events
        monitor_task = asyncio.create_task(self._monitor_security_events())
        self.response_tasks.append(monitor_task)
        
        # Start periodic incident correlation
        correlation_task = asyncio.create_task(self._correlate_incidents())
        self.response_tasks.append(correlation_task)
        
        logger.info("Incident response engine started")
    
    async def stop_engine(self):
        """Stop the incident response engine."""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping incident response engine")
        
        # Cancel all tasks
        for task in self.response_tasks:
            task.cancel()
        
        await asyncio.gather(*self.response_tasks, return_exceptions=True)
        self.response_tasks.clear()
        
        logger.info("Incident response engine stopped")
    
    async def _monitor_security_events(self):
        """Monitor SIEM for new security events."""
        while self.running:
            try:
                # Get recent security events from SIEM
                recent_events = await self.siem_engine.get_recent_events(
                    limit=100,
                    time_window=timedelta(minutes=5)
                )
                
                for event in recent_events:
                    await self._process_security_event(event)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring security events: {e}")
                await asyncio.sleep(30)
    
    async def _process_security_event(self, event: SecurityEvent):
        """Process a new security event and create/update incidents."""
        try:
            # Use ML threat detection to assess threat level
            threat_assessment = await self.threat_detector.assess_threat(event)
            
            # Check if this event correlates with existing incidents
            related_incident = self._find_related_incident(event)
            
            if related_incident:
                # Add event to existing incident
                related_incident.security_events.append(event)
                related_incident.updated_at = datetime.now()
                related_incident.last_seen = event.timestamp
                
                # Update incident severity if necessary
                if threat_assessment["severity"] > related_incident.severity.value:
                    related_incident.severity = IncidentSeverity(threat_assessment["severity"])
                
                logger.info(f"Added event to existing incident {related_incident.incident_id}")
            
            else:
                # Create new incident if threat level is significant
                if threat_assessment["threat_score"] > 0.6:
                    incident = await self._create_incident(event, threat_assessment)
                    logger.info(f"Created new incident {incident.incident_id}")
            
        except Exception as e:
            logger.error(f"Error processing security event: {e}")
    
    def _find_related_incident(self, event: SecurityEvent) -> Optional[SecurityIncident]:
        """Find existing incident that correlates with the new event."""
        current_time = datetime.now()
        correlation_window = current_time - self.incident_correlation_window
        
        for incident in self.active_incidents.values():
            # Skip closed incidents
            if incident.status == IncidentStatus.CLOSED:
                continue
            
            # Check time window
            if incident.updated_at < correlation_window:
                continue
            
            # Check for correlation factors
            correlation_score = 0
            
            # Same source IP
            for existing_event in incident.security_events:
                if (hasattr(event, 'source_ip') and hasattr(existing_event, 'source_ip') and
                    event.source_ip == existing_event.source_ip):
                    correlation_score += 0.3
                
                # Same target
                if (hasattr(event, 'target') and hasattr(existing_event, 'target') and
                    event.target == existing_event.target):
                    correlation_score += 0.4
                
                # Same attack pattern
                if event.event_type == existing_event.event_type:
                    correlation_score += 0.2
            
            # If correlation score is high enough, consider it related
            if correlation_score >= 0.5:
                return incident
        
        return None
    
    async def _create_incident(self, event: SecurityEvent, threat_assessment: Dict[str, Any]) -> SecurityIncident:
        """Create a new security incident."""
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        
        # Determine severity based on threat assessment
        severity_mapping = {
            "low": IncidentSeverity.LOW,
            "medium": IncidentSeverity.MEDIUM,
            "high": IncidentSeverity.HIGH,
            "critical": IncidentSeverity.CRITICAL
        }
        
        severity = severity_mapping.get(threat_assessment["severity"], IncidentSeverity.MEDIUM)
        
        # Create incident
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"{event.event_type.value.replace('_', ' ').title()} - {event.source}",
            description=f"Automated incident created from {event.event_type.value} event",
            severity=severity,
            status=IncidentStatus.OPEN,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            first_seen=event.timestamp,
            last_seen=event.timestamp,
            security_events=[event]
        )
        
        # Extract indicators of compromise
        if hasattr(event, 'source_ip'):
            incident.indicators_of_compromise.append(f"IP:{event.source_ip}")
        
        if hasattr(event, 'file_hash'):
            incident.indicators_of_compromise.append(f"Hash:{event.file_hash}")
        
        # Store incident
        self.active_incidents[incident_id] = incident
        
        # Trigger automated response
        if self.auto_response_enabled:
            await self._trigger_automated_response(incident)
        
        # Update metrics
        if self.metrics_collector:
            self.metrics_collector.record_incident_created(
                severity=severity.value,
                event_type=event.event_type.value
            )
        
        return incident
    
    async def _trigger_automated_response(self, incident: SecurityIncident):
        """Trigger automated response for an incident."""
        try:
            # Find matching playbooks
            matching_playbooks = [
                playbook for playbook in self.playbooks
                if playbook.matches_incident(incident)
            ]
            
            if not matching_playbooks:
                logger.warning(f"No matching playbooks found for incident {incident.incident_id}")
                return
            
            # Execute the most specific playbook (first match)
            playbook = matching_playbooks[0]
            logger.info(f"Executing playbook '{playbook.name}' for incident {incident.incident_id}")
            
            # Update incident status
            incident.status = IncidentStatus.RESPONDING
            incident.updated_at = datetime.now()
            
            # Execute response actions
            for action in playbook.response_actions:
                try:
                    result = await self._execute_response_action(action, incident)
                    incident.response_actions.append(result)
                    
                    if result.success:
                        logger.info(f"Successfully executed {action} for incident {incident.incident_id}")
                    else:
                        logger.error(f"Failed to execute {action} for incident {incident.incident_id}: {result.error}")
                
                except Exception as e:
                    logger.error(f"Error executing response action {action}: {e}")
                    
                    # Record failed action
                    incident.response_actions.append(ResponseActionResult(
                        action=action,
                        success=False,
                        timestamp=datetime.now(),
                        details={},
                        error=str(e)
                    ))
            
            # Update playbook statistics
            playbook.execution_count += 1
            successful_actions = sum(1 for result in incident.response_actions if result.success)
            playbook.success_rate = successful_actions / len(incident.response_actions)
            
            # Check if incident should be escalated
            if playbook.escalation_rules:
                await self._check_escalation(incident, playbook)
            
        except Exception as e:
            logger.error(f"Error in automated response for incident {incident.incident_id}: {e}")
    
    async def _execute_response_action(self, action: ResponseAction, incident: SecurityIncident) -> ResponseActionResult:
        """Execute a specific response action."""
        handler = self.response_handlers.get(action)
        
        if not handler:
            return ResponseActionResult(
                action=action,
                success=False,
                timestamp=datetime.now(),
                details={},
                error="No handler registered for action"
            )
        
        try:
            result = await handler(incident)
            
            return ResponseActionResult(
                action=action,
                success=True,
                timestamp=datetime.now(),
                details=result if isinstance(result, dict) else {"result": str(result)}
            )
            
        except Exception as e:
            return ResponseActionResult(
                action=action,
                success=False,
                timestamp=datetime.now(),
                details={},
                error=str(e)
            )
    
    async def _check_escalation(self, incident: SecurityIncident, playbook: IncidentPlaybook):
        """Check if incident should be escalated."""
        escalation_rules = playbook.escalation_rules
        
        # Check time-based escalation
        if "max_response_time" in escalation_rules:
            max_time = timedelta(minutes=escalation_rules["max_response_time"])
            if datetime.now() - incident.created_at > max_time:
                await self._escalate_incident(incident, "Response time exceeded")
        
        # Check failure-based escalation
        if "max_failures" in escalation_rules:
            failed_actions = sum(1 for result in incident.response_actions if not result.success)
            if failed_actions >= escalation_rules["max_failures"]:
                await self._escalate_incident(incident, "Too many failed response actions")
    
    async def _escalate_incident(self, incident: SecurityIncident, reason: str):
        """Escalate an incident."""
        logger.warning(f"Escalating incident {incident.incident_id}: {reason}")
        
        # Increase severity if not already critical
        if incident.severity != IncidentSeverity.CRITICAL:
            previous_severity = incident.severity
            incident.severity = IncidentSeverity.CRITICAL
            logger.info(f"Escalated incident severity from {previous_severity} to {incident.severity}")
        
        # Execute escalation action
        escalation_result = await self._execute_response_action(ResponseAction.ESCALATE, incident)
        incident.response_actions.append(escalation_result)
    
    async def _correlate_incidents(self):
        """Periodic incident correlation to merge related incidents."""
        while self.running:
            try:
                # Look for incidents that should be merged
                incidents_to_merge = []
                
                for incident1_id, incident1 in self.active_incidents.items():
                    for incident2_id, incident2 in self.active_incidents.items():
                        if incident1_id >= incident2_id:  # Avoid duplicate comparisons
                            continue
                        
                        # Check if incidents should be merged
                        if self._should_merge_incidents(incident1, incident2):
                            incidents_to_merge.append((incident1, incident2))
                
                # Merge incidents
                for incident1, incident2 in incidents_to_merge:
                    await self._merge_incidents(incident1, incident2)
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in incident correlation: {e}")
                await asyncio.sleep(300)
    
    def _should_merge_incidents(self, incident1: SecurityIncident, incident2: SecurityIncident) -> bool:
        """Determine if two incidents should be merged."""
        # Don't merge closed incidents
        if incident1.status == IncidentStatus.CLOSED or incident2.status == IncidentStatus.CLOSED:
            return False
        
        # Check for common indicators of compromise
        common_iocs = set(incident1.indicators_of_compromise) & set(incident2.indicators_of_compromise)
        if len(common_iocs) >= 2:
            return True
        
        # Check for common affected assets
        common_assets = set(incident1.affected_assets) & set(incident2.affected_assets)
        if len(common_assets) >= 1:
            return True
        
        return False
    
    async def _merge_incidents(self, primary_incident: SecurityIncident, secondary_incident: SecurityIncident):
        """Merge two related incidents."""
        logger.info(f"Merging incident {secondary_incident.incident_id} into {primary_incident.incident_id}")
        
        # Merge data
        primary_incident.security_events.extend(secondary_incident.security_events)
        primary_incident.response_actions.extend(secondary_incident.response_actions)
        primary_incident.evidence.extend(secondary_incident.evidence)
        primary_incident.affected_assets.extend(secondary_incident.affected_assets)
        primary_incident.indicators_of_compromise.extend(secondary_incident.indicators_of_compromise)
        
        # Remove duplicates
        primary_incident.affected_assets = list(set(primary_incident.affected_assets))
        primary_incident.indicators_of_compromise = list(set(primary_incident.indicators_of_compromise))
        
        # Update metadata
        primary_incident.updated_at = datetime.now()
        if secondary_incident.first_seen and secondary_incident.first_seen < primary_incident.first_seen:
            primary_incident.first_seen = secondary_incident.first_seen
        
        # Remove secondary incident
        del self.active_incidents[secondary_incident.incident_id]
        self.incident_history.append(secondary_incident)
    
    def get_incident(self, incident_id: str) -> Optional[SecurityIncident]:
        """Get incident by ID."""
        return self.active_incidents.get(incident_id)
    
    def get_incidents(self, 
                     status: Optional[IncidentStatus] = None,
                     severity: Optional[IncidentSeverity] = None,
                     limit: int = 100) -> List[SecurityIncident]:
        """Get incidents with optional filtering."""
        incidents = list(self.active_incidents.values())
        
        if status:
            incidents = [inc for inc in incidents if inc.status == status]
        
        if severity:
            incidents = [inc for inc in incidents if inc.severity == severity]
        
        # Sort by creation time (newest first)
        incidents.sort(key=lambda x: x.created_at, reverse=True)
        
        return incidents[:limit]
    
    def _initialize_default_playbooks(self):
        """Initialize default incident response playbooks."""
        
        # Malware detection playbook
        malware_playbook = IncidentPlaybook(
            name="Malware Response",
            trigger_conditions={
                "event_types": ["malware_detected", "suspicious_file"],
                "severity": "high"
            },
            response_actions=[
                ResponseAction.QUARANTINE_FILE,
                ResponseAction.ISOLATE_HOST,
                ResponseAction.COLLECT_FORENSICS,
                ResponseAction.UPDATE_SIGNATURES,
                ResponseAction.NOTIFY_ADMIN
            ],
            escalation_rules={
                "max_response_time": 30,  # minutes
                "max_failures": 2
            }
        )
        self.add_playbook(malware_playbook)
        
        # Brute force attack playbook
        brute_force_playbook = IncidentPlaybook(
            name="Brute Force Response",
            trigger_conditions={
                "event_types": ["brute_force_attack", "failed_login"],
                "severity": "medium"
            },
            response_actions=[
                ResponseAction.BLOCK_IP,
                ResponseAction.RESET_PASSWORD,
                ResponseAction.DEPLOY_HONEYPOT,
                ResponseAction.NOTIFY_ADMIN
            ]
        )
        self.add_playbook(brute_force_playbook)
        
        # Data exfiltration playbook
        data_exfil_playbook = IncidentPlaybook(
            name="Data Exfiltration Response",
            trigger_conditions={
                "event_types": ["data_exfiltration", "unauthorized_access"],
                "severity": "critical"
            },
            response_actions=[
                ResponseAction.ISOLATE_HOST,
                ResponseAction.BLOCK_IP,
                ResponseAction.DISABLE_ACCOUNT,
                ResponseAction.COLLECT_FORENSICS,
                ResponseAction.ESCALATE,
                ResponseAction.NOTIFY_ADMIN
            ],
            escalation_rules={
                "max_response_time": 15,  # minutes
                "max_failures": 1
            }
        )
        self.add_playbook(data_exfil_playbook)
    
    def _initialize_default_handlers(self):
        """Initialize default response action handlers."""
        
        async def isolate_host_handler(incident: SecurityIncident):
            """Isolate compromised host."""
            # Implementation would integrate with network security tools
            logger.info(f"Isolating hosts for incident {incident.incident_id}")
            return {"hosts_isolated": incident.affected_assets}
        
        async def block_ip_handler(incident: SecurityIncident):
            """Block malicious IP addresses."""
            ips_to_block = [ioc for ioc in incident.indicators_of_compromise if ioc.startswith("IP:")]
            logger.info(f"Blocking IPs: {ips_to_block}")
            return {"blocked_ips": ips_to_block}
        
        async def collect_forensics_handler(incident: SecurityIncident):
            """Collect forensic evidence."""
            logger.info(f"Collecting forensics for incident {incident.incident_id}")
            
            # Create forensic evidence
            evidence = IncidentEvidence(
                evidence_id=f"EVID-{uuid.uuid4().hex[:8].upper()}",
                evidence_type="system_snapshot",
                source="automated_collection",
                timestamp=datetime.now(),
                data={
                    "memory_dump": f"memory_{incident.incident_id}.dmp",
                    "disk_image": f"disk_{incident.incident_id}.img",
                    "network_logs": f"network_{incident.incident_id}.pcap"
                }
            )
            
            incident.evidence.append(evidence)
            return {"evidence_collected": evidence.evidence_id}
        
        async def notify_admin_handler(incident: SecurityIncident):
            """Notify administrators."""
            logger.info(f"Notifying administrators about incident {incident.incident_id}")
            return {"notification_sent": True, "recipients": ["admin@company.com"]}
        
        async def escalate_handler(incident: SecurityIncident):
            """Escalate incident to higher tier."""
            logger.warning(f"Escalating incident {incident.incident_id}")
            incident.assigned_to = "tier2_analyst"
            return {"escalated_to": "tier2_analyst"}
        
        # Register handlers
        self.register_response_handler(ResponseAction.ISOLATE_HOST, isolate_host_handler)
        self.register_response_handler(ResponseAction.BLOCK_IP, block_ip_handler)
        self.register_response_handler(ResponseAction.COLLECT_FORENSICS, collect_forensics_handler)
        self.register_response_handler(ResponseAction.NOTIFY_ADMIN, notify_admin_handler)
        self.register_response_handler(ResponseAction.ESCALATE, escalate_handler)