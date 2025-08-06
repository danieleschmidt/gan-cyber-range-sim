"""Advanced incident response automation and orchestration."""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class IncidentStatus(Enum):
    """Incident status states."""
    NEW = "new"
    ASSIGNED = "assigned"
    INVESTIGATING = "investigating"
    CONTAINING = "containing"
    ERADICATING = "eradicating"
    RECOVERING = "recovering"
    CLOSED = "closed"


class IncidentPriority(Enum):
    """Incident priority levels."""
    P1 = "p1"  # Critical - immediate response required
    P2 = "p2"  # High - response within 1 hour
    P3 = "p3"  # Medium - response within 4 hours
    P4 = "p4"  # Low - response within 24 hours


class ResponseAction(Enum):
    """Types of automated response actions."""
    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    DISABLE_ACCOUNT = "disable_account"
    COLLECT_EVIDENCE = "collect_evidence"
    DEPLOY_COUNTERMEASURES = "deploy_countermeasures"
    NOTIFY_STAKEHOLDERS = "notify_stakeholders"
    CREATE_TICKET = "create_ticket"
    EXECUTE_PLAYBOOK = "execute_playbook"


@dataclass
class IncidentArtifact:
    """Evidence or artifact related to an incident."""
    id: str
    name: str
    type: str  # log, file, network_capture, memory_dump, etc.
    path: str
    hash: Optional[str] = None
    size: Optional[int] = None
    collected_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseTask:
    """Individual response task within an incident."""
    id: str
    title: str
    description: str
    action_type: ResponseAction
    priority: int
    assigned_to: str
    status: str  # pending, in_progress, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass 
class SecurityIncident:
    """Comprehensive security incident data structure."""
    id: str
    title: str
    description: str
    severity: str  # low, medium, high, critical
    priority: IncidentPriority
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None
    
    # Detection information
    detection_source: str
    detection_method: str
    confidence: float
    
    # Affected assets and scope
    affected_assets: List[str] = field(default_factory=list)
    impacted_users: List[str] = field(default_factory=list)
    business_impact: str = "unknown"
    
    # Technical details
    indicators_of_compromise: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    kill_chain_stage: str = "unknown"
    
    # Response tracking
    assigned_to: str = "unassigned"
    response_team: List[str] = field(default_factory=list)
    tasks: List[ResponseTask] = field(default_factory=list)
    artifacts: List[IncidentArtifact] = field(default_factory=list)
    
    # Communication and documentation
    communications: List[Dict[str, Any]] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    
    # Metrics
    detection_time: Optional[datetime] = None
    response_time: Optional[datetime] = None
    containment_time: Optional[datetime] = None
    recovery_time: Optional[datetime] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlaybookEngine:
    """Automated response playbook engine."""
    
    def __init__(self):
        self.playbooks = {}
        self.logger = logging.getLogger("PlaybookEngine")
        self._initialize_default_playbooks()
    
    def _initialize_default_playbooks(self) -> None:
        """Initialize default incident response playbooks."""
        self.playbooks = {
            "malware_infection": {
                "name": "Malware Infection Response",
                "description": "Standard response for malware infections",
                "trigger_conditions": ["malware_detected", "suspicious_file_execution"],
                "tasks": [
                    {
                        "title": "Isolate infected host",
                        "action_type": ResponseAction.ISOLATE_HOST,
                        "priority": 1,
                        "automated": True,
                        "timeout_minutes": 5
                    },
                    {
                        "title": "Collect memory dump",
                        "action_type": ResponseAction.COLLECT_EVIDENCE,
                        "priority": 2,
                        "automated": True,
                        "timeout_minutes": 15,
                        "dependencies": ["task_1"]
                    },
                    {
                        "title": "Scan for lateral movement",
                        "action_type": ResponseAction.DEPLOY_COUNTERMEASURES,
                        "priority": 3,
                        "automated": True,
                        "timeout_minutes": 10
                    },
                    {
                        "title": "Notify security team",
                        "action_type": ResponseAction.NOTIFY_STAKEHOLDERS,
                        "priority": 4,
                        "automated": True,
                        "timeout_minutes": 2
                    }
                ]
            },
            
            "data_exfiltration": {
                "name": "Data Exfiltration Response",
                "description": "Response to suspected data exfiltration",
                "trigger_conditions": ["large_data_transfer", "unauthorized_access"],
                "tasks": [
                    {
                        "title": "Block outbound connections",
                        "action_type": ResponseAction.DEPLOY_COUNTERMEASURES,
                        "priority": 1,
                        "automated": True,
                        "timeout_minutes": 2
                    },
                    {
                        "title": "Isolate affected systems",
                        "action_type": ResponseAction.ISOLATE_HOST,
                        "priority": 1,
                        "automated": True,
                        "timeout_minutes": 5
                    },
                    {
                        "title": "Collect network logs",
                        "action_type": ResponseAction.COLLECT_EVIDENCE,
                        "priority": 2,
                        "automated": True,
                        "timeout_minutes": 10
                    },
                    {
                        "title": "Notify legal team",
                        "action_type": ResponseAction.NOTIFY_STAKEHOLDERS,
                        "priority": 3,
                        "automated": False,
                        "timeout_minutes": 30
                    },
                    {
                        "title": "Create forensic ticket",
                        "action_type": ResponseAction.CREATE_TICKET,
                        "priority": 4,
                        "automated": True,
                        "timeout_minutes": 5
                    }
                ]
            },
            
            "credential_compromise": {
                "name": "Credential Compromise Response", 
                "description": "Response to compromised user credentials",
                "trigger_conditions": ["brute_force_success", "unusual_login_location"],
                "tasks": [
                    {
                        "title": "Disable compromised account",
                        "action_type": ResponseAction.DISABLE_ACCOUNT,
                        "priority": 1,
                        "automated": True,
                        "timeout_minutes": 2
                    },
                    {
                        "title": "Block source IP",
                        "action_type": ResponseAction.BLOCK_IP,
                        "priority": 1,
                        "automated": True,
                        "timeout_minutes": 2
                    },
                    {
                        "title": "Collect authentication logs",
                        "action_type": ResponseAction.COLLECT_EVIDENCE,
                        "priority": 2,
                        "automated": True,
                        "timeout_minutes": 5
                    },
                    {
                        "title": "Force password reset",
                        "action_type": ResponseAction.DEPLOY_COUNTERMEASURES,
                        "priority": 3,
                        "automated": True,
                        "timeout_minutes": 10
                    },
                    {
                        "title": "Notify user and manager",
                        "action_type": ResponseAction.NOTIFY_STAKEHOLDERS,
                        "priority": 4,
                        "automated": False,
                        "timeout_minutes": 15
                    }
                ]
            },
            
            "advanced_persistent_threat": {
                "name": "APT Response Playbook",
                "description": "Response to advanced persistent threat activity",
                "trigger_conditions": ["apt_indicators", "living_off_land", "lateral_movement"],
                "tasks": [
                    {
                        "title": "Activate incident response team",
                        "action_type": ResponseAction.NOTIFY_STAKEHOLDERS,
                        "priority": 1,
                        "automated": True,
                        "timeout_minutes": 5
                    },
                    {
                        "title": "Preserve evidence immediately", 
                        "action_type": ResponseAction.COLLECT_EVIDENCE,
                        "priority": 1,
                        "automated": True,
                        "timeout_minutes": 10
                    },
                    {
                        "title": "Deploy enhanced monitoring",
                        "action_type": ResponseAction.DEPLOY_COUNTERMEASURES,
                        "priority": 2,
                        "automated": True,
                        "timeout_minutes": 15
                    },
                    {
                        "title": "Coordinate with threat intelligence",
                        "action_type": ResponseAction.NOTIFY_STAKEHOLDERS,
                        "priority": 3,
                        "automated": False,
                        "timeout_minutes": 30
                    },
                    {
                        "title": "Implement network segmentation",
                        "action_type": ResponseAction.DEPLOY_COUNTERMEASURES,
                        "priority": 4,
                        "automated": False,
                        "timeout_minutes": 60
                    }
                ]
            }
        }
    
    async def execute_playbook(self, playbook_name: str, incident: SecurityIncident) -> List[ResponseTask]:
        """Execute a response playbook for an incident."""
        if playbook_name not in self.playbooks:
            self.logger.error(f"Playbook not found: {playbook_name}")
            return []
        
        playbook = self.playbooks[playbook_name]
        tasks = []
        
        self.logger.info(f"Executing playbook '{playbook_name}' for incident {incident.id}")
        
        for i, task_template in enumerate(playbook["tasks"]):
            task_id = f"{incident.id}_task_{i+1}"
            task = ResponseTask(
                id=task_id,
                title=task_template["title"],
                description=f"Automated task from {playbook_name} playbook",
                action_type=task_template["action_type"],
                priority=task_template["priority"],
                assigned_to="automation" if task_template.get("automated", False) else "analyst",
                status="pending",
                created_at=datetime.now(),
                dependencies=task_template.get("dependencies", [])
            )
            
            tasks.append(task)
            
            # Execute automated tasks immediately
            if task_template.get("automated", False):
                await self._execute_automated_task(task, incident)
        
        return tasks
    
    async def _execute_automated_task(self, task: ResponseTask, incident: SecurityIncident) -> None:
        """Execute an automated response task."""
        self.logger.info(f"Executing automated task: {task.title}")
        
        task.status = "in_progress"
        task.started_at = datetime.now()
        
        try:
            # Execute based on action type
            if task.action_type == ResponseAction.ISOLATE_HOST:
                result = await self._isolate_host(incident.affected_assets)
            elif task.action_type == ResponseAction.BLOCK_IP:
                result = await self._block_ip(incident.indicators_of_compromise)
            elif task.action_type == ResponseAction.DISABLE_ACCOUNT:
                result = await self._disable_account(incident.impacted_users)
            elif task.action_type == ResponseAction.COLLECT_EVIDENCE:
                result = await self._collect_evidence(incident)
            elif task.action_type == ResponseAction.DEPLOY_COUNTERMEASURES:
                result = await self._deploy_countermeasures(incident)
            elif task.action_type == ResponseAction.NOTIFY_STAKEHOLDERS:
                result = await self._notify_stakeholders(incident, task.title)
            elif task.action_type == ResponseAction.CREATE_TICKET:
                result = await self._create_ticket(incident)
            else:
                result = {"status": "not_implemented", "message": f"Action type {task.action_type} not implemented"}
            
            task.result = result
            task.status = "completed" if result.get("status") == "success" else "failed"
            task.completed_at = datetime.now()
            
            self.logger.info(f"Task {task.title} completed with status: {task.status}")
        
        except Exception as e:
            task.result = {"status": "error", "message": str(e)}
            task.status = "failed"
            task.completed_at = datetime.now()
            self.logger.error(f"Task {task.title} failed: {e}")
    
    async def _isolate_host(self, affected_assets: List[str]) -> Dict[str, Any]:
        """Isolate affected hosts from the network."""
        isolated_hosts = []
        failed_hosts = []
        
        for asset in affected_assets:
            try:
                # Simulate host isolation
                await asyncio.sleep(0.1)  # Simulate API call
                isolated_hosts.append(asset)
                self.logger.info(f"Isolated host: {asset}")
            except Exception as e:
                failed_hosts.append({"host": asset, "error": str(e)})
        
        return {
            "status": "success" if not failed_hosts else "partial",
            "isolated_hosts": isolated_hosts,
            "failed_hosts": failed_hosts,
            "total_attempts": len(affected_assets)
        }
    
    async def _block_ip(self, indicators: List[str]) -> Dict[str, Any]:
        """Block malicious IP addresses."""
        blocked_ips = []
        failed_ips = []
        
        # Extract IP addresses from indicators
        ips = [indicator.split(":")[-1] for indicator in indicators if indicator.startswith("ip:")]
        
        for ip in ips:
            try:
                # Simulate IP blocking
                await asyncio.sleep(0.1)
                blocked_ips.append(ip)
                self.logger.info(f"Blocked IP: {ip}")
            except Exception as e:
                failed_ips.append({"ip": ip, "error": str(e)})
        
        return {
            "status": "success" if not failed_ips else "partial",
            "blocked_ips": blocked_ips,
            "failed_ips": failed_ips,
            "total_attempts": len(ips)
        }
    
    async def _disable_account(self, impacted_users: List[str]) -> Dict[str, Any]:
        """Disable compromised user accounts."""
        disabled_accounts = []
        failed_accounts = []
        
        for user in impacted_users:
            try:
                # Simulate account disabling
                await asyncio.sleep(0.1)
                disabled_accounts.append(user)
                self.logger.info(f"Disabled account: {user}")
            except Exception as e:
                failed_accounts.append({"user": user, "error": str(e)})
        
        return {
            "status": "success" if not failed_accounts else "partial", 
            "disabled_accounts": disabled_accounts,
            "failed_accounts": failed_accounts,
            "total_attempts": len(impacted_users)
        }
    
    async def _collect_evidence(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Collect forensic evidence related to the incident."""
        collected_artifacts = []
        
        # Simulate evidence collection
        for asset in incident.affected_assets:
            artifact_id = f"artifact_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{asset}"
            artifact = IncidentArtifact(
                id=artifact_id,
                name=f"Memory dump - {asset}",
                type="memory_dump",
                path=f"/forensics/{incident.id}/{artifact_id}.dmp",
                collected_at=datetime.now(),
                metadata={
                    "asset": asset,
                    "collection_method": "automated",
                    "incident_id": incident.id
                }
            )
            collected_artifacts.append(artifact)
        
        await asyncio.sleep(0.2)  # Simulate collection time
        
        return {
            "status": "success",
            "artifacts_collected": len(collected_artifacts),
            "artifacts": [
                {
                    "id": art.id,
                    "name": art.name,
                    "type": art.type,
                    "path": art.path
                }
                for art in collected_artifacts
            ]
        }
    
    async def _deploy_countermeasures(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Deploy security countermeasures."""
        countermeasures = []
        
        # Determine countermeasures based on incident type and MITRE techniques
        if "T1059" in incident.mitre_techniques:  # Command and Scripting Interpreter
            countermeasures.append("Enhanced PowerShell logging")
            countermeasures.append("Script execution monitoring")
        
        if "T1071" in incident.mitre_techniques:  # Application Layer Protocol
            countermeasures.append("Network traffic analysis")
            countermeasures.append("C2 domain blocking")
        
        if "T1041" in incident.mitre_techniques:  # Exfiltration Over C2 Channel
            countermeasures.append("Data loss prevention rules")
            countermeasures.append("Egress traffic monitoring")
        
        # Default countermeasures
        if not countermeasures:
            countermeasures = ["Enhanced monitoring", "Signature updates"]
        
        await asyncio.sleep(0.3)  # Simulate deployment time
        
        return {
            "status": "success",
            "countermeasures_deployed": countermeasures,
            "deployment_time": datetime.now().isoformat()
        }
    
    async def _notify_stakeholders(self, incident: SecurityIncident, notification_type: str) -> Dict[str, Any]:
        """Send notifications to relevant stakeholders."""
        notifications_sent = []
        
        # Determine notification recipients based on incident severity and type
        recipients = ["security_team@company.com"]
        
        if incident.severity in ["high", "critical"]:
            recipients.extend(["ciso@company.com", "incident_response@company.com"])
        
        if incident.priority == IncidentPriority.P1:
            recipients.append("emergency_contact@company.com")
        
        if "data_exfiltration" in incident.title.lower():
            recipients.extend(["legal@company.com", "privacy@company.com"])
        
        # Simulate notification sending
        for recipient in recipients:
            notification = {
                "recipient": recipient,
                "subject": f"Security Incident {incident.id}: {incident.title}",
                "content": f"Incident: {incident.description}\nSeverity: {incident.severity}\nStatus: {incident.status.value}",
                "sent_at": datetime.now().isoformat()
            }
            notifications_sent.append(notification)
        
        await asyncio.sleep(0.1)
        
        return {
            "status": "success",
            "notifications_sent": len(notifications_sent),
            "recipients": [n["recipient"] for n in notifications_sent]
        }
    
    async def _create_ticket(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Create external ticket for incident tracking."""
        ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d')}-{incident.id[-4:]}"
        
        ticket_data = {
            "id": ticket_id,
            "title": f"Security Incident: {incident.title}",
            "description": incident.description,
            "severity": incident.severity,
            "priority": incident.priority.value,
            "assignee": incident.assigned_to,
            "created_at": datetime.now().isoformat(),
            "source": "security_automation"
        }
        
        await asyncio.sleep(0.1)  # Simulate ticket creation
        
        return {
            "status": "success",
            "ticket_id": ticket_id,
            "ticket_url": f"https://ticketing.company.com/tickets/{ticket_id}"
        }


class IncidentResponseOrchestrator:
    """Main incident response orchestration engine."""
    
    def __init__(self):
        self.playbook_engine = PlaybookEngine()
        self.active_incidents = {}
        self.incident_history = []
        self.response_metrics = defaultdict(list)
        self.logger = logging.getLogger("IncidentResponseOrchestrator")
    
    async def create_incident(
        self,
        title: str,
        description: str,
        severity: str,
        detection_source: str,
        detection_method: str,
        confidence: float,
        affected_assets: List[str] = None,
        indicators_of_compromise: List[str] = None,
        mitre_techniques: List[str] = None
    ) -> SecurityIncident:
        """Create a new security incident."""
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Determine priority based on severity and other factors
        priority = self._calculate_priority(severity, affected_assets or [], confidence)
        
        incident = SecurityIncident(
            id=incident_id,
            title=title,
            description=description,
            severity=severity,
            priority=priority,
            status=IncidentStatus.NEW,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            detection_source=detection_source,
            detection_method=detection_method,
            confidence=confidence,
            affected_assets=affected_assets or [],
            indicators_of_compromise=indicators_of_compromise or [],
            mitre_techniques=mitre_techniques or [],
            detection_time=datetime.now()
        )
        
        # Add to active incidents
        self.active_incidents[incident_id] = incident
        
        # Log incident creation
        self.logger.warning(
            f"Created security incident {incident_id}: {title} "
            f"(Severity: {severity}, Priority: {priority.value})"
        )
        
        # Trigger automated response
        await self._trigger_automated_response(incident)
        
        return incident
    
    def _calculate_priority(self, severity: str, affected_assets: List[str], confidence: float) -> IncidentPriority:
        """Calculate incident priority based on multiple factors."""
        # Start with severity mapping
        severity_priority = {
            "critical": IncidentPriority.P1,
            "high": IncidentPriority.P2,
            "medium": IncidentPriority.P3,
            "low": IncidentPriority.P4
        }.get(severity.lower(), IncidentPriority.P3)
        
        # Adjust based on affected assets
        critical_assets = ["domain_controller", "database_server", "payment_system"]
        if any(asset in critical_assets for asset in affected_assets):
            if severity_priority.value > IncidentPriority.P2.value:
                severity_priority = IncidentPriority.P2
        
        # Adjust based on confidence
        if confidence >= 0.9 and severity in ["high", "critical"]:
            severity_priority = IncidentPriority.P1
        
        return severity_priority
    
    async def _trigger_automated_response(self, incident: SecurityIncident) -> None:
        """Trigger automated response based on incident characteristics."""
        # Determine appropriate playbook
        playbook_name = self._select_playbook(incident)
        
        if playbook_name:
            # Execute playbook
            tasks = await self.playbook_engine.execute_playbook(playbook_name, incident)
            incident.tasks.extend(tasks)
            
            # Update incident status
            incident.status = IncidentStatus.INVESTIGATING
            incident.response_time = datetime.now()
            incident.updated_at = datetime.now()
            
            self.logger.info(
                f"Triggered automated response for incident {incident.id} "
                f"using playbook '{playbook_name}' ({len(tasks)} tasks)"
            )
    
    def _select_playbook(self, incident: SecurityIncident) -> Optional[str]:
        """Select appropriate playbook based on incident characteristics."""
        # Check for malware indicators
        if any(technique in ["T1059", "T1055", "T1204"] for technique in incident.mitre_techniques):
            return "malware_infection"
        
        # Check for data exfiltration
        if any(technique in ["T1041", "T1020"] for technique in incident.mitre_techniques):
            return "data_exfiltration"
        
        # Check for credential compromise
        if any(technique in ["T1110", "T1078"] for technique in incident.mitre_techniques):
            return "credential_compromise"
        
        # Check for APT indicators
        if (len(incident.mitre_techniques) >= 3 or
            any(technique in ["T1021", "T1068", "T1562"] for technique in incident.mitre_techniques)):
            return "advanced_persistent_threat"
        
        # Default to malware playbook for unknown cases
        return "malware_infection"
    
    async def update_incident_status(self, incident_id: str, new_status: IncidentStatus) -> bool:
        """Update the status of an incident."""
        if incident_id not in self.active_incidents:
            self.logger.error(f"Incident not found: {incident_id}")
            return False
        
        incident = self.active_incidents[incident_id]
        old_status = incident.status
        incident.status = new_status
        incident.updated_at = datetime.now()
        
        # Update timing metrics based on status transitions
        if new_status == IncidentStatus.CONTAINING and not incident.containment_time:
            incident.containment_time = datetime.now()
        elif new_status == IncidentStatus.CLOSED:
            incident.closed_at = datetime.now()
            incident.recovery_time = datetime.now()
            
            # Move to history
            self.incident_history.append(incident)
            del self.active_incidents[incident_id]
        
        self.logger.info(
            f"Updated incident {incident_id} status: {old_status.value} -> {new_status.value}"
        )
        
        return True
    
    async def add_incident_artifact(
        self,
        incident_id: str,
        artifact_name: str,
        artifact_type: str,
        artifact_path: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add an artifact to an incident."""
        if incident_id not in self.active_incidents:
            return False
        
        incident = self.active_incidents[incident_id]
        
        artifact = IncidentArtifact(
            id=f"artifact_{len(incident.artifacts) + 1}",
            name=artifact_name,
            type=artifact_type,
            path=artifact_path,
            collected_at=datetime.now(),
            metadata=metadata or {}
        )
        
        incident.artifacts.append(artifact)
        incident.updated_at = datetime.now()
        
        self.logger.info(f"Added artifact '{artifact_name}' to incident {incident_id}")
        return True
    
    def get_incident_summary(self) -> Dict[str, Any]:
        """Get summary of current incidents and metrics."""
        active_count = len(self.active_incidents)
        status_breakdown = defaultdict(int)
        severity_breakdown = defaultdict(int)
        priority_breakdown = defaultdict(int)
        
        for incident in self.active_incidents.values():
            status_breakdown[incident.status.value] += 1
            severity_breakdown[incident.severity] += 1
            priority_breakdown[incident.priority.value] += 1
        
        # Calculate average response times from history
        response_times = []
        containment_times = []
        
        for incident in self.incident_history[-50:]:  # Last 50 incidents
            if incident.response_time and incident.detection_time:
                response_delta = incident.response_time - incident.detection_time
                response_times.append(response_delta.total_seconds() / 60)  # Minutes
            
            if incident.containment_time and incident.detection_time:
                containment_delta = incident.containment_time - incident.detection_time
                containment_times.append(containment_delta.total_seconds() / 60)  # Minutes
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        avg_containment_time = sum(containment_times) / len(containment_times) if containment_times else 0
        
        return {
            "active_incidents": active_count,
            "total_incidents_24h": len([
                inc for inc in self.incident_history
                if inc.created_at > datetime.now() - timedelta(hours=24)
            ]),
            "status_breakdown": dict(status_breakdown),
            "severity_breakdown": dict(severity_breakdown),
            "priority_breakdown": dict(priority_breakdown),
            "metrics": {
                "avg_response_time_minutes": round(avg_response_time, 2),
                "avg_containment_time_minutes": round(avg_containment_time, 2),
                "incidents_closed_24h": len([
                    inc for inc in self.incident_history
                    if inc.closed_at and inc.closed_at > datetime.now() - timedelta(hours=24)
                ])
            },
            "recent_incidents": [
                {
                    "id": inc.id,
                    "title": inc.title,
                    "severity": inc.severity,
                    "status": inc.status.value,
                    "created_at": inc.created_at.isoformat()
                }
                for inc in sorted(
                    self.active_incidents.values(),
                    key=lambda x: x.created_at,
                    reverse=True
                )[:10]
            ]
        }
    
    async def run_incident_metrics_collection(self) -> None:
        """Collect and store incident response metrics."""
        while True:
            try:
                # Collect current metrics
                current_time = datetime.now()
                
                metrics = {
                    "timestamp": current_time.isoformat(),
                    "active_incidents": len(self.active_incidents),
                    "p1_incidents": len([i for i in self.active_incidents.values() if i.priority == IncidentPriority.P1]),
                    "p2_incidents": len([i for i in self.active_incidents.values() if i.priority == IncidentPriority.P2])
                }
                
                # Store metrics for trend analysis
                self.response_metrics["incident_counts"].append(metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = current_time - timedelta(hours=24)
                for metric_type in self.response_metrics:
                    self.response_metrics[metric_type] = [
                        m for m in self.response_metrics[metric_type]
                        if datetime.fromisoformat(m["timestamp"]) > cutoff_time
                    ]
                
                # Sleep for 5 minutes before next collection
                await asyncio.sleep(300)
            
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(60)  # Shorter retry interval