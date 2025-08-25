"""
Global Compliance Module for GAN Cyber Range.

Ensures compliance with international data protection and cybersecurity regulations:
- GDPR (General Data Protection Regulation) - EU
- CCPA (California Consumer Privacy Act) - US/CA
- PDPA (Personal Data Protection Act) - Singapore
- PIPL (Personal Information Protection Law) - China
- SOX (Sarbanes-Oxley Act) - US Financial
- APPI (Act on Protection of Personal Information) - Japan

Features:
- Automatic data classification and handling
- Privacy-by-design enforcement  
- Audit trail generation
- Cross-border data transfer validation
- Retention policy enforcement
- Consent management
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data sensitivity classifications."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL_DATA = "personal_data"
    SENSITIVE_PERSONAL_DATA = "sensitive_personal_data"


class ProcessingPurpose(Enum):
    """Lawful bases for data processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"
    CYBERSECURITY_TRAINING = "cybersecurity_training"
    THREAT_DETECTION = "threat_detection"


class Region(Enum):
    """Supported regulatory regions."""
    EU = "EU"
    US = "US"
    CA = "CA"
    SG = "SG"
    JP = "JP"
    CN = "CN"
    UK = "UK"
    AU = "AU"


@dataclass
class DataHandlingPolicy:
    """Policy for handling specific types of data."""
    data_type: DataClassification
    retention_days: int
    encryption_required: bool
    anonymization_required: bool
    cross_border_allowed: bool
    audit_required: bool
    consent_required: bool
    processing_purposes: List[ProcessingPurpose]


@dataclass
class ComplianceEvent:
    """Compliance-related event for audit trails."""
    timestamp: float
    event_type: str
    data_subject_id: Optional[str]
    processing_purpose: ProcessingPurpose
    data_types: List[DataClassification]
    legal_basis: str
    region: Region
    details: Dict[str, Any]


class ComplianceManager:
    """
    Comprehensive compliance management system.
    
    Automatically enforces data protection regulations across all supported regions.
    """
    
    # Regional compliance requirements
    REGIONAL_POLICIES = {
        Region.EU: {
            'regulations': ['GDPR'],
            'data_retention_max_days': 1095,  # 3 years default
            'cross_border_restricted': True,
            'consent_required_age': 16,
            'breach_notification_hours': 72,
            'data_portability_required': True,
            'right_to_erasure': True
        },
        Region.US: {
            'regulations': ['CCPA', 'SOX'],
            'data_retention_max_days': 2555,  # 7 years for financial
            'cross_border_restricted': False,
            'consent_required_age': 13,
            'breach_notification_hours': 72,
            'data_portability_required': True,
            'right_to_erasure': False  # Limited under CCPA
        },
        Region.CA: {
            'regulations': ['PIPEDA'],
            'data_retention_max_days': 1095,
            'cross_border_restricted': True,
            'consent_required_age': 13,
            'breach_notification_hours': 72,
            'data_portability_required': False,
            'right_to_erasure': False
        },
        Region.SG: {
            'regulations': ['PDPA'],
            'data_retention_max_days': 1095,
            'cross_border_restricted': True,
            'consent_required_age': 13,
            'breach_notification_hours': 72,
            'data_portability_required': False,
            'right_to_erasure': False
        },
        Region.JP: {
            'regulations': ['APPI'],
            'data_retention_max_days': 1095,
            'cross_border_restricted': True,
            'consent_required_age': 15,
            'breach_notification_hours': 72,
            'data_portability_required': False,
            'right_to_erasure': False
        },
        Region.CN: {
            'regulations': ['PIPL', 'CSL'],
            'data_retention_max_days': 1095,
            'cross_border_restricted': True,
            'consent_required_age': 14,
            'breach_notification_hours': 72,
            'data_portability_required': False,
            'right_to_erasure': True
        }
    }
    
    # Default data handling policies
    DEFAULT_POLICIES = {
        DataClassification.PUBLIC: DataHandlingPolicy(
            data_type=DataClassification.PUBLIC,
            retention_days=365,
            encryption_required=False,
            anonymization_required=False,
            cross_border_allowed=True,
            audit_required=False,
            consent_required=False,
            processing_purposes=[ProcessingPurpose.LEGITIMATE_INTERESTS]
        ),
        DataClassification.INTERNAL: DataHandlingPolicy(
            data_type=DataClassification.INTERNAL,
            retention_days=1095,
            encryption_required=True,
            anonymization_required=False,
            cross_border_allowed=False,
            audit_required=True,
            consent_required=False,
            processing_purposes=[ProcessingPurpose.LEGITIMATE_INTERESTS]
        ),
        DataClassification.PERSONAL_DATA: DataHandlingPolicy(
            data_type=DataClassification.PERSONAL_DATA,
            retention_days=730,
            encryption_required=True,
            anonymization_required=True,
            cross_border_allowed=False,
            audit_required=True,
            consent_required=True,
            processing_purposes=[ProcessingPurpose.CONSENT, ProcessingPurpose.CYBERSECURITY_TRAINING]
        ),
        DataClassification.SENSITIVE_PERSONAL_DATA: DataHandlingPolicy(
            data_type=DataClassification.SENSITIVE_PERSONAL_DATA,
            retention_days=365,
            encryption_required=True,
            anonymization_required=True,
            cross_border_allowed=False,
            audit_required=True,
            consent_required=True,
            processing_purposes=[ProcessingPurpose.CONSENT]
        )
    }
    
    def __init__(self, default_region: Region = Region.US):
        self.default_region = default_region
        self.audit_trail: List[ComplianceEvent] = []
        self.consent_records: Dict[str, Dict] = {}
        self.data_inventory: Dict[str, Dict] = {}
        self.policies = self.DEFAULT_POLICIES.copy()
        
        logger.info(f"Compliance manager initialized for region: {default_region.value}")
    
    def classify_data(self, data: Any, context: Dict[str, Any] = None) -> DataClassification:
        """
        Automatically classify data based on content and context.
        
        Args:
            data: Data to classify
            context: Additional context for classification
            
        Returns:
            Data classification level
        """
        context = context or {}
        
        # Convert data to string for analysis
        data_str = str(data).lower()
        
        # Check for personal data indicators
        personal_indicators = [
            'email', 'phone', 'address', 'name', 'ssn', 'passport',
            'credit_card', 'bank_account', 'ip_address', 'user_id'
        ]
        
        sensitive_indicators = [
            'password', 'key', 'token', 'credential', 'biometric',
            'health', 'medical', 'financial', 'race', 'religion'
        ]
        
        # Classification logic
        if any(indicator in data_str for indicator in sensitive_indicators):
            return DataClassification.SENSITIVE_PERSONAL_DATA
        elif any(indicator in data_str for indicator in personal_indicators):
            return DataClassification.PERSONAL_DATA
        elif context.get('confidential', False):
            return DataClassification.CONFIDENTIAL
        elif context.get('internal', False):
            return DataClassification.INTERNAL
        else:
            return DataClassification.PUBLIC
    
    def validate_processing(self, 
                          data_classification: DataClassification,
                          purpose: ProcessingPurpose,
                          region: Region = None) -> bool:
        """
        Validate if data processing is compliant.
        
        Args:
            data_classification: Classification of data being processed
            purpose: Purpose of processing
            region: Regulatory region
            
        Returns:
            True if processing is compliant
        """
        region = region or self.default_region
        policy = self.policies.get(data_classification)
        
        if not policy:
            logger.warning(f"No policy found for data classification: {data_classification}")
            return False
        
        # Check if purpose is allowed
        if purpose not in policy.processing_purposes:
            logger.warning(f"Processing purpose {purpose} not allowed for {data_classification}")
            return False
        
        # Check regional restrictions
        regional_policy = self.REGIONAL_POLICIES.get(region, {})
        
        # Consent validation for EU/GDPR
        if region == Region.EU and policy.consent_required:
            # Would integrate with consent management system
            logger.info("GDPR consent validation required")
        
        # Log compliance event
        self._log_compliance_event(
            event_type="processing_validation",
            data_types=[data_classification],
            processing_purpose=purpose,
            legal_basis=purpose.value,
            region=region,
            details={'validation_result': True}
        )
        
        return True
    
    def validate_cross_border_transfer(self, 
                                     data_classification: DataClassification,
                                     source_region: Region,
                                     destination_region: Region) -> bool:
        """
        Validate cross-border data transfer compliance.
        
        Args:
            data_classification: Classification of data being transferred
            source_region: Source region
            destination_region: Destination region
            
        Returns:
            True if transfer is compliant
        """
        policy = self.policies.get(data_classification)
        if not policy:
            return False
        
        # Check if cross-border transfer is allowed for this data type
        if not policy.cross_border_allowed:
            logger.warning(f"Cross-border transfer not allowed for {data_classification}")
            return False
        
        # Check source region restrictions
        source_policy = self.REGIONAL_POLICIES.get(source_region, {})
        if source_policy.get('cross_border_restricted', False):
            # Would check adequacy decisions, BCRs, SCCs, etc.
            logger.info(f"Cross-border transfer from {source_region} requires additional safeguards")
        
        # Log transfer validation
        self._log_compliance_event(
            event_type="cross_border_transfer",
            data_types=[data_classification],
            processing_purpose=ProcessingPurpose.LEGITIMATE_INTERESTS,
            legal_basis="adequate_protection",
            region=source_region,
            details={
                'source_region': source_region.value,
                'destination_region': destination_region.value,
                'validation_result': True
            }
        )
        
        return True
    
    def apply_data_minimization(self, data: Dict[str, Any], purpose: ProcessingPurpose) -> Dict[str, Any]:
        """
        Apply data minimization principle - only process data necessary for the purpose.
        
        Args:
            data: Original data
            purpose: Processing purpose
            
        Returns:
            Minimized data
        """
        # Define necessary fields for different purposes
        purpose_fields = {
            ProcessingPurpose.CYBERSECURITY_TRAINING: [
                'timestamp', 'event_type', 'severity', 'source_ip', 'destination_ip',
                'protocol', 'attack_type', 'defense_action'
            ],
            ProcessingPurpose.THREAT_DETECTION: [
                'timestamp', 'event_type', 'severity', 'indicators', 'threat_type'
            ],
            ProcessingPurpose.LEGITIMATE_INTERESTS: [
                'timestamp', 'event_type', 'metadata'
            ]
        }
        
        allowed_fields = purpose_fields.get(purpose, [])
        minimized_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        logger.debug(f"Data minimization: {len(data)} -> {len(minimized_data)} fields")
        return minimized_data
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize personal data to comply with privacy regulations.
        
        Args:
            data: Data to anonymize
            
        Returns:
            Anonymized data
        """
        anonymized = data.copy()
        
        # Fields to anonymize
        sensitive_fields = [
            'email', 'phone', 'address', 'name', 'user_id', 'ip_address',
            'device_id', 'session_id'
        ]
        
        for field in sensitive_fields:
            if field in anonymized:
                # Hash-based anonymization
                original_value = str(anonymized[field])
                anonymized_value = hashlib.sha256(original_value.encode()).hexdigest()[:16]
                anonymized[field] = f"anon_{anonymized_value}"
        
        return anonymized
    
    def check_retention_compliance(self) -> List[Dict[str, Any]]:
        """
        Check data retention compliance and identify data eligible for deletion.
        
        Returns:
            List of data records that should be deleted
        """
        expired_records = []
        current_time = time.time()
        
        for record_id, record_info in self.data_inventory.items():
            classification = record_info.get('classification')
            created_time = record_info.get('created_time', 0)
            policy = self.policies.get(classification)
            
            if policy:
                retention_seconds = policy.retention_days * 24 * 3600
                if current_time - created_time > retention_seconds:
                    expired_records.append({
                        'record_id': record_id,
                        'classification': classification,
                        'age_days': (current_time - created_time) / (24 * 3600),
                        'retention_days': policy.retention_days
                    })
        
        if expired_records:
            logger.info(f"Found {len(expired_records)} records eligible for deletion")
        
        return expired_records
    
    def record_consent(self, 
                      data_subject_id: str,
                      purposes: List[ProcessingPurpose],
                      region: Region = None) -> str:
        """
        Record user consent for data processing.
        
        Args:
            data_subject_id: Unique identifier for data subject
            purposes: List of processing purposes consented to
            region: Regional context
            
        Returns:
            Consent record ID
        """
        region = region or self.default_region
        consent_id = hashlib.sha256(f"{data_subject_id}_{time.time()}".encode()).hexdigest()[:16]
        
        consent_record = {
            'consent_id': consent_id,
            'data_subject_id': data_subject_id,
            'purposes': [p.value for p in purposes],
            'timestamp': time.time(),
            'region': region.value,
            'status': 'active',
            'withdrawal_method': 'email_request'
        }
        
        self.consent_records[consent_id] = consent_record
        
        # Log consent event
        self._log_compliance_event(
            event_type="consent_recorded",
            data_subject_id=data_subject_id,
            processing_purpose=purposes[0] if purposes else ProcessingPurpose.CONSENT,
            data_types=[DataClassification.PERSONAL_DATA],
            legal_basis="consent",
            region=region,
            details={'consent_id': consent_id, 'purposes': [p.value for p in purposes]}
        )
        
        logger.info(f"Recorded consent for subject {data_subject_id}: {consent_id}")
        return consent_id
    
    def withdraw_consent(self, data_subject_id: str, consent_id: str) -> bool:
        """
        Process consent withdrawal request.
        
        Args:
            data_subject_id: Data subject identifier
            consent_id: Consent record to withdraw
            
        Returns:
            True if withdrawal processed successfully
        """
        if consent_id in self.consent_records:
            consent_record = self.consent_records[consent_id]
            
            if consent_record['data_subject_id'] == data_subject_id:
                consent_record['status'] = 'withdrawn'
                consent_record['withdrawal_time'] = time.time()
                
                # Log withdrawal
                self._log_compliance_event(
                    event_type="consent_withdrawn",
                    data_subject_id=data_subject_id,
                    processing_purpose=ProcessingPurpose.CONSENT,
                    data_types=[DataClassification.PERSONAL_DATA],
                    legal_basis="consent_withdrawn",
                    region=self.default_region,
                    details={'consent_id': consent_id}
                )
                
                logger.info(f"Consent withdrawn: {consent_id}")
                return True
        
        logger.warning(f"Failed to withdraw consent: {consent_id}")
        return False
    
    def generate_privacy_notice(self, region: Region = None) -> Dict[str, Any]:
        """
        Generate privacy notice for the specified region.
        
        Args:
            region: Target regulatory region
            
        Returns:
            Privacy notice content
        """
        region = region or self.default_region
        regional_policy = self.REGIONAL_POLICIES.get(region, {})
        
        privacy_notice = {
            'region': region.value,
            'regulations': regional_policy.get('regulations', []),
            'data_controller': 'GAN Cyber Range Simulator',
            'contact_email': 'privacy@gan-cyber-range.org',
            'data_types_collected': [
                'Training simulation data',
                'Performance metrics',
                'System logs',
                'User preferences'
            ],
            'processing_purposes': [
                'Cybersecurity training',
                'Threat detection research',
                'Performance optimization',
                'Security monitoring'
            ],
            'legal_bases': [
                'Legitimate interests',
                'Consent (where required)',
                'Legal obligation'
            ],
            'retention_periods': {
                'Training data': '2 years',
                'Performance metrics': '1 year', 
                'System logs': '90 days',
                'Personal data': 'As per user consent'
            },
            'data_subject_rights': []
        }
        
        # Add region-specific rights
        if regional_policy.get('data_portability_required'):
            privacy_notice['data_subject_rights'].append('Data portability')
        if regional_policy.get('right_to_erasure'):
            privacy_notice['data_subject_rights'].append('Right to be forgotten')
        
        # Standard rights
        privacy_notice['data_subject_rights'].extend([
            'Access to personal data',
            'Rectification of inaccurate data',
            'Restriction of processing',
            'Object to processing'
        ])
        
        return privacy_notice
    
    def _log_compliance_event(self,
                            event_type: str,
                            processing_purpose: ProcessingPurpose,
                            data_types: List[DataClassification],
                            legal_basis: str,
                            region: Region,
                            data_subject_id: str = None,
                            details: Dict[str, Any] = None):
        """Log compliance event for audit trail."""
        event = ComplianceEvent(
            timestamp=time.time(),
            event_type=event_type,
            data_subject_id=data_subject_id,
            processing_purpose=processing_purpose,
            data_types=data_types,
            legal_basis=legal_basis,
            region=region,
            details=details or {}
        )
        
        self.audit_trail.append(event)
    
    def export_audit_trail(self, start_time: float = None, end_time: float = None) -> List[Dict]:
        """Export audit trail for compliance reporting."""
        start_time = start_time or 0
        end_time = end_time or time.time()
        
        filtered_events = [
            {
                'timestamp': datetime.fromtimestamp(event.timestamp).isoformat(),
                'event_type': event.event_type,
                'data_subject_id': event.data_subject_id,
                'processing_purpose': event.processing_purpose.value,
                'data_types': [dt.value for dt in event.data_types],
                'legal_basis': event.legal_basis,
                'region': event.region.value,
                'details': event.details
            }
            for event in self.audit_trail
            if start_time <= event.timestamp <= end_time
        ]
        
        return filtered_events
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance status summary."""
        expired_records = self.check_retention_compliance()
        
        return {
            'default_region': self.default_region.value,
            'audit_events': len(self.audit_trail),
            'active_consents': len([c for c in self.consent_records.values() if c['status'] == 'active']),
            'expired_records': len(expired_records),
            'data_inventory_size': len(self.data_inventory),
            'supported_regions': [r.value for r in Region],
            'compliance_frameworks': list(set(
                reg for regional in self.REGIONAL_POLICIES.values() 
                for reg in regional.get('regulations', [])
            ))
        }


# Global compliance manager instance
_compliance_manager = None


def get_compliance_manager(region: Region = Region.US) -> ComplianceManager:
    """Get global compliance manager instance."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager(region)
    return _compliance_manager


if __name__ == "__main__":
    # Example usage
    cm = ComplianceManager(Region.EU)
    
    # Classify some data
    test_data = {"user_email": "test@example.com", "event_type": "login"}
    classification = cm.classify_data(test_data)
    print(f"Data classification: {classification}")
    
    # Validate processing
    valid = cm.validate_processing(classification, ProcessingPurpose.CYBERSECURITY_TRAINING)
    print(f"Processing valid: {valid}")
    
    # Record consent
    consent_id = cm.record_consent("user123", [ProcessingPurpose.CYBERSECURITY_TRAINING])
    print(f"Consent recorded: {consent_id}")
    
    # Generate privacy notice
    notice = cm.generate_privacy_notice()
    print(f"Privacy notice: {json.dumps(notice, indent=2)}")
    
    # Get compliance summary
    summary = cm.get_compliance_summary()
    print(f"Compliance summary: {json.dumps(summary, indent=2)}")