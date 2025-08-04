"""Input validation and security checks."""

import re
import ipaddress
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ValidationError(Exception):
    """Custom validation error."""
    pass


class SecurityLevel(Enum):
    """Security validation levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    message: str
    level: SecurityLevel = SecurityLevel.LOW
    metadata: Dict[str, Any] = None


class InputValidator:
    """General input validation."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        
        # Common patterns
        self.patterns = {
            "alphanumeric": re.compile(r"^[a-zA-Z0-9_-]+$"),
            "service_name": re.compile(r"^[a-z0-9-]+$"),
            "namespace": re.compile(r"^[a-z0-9-]+$"),
            "kubernetes_name": re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"),
            "container_name": re.compile(r"^[a-z0-9-]+$"),
            "port": re.compile(r"^([1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$"),
            "duration": re.compile(r"^\d+(\.\d+)?[smhd]?$"),
        }
        
        # Dangerous patterns to block
        self.dangerous_patterns = [
            re.compile(r"[;&|`$()]"),  # Shell metacharacters
            re.compile(r"\.\.\/"),     # Path traversal
            re.compile(r"<script"),    # XSS
            re.compile(r"union.*select", re.IGNORECASE),  # SQL injection
            re.compile(r"eval\s*\("),  # Code injection
            re.compile(r"exec\s*\("),  # Code execution
        ]
    
    def validate_string(
        self,
        value: str,
        pattern_name: Optional[str] = None,
        min_length: int = 1,
        max_length: int = 255,
        allow_empty: bool = False
    ) -> ValidationResult:
        """Validate string input."""
        if not isinstance(value, str):
            return ValidationResult(
                valid=False,
                message="Value must be a string",
                level=SecurityLevel.HIGH
            )
        
        if not allow_empty and not value.strip():
            return ValidationResult(
                valid=False,
                message="Value cannot be empty",
                level=SecurityLevel.MEDIUM
            )
        
        if len(value) < min_length:
            return ValidationResult(
                valid=False,
                message=f"Value too short (minimum {min_length} characters)",
                level=SecurityLevel.MEDIUM
            )
        
        if len(value) > max_length:
            return ValidationResult(
                valid=False,
                message=f"Value too long (maximum {max_length} characters)",
                level=SecurityLevel.MEDIUM
            )
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(value):
                return ValidationResult(
                    valid=False,
                    message="Value contains potentially dangerous characters",
                    level=SecurityLevel.CRITICAL
                )
        
        # Check specific pattern if provided
        if pattern_name and pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            if not pattern.match(value):
                return ValidationResult(
                    valid=False,
                    message=f"Value does not match required pattern: {pattern_name}",
                    level=SecurityLevel.MEDIUM
                )
        
        return ValidationResult(valid=True, message="Valid string")
    
    def validate_ip_address(self, value: str) -> ValidationResult:
        """Validate IP address."""
        try:
            ip = ipaddress.ip_address(value)
            
            # Check for private/reserved ranges in high security mode
            if self.security_level == SecurityLevel.CRITICAL:
                if ip.is_private or ip.is_reserved or ip.is_loopback:
                    return ValidationResult(
                        valid=False,
                        message="Private/reserved IP addresses not allowed",
                        level=SecurityLevel.HIGH
                    )
            
            return ValidationResult(
                valid=True,
                message="Valid IP address",
                metadata={"ip_version": ip.version, "is_private": ip.is_private}
            )
        
        except ValueError as e:
            return ValidationResult(
                valid=False,
                message=f"Invalid IP address: {e}",
                level=SecurityLevel.HIGH
            )
    
    def validate_port(self, value: Union[int, str]) -> ValidationResult:
        """Validate port number."""
        try:
            port = int(value)
            
            if port < 1 or port > 65535:
                return ValidationResult(
                    valid=False,
                    message="Port must be between 1 and 65535",
                    level=SecurityLevel.MEDIUM
                )
            
            # Check for privileged ports in high security mode
            if self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                if port < 1024:
                    return ValidationResult(
                        valid=False,
                        message="Privileged ports (< 1024) not allowed",
                        level=SecurityLevel.HIGH
                    )
            
            return ValidationResult(
                valid=True,
                message="Valid port",
                metadata={"port": port, "privileged": port < 1024}
            )
        
        except (ValueError, TypeError):
            return ValidationResult(
                valid=False,
                message="Port must be a valid integer",
                level=SecurityLevel.MEDIUM
            )
    
    def validate_list(
        self,
        value: List[Any],
        item_validator: Optional[callable] = None,
        min_items: int = 0,
        max_items: int = 100
    ) -> ValidationResult:
        """Validate list input."""
        if not isinstance(value, list):
            return ValidationResult(
                valid=False,
                message="Value must be a list",
                level=SecurityLevel.MEDIUM
            )
        
        if len(value) < min_items:
            return ValidationResult(
                valid=False,
                message=f"List too short (minimum {min_items} items)",
                level=SecurityLevel.MEDIUM
            )
        
        if len(value) > max_items:
            return ValidationResult(
                valid=False,
                message=f"List too long (maximum {max_items} items)",
                level=SecurityLevel.MEDIUM
            )
        
        # Validate individual items if validator provided
        if item_validator:
            for i, item in enumerate(value):
                result = item_validator(item)
                if not result.valid:
                    return ValidationResult(
                        valid=False,
                        message=f"Item {i}: {result.message}",
                        level=result.level
                    )
        
        return ValidationResult(valid=True, message="Valid list")
    
    def validate_dictionary(
        self,
        value: Dict[str, Any],
        required_keys: Optional[List[str]] = None,
        allowed_keys: Optional[List[str]] = None,
        key_validator: Optional[callable] = None,
        value_validator: Optional[callable] = None
    ) -> ValidationResult:
        """Validate dictionary input."""
        if not isinstance(value, dict):
            return ValidationResult(
                valid=False,
                message="Value must be a dictionary",
                level=SecurityLevel.MEDIUM
            )
        
        # Check required keys
        if required_keys:
            missing_keys = set(required_keys) - set(value.keys())
            if missing_keys:
                return ValidationResult(
                    valid=False,
                    message=f"Missing required keys: {', '.join(missing_keys)}",
                    level=SecurityLevel.HIGH
                )
        
        # Check allowed keys
        if allowed_keys:
            extra_keys = set(value.keys()) - set(allowed_keys)
            if extra_keys:
                return ValidationResult(
                    valid=False,
                    message=f"Unexpected keys: {', '.join(extra_keys)}",
                    level=SecurityLevel.MEDIUM
                )
        
        # Validate keys and values
        for key, val in value.items():
            if key_validator:
                key_result = key_validator(key)
                if not key_result.valid:
                    return ValidationResult(
                        valid=False,
                        message=f"Invalid key '{key}': {key_result.message}",
                        level=key_result.level
                    )
            
            if value_validator:
                val_result = value_validator(val)
                if not val_result.valid:
                    return ValidationResult(
                        valid=False,
                        message=f"Invalid value for key '{key}': {val_result.message}",
                        level=val_result.level
                    )
        
        return ValidationResult(valid=True, message="Valid dictionary")


class SecurityValidator:
    """Security-specific validation."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.input_validator = InputValidator(security_level)
        
        # Security-specific patterns
        self.security_patterns = {
            "malicious_commands": [
                re.compile(r"rm\s+-rf", re.IGNORECASE),
                re.compile(r"curl\s+.*\|\s*sh", re.IGNORECASE),
                re.compile(r"wget\s+.*\|\s*sh", re.IGNORECASE),
                re.compile(r"nc\s+.*-e", re.IGNORECASE),
                re.compile(r"/bin/sh", re.IGNORECASE),
                re.compile(r"/bin/bash", re.IGNORECASE),
                re.compile(r"powershell", re.IGNORECASE),
                re.compile(r"cmd\.exe", re.IGNORECASE),
            ],
            "sensitive_files": [
                re.compile(r"/etc/passwd"),
                re.compile(r"/etc/shadow"),
                re.compile(r"\.ssh/id_rsa"),
                re.compile(r"\.aws/credentials"),
                re.compile(r"config\.json"),
                re.compile(r"\.env"),
            ]
        }
    
    def validate_agent_action(self, action_data: Dict[str, Any]) -> ValidationResult:
        """Validate agent action data."""
        # Validate required fields
        required_fields = ["type", "target"]
        result = self.input_validator.validate_dictionary(
            action_data,
            required_keys=required_fields
        )
        if not result.valid:
            return result
        
        # Validate action type
        action_type = action_data.get("type", "")
        allowed_actions = [
            "reconnaissance", "vulnerability_scan", "exploit_attempt",
            "privilege_escalation", "persistence", "data_exfiltration",
            "lateral_movement", "threat_detection", "vulnerability_patching",
            "access_control", "network_monitoring", "incident_response",
            "system_hardening", "honeypot_deployment"
        ]
        
        if action_type not in allowed_actions:
            return ValidationResult(
                valid=False,
                message=f"Invalid action type: {action_type}",
                level=SecurityLevel.HIGH
            )
        
        # Validate target
        target = action_data.get("target", "")
        result = self.input_validator.validate_string(
            target,
            pattern_name="service_name",
            max_length=64
        )
        if not result.valid:
            return result
        
        # Check for malicious content in payload
        payload = action_data.get("payload", {})
        if isinstance(payload, dict):
            result = self._check_malicious_content(payload)
            if not result.valid:
                return result
        
        return ValidationResult(valid=True, message="Valid agent action")
    
    def validate_service_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate service configuration."""
        # Validate required fields
        required_fields = ["name", "image"]
        result = self.input_validator.validate_dictionary(
            config,
            required_keys=required_fields
        )
        if not result.valid:
            return result
        
        # Validate service name
        result = self.input_validator.validate_string(
            config["name"],
            pattern_name="kubernetes_name",
            max_length=63
        )
        if not result.valid:
            return result
        
        # Validate image name
        image = config["image"]
        if not isinstance(image, str) or len(image) > 256:
            return ValidationResult(
                valid=False,
                message="Invalid container image name",
                level=SecurityLevel.HIGH
            )
        
        # Check for suspicious image names
        suspicious_images = ["busybox", "alpine", "ubuntu:latest", "scratch"]
        if any(suspicious in image.lower() for suspicious in suspicious_images):
            if self.security_level == SecurityLevel.CRITICAL:
                return ValidationResult(
                    valid=False,
                    message="Suspicious container image not allowed",
                    level=SecurityLevel.HIGH
                )
        
        # Validate ports if present
        if "ports" in config:
            ports = config["ports"]
            if isinstance(ports, list):
                for port in ports:
                    result = self.input_validator.validate_port(port)
                    if not result.valid:
                        return result
        
        return ValidationResult(valid=True, message="Valid service configuration")
    
    def validate_kubernetes_resource(self, resource: Dict[str, Any]) -> ValidationResult:
        """Validate Kubernetes resource definition."""
        # Check for required Kubernetes fields
        if "metadata" not in resource:
            return ValidationResult(
                valid=False,
                message="Missing metadata field",
                level=SecurityLevel.HIGH
            )
        
        metadata = resource["metadata"]
        if "name" not in metadata:
            return ValidationResult(
                valid=False,
                message="Missing resource name",
                level=SecurityLevel.HIGH
            )
        
        # Validate resource name
        result = self.input_validator.validate_string(
            metadata["name"],
            pattern_name="kubernetes_name",
            max_length=63
        )
        if not result.valid:
            return result
        
        # Check for dangerous security contexts
        if "spec" in resource:
            spec = resource["spec"]
            
            # Check pod security context
            if "template" in spec and "spec" in spec["template"]:
                pod_spec = spec["template"]["spec"]
                
                # Check for privileged containers
                if "containers" in pod_spec:
                    for container in pod_spec["containers"]:
                        if "securityContext" in container:
                            sec_ctx = container["securityContext"]
                            if sec_ctx.get("privileged", False):
                                return ValidationResult(
                                    valid=False,
                                    message="Privileged containers not allowed",
                                    level=SecurityLevel.CRITICAL
                                )
                            
                            if sec_ctx.get("allowPrivilegeEscalation", False):
                                return ValidationResult(
                                    valid=False,
                                    message="Privilege escalation not allowed",
                                    level=SecurityLevel.CRITICAL
                                )
        
        return ValidationResult(valid=True, message="Valid Kubernetes resource")
    
    def _check_malicious_content(self, data: Any) -> ValidationResult:
        """Check for malicious content in data."""
        if isinstance(data, str):
            # Check for malicious commands
            for pattern_list in self.security_patterns.values():
                for pattern in pattern_list:
                    if pattern.search(data):
                        return ValidationResult(
                            valid=False,
                            message="Potentially malicious content detected",
                            level=SecurityLevel.CRITICAL
                        )
        
        elif isinstance(data, dict):
            for key, value in data.items():
                result = self._check_malicious_content(key)
                if not result.valid:
                    return result
                
                result = self._check_malicious_content(value)
                if not result.valid:
                    return result
        
        elif isinstance(data, list):
            for item in data:
                result = self._check_malicious_content(item)
                if not result.valid:
                    return result
        
        return ValidationResult(valid=True, message="Content appears safe")
    
    def validate_network_access(
        self,
        source_ip: str,
        target_ip: str,
        port: int
    ) -> ValidationResult:
        """Validate network access request."""
        # Validate source IP
        result = self.input_validator.validate_ip_address(source_ip)
        if not result.valid:
            return ValidationResult(
                valid=False,
                message=f"Invalid source IP: {result.message}",
                level=result.level
            )
        
        # Validate target IP
        result = self.input_validator.validate_ip_address(target_ip)
        if not result.valid:
            return ValidationResult(
                valid=False,
                message=f"Invalid target IP: {result.message}",
                level=result.level
            )
        
        # Validate port
        result = self.input_validator.validate_port(port)
        if not result.valid:
            return result
        
        # Check for same IP (potential loop)
        if source_ip == target_ip:
            return ValidationResult(
                valid=False,
                message="Source and target IP cannot be the same",
                level=SecurityLevel.MEDIUM
            )
        
        return ValidationResult(valid=True, message="Valid network access")