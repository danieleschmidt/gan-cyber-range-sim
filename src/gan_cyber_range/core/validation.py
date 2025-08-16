"""Input validation and security validation framework."""

import re
import ipaddress
import json
from typing import Any, Dict, List, Optional, Union, Callable
from urllib.parse import urlparse
from pathlib import Path
import logging


class ValidationError(Exception):
    """Validation error exception."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class ValidationRule:
    """Base validation rule."""
    
    def __init__(self, name: str, error_message: str):
        self.name = name
        self.error_message = error_message
    
    def validate(self, value: Any) -> bool:
        """Validate value - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement validate")
    
    def __call__(self, value: Any) -> bool:
        """Make rule callable."""
        return self.validate(value)


class RequiredRule(ValidationRule):
    """Validate that value is not None or empty."""
    
    def __init__(self):
        super().__init__("required", "Field is required")
    
    def validate(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, list, dict)) and len(value) == 0:
            return False
        return True


class TypeRule(ValidationRule):
    """Validate value type."""
    
    def __init__(self, expected_type: type):
        self.expected_type = expected_type
        super().__init__(
            f"type_{expected_type.__name__}",
            f"Must be of type {expected_type.__name__}"
        )
    
    def validate(self, value: Any) -> bool:
        return isinstance(value, self.expected_type)


class RangeRule(ValidationRule):
    """Validate numeric range."""
    
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(
            "range",
            f"Must be between {min_val} and {max_val}"
        )
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        
        if self.min_val is not None and value < self.min_val:
            return False
        if self.max_val is not None and value > self.max_val:
            return False
        
        return True


class RegexRule(ValidationRule):
    """Validate against regex pattern."""
    
    def __init__(self, pattern: str, flags: int = 0):
        self.pattern = re.compile(pattern, flags)
        super().__init__(
            "regex",
            f"Must match pattern: {pattern}"
        )
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return bool(self.pattern.match(value))


class EmailRule(ValidationRule):
    """Validate email format."""
    
    def __init__(self):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        self.pattern = re.compile(email_pattern)
        super().__init__("email", "Must be valid email address")
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return bool(self.pattern.match(value))


class URLRule(ValidationRule):
    """Validate URL format."""
    
    def __init__(self, schemes: Optional[List[str]] = None):
        self.schemes = schemes or ['http', 'https']
        super().__init__("url", "Must be valid URL")
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        
        try:
            parsed = urlparse(value)
            return (
                parsed.scheme in self.schemes and
                bool(parsed.netloc)
            )
        except Exception:
            return False


class IPAddressRule(ValidationRule):
    """Validate IP address format."""
    
    def __init__(self, version: Optional[int] = None):
        self.version = version  # 4, 6, or None for both
        super().__init__("ip_address", "Must be valid IP address")
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        
        try:
            addr = ipaddress.ip_address(value)
            if self.version is None:
                return True
            return addr.version == self.version
        except ValueError:
            return False


class SecurityRule(ValidationRule):
    """Security-focused validation rules."""
    
    def __init__(self, rule_type: str):
        self.rule_type = rule_type
        super().__init__(f"security_{rule_type}", f"Security validation failed: {rule_type}")
    
    def validate(self, value: Any) -> bool:
        if self.rule_type == "no_script_tags":
            return self._check_no_script_tags(value)
        elif self.rule_type == "no_sql_injection":
            return self._check_no_sql_injection(value)
        elif self.rule_type == "safe_filename":
            return self._check_safe_filename(value)
        elif self.rule_type == "no_command_injection":
            return self._check_no_command_injection(value)
        return True
    
    def _check_no_script_tags(self, value: Any) -> bool:
        """Check for script tags and potentially dangerous HTML."""
        if not isinstance(value, str):
            return True
        
        dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe.*?>',
            r'<object.*?>',
            r'<embed.*?>'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE | re.DOTALL):
                return False
        
        return True
    
    def _check_no_sql_injection(self, value: Any) -> bool:
        """Check for SQL injection patterns."""
        if not isinstance(value, str):
            return True
        
        sql_patterns = [
            r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
            r"(\b(or|and)\s+\d+\s*=\s*\d+)",
            r"('|\")?\s*;\s*--",
            r"(\b(xp_|sp_)\w+)",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    def _check_safe_filename(self, value: Any) -> bool:
        """Check for safe filename (no path traversal)."""
        if not isinstance(value, str):
            return True
        
        # Check for path traversal
        if '..' in value or '/' in value or '\\' in value:
            return False
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        if any(char in value for char in dangerous_chars):
            return False
        
        return True
    
    def _check_no_command_injection(self, value: Any) -> bool:
        """Check for command injection patterns."""
        if not isinstance(value, str):
            return True
        
        command_patterns = [
            r'[;&|`]',
            r'\$\(',
            r'\${',
            r'<\s*\(',
            r'>\s*\(',
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, value):
                return False
        
        return True


class InputValidator:
    """General input validation framework."""
    
    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, field: str, rule: ValidationRule):
        """Add validation rule for a field."""
        if field not in self.rules:
            self.rules[field] = []
        self.rules[field].append(rule)
    
    def validate(self, data: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
        """Validate data against rules."""
        errors = {}
        validated_data = {}
        
        # Check all rules
        for field, rules in self.rules.items():
            value = data.get(field)
            field_errors = []
            
            for rule in rules:
                try:
                    if not rule.validate(value):
                        field_errors.append(rule.error_message)
                except Exception as e:
                    field_errors.append(f"Validation error: {str(e)}")
            
            if field_errors:
                errors[field] = field_errors
            else:
                validated_data[field] = value
        
        # Check for unexpected fields in strict mode
        if strict:
            unexpected_fields = set(data.keys()) - set(self.rules.keys())
            if unexpected_fields:
                errors['_unexpected'] = list(unexpected_fields)
        
        if errors:
            raise ValidationError(f"Validation failed: {errors}")
        
        return validated_data


class ConfigValidator(InputValidator):
    """Configuration validation with predefined rules."""
    
    def __init__(self):
        super().__init__()
        self._setup_config_rules()
    
    def _setup_config_rules(self):
        """Setup common configuration validation rules."""
        # API configuration
        self.add_rule('api_host', RequiredRule())
        self.add_rule('api_host', TypeRule(str))
        self.add_rule('api_port', RequiredRule())
        self.add_rule('api_port', TypeRule(int))
        self.add_rule('api_port', RangeRule(1, 65535))
        
        # Agent configuration
        self.add_rule('agent_timeout', TypeRule((int, float)))
        self.add_rule('agent_timeout', RangeRule(1, 3600))
        self.add_rule('max_agents', TypeRule(int))
        self.add_rule('max_agents', RangeRule(1, 1000))
        
        # Security configuration
        self.add_rule('jwt_secret', RequiredRule())
        self.add_rule('jwt_secret', TypeRule(str))
        self.add_rule('admin_email', EmailRule())
        
        # Kubernetes configuration
        self.add_rule('k8s_namespace', TypeRule(str))
        self.add_rule('k8s_namespace', RegexRule(r'^[a-z0-9-]+$'))


class SecurityValidator(InputValidator):
    """Security-focused input validator."""
    
    def __init__(self):
        super().__init__()
        self._setup_security_rules()
    
    def _setup_security_rules(self):
        """Setup security validation rules."""
        # Common security rules for all string inputs
        self.add_rule('*', SecurityRule('no_script_tags'))
        self.add_rule('*', SecurityRule('no_sql_injection'))
        self.add_rule('*', SecurityRule('no_command_injection'))
        
        # File-specific rules
        self.add_rule('filename', SecurityRule('safe_filename'))
        
        # Network-specific rules
        self.add_rule('ip_address', IPAddressRule())
        self.add_rule('url', URLRule())
    
    def validate(self, data: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
        """Validate with security rules applied to all fields."""
        # Apply universal security rules to all string fields
        universal_rules = self.rules.get('*', [])
        
        for field, value in data.items():
            if isinstance(value, str) and field not in self.rules:
                # Apply universal security rules to unspecified string fields
                for rule in universal_rules:
                    if field not in self.rules:
                        self.rules[field] = []
                    if rule not in self.rules[field]:
                        self.rules[field].append(rule)
        
        return super().validate(data, strict)


# Pre-configured validators
def create_api_validator() -> InputValidator:
    """Create validator for API inputs."""
    validator = InputValidator()
    
    # Common API field validations
    validator.add_rule('user_id', RequiredRule())
    validator.add_rule('user_id', TypeRule(str))
    validator.add_rule('user_id', RegexRule(r'^[a-zA-Z0-9_-]+$'))
    
    validator.add_rule('agent_type', RequiredRule())
    validator.add_rule('agent_type', TypeRule(str))
    validator.add_rule('agent_type', RegexRule(r'^(red|blue)_team$'))
    
    validator.add_rule('scenario_name', RequiredRule())
    validator.add_rule('scenario_name', TypeRule(str))
    validator.add_rule('scenario_name', SecurityRule('safe_filename'))
    
    return validator


def create_agent_validator() -> InputValidator:
    """Create validator for agent configurations."""
    validator = InputValidator()
    
    validator.add_rule('llm_model', RequiredRule())
    validator.add_rule('llm_model', TypeRule(str))
    validator.add_rule('llm_model', RegexRule(r'^(gpt-4|claude-3|gemini).*$'))
    
    validator.add_rule('max_iterations', TypeRule(int))
    validator.add_rule('max_iterations', RangeRule(1, 1000))
    
    validator.add_rule('tools', TypeRule(list))
    
    return validator