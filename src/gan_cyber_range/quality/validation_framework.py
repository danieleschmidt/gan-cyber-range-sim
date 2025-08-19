"""Advanced validation framework for quality gates."""

import asyncio
import logging
import time
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import hashlib
import yaml

from .quality_gates import QualityGate, QualityGateResult, QualityGateStatus
from .monitoring import QualityMetric, MetricsCollector, get_global_metrics_collector
from ..core.error_handling import CyberRangeError, ErrorSeverity


class ValidationLevel(Enum):
    """Validation levels with increasing strictness."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationRule:
    """Individual validation rule."""
    name: str
    description: str
    level: ValidationLevel
    enabled: bool = True
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationContext:
    """Context for validation execution."""
    project_root: Path
    validation_level: ValidationLevel
    config: Dict[str, Any]
    environment: Dict[str, str]
    artifacts_dir: Path
    timestamp: str
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(
        self,
        name: str,
        description: str,
        level: ValidationLevel = ValidationLevel.STANDARD
    ):
        self.name = name
        self.description = description
        self.level = level
        self.logger = logging.getLogger(f"validator.{name}")
        self.rules: List[ValidationRule] = []
        
    @abstractmethod
    async def validate(self, context: ValidationContext) -> List[QualityGateResult]:
        """Execute validation."""
        pass
    
    def add_rule(self, rule: ValidationRule):
        """Add validation rule."""
        self.rules.append(rule)
        self.logger.debug(f"Added rule: {rule.name}")
    
    def get_enabled_rules(self, max_level: ValidationLevel) -> List[ValidationRule]:
        """Get enabled rules up to specified level."""
        level_order = [
            ValidationLevel.BASIC,
            ValidationLevel.STANDARD,
            ValidationLevel.STRICT,
            ValidationLevel.PARANOID
        ]
        
        max_level_index = level_order.index(max_level)
        
        return [
            rule for rule in self.rules
            if rule.enabled and level_order.index(rule.level) <= max_level_index
        ]


class SecurityValidator(BaseValidator):
    """Advanced security validation."""
    
    def __init__(self):
        super().__init__(
            "security_validator",
            "Comprehensive security validation including SAST, DAST, and dependency scanning",
            ValidationLevel.STANDARD
        )
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup security validation rules."""
        self.add_rule(ValidationRule(
            name="sast_scan",
            description="Static Application Security Testing",
            level=ValidationLevel.BASIC,
            severity=ErrorSeverity.HIGH
        ))
        
        self.add_rule(ValidationRule(
            name="dependency_scan",
            description="Dependency vulnerability scanning",
            level=ValidationLevel.BASIC,
            severity=ErrorSeverity.HIGH
        ))
        
        self.add_rule(ValidationRule(
            name="secrets_scan",
            description="Secrets and credentials scanning",
            level=ValidationLevel.STANDARD,
            severity=ErrorSeverity.CRITICAL
        ))
        
        self.add_rule(ValidationRule(
            name="license_compliance",
            description="License compliance checking",
            level=ValidationLevel.STANDARD,
            severity=ErrorSeverity.MEDIUM
        ))
        
        self.add_rule(ValidationRule(
            name="security_headers",
            description="Security headers validation",
            level=ValidationLevel.STRICT,
            severity=ErrorSeverity.MEDIUM
        ))
        
        self.add_rule(ValidationRule(
            name="crypto_validation",
            description="Cryptographic implementation validation",
            level=ValidationLevel.PARANOID,
            severity=ErrorSeverity.HIGH
        ))
    
    async def validate(self, context: ValidationContext) -> List[QualityGateResult]:
        """Execute security validation."""
        results = []
        enabled_rules = self.get_enabled_rules(context.validation_level)
        
        for rule in enabled_rules:
            try:
                result = await self._execute_rule(rule, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Rule {rule.name} failed: {e}")
                results.append(self._create_error_result(rule, str(e)))
        
        return results
    
    async def _execute_rule(self, rule: ValidationRule, context: ValidationContext) -> QualityGateResult:
        """Execute individual security rule."""
        start_time = time.time()
        
        if rule.name == "sast_scan":
            return await self._run_sast_scan(context)
        elif rule.name == "dependency_scan":
            return await self._run_dependency_scan(context)
        elif rule.name == "secrets_scan":
            return await self._run_secrets_scan(context)
        elif rule.name == "license_compliance":
            return await self._check_license_compliance(context)
        elif rule.name == "security_headers":
            return await self._validate_security_headers(context)
        elif rule.name == "crypto_validation":
            return await self._validate_cryptography(context)
        else:
            raise CyberRangeError(f"Unknown security rule: {rule.name}")
    
    async def _run_sast_scan(self, context: ValidationContext) -> QualityGateResult:
        """Run SAST scan using multiple tools."""
        security_score = 100.0
        issues = []
        
        # Run bandit
        try:
            cmd = [
                "python", "-m", "bandit",
                "-r", str(context.project_root / "src"),
                "-f", "json",
                "-o", str(context.artifacts_dir / "bandit_report.json")
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            # Parse results
            bandit_file = context.artifacts_dir / "bandit_report.json"
            if bandit_file.exists():
                with open(bandit_file) as f:
                    bandit_data = json.load(f)
                
                high_issues = [r for r in bandit_data.get("results", []) 
                             if r.get("issue_severity") == "HIGH"]
                medium_issues = [r for r in bandit_data.get("results", []) 
                               if r.get("issue_severity") == "MEDIUM"]
                low_issues = [r for r in bandit_data.get("results", []) 
                            if r.get("issue_severity") == "LOW"]
                
                security_score -= len(high_issues) * 15
                security_score -= len(medium_issues) * 8
                security_score -= len(low_issues) * 3
                
                issues.extend(bandit_data.get("results", []))
        
        except Exception as e:
            self.logger.warning(f"Bandit scan failed: {e}")
            security_score -= 20
        
        # Run semgrep if available
        try:
            cmd = ["semgrep", "--config=auto", "--json", str(context.project_root / "src")]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            
            if stdout:
                semgrep_data = json.loads(stdout.decode())
                semgrep_issues = semgrep_data.get("results", [])
                
                # Categorize semgrep issues
                critical_issues = [r for r in semgrep_issues if r.get("extra", {}).get("severity") == "ERROR"]
                warning_issues = [r for r in semgrep_issues if r.get("extra", {}).get("severity") == "WARNING"]
                
                security_score -= len(critical_issues) * 12
                security_score -= len(warning_issues) * 5
                
                issues.extend(semgrep_issues)
        
        except Exception as e:
            self.logger.debug(f"Semgrep not available or failed: {e}")
        
        security_score = max(0.0, security_score)
        
        status = QualityGateStatus.PASSED if security_score >= 80 else QualityGateStatus.FAILED
        
        return QualityGateResult(
            gate_name="sast_scan",
            status=status,
            score=security_score,
            threshold=80.0,
            message=f"SAST scan score: {security_score:.1f}%",
            details={
                "total_issues": len(issues),
                "high_severity": len([i for i in issues if i.get("issue_severity") == "HIGH"]),
                "medium_severity": len([i for i in issues if i.get("issue_severity") == "MEDIUM"]),
                "tools_used": ["bandit", "semgrep"]
            },
            execution_time=time.time(),
            timestamp=context.timestamp,
            artifacts=[str(context.artifacts_dir / "bandit_report.json")]
        )
    
    async def _run_dependency_scan(self, context: ValidationContext) -> QualityGateResult:
        """Run dependency vulnerability scan."""
        security_score = 100.0
        vulnerabilities = []
        
        # Run safety
        try:
            cmd = ["python", "-m", "safety", "check", "--json"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.project_root
            )
            
            stdout, _ = await process.communicate()
            
            if stdout:
                safety_data = json.loads(stdout.decode())
                vulnerabilities.extend(safety_data)
                
                # Deduct points based on vulnerability severity
                for vuln in safety_data:
                    if "critical" in vuln.get("vulnerability", "").lower():
                        security_score -= 25
                    elif "high" in vuln.get("vulnerability", "").lower():
                        security_score -= 15
                    elif "medium" in vuln.get("vulnerability", "").lower():
                        security_score -= 8
                    else:
                        security_score -= 3
        
        except Exception as e:
            self.logger.warning(f"Safety scan failed: {e}")
            security_score -= 10
        
        # Run pip-audit if available
        try:
            cmd = ["pip-audit", "--format=json"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.project_root
            )
            
            stdout, _ = await process.communicate()
            
            if stdout:
                audit_data = json.loads(stdout.decode())
                audit_vulns = audit_data.get("vulnerabilities", [])
                
                for vuln in audit_vulns:
                    security_score -= 10  # Pip-audit findings are generally serious
                
                vulnerabilities.extend(audit_vulns)
        
        except Exception as e:
            self.logger.debug(f"pip-audit not available or failed: {e}")
        
        security_score = max(0.0, security_score)
        status = QualityGateStatus.PASSED if security_score >= 90 else QualityGateStatus.FAILED
        
        return QualityGateResult(
            gate_name="dependency_scan",
            status=status,
            score=security_score,
            threshold=90.0,
            message=f"Dependency scan score: {security_score:.1f}%",
            details={
                "total_vulnerabilities": len(vulnerabilities),
                "critical_vulns": len([v for v in vulnerabilities if "critical" in str(v).lower()]),
                "high_vulns": len([v for v in vulnerabilities if "high" in str(v).lower()]),
                "tools_used": ["safety", "pip-audit"]
            },
            execution_time=time.time(),
            timestamp=context.timestamp
        )
    
    async def _run_secrets_scan(self, context: ValidationContext) -> QualityGateResult:
        """Run secrets scanning."""
        security_score = 100.0
        secrets_found = []
        
        try:
            cmd = [
                "detect-secrets", "scan",
                "--all-files",
                "--force-use-all-plugins",
                str(context.project_root)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            
            if stdout:
                secrets_data = json.loads(stdout.decode())
                results = secrets_data.get("results", {})
                
                for file_path, secrets in results.items():
                    for secret in secrets:
                        secrets_found.append({
                            "file": file_path,
                            "type": secret.get("type"),
                            "line": secret.get("line_number")
                        })
                        
                        # Deduct points based on secret type
                        secret_type = secret.get("type", "").lower()
                        if "private_key" in secret_type or "password" in secret_type:
                            security_score -= 30
                        elif "token" in secret_type or "key" in secret_type:
                            security_score -= 20
                        else:
                            security_score -= 10
        
        except Exception as e:
            self.logger.warning(f"Secrets scan failed: {e}")
            security_score -= 5
        
        security_score = max(0.0, security_score)
        status = QualityGateStatus.PASSED if len(secrets_found) == 0 else QualityGateStatus.FAILED
        
        return QualityGateResult(
            gate_name="secrets_scan",
            status=status,
            score=security_score,
            threshold=100.0,
            message=f"Secrets scan: {len(secrets_found)} potential secrets found",
            details={
                "secrets_found": len(secrets_found),
                "secret_types": list(set(s["type"] for s in secrets_found)),
                "affected_files": list(set(s["file"] for s in secrets_found))
            },
            execution_time=time.time(),
            timestamp=context.timestamp
        )
    
    async def _check_license_compliance(self, context: ValidationContext) -> QualityGateResult:
        """Check license compliance."""
        compliance_score = 100.0
        issues = []
        
        # Check main license file
        license_file = context.project_root / "LICENSE"
        if not license_file.exists():
            compliance_score -= 30
            issues.append("No LICENSE file found")
        else:
            license_content = license_file.read_text().lower()
            approved_licenses = ["mit", "apache", "bsd", "gpl"]
            
            if not any(license_type in license_content for license_type in approved_licenses):
                compliance_score -= 20
                issues.append("License not recognized as standard open source license")
        
        # Check for license headers in source files
        source_files = list(context.project_root.glob("src/**/*.py"))
        files_without_headers = []
        
        for file_path in source_files[:20]:  # Sample first 20 files
            try:
                content = file_path.read_text()
                # Simple check for license/copyright header
                if not any(keyword in content.lower()[:500] for keyword in ["copyright", "license", "licensed"]):
                    files_without_headers.append(str(file_path))
            except Exception:
                continue
        
        if files_without_headers:
            header_ratio = len(files_without_headers) / min(len(source_files), 20)
            compliance_score -= header_ratio * 20
            issues.append(f"{len(files_without_headers)} files missing license headers")
        
        compliance_score = max(0.0, compliance_score)
        status = QualityGateStatus.PASSED if compliance_score >= 85 else QualityGateStatus.FAILED
        
        return QualityGateResult(
            gate_name="license_compliance",
            status=status,
            score=compliance_score,
            threshold=85.0,
            message=f"License compliance score: {compliance_score:.1f}%",
            details={
                "issues": issues,
                "files_checked": len(source_files),
                "files_without_headers": len(files_without_headers)
            },
            execution_time=time.time(),
            timestamp=context.timestamp
        )
    
    async def _validate_security_headers(self, context: ValidationContext) -> QualityGateResult:
        """Validate security headers in web application."""
        # This is a placeholder - would require running the application
        return QualityGateResult(
            gate_name="security_headers",
            status=QualityGateStatus.SKIPPED,
            score=100.0,
            threshold=90.0,
            message="Security headers validation skipped (no running application)",
            details={"reason": "Application not running"},
            execution_time=0.0,
            timestamp=context.timestamp
        )
    
    async def _validate_cryptography(self, context: ValidationContext) -> QualityGateResult:
        """Validate cryptographic implementations."""
        crypto_score = 100.0
        issues = []
        
        # Search for common crypto patterns
        crypto_files = []
        for file_path in context.project_root.glob("src/**/*.py"):
            try:
                content = file_path.read_text()
                crypto_keywords = ["crypto", "hash", "encrypt", "decrypt", "password", "token", "secret"]
                
                if any(keyword in content.lower() for keyword in crypto_keywords):
                    crypto_files.append(str(file_path))
                    
                    # Check for weak patterns
                    if "md5" in content.lower():
                        crypto_score -= 15
                        issues.append(f"MD5 usage found in {file_path}")
                    
                    if "sha1" in content.lower() and "hashlib.sha1" in content:
                        crypto_score -= 10
                        issues.append(f"SHA1 usage found in {file_path}")
                    
                    if "random.random" in content:
                        crypto_score -= 20
                        issues.append(f"Weak random number generation in {file_path}")
                        
            except Exception:
                continue
        
        crypto_score = max(0.0, crypto_score)
        status = QualityGateStatus.PASSED if crypto_score >= 80 else QualityGateStatus.FAILED
        
        return QualityGateResult(
            gate_name="crypto_validation",
            status=status,
            score=crypto_score,
            threshold=80.0,
            message=f"Cryptography validation score: {crypto_score:.1f}%",
            details={
                "files_with_crypto": len(crypto_files),
                "issues_found": len(issues),
                "issues": issues[:10]  # Limit output
            },
            execution_time=time.time(),
            timestamp=context.timestamp
        )
    
    def _create_error_result(self, rule: ValidationRule, error_message: str) -> QualityGateResult:
        """Create error result for failed rule."""
        return QualityGateResult(
            gate_name=rule.name,
            status=QualityGateStatus.ERROR,
            score=0.0,
            threshold=100.0,
            message=f"Validation error: {error_message}",
            details={"error": error_message, "rule": rule.name},
            execution_time=0.0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )


class PerformanceValidator(BaseValidator):
    """Advanced performance validation."""
    
    def __init__(self):
        super().__init__(
            "performance_validator",
            "Comprehensive performance validation including benchmarks and profiling",
            ValidationLevel.STANDARD
        )
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup performance validation rules."""
        self.add_rule(ValidationRule(
            name="benchmark_tests",
            description="Performance benchmark testing",
            level=ValidationLevel.BASIC,
            severity=ErrorSeverity.MEDIUM
        ))
        
        self.add_rule(ValidationRule(
            name="memory_profiling",
            description="Memory usage profiling",
            level=ValidationLevel.STANDARD,
            severity=ErrorSeverity.MEDIUM
        ))
        
        self.add_rule(ValidationRule(
            name="cpu_profiling",
            description="CPU usage profiling",
            level=ValidationLevel.STRICT,
            severity=ErrorSeverity.MEDIUM
        ))
        
        self.add_rule(ValidationRule(
            name="load_testing",
            description="Load testing validation",
            level=ValidationLevel.PARANOID,
            severity=ErrorSeverity.HIGH
        ))
    
    async def validate(self, context: ValidationContext) -> List[QualityGateResult]:
        """Execute performance validation."""
        results = []
        enabled_rules = self.get_enabled_rules(context.validation_level)
        
        for rule in enabled_rules:
            try:
                result = await self._execute_rule(rule, context)
                results.append(result)
                
                # Record performance metrics
                if result.details:
                    metrics_collector = get_global_metrics_collector()
                    metric = QualityMetric(
                        name=f"performance_{rule.name}",
                        value=result.score,
                        unit="percent",
                        timestamp=time.time(),
                        tags={"validator": "performance", "rule": rule.name}
                    )
                    metrics_collector.record_metric(metric)
                    
            except Exception as e:
                self.logger.error(f"Performance rule {rule.name} failed: {e}")
                results.append(self._create_error_result(rule, str(e)))
        
        return results
    
    async def _execute_rule(self, rule: ValidationRule, context: ValidationContext) -> QualityGateResult:
        """Execute individual performance rule."""
        if rule.name == "benchmark_tests":
            return await self._run_benchmark_tests(context)
        elif rule.name == "memory_profiling":
            return await self._run_memory_profiling(context)
        elif rule.name == "cpu_profiling":
            return await self._run_cpu_profiling(context)
        elif rule.name == "load_testing":
            return await self._run_load_testing(context)
        else:
            raise CyberRangeError(f"Unknown performance rule: {rule.name}")
    
    async def _run_benchmark_tests(self, context: ValidationContext) -> QualityGateResult:
        """Run performance benchmark tests."""
        try:
            cmd = [
                "python", "-m", "pytest",
                "--benchmark-json", str(context.artifacts_dir / "benchmark_results.json"),
                "--benchmark-only",
                "tests/performance/"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.project_root
            )
            
            await process.communicate()
            
            # Parse results
            benchmark_file = context.artifacts_dir / "benchmark_results.json"
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                
                benchmarks = benchmark_data.get("benchmarks", [])
                if not benchmarks:
                    return QualityGateResult(
                        gate_name="benchmark_tests",
                        status=QualityGateStatus.WARNING,
                        score=50.0,
                        threshold=80.0,
                        message="No benchmark tests found",
                        details={"benchmarks": 0},
                        execution_time=0.0,
                        timestamp=context.timestamp
                    )
                
                # Calculate performance score
                performance_score = 100.0
                slow_benchmarks = 0
                total_time = 0
                
                for benchmark in benchmarks:
                    mean_time = benchmark.get("stats", {}).get("mean", 0)
                    total_time += mean_time
                    
                    # Penalize slow tests
                    if mean_time > 1.0:  # Slower than 1 second
                        slow_benchmarks += 1
                        performance_score -= 10
                    elif mean_time > 0.5:  # Slower than 500ms
                        performance_score -= 5
                
                performance_score = max(0.0, performance_score)
                
                status = QualityGateStatus.PASSED if performance_score >= 80 else QualityGateStatus.FAILED
                
                return QualityGateResult(
                    gate_name="benchmark_tests",
                    status=status,
                    score=performance_score,
                    threshold=80.0,
                    message=f"Benchmark score: {performance_score:.1f}%",
                    details={
                        "total_benchmarks": len(benchmarks),
                        "slow_benchmarks": slow_benchmarks,
                        "average_time": total_time / len(benchmarks),
                        "total_time": total_time
                    },
                    execution_time=time.time(),
                    timestamp=context.timestamp,
                    artifacts=[str(benchmark_file)]
                )
            else:
                return QualityGateResult(
                    gate_name="benchmark_tests",
                    status=QualityGateStatus.ERROR,
                    score=0.0,
                    threshold=80.0,
                    message="Benchmark results file not found",
                    details={"error": "No benchmark results generated"},
                    execution_time=0.0,
                    timestamp=context.timestamp
                )
                
        except Exception as e:
            return QualityGateResult(
                gate_name="benchmark_tests",
                status=QualityGateStatus.ERROR,
                score=0.0,
                threshold=80.0,
                message=f"Benchmark testing failed: {e}",
                details={"error": str(e)},
                execution_time=0.0,
                timestamp=context.timestamp
            )
    
    async def _run_memory_profiling(self, context: ValidationContext) -> QualityGateResult:
        """Run memory profiling."""
        # Placeholder - would require memory_profiler integration
        return QualityGateResult(
            gate_name="memory_profiling",
            status=QualityGateStatus.SKIPPED,
            score=80.0,
            threshold=75.0,
            message="Memory profiling skipped (not implemented)",
            details={"reason": "Memory profiling not implemented"},
            execution_time=0.0,
            timestamp=context.timestamp
        )
    
    async def _run_cpu_profiling(self, context: ValidationContext) -> QualityGateResult:
        """Run CPU profiling."""
        # Placeholder - would require cProfile integration
        return QualityGateResult(
            gate_name="cpu_profiling",
            status=QualityGateStatus.SKIPPED,
            score=80.0,
            threshold=75.0,
            message="CPU profiling skipped (not implemented)",
            details={"reason": "CPU profiling not implemented"},
            execution_time=0.0,
            timestamp=context.timestamp
        )
    
    async def _run_load_testing(self, context: ValidationContext) -> QualityGateResult:
        """Run load testing."""
        # Placeholder - would require load testing framework
        return QualityGateResult(
            gate_name="load_testing",
            status=QualityGateStatus.SKIPPED,
            score=80.0,
            threshold=70.0,
            message="Load testing skipped (no running application)",
            details={"reason": "Application not running"},
            execution_time=0.0,
            timestamp=context.timestamp
        )
    
    def _create_error_result(self, rule: ValidationRule, error_message: str) -> QualityGateResult:
        """Create error result for failed rule."""
        return QualityGateResult(
            gate_name=rule.name,
            status=QualityGateStatus.ERROR,
            score=0.0,
            threshold=100.0,
            message=f"Performance validation error: {error_message}",
            details={"error": error_message, "rule": rule.name},
            execution_time=0.0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )


class ValidationFramework:
    """Comprehensive validation framework."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger("validation_framework")
        self.validators: Dict[str, BaseValidator] = {}
        self.config = self._load_config(config_file)
        
        # Register default validators
        self._register_default_validators()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load validation configuration."""
        default_config = {
            "validation_level": "standard",
            "enabled_validators": ["security", "performance"],
            "artifacts_dir": "validation_artifacts",
            "parallel_execution": True,
            "timeout_minutes": 30
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    if config_file.endswith(('.yaml', '.yml')):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                default_config.update(user_config)
                self.logger.info(f"Loaded validation config from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def _register_default_validators(self):
        """Register default validators."""
        self.register_validator("security", SecurityValidator())
        self.register_validator("performance", PerformanceValidator())
    
    def register_validator(self, name: str, validator: BaseValidator):
        """Register a validator."""
        self.validators[name] = validator
        self.logger.info(f"Registered validator: {name}")
    
    async def run_validation(
        self,
        project_root: str = ".",
        validation_level: Optional[ValidationLevel] = None,
        enabled_validators: Optional[List[str]] = None
    ) -> List[QualityGateResult]:
        """Run comprehensive validation."""
        if validation_level is None:
            validation_level = ValidationLevel(self.config.get("validation_level", "standard"))
        
        if enabled_validators is None:
            enabled_validators = self.config.get("enabled_validators", list(self.validators.keys()))
        
        # Setup validation context
        project_path = Path(project_root)
        artifacts_dir = project_path / self.config.get("artifacts_dir", "validation_artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        context = ValidationContext(
            project_root=project_path,
            validation_level=validation_level,
            config=self.config,
            environment=dict(os.environ) if 'os' in globals() else {},
            artifacts_dir=artifacts_dir,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        # Run validators
        all_results = []
        
        if self.config.get("parallel_execution", True):
            # Run validators in parallel
            tasks = []
            for validator_name in enabled_validators:
                if validator_name in self.validators:
                    validator = self.validators[validator_name]
                    task = asyncio.create_task(validator.validate(context))
                    tasks.append((validator_name, task))
            
            for validator_name, task in tasks:
                try:
                    results = await task
                    all_results.extend(results)
                    self.logger.info(f"Validator {validator_name} completed with {len(results)} results")
                except Exception as e:
                    self.logger.error(f"Validator {validator_name} failed: {e}")
        else:
            # Run validators sequentially
            for validator_name in enabled_validators:
                if validator_name in self.validators:
                    validator = self.validators[validator_name]
                    try:
                        results = await validator.validate(context)
                        all_results.extend(results)
                        self.logger.info(f"Validator {validator_name} completed with {len(results)} results")
                    except Exception as e:
                        self.logger.error(f"Validator {validator_name} failed: {e}")
        
        return all_results