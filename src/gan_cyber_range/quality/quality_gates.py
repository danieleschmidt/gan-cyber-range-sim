"""Core quality gate implementations for progressive validation."""

import asyncio
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import yaml

from ..core.error_handling import CyberRangeError, ErrorSeverity


class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_name: str
    status: QualityGateStatus
    score: float  # 0-100 scale
    threshold: float
    message: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: str
    artifacts: List[str] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


class QualityGate(ABC):
    """Abstract base class for quality gates."""
    
    def __init__(
        self,
        name: str,
        threshold: float = 80.0,
        required: bool = True,
        timeout: int = 300
    ):
        self.name = name
        self.threshold = threshold
        self.required = required
        self.timeout = timeout
        self.logger = logging.getLogger(f"quality_gate.{name}")
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute the quality gate check."""
        pass
    
    def _create_result(
        self,
        status: QualityGateStatus,
        score: float,
        message: str,
        details: Dict[str, Any] = None,
        execution_time: float = 0.0,
        artifacts: List[str] = None
    ) -> QualityGateResult:
        """Helper to create quality gate result."""
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            threshold=self.threshold,
            message=message,
            details=details or {},
            execution_time=execution_time,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            artifacts=artifacts or []
        )


class TestCoverageGate(QualityGate):
    """Quality gate for test coverage validation."""
    
    def __init__(self, threshold: float = 85.0, **kwargs):
        super().__init__("test_coverage", threshold, **kwargs)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute test coverage check."""
        start_time = time.time()
        
        try:
            # Run pytest with coverage
            cmd = [
                "python", "-m", "pytest",
                "--cov=gan_cyber_range",
                "--cov-report=json",
                "--cov-report=term",
                "tests/"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.get("project_root", ".")
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse coverage report
            coverage_file = Path(context.get("project_root", ".")) / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data["totals"]["percent_covered"]
                
                status = (
                    QualityGateStatus.PASSED if total_coverage >= self.threshold
                    else QualityGateStatus.FAILED
                )
                
                return self._create_result(
                    status=status,
                    score=total_coverage,
                    message=f"Test coverage: {total_coverage:.1f}%",
                    details={
                        "total_coverage": total_coverage,
                        "lines_covered": coverage_data["totals"]["covered_lines"],
                        "total_lines": coverage_data["totals"]["num_statements"],
                        "files": coverage_data["files"]
                    },
                    execution_time=execution_time,
                    artifacts=["coverage.json", "htmlcov/"]
                )
            else:
                return self._create_result(
                    status=QualityGateStatus.ERROR,
                    score=0.0,
                    message="Coverage report not found",
                    execution_time=execution_time
                )
                
        except asyncio.TimeoutError:
            return self._create_result(
                status=QualityGateStatus.ERROR,
                score=0.0,
                message=f"Test execution timed out after {self.timeout}s",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return self._create_result(
                status=QualityGateStatus.ERROR,
                score=0.0,
                message=f"Test execution failed: {str(e)}",
                execution_time=time.time() - start_time
            )


class SecurityGate(QualityGate):
    """Quality gate for security validation."""
    
    def __init__(self, threshold: float = 90.0, **kwargs):
        super().__init__("security_scan", threshold, **kwargs)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute security scan."""
        start_time = time.time()
        
        try:
            # Run bandit security scan
            bandit_cmd = [
                "python", "-m", "bandit",
                "-r", "src/",
                "-f", "json",
                "-o", "bandit_report.json"
            ]
            
            bandit_process = await asyncio.create_subprocess_exec(
                *bandit_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.get("project_root", ".")
            )
            
            await bandit_process.communicate()
            
            # Run safety check for dependencies
            safety_cmd = ["python", "-m", "safety", "check", "--json"]
            
            safety_process = await asyncio.create_subprocess_exec(
                *safety_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.get("project_root", ".")
            )
            
            safety_stdout, _ = await safety_process.communicate()
            
            execution_time = time.time() - start_time
            
            # Parse results
            security_score = 100.0
            issues = []
            
            # Parse bandit results
            bandit_file = Path(context.get("project_root", ".")) / "bandit_report.json"
            if bandit_file.exists():
                with open(bandit_file) as f:
                    bandit_data = json.load(f)
                
                high_severity = len([r for r in bandit_data.get("results", []) 
                                   if r.get("issue_severity") == "HIGH"])
                medium_severity = len([r for r in bandit_data.get("results", []) 
                                     if r.get("issue_severity") == "MEDIUM"])
                
                # Deduct points for security issues
                security_score -= (high_severity * 10 + medium_severity * 5)
                issues.extend(bandit_data.get("results", []))
            
            # Parse safety results
            if safety_stdout:
                try:
                    safety_data = json.loads(safety_stdout.decode())
                    vulnerable_packages = len(safety_data)
                    security_score -= vulnerable_packages * 15
                    issues.extend(safety_data)
                except json.JSONDecodeError:
                    pass
            
            security_score = max(0.0, security_score)
            
            status = (
                QualityGateStatus.PASSED if security_score >= self.threshold
                else QualityGateStatus.FAILED
            )
            
            return self._create_result(
                status=status,
                score=security_score,
                message=f"Security score: {security_score:.1f}%",
                details={
                    "security_score": security_score,
                    "total_issues": len(issues),
                    "high_severity_issues": len([i for i in issues if i.get("issue_severity") == "HIGH"]),
                    "medium_severity_issues": len([i for i in issues if i.get("issue_severity") == "MEDIUM"]),
                    "issues": issues[:10]  # Limit details
                },
                execution_time=execution_time,
                artifacts=["bandit_report.json"]
            )
            
        except Exception as e:
            return self._create_result(
                status=QualityGateStatus.ERROR,
                score=0.0,
                message=f"Security scan failed: {str(e)}",
                execution_time=time.time() - start_time
            )


class PerformanceGate(QualityGate):
    """Quality gate for performance validation."""
    
    def __init__(self, threshold: float = 80.0, **kwargs):
        super().__init__("performance_benchmark", threshold, **kwargs)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute performance benchmarks."""
        start_time = time.time()
        
        try:
            # Run performance benchmarks
            cmd = [
                "python", "-m", "pytest",
                "--benchmark-json=benchmark_results.json",
                "--benchmark-only",
                "tests/performance/"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.get("project_root", ".")
            )
            
            await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse benchmark results
            benchmark_file = Path(context.get("project_root", ".")) / "benchmark_results.json"
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                
                benchmarks = benchmark_data.get("benchmarks", [])
                
                # Calculate performance score based on benchmarks
                performance_score = 100.0
                slow_tests = 0
                
                for benchmark in benchmarks:
                    mean_time = benchmark.get("stats", {}).get("mean", 0)
                    if mean_time > 1.0:  # Slower than 1 second
                        slow_tests += 1
                        performance_score -= 5
                
                performance_score = max(0.0, performance_score)
                
                status = (
                    QualityGateStatus.PASSED if performance_score >= self.threshold
                    else QualityGateStatus.FAILED
                )
                
                return self._create_result(
                    status=status,
                    score=performance_score,
                    message=f"Performance score: {performance_score:.1f}%",
                    details={
                        "performance_score": performance_score,
                        "total_benchmarks": len(benchmarks),
                        "slow_tests": slow_tests,
                        "average_execution_time": sum(b.get("stats", {}).get("mean", 0) for b in benchmarks) / len(benchmarks) if benchmarks else 0
                    },
                    execution_time=execution_time,
                    artifacts=["benchmark_results.json"]
                )
            else:
                return self._create_result(
                    status=QualityGateStatus.WARNING,
                    score=self.threshold,
                    message="No performance benchmarks found",
                    execution_time=execution_time
                )
                
        except asyncio.TimeoutError:
            return self._create_result(
                status=QualityGateStatus.ERROR,
                score=0.0,
                message=f"Performance tests timed out after {self.timeout}s",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return self._create_result(
                status=QualityGateStatus.ERROR,
                score=0.0,
                message=f"Performance tests failed: {str(e)}",
                execution_time=time.time() - start_time
            )


class CodeQualityGate(QualityGate):
    """Quality gate for code quality validation."""
    
    def __init__(self, threshold: float = 85.0, **kwargs):
        super().__init__("code_quality", threshold, **kwargs)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute code quality checks."""
        start_time = time.time()
        
        try:
            # Run ruff for linting
            ruff_cmd = ["python", "-m", "ruff", "check", "src/", "--output-format=json"]
            
            ruff_process = await asyncio.create_subprocess_exec(
                *ruff_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.get("project_root", ".")
            )
            
            ruff_stdout, _ = await ruff_process.communicate()
            
            # Run mypy for type checking
            mypy_cmd = ["python", "-m", "mypy", "src/", "--json-report", "mypy_report.json"]
            
            mypy_process = await asyncio.create_subprocess_exec(
                *mypy_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.get("project_root", ".")
            )
            
            await mypy_process.communicate()
            
            execution_time = time.time() - start_time
            
            # Parse results
            quality_score = 100.0
            issues = []
            
            # Parse ruff results
            if ruff_stdout:
                try:
                    ruff_issues = json.loads(ruff_stdout.decode())
                    error_count = len([i for i in ruff_issues if i.get("severity") == "error"])
                    warning_count = len([i for i in ruff_issues if i.get("severity") == "warning"])
                    
                    quality_score -= (error_count * 2 + warning_count * 1)
                    issues.extend(ruff_issues)
                except json.JSONDecodeError:
                    pass
            
            # Parse mypy results
            mypy_file = Path(context.get("project_root", ".")) / "mypy_report.json"
            if mypy_file.exists():
                with open(mypy_file) as f:
                    mypy_data = json.load(f)
                
                type_errors = len(mypy_data.get("reports", {}).get("errors", []))
                quality_score -= type_errors * 3
                issues.extend(mypy_data.get("reports", {}).get("errors", []))
            
            quality_score = max(0.0, quality_score)
            
            status = (
                QualityGateStatus.PASSED if quality_score >= self.threshold
                else QualityGateStatus.FAILED
            )
            
            return self._create_result(
                status=status,
                score=quality_score,
                message=f"Code quality score: {quality_score:.1f}%",
                details={
                    "quality_score": quality_score,
                    "total_issues": len(issues),
                    "ruff_issues": len([i for i in issues if "ruff" in str(i)]),
                    "mypy_issues": len([i for i in issues if "mypy" in str(i)])
                },
                execution_time=execution_time,
                artifacts=["mypy_report.json"]
            )
            
        except Exception as e:
            return self._create_result(
                status=QualityGateStatus.ERROR,
                score=0.0,
                message=f"Code quality check failed: {str(e)}",
                execution_time=time.time() - start_time
            )


class ComplianceGate(QualityGate):
    """Quality gate for compliance validation."""
    
    def __init__(self, threshold: float = 95.0, **kwargs):
        super().__init__("compliance_check", threshold, **kwargs)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute compliance checks."""
        start_time = time.time()
        
        try:
            compliance_score = 100.0
            compliance_details = {}
            
            project_root = Path(context.get("project_root", "."))
            
            # Check required files
            required_files = [
                "LICENSE",
                "README.md",
                "pyproject.toml",
                "requirements.txt",
                "SECURITY.md",
                "CODE_OF_CONDUCT.md",
                "CONTRIBUTING.md"
            ]
            
            missing_files = []
            for file in required_files:
                if not (project_root / file).exists():
                    missing_files.append(file)
                    compliance_score -= 5
            
            compliance_details["missing_files"] = missing_files
            
            # Check license compatibility
            license_file = project_root / "LICENSE"
            if license_file.exists():
                license_content = license_file.read_text()
                if "MIT" not in license_content and "Apache" not in license_content:
                    compliance_score -= 10
                    compliance_details["license_warning"] = "Non-standard license detected"
            
            # Check security policy
            security_file = project_root / "SECURITY.md"
            if security_file.exists():
                security_content = security_file.read_text()
                if "security@" not in security_content.lower():
                    compliance_score -= 5
                    compliance_details["security_policy_warning"] = "No security contact found"
            
            execution_time = time.time() - start_time
            compliance_score = max(0.0, compliance_score)
            
            status = (
                QualityGateStatus.PASSED if compliance_score >= self.threshold
                else QualityGateStatus.FAILED
            )
            
            return self._create_result(
                status=status,
                score=compliance_score,
                message=f"Compliance score: {compliance_score:.1f}%",
                details=compliance_details,
                execution_time=execution_time
            )
            
        except Exception as e:
            return self._create_result(
                status=QualityGateStatus.ERROR,
                score=0.0,
                message=f"Compliance check failed: {str(e)}",
                execution_time=time.time() - start_time
            )


class QualityGateRunner:
    """Orchestrates execution of quality gates."""
    
    def __init__(self, gates: List[QualityGate]):
        self.gates = gates
        self.logger = logging.getLogger("quality_gate_runner")
    
    async def run_all(
        self,
        context: Dict[str, Any],
        fail_fast: bool = False,
        parallel: bool = True
    ) -> List[QualityGateResult]:
        """Run all quality gates."""
        self.logger.info(f"Running {len(self.gates)} quality gates")
        
        if parallel:
            # Run gates in parallel
            tasks = [gate.execute(context) for gate in self.gates]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(
                        QualityGateResult(
                            gate_name=self.gates[i].name,
                            status=QualityGateStatus.ERROR,
                            score=0.0,
                            threshold=self.gates[i].threshold,
                            message=f"Gate execution failed: {result}",
                            details={"exception": str(result)},
                            execution_time=0.0,
                            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        )
                    )
                else:
                    final_results.append(result)
        else:
            # Run gates sequentially
            final_results = []
            for gate in self.gates:
                try:
                    result = await gate.execute(context)
                    final_results.append(result)
                    
                    # Fail fast if enabled and gate failed
                    if (fail_fast and 
                        result.status == QualityGateStatus.FAILED and 
                        gate.required):
                        self.logger.warning(f"Failing fast due to {gate.name} failure")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Gate {gate.name} failed with exception: {e}")
                    final_results.append(
                        QualityGateResult(
                            gate_name=gate.name,
                            status=QualityGateStatus.ERROR,
                            score=0.0,
                            threshold=gate.threshold,
                            message=f"Gate execution failed: {e}",
                            details={"exception": str(e)},
                            execution_time=0.0,
                            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        )
                    )
        
        # Log summary
        passed = len([r for r in final_results if r.status == QualityGateStatus.PASSED])
        failed = len([r for r in final_results if r.status == QualityGateStatus.FAILED])
        warnings = len([r for r in final_results if r.status == QualityGateStatus.WARNING])
        errors = len([r for r in final_results if r.status == QualityGateStatus.ERROR])
        
        self.logger.info(
            f"Quality gates complete: {passed} passed, {failed} failed, "
            f"{warnings} warnings, {errors} errors"
        )
        
        return final_results
    
    def get_default_gates(self) -> List[QualityGate]:
        """Get default set of quality gates."""
        return [
            TestCoverageGate(threshold=85.0),
            SecurityGate(threshold=90.0),
            CodeQualityGate(threshold=85.0),
            PerformanceGate(threshold=80.0),
            ComplianceGate(threshold=95.0)
        ]