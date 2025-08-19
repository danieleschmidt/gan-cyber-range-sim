"""Progressive validation system for autonomous SDLC quality gates."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import json
from pathlib import Path

from .quality_gates import QualityGate, QualityGateResult, QualityGateStatus, QualityGateRunner
from ..core.error_handling import CyberRangeError, ErrorSeverity


class ValidationStage(Enum):
    """Progressive validation stages."""
    GENERATION_1 = "generation_1"  # Make it work (basic functionality)
    GENERATION_2 = "generation_2"  # Make it robust (error handling, validation)
    GENERATION_3 = "generation_3"  # Make it scale (optimization, performance)
    PRODUCTION = "production"      # Production readiness


@dataclass
class ValidationMetrics:
    """Metrics collected during validation."""
    stage: ValidationStage
    total_gates: int
    passed_gates: int
    failed_gates: int
    warning_gates: int
    error_gates: int
    overall_score: float
    execution_time: float
    timestamp: str
    critical_failures: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return (self.passed_gates / self.total_gates * 100) if self.total_gates > 0 else 0.0
    
    @property
    def is_passing(self) -> bool:
        """Check if validation is passing."""
        return self.failed_gates == 0 and len(self.critical_failures) == 0


@dataclass
class ValidationResult:
    """Result of progressive validation."""
    stage: ValidationStage
    status: QualityGateStatus
    metrics: ValidationMetrics
    gate_results: List[QualityGateResult]
    recommendations: List[str]
    next_stage_ready: bool
    report_path: Optional[str] = None


class ProgressiveValidator:
    """Progressive validation system for autonomous SDLC."""
    
    def __init__(
        self,
        project_root: str = ".",
        enable_auto_fix: bool = True,
        fail_fast: bool = False
    ):
        self.project_root = Path(project_root)
        self.enable_auto_fix = enable_auto_fix
        self.fail_fast = fail_fast
        self.logger = logging.getLogger("progressive_validator")
        
        # Stage-specific quality gate configurations
        self.stage_gates = self._configure_stage_gates()
        
        # Auto-fix strategies
        self.auto_fix_strategies: Dict[str, Callable] = {}
        self._register_auto_fix_strategies()
    
    def _configure_stage_gates(self) -> Dict[ValidationStage, List[QualityGate]]:
        """Configure quality gates for each validation stage."""
        from .quality_gates import (
            TestCoverageGate, SecurityGate, CodeQualityGate, 
            PerformanceGate, ComplianceGate
        )
        
        return {
            ValidationStage.GENERATION_1: [
                # Basic functionality gates - lower thresholds
                TestCoverageGate(threshold=70.0, required=True),
                CodeQualityGate(threshold=75.0, required=True),
                SecurityGate(threshold=80.0, required=False),  # Non-blocking initially
            ],
            
            ValidationStage.GENERATION_2: [
                # Robust implementation gates - medium thresholds
                TestCoverageGate(threshold=80.0, required=True),
                CodeQualityGate(threshold=85.0, required=True),
                SecurityGate(threshold=90.0, required=True),
                ComplianceGate(threshold=90.0, required=True),
            ],
            
            ValidationStage.GENERATION_3: [
                # Scalable implementation gates - high thresholds
                TestCoverageGate(threshold=90.0, required=True),
                CodeQualityGate(threshold=90.0, required=True),
                SecurityGate(threshold=95.0, required=True),
                PerformanceGate(threshold=85.0, required=True),
                ComplianceGate(threshold=95.0, required=True),
            ],
            
            ValidationStage.PRODUCTION: [
                # Production readiness gates - highest thresholds
                TestCoverageGate(threshold=95.0, required=True),
                CodeQualityGate(threshold=95.0, required=True),
                SecurityGate(threshold=98.0, required=True),
                PerformanceGate(threshold=90.0, required=True),
                ComplianceGate(threshold=98.0, required=True),
            ]
        }
    
    def _register_auto_fix_strategies(self):
        """Register auto-fix strategies for common issues."""
        self.auto_fix_strategies.update({
            "test_coverage": self._auto_fix_test_coverage,
            "code_quality": self._auto_fix_code_quality,
            "security_scan": self._auto_fix_security,
            "compliance_check": self._auto_fix_compliance
        })
    
    async def validate_stage(
        self,
        stage: ValidationStage,
        auto_fix: bool = None
    ) -> ValidationResult:
        """Validate a specific stage."""
        start_time = time.time()
        
        if auto_fix is None:
            auto_fix = self.enable_auto_fix
        
        self.logger.info(f"Starting validation for {stage.value}")
        
        # Get gates for this stage
        gates = self.stage_gates.get(stage, [])
        if not gates:
            raise CyberRangeError(
                f"No quality gates configured for stage: {stage.value}",
                severity=ErrorSeverity.HIGH
            )
        
        # Prepare context
        context = {
            "project_root": str(self.project_root),
            "stage": stage.value,
            "auto_fix_enabled": auto_fix
        }
        
        # Run quality gates
        runner = QualityGateRunner(gates)
        gate_results = await runner.run_all(
            context=context,
            fail_fast=self.fail_fast,
            parallel=True
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(stage, gate_results, time.time() - start_time)
        
        # Determine overall status
        overall_status = self._determine_overall_status(gate_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stage, gate_results, metrics)
        
        # Determine if ready for next stage
        next_stage_ready = self._is_next_stage_ready(stage, metrics, gate_results)
        
        # Auto-fix if enabled and needed
        if auto_fix and not metrics.is_passing:
            self.logger.info("Attempting auto-fix for failed gates")
            await self._attempt_auto_fixes(gate_results)
            
            # Re-run failed gates after auto-fix
            failed_gates = [g for g, r in zip(gates, gate_results) 
                          if r.status == QualityGateStatus.FAILED]
            
            if failed_gates:
                self.logger.info(f"Re-running {len(failed_gates)} gates after auto-fix")
                retry_runner = QualityGateRunner(failed_gates)
                retry_results = await retry_runner.run_all(context=context)
                
                # Update results
                retry_map = {r.gate_name: r for r in retry_results}
                for i, result in enumerate(gate_results):
                    if result.gate_name in retry_map:
                        gate_results[i] = retry_map[result.gate_name]
                
                # Recalculate metrics
                metrics = self._calculate_metrics(stage, gate_results, time.time() - start_time)
                overall_status = self._determine_overall_status(gate_results)
                next_stage_ready = self._is_next_stage_ready(stage, metrics, gate_results)
        
        # Generate report
        report_path = await self._generate_validation_report(
            stage, metrics, gate_results, recommendations
        )
        
        result = ValidationResult(
            stage=stage,
            status=overall_status,
            metrics=metrics,
            gate_results=gate_results,
            recommendations=recommendations,
            next_stage_ready=next_stage_ready,
            report_path=report_path
        )
        
        self.logger.info(
            f"Validation complete for {stage.value}: "
            f"{overall_status.value} (score: {metrics.overall_score:.1f}%)"
        )
        
        return result
    
    async def validate_progressive(
        self,
        start_stage: ValidationStage = ValidationStage.GENERATION_1,
        target_stage: ValidationStage = ValidationStage.PRODUCTION
    ) -> List[ValidationResult]:
        """Run progressive validation through multiple stages."""
        stages = [
            ValidationStage.GENERATION_1,
            ValidationStage.GENERATION_2,
            ValidationStage.GENERATION_3,
            ValidationStage.PRODUCTION
        ]
        
        start_idx = stages.index(start_stage)
        target_idx = stages.index(target_stage)
        
        if start_idx > target_idx:
            raise CyberRangeError(
                f"Start stage {start_stage.value} cannot be after target stage {target_stage.value}",
                severity=ErrorSeverity.HIGH
            )
        
        results = []
        current_stage_idx = start_idx
        
        while current_stage_idx <= target_idx:
            stage = stages[current_stage_idx]
            
            self.logger.info(f"Progressive validation: Stage {current_stage_idx + 1}/{target_idx + 1} ({stage.value})")
            
            result = await self.validate_stage(stage)
            results.append(result)
            
            # Check if we can proceed to next stage
            if not result.next_stage_ready and current_stage_idx < target_idx:
                self.logger.warning(
                    f"Stage {stage.value} validation incomplete. "
                    f"Cannot proceed to next stage."
                )
                if self.fail_fast:
                    break
            
            current_stage_idx += 1
        
        return results
    
    def _calculate_metrics(
        self,
        stage: ValidationStage,
        gate_results: List[QualityGateResult],
        execution_time: float
    ) -> ValidationMetrics:
        """Calculate validation metrics."""
        passed = len([r for r in gate_results if r.status == QualityGateStatus.PASSED])
        failed = len([r for r in gate_results if r.status == QualityGateStatus.FAILED])
        warnings = len([r for r in gate_results if r.status == QualityGateStatus.WARNING])
        errors = len([r for r in gate_results if r.status == QualityGateStatus.ERROR])
        
        # Calculate overall score (weighted average)
        total_score = sum(r.score for r in gate_results)
        overall_score = total_score / len(gate_results) if gate_results else 0.0
        
        # Identify critical failures
        critical_failures = [
            r.gate_name for r in gate_results
            if r.status == QualityGateStatus.FAILED and r.score < 50.0
        ]
        
        # Collect performance metrics
        performance_metrics = {
            "total_execution_time": execution_time,
            "average_gate_time": sum(r.execution_time for r in gate_results) / len(gate_results) if gate_results else 0.0,
            "slowest_gate_time": max((r.execution_time for r in gate_results), default=0.0)
        }
        
        return ValidationMetrics(
            stage=stage,
            total_gates=len(gate_results),
            passed_gates=passed,
            failed_gates=failed,
            warning_gates=warnings,
            error_gates=errors,
            overall_score=overall_score,
            execution_time=execution_time,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            critical_failures=critical_failures,
            performance_metrics=performance_metrics
        )
    
    def _determine_overall_status(self, gate_results: List[QualityGateResult]) -> QualityGateStatus:
        """Determine overall validation status."""
        if any(r.status == QualityGateStatus.ERROR for r in gate_results):
            return QualityGateStatus.ERROR
        elif any(r.status == QualityGateStatus.FAILED for r in gate_results):
            return QualityGateStatus.FAILED
        elif any(r.status == QualityGateStatus.WARNING for r in gate_results):
            return QualityGateStatus.WARNING
        else:
            return QualityGateStatus.PASSED
    
    def _generate_recommendations(
        self,
        stage: ValidationStage,
        gate_results: List[QualityGateResult],
        metrics: ValidationMetrics
    ) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        # Stage-specific recommendations
        if stage == ValidationStage.GENERATION_1:
            if metrics.overall_score < 70:
                recommendations.append("Focus on implementing core functionality with basic error handling")
            if any(r.gate_name == "test_coverage" and r.status == QualityGateStatus.FAILED for r in gate_results):
                recommendations.append("Add unit tests for core functionality")
        
        elif stage == ValidationStage.GENERATION_2:
            if metrics.overall_score < 85:
                recommendations.append("Enhance error handling and input validation")
            if any(r.gate_name == "security_scan" and r.status == QualityGateStatus.FAILED for r in gate_results):
                recommendations.append("Address security vulnerabilities before proceeding")
        
        elif stage == ValidationStage.GENERATION_3:
            if metrics.overall_score < 90:
                recommendations.append("Optimize performance and add monitoring")
            if any(r.gate_name == "performance_benchmark" and r.status == QualityGateStatus.FAILED for r in gate_results):
                recommendations.append("Performance optimization required for production readiness")
        
        # General recommendations
        failed_gates = [r for r in gate_results if r.status == QualityGateStatus.FAILED]
        for result in failed_gates:
            if result.gate_name == "test_coverage":
                recommendations.append(f"Increase test coverage to {result.threshold}% (currently {result.score:.1f}%)")
            elif result.gate_name == "security_scan":
                recommendations.append("Fix security vulnerabilities identified in scan")
            elif result.gate_name == "code_quality":
                recommendations.append("Address code quality issues (linting, type checking)")
        
        return recommendations
    
    def _is_next_stage_ready(
        self,
        current_stage: ValidationStage,
        metrics: ValidationMetrics,
        gate_results: List[QualityGateResult]
    ) -> bool:
        """Determine if ready for next stage."""
        # Check for critical failures
        if metrics.critical_failures:
            return False
        
        # Check required gates
        required_failed = any(
            r.status == QualityGateStatus.FAILED
            for r, gate in zip(gate_results, self.stage_gates[current_stage])
            if gate.required
        )
        
        if required_failed:
            return False
        
        # Stage-specific readiness criteria
        if current_stage == ValidationStage.GENERATION_1:
            return metrics.overall_score >= 70.0
        elif current_stage == ValidationStage.GENERATION_2:
            return metrics.overall_score >= 85.0
        elif current_stage == ValidationStage.GENERATION_3:
            return metrics.overall_score >= 90.0
        
        return True
    
    async def _attempt_auto_fixes(self, gate_results: List[QualityGateResult]):
        """Attempt automatic fixes for failed gates."""
        for result in gate_results:
            if result.status == QualityGateStatus.FAILED and result.gate_name in self.auto_fix_strategies:
                try:
                    self.logger.info(f"Attempting auto-fix for {result.gate_name}")
                    await self.auto_fix_strategies[result.gate_name](result)
                except Exception as e:
                    self.logger.warning(f"Auto-fix failed for {result.gate_name}: {e}")
    
    async def _auto_fix_test_coverage(self, result: QualityGateResult):
        """Auto-fix for test coverage issues."""
        # This would generate basic test templates
        self.logger.info("Auto-fix: Generating test templates for uncovered code")
        # Implementation would analyze coverage report and generate test stubs
    
    async def _auto_fix_code_quality(self, result: QualityGateResult):
        """Auto-fix for code quality issues."""
        # Run automatic formatters
        cmd = ["python", "-m", "black", "src/"]
        process = await asyncio.create_subprocess_exec(*cmd, cwd=self.project_root)
        await process.communicate()
        
        cmd = ["python", "-m", "isort", "src/"]
        process = await asyncio.create_subprocess_exec(*cmd, cwd=self.project_root)
        await process.communicate()
    
    async def _auto_fix_security(self, result: QualityGateResult):
        """Auto-fix for security issues."""
        self.logger.info("Auto-fix: Applying security patches where possible")
        # Implementation would fix common security issues automatically
    
    async def _auto_fix_compliance(self, result: QualityGateResult):
        """Auto-fix for compliance issues."""
        self.logger.info("Auto-fix: Creating missing compliance files")
        # Implementation would generate missing compliance files
    
    async def _generate_validation_report(
        self,
        stage: ValidationStage,
        metrics: ValidationMetrics,
        gate_results: List[QualityGateResult],
        recommendations: List[str]
    ) -> str:
        """Generate detailed validation report."""
        report_data = {
            "stage": stage.value,
            "timestamp": metrics.timestamp,
            "metrics": {
                "overall_score": metrics.overall_score,
                "success_rate": metrics.success_rate,
                "total_gates": metrics.total_gates,
                "passed_gates": metrics.passed_gates,
                "failed_gates": metrics.failed_gates,
                "warning_gates": metrics.warning_gates,
                "error_gates": metrics.error_gates,
                "execution_time": metrics.execution_time,
                "critical_failures": metrics.critical_failures,
                "performance_metrics": metrics.performance_metrics
            },
            "gate_results": [
                {
                    "name": r.gate_name,
                    "status": r.status.value,
                    "score": r.score,
                    "threshold": r.threshold,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "artifacts": r.artifacts
                }
                for r in gate_results
            ],
            "recommendations": recommendations,
            "next_stage_ready": self._is_next_stage_ready(stage, metrics, gate_results)
        }
        
        # Save report
        report_dir = self.project_root / "quality_reports"
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"validation_report_{stage.value}_{int(time.time())}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Validation report saved: {report_file}")
        return str(report_file)