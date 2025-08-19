"""Automated quality pipeline for continuous validation."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json
import yaml

from .progressive_validator import ProgressiveValidator, ValidationStage, ValidationResult
from .quality_gates import QualityGateStatus
from ..core.error_handling import CyberRangeError, ErrorSeverity


class PipelineStage(Enum):
    """Pipeline execution stages."""
    PRE_VALIDATION = "pre_validation"
    QUALITY_GATES = "quality_gates"
    PROGRESSIVE_VALIDATION = "progressive_validation"
    POST_VALIDATION = "post_validation"
    REPORTING = "reporting"
    DEPLOYMENT_PREP = "deployment_prep"


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_id: str
    status: QualityGateStatus
    stages_completed: List[PipelineStage]
    validation_results: List[ValidationResult]
    total_execution_time: float
    artifacts: List[str] = field(default_factory=list)
    deployment_ready: bool = False
    error_messages: List[str] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        """Calculate overall pipeline score."""
        if not self.validation_results:
            return 0.0
        
        total_score = sum(r.metrics.overall_score for r in self.validation_results)
        return total_score / len(self.validation_results)
    
    @property
    def success_rate(self) -> float:
        """Calculate pipeline success rate."""
        if not self.validation_results:
            return 0.0
        
        passed = len([r for r in self.validation_results if r.status == QualityGateStatus.PASSED])
        return (passed / len(self.validation_results)) * 100


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    pipeline_result: PipelineResult
    detailed_metrics: Dict[str, Any]
    recommendations: List[str]
    next_steps: List[str]
    compliance_status: Dict[str, bool]
    performance_benchmarks: Dict[str, float]
    security_assessment: Dict[str, Any]
    report_path: str
    generated_at: str


class AutomatedQualityPipeline:
    """Automated quality pipeline for continuous validation and deployment."""
    
    def __init__(
        self,
        project_root: str = ".",
        config_file: Optional[str] = None,
        enable_notifications: bool = True,
        auto_deploy: bool = False
    ):
        self.project_root = Path(project_root)
        self.enable_notifications = enable_notifications
        self.auto_deploy = auto_deploy
        self.logger = logging.getLogger("automated_quality_pipeline")
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.validator = ProgressiveValidator(
            project_root=project_root,
            enable_auto_fix=self.config.get("enable_auto_fix", True),
            fail_fast=self.config.get("fail_fast", False)
        )
        
        # Pipeline hooks
        self.stage_hooks: Dict[PipelineStage, List[Callable]] = {
            stage: [] for stage in PipelineStage
        }
        
        # Notification handlers
        self.notification_handlers: List[Callable] = []
        
        # Register default hooks
        self._register_default_hooks()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load pipeline configuration."""
        default_config = {
            "enable_auto_fix": True,
            "fail_fast": False,
            "parallel_execution": True,
            "target_stage": "production",
            "notification_channels": ["console"],
            "quality_thresholds": {
                "minimum_overall_score": 85.0,
                "minimum_success_rate": 90.0,
                "maximum_critical_failures": 0
            },
            "deployment_criteria": {
                "require_all_stages_passed": True,
                "minimum_score": 95.0,
                "security_scan_required": True,
                "performance_benchmark_required": True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_file}: {e}")
        
        return default_config
    
    def _register_default_hooks(self):
        """Register default pipeline hooks."""
        # Pre-validation hooks
        self.register_hook(PipelineStage.PRE_VALIDATION, self._setup_environment)
        self.register_hook(PipelineStage.PRE_VALIDATION, self._validate_dependencies)
        
        # Post-validation hooks
        self.register_hook(PipelineStage.POST_VALIDATION, self._collect_artifacts)
        self.register_hook(PipelineStage.POST_VALIDATION, self._update_metrics)
        
        # Reporting hooks
        self.register_hook(PipelineStage.REPORTING, self._generate_comprehensive_report)
        self.register_hook(PipelineStage.REPORTING, self._send_notifications)
        
        # Deployment prep hooks
        self.register_hook(PipelineStage.DEPLOYMENT_PREP, self._prepare_deployment_artifacts)
        self.register_hook(PipelineStage.DEPLOYMENT_PREP, self._validate_deployment_readiness)
    
    def register_hook(self, stage: PipelineStage, hook_func: Callable):
        """Register a hook for a pipeline stage."""
        self.stage_hooks[stage].append(hook_func)
        self.logger.debug(f"Registered hook {hook_func.__name__} for stage {stage.value}")
    
    def register_notification_handler(self, handler: Callable):
        """Register a notification handler."""
        self.notification_handlers.append(handler)
    
    async def run_pipeline(
        self,
        trigger_event: str = "manual",
        target_stage: Optional[ValidationStage] = None,
        pipeline_id: Optional[str] = None
    ) -> PipelineResult:
        """Run the complete quality pipeline."""
        start_time = time.time()
        
        if pipeline_id is None:
            pipeline_id = f"pipeline_{int(time.time())}"
        
        if target_stage is None:
            target_stage = ValidationStage(self.config.get("target_stage", "production"))
        
        self.logger.info(f"Starting quality pipeline {pipeline_id} (trigger: {trigger_event})")
        
        result = PipelineResult(
            pipeline_id=pipeline_id,
            status=QualityGateStatus.ERROR,  # Will be updated
            stages_completed=[],
            validation_results=[],
            total_execution_time=0.0
        )
        
        try:
            # Execute pipeline stages
            await self._execute_stage(PipelineStage.PRE_VALIDATION, result)
            await self._execute_stage(PipelineStage.QUALITY_GATES, result)
            await self._execute_stage(PipelineStage.PROGRESSIVE_VALIDATION, result, target_stage)
            await self._execute_stage(PipelineStage.POST_VALIDATION, result)
            await self._execute_stage(PipelineStage.REPORTING, result)
            
            # Check deployment readiness
            if self._is_deployment_ready(result):
                await self._execute_stage(PipelineStage.DEPLOYMENT_PREP, result)
                result.deployment_ready = True
            
            # Determine final status
            result.status = self._determine_pipeline_status(result)
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            result.status = QualityGateStatus.ERROR
            result.error_messages.append(str(e))
        
        finally:
            result.total_execution_time = time.time() - start_time
            
            self.logger.info(
                f"Pipeline {pipeline_id} completed: {result.status.value} "
                f"(score: {result.overall_score:.1f}%, time: {result.total_execution_time:.1f}s)"
            )
        
        return result
    
    async def _execute_stage(
        self,
        stage: PipelineStage,
        result: PipelineResult,
        *args
    ):
        """Execute a pipeline stage with hooks."""
        self.logger.info(f"Executing stage: {stage.value}")
        stage_start_time = time.time()
        
        try:
            # Execute stage-specific logic
            if stage == PipelineStage.PROGRESSIVE_VALIDATION:
                target_stage = args[0] if args else ValidationStage.PRODUCTION
                validation_results = await self.validator.validate_progressive(
                    target_stage=target_stage
                )
                result.validation_results.extend(validation_results)
            
            # Execute hooks
            for hook in self.stage_hooks[stage]:
                try:
                    await hook(result, *args)
                except Exception as e:
                    self.logger.warning(f"Hook {hook.__name__} failed: {e}")
            
            result.stages_completed.append(stage)
            
            stage_time = time.time() - stage_start_time
            self.logger.debug(f"Stage {stage.value} completed in {stage_time:.1f}s")
            
        except Exception as e:
            self.logger.error(f"Stage {stage.value} failed: {e}")
            result.error_messages.append(f"Stage {stage.value} failed: {e}")
            raise
    
    def _determine_pipeline_status(self, result: PipelineResult) -> QualityGateStatus:
        """Determine overall pipeline status."""
        if result.error_messages:
            return QualityGateStatus.ERROR
        
        if not result.validation_results:
            return QualityGateStatus.ERROR
        
        # Check if all validations passed
        all_passed = all(
            r.status == QualityGateStatus.PASSED for r in result.validation_results
        )
        
        if all_passed:
            return QualityGateStatus.PASSED
        
        # Check for any failures
        any_failed = any(
            r.status == QualityGateStatus.FAILED for r in result.validation_results
        )
        
        if any_failed:
            return QualityGateStatus.FAILED
        
        # Must have warnings
        return QualityGateStatus.WARNING
    
    def _is_deployment_ready(self, result: PipelineResult) -> bool:
        """Check if pipeline result meets deployment criteria."""
        criteria = self.config.get("deployment_criteria", {})
        
        # Check if all stages passed (if required)
        if criteria.get("require_all_stages_passed", True):
            if not all(r.status == QualityGateStatus.PASSED for r in result.validation_results):
                return False
        
        # Check minimum score
        min_score = criteria.get("minimum_score", 95.0)
        if result.overall_score < min_score:
            return False
        
        # Check for critical failures
        max_critical = criteria.get("maximum_critical_failures", 0)
        critical_failures = sum(
            len(r.metrics.critical_failures) for r in result.validation_results
        )
        if critical_failures > max_critical:
            return False
        
        return True
    
    # Default hook implementations
    async def _setup_environment(self, result: PipelineResult, *args):
        """Setup pipeline environment."""
        self.logger.info("Setting up pipeline environment")
        
        # Create directories
        (self.project_root / "quality_reports").mkdir(exist_ok=True)
        (self.project_root / "artifacts").mkdir(exist_ok=True)
        
        # Set environment variables
        import os
        os.environ["PIPELINE_ID"] = result.pipeline_id
        os.environ["QUALITY_PIPELINE_RUNNING"] = "true"
    
    async def _validate_dependencies(self, result: PipelineResult, *args):
        """Validate pipeline dependencies."""
        self.logger.info("Validating dependencies")
        
        # Check required tools
        required_tools = ["python", "pytest", "bandit", "ruff"]
        missing_tools = []
        
        for tool in required_tools:
            try:
                process = await asyncio.create_subprocess_exec(
                    "which", tool,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                if process.returncode != 0:
                    missing_tools.append(tool)
            except Exception:
                missing_tools.append(tool)
        
        if missing_tools:
            raise CyberRangeError(
                f"Missing required tools: {missing_tools}",
                severity=ErrorSeverity.HIGH
            )
    
    async def _collect_artifacts(self, result: PipelineResult, *args):
        """Collect pipeline artifacts."""
        self.logger.info("Collecting artifacts")
        
        artifacts = []
        
        # Collect validation reports
        for validation_result in result.validation_results:
            if validation_result.report_path:
                artifacts.append(validation_result.report_path)
        
        # Collect test artifacts
        test_artifacts = [
            "coverage.json",
            "htmlcov/",
            "pytest_report.html",
            "benchmark_results.json",
            "bandit_report.json"
        ]
        
        for artifact in test_artifacts:
            artifact_path = self.project_root / artifact
            if artifact_path.exists():
                artifacts.append(str(artifact_path))
        
        result.artifacts.extend(artifacts)
    
    async def _update_metrics(self, result: PipelineResult, *args):
        """Update pipeline metrics."""
        self.logger.info("Updating metrics")
        
        # Store metrics for trending
        metrics_file = self.project_root / "quality_reports" / "pipeline_metrics.json"
        
        current_metrics = {
            "pipeline_id": result.pipeline_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "overall_score": result.overall_score,
            "success_rate": result.success_rate,
            "execution_time": result.total_execution_time,
            "deployment_ready": result.deployment_ready
        }
        
        # Append to historical metrics
        historical_metrics = []
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    historical_metrics = json.load(f)
            except Exception:
                pass
        
        historical_metrics.append(current_metrics)
        
        # Keep only last 100 entries
        historical_metrics = historical_metrics[-100:]
        
        with open(metrics_file, 'w') as f:
            json.dump(historical_metrics, f, indent=2)
    
    async def _generate_comprehensive_report(self, result: PipelineResult, *args):
        """Generate comprehensive quality report."""
        self.logger.info("Generating comprehensive report")
        
        # Detailed metrics
        detailed_metrics = {
            "pipeline_execution": {
                "total_time": result.total_execution_time,
                "stages_completed": [s.value for s in result.stages_completed],
                "overall_score": result.overall_score,
                "success_rate": result.success_rate
            },
            "validation_stages": [
                {
                    "stage": r.stage.value,
                    "status": r.status.value,
                    "score": r.metrics.overall_score,
                    "gates_passed": r.metrics.passed_gates,
                    "gates_failed": r.metrics.failed_gates,
                    "execution_time": r.metrics.execution_time
                }
                for r in result.validation_results
            ]
        }
        
        # Recommendations
        all_recommendations = []
        for validation_result in result.validation_results:
            all_recommendations.extend(validation_result.recommendations)
        
        # Next steps
        next_steps = []
        if result.deployment_ready:
            next_steps.append("Ready for deployment")
        else:
            next_steps.append("Address failing quality gates before deployment")
            next_steps.extend(all_recommendations[:5])  # Top 5 recommendations
        
        # Compliance status
        compliance_status = {
            "security_scan_passed": any(
                r.gate_name == "security_scan" and r.status == QualityGateStatus.PASSED
                for vr in result.validation_results
                for r in vr.gate_results
            ),
            "test_coverage_adequate": any(
                r.gate_name == "test_coverage" and r.score >= 85.0
                for vr in result.validation_results
                for r in vr.gate_results
            ),
            "code_quality_passed": any(
                r.gate_name == "code_quality" and r.status == QualityGateStatus.PASSED
                for vr in result.validation_results
                for r in vr.gate_results
            )
        }
        
        # Performance benchmarks
        performance_benchmarks = {}
        for validation_result in result.validation_results:
            performance_benchmarks.update(validation_result.metrics.performance_metrics)
        
        # Security assessment
        security_issues = []
        for validation_result in result.validation_results:
            for gate_result in validation_result.gate_results:
                if gate_result.gate_name == "security_scan" and gate_result.details:
                    security_issues.extend(gate_result.details.get("issues", []))
        
        security_assessment = {
            "total_issues": len(security_issues),
            "high_severity": len([i for i in security_issues if i.get("issue_severity") == "HIGH"]),
            "medium_severity": len([i for i in security_issues if i.get("issue_severity") == "MEDIUM"]),
            "scan_status": "passed" if not security_issues else "issues_found"
        }
        
        # Generate report
        report = QualityReport(
            pipeline_result=result,
            detailed_metrics=detailed_metrics,
            recommendations=list(set(all_recommendations)),  # Deduplicate
            next_steps=next_steps,
            compliance_status=compliance_status,
            performance_benchmarks=performance_benchmarks,
            security_assessment=security_assessment,
            report_path="",  # Will be set after saving
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        # Save report
        report_file = (
            self.project_root / "quality_reports" / 
            f"comprehensive_report_{result.pipeline_id}.json"
        )
        
        report_dict = {
            "pipeline_result": {
                "pipeline_id": report.pipeline_result.pipeline_id,
                "status": report.pipeline_result.status.value,
                "overall_score": report.pipeline_result.overall_score,
                "success_rate": report.pipeline_result.success_rate,
                "deployment_ready": report.pipeline_result.deployment_ready,
                "total_execution_time": report.pipeline_result.total_execution_time
            },
            "detailed_metrics": report.detailed_metrics,
            "recommendations": report.recommendations,
            "next_steps": report.next_steps,
            "compliance_status": report.compliance_status,
            "performance_benchmarks": report.performance_benchmarks,
            "security_assessment": report.security_assessment,
            "generated_at": report.generated_at
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        report.report_path = str(report_file)
        self.logger.info(f"Comprehensive report generated: {report_file}")
    
    async def _send_notifications(self, result: PipelineResult, *args):
        """Send pipeline notifications."""
        if not self.enable_notifications:
            return
        
        self.logger.info("Sending notifications")
        
        notification_data = {
            "pipeline_id": result.pipeline_id,
            "status": result.status.value,
            "overall_score": result.overall_score,
            "deployment_ready": result.deployment_ready,
            "summary": f"Pipeline {result.status.value} with score {result.overall_score:.1f}%"
        }
        
        for handler in self.notification_handlers:
            try:
                await handler(notification_data)
            except Exception as e:
                self.logger.warning(f"Notification handler failed: {e}")
    
    async def _prepare_deployment_artifacts(self, result: PipelineResult, *args):
        """Prepare deployment artifacts."""
        self.logger.info("Preparing deployment artifacts")
        
        deployment_dir = self.project_root / "deployment_artifacts"
        deployment_dir.mkdir(exist_ok=True)
        
        # Create deployment manifest
        manifest = {
            "pipeline_id": result.pipeline_id,
            "quality_score": result.overall_score,
            "validation_passed": result.status == QualityGateStatus.PASSED,
            "deployment_ready": result.deployment_ready,
            "artifacts": result.artifacts,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        manifest_file = deployment_dir / "deployment_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        result.artifacts.append(str(manifest_file))
    
    async def _validate_deployment_readiness(self, result: PipelineResult, *args):
        """Final validation of deployment readiness."""
        self.logger.info("Validating deployment readiness")
        
        if not self._is_deployment_ready(result):
            raise CyberRangeError(
                "Deployment readiness validation failed",
                severity=ErrorSeverity.HIGH
            )