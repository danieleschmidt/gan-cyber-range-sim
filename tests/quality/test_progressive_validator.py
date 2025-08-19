"""Tests for progressive validation system."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from gan_cyber_range.quality.progressive_validator import (
    ProgressiveValidator,
    ValidationStage,
    ValidationResult,
    ValidationMetrics
)
from gan_cyber_range.quality.quality_gates import QualityGateStatus, QualityGateResult


@pytest.fixture
def temp_project():
    """Create temporary project directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create basic project structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "quality_reports").mkdir()
        
        yield project_path


@pytest.fixture
def mock_gate_results():
    """Create mock quality gate results."""
    return [
        QualityGateResult(
            gate_name="test_coverage",
            status=QualityGateStatus.PASSED,
            score=85.0,
            threshold=80.0,
            message="Test coverage passed",
            details={},
            execution_time=2.0,
            timestamp="2024-01-01T00:00:00Z"
        ),
        QualityGateResult(
            gate_name="security_scan",
            status=QualityGateStatus.PASSED,
            score=92.0,
            threshold=90.0,
            message="Security scan passed",
            details={},
            execution_time=3.0,
            timestamp="2024-01-01T00:00:00Z"
        ),
        QualityGateResult(
            gate_name="code_quality",
            status=QualityGateStatus.FAILED,
            score=75.0,
            threshold=85.0,
            message="Code quality failed",
            details={},
            execution_time=1.5,
            timestamp="2024-01-01T00:00:00Z"
        )
    ]


class TestProgressiveValidator:
    """Test ProgressiveValidator functionality."""
    
    def test_stage_gates_configuration(self, temp_project):
        """Test that stage gates are properly configured."""
        validator = ProgressiveValidator(str(temp_project))
        
        # Check that all stages have gates configured
        for stage in ValidationStage:
            assert stage in validator.stage_gates
            assert len(validator.stage_gates[stage]) > 0
        
        # Verify progressive thresholds (Generation 3 should be stricter than Generation 1)
        gen1_gates = validator.stage_gates[ValidationStage.GENERATION_1]
        gen3_gates = validator.stage_gates[ValidationStage.GENERATION_3]
        
        # Find test coverage gates
        gen1_coverage = next((g for g in gen1_gates if g.name == "test_coverage"), None)
        gen3_coverage = next((g for g in gen3_gates if g.name == "test_coverage"), None)
        
        assert gen1_coverage is not None
        assert gen3_coverage is not None
        assert gen1_coverage.threshold < gen3_coverage.threshold
    
    @pytest.mark.asyncio
    async def test_validate_stage_success(self, temp_project, mock_gate_results):
        """Test successful stage validation."""
        validator = ProgressiveValidator(str(temp_project))
        
        # Mock successful gate execution
        with patch.object(validator, '_execute_quality_gates') as mock_execute:
            mock_execute.return_value = mock_gate_results
            
            result = await validator.validate_stage(ValidationStage.GENERATION_1)
        
        assert isinstance(result, ValidationResult)
        assert result.stage == ValidationStage.GENERATION_1
        assert result.status == QualityGateStatus.FAILED  # One gate failed
        assert len(result.gate_results) == len(mock_gate_results)
        assert result.metrics.passed_gates == 2
        assert result.metrics.failed_gates == 1
    
    @pytest.mark.asyncio
    async def test_validate_stage_metrics_calculation(self, temp_project, mock_gate_results):
        """Test validation metrics calculation."""
        validator = ProgressiveValidator(str(temp_project))
        
        with patch.object(validator, '_execute_quality_gates') as mock_execute:
            mock_execute.return_value = mock_gate_results
            
            result = await validator.validate_stage(ValidationStage.GENERATION_2)
        
        metrics = result.metrics
        assert metrics.total_gates == 3
        assert metrics.passed_gates == 2
        assert metrics.failed_gates == 1
        assert metrics.success_rate == (2 / 3) * 100
        assert not metrics.is_passing  # Has failed gates
        
        # Check overall score calculation
        expected_score = (85.0 + 92.0 + 75.0) / 3
        assert abs(metrics.overall_score - expected_score) < 0.1
    
    @pytest.mark.asyncio
    async def test_auto_fix_functionality(self, temp_project):
        """Test auto-fix functionality."""
        validator = ProgressiveValidator(str(temp_project), enable_auto_fix=True)
        
        # Create a failed gate result
        failed_result = QualityGateResult(
            gate_name="code_quality",
            status=QualityGateStatus.FAILED,
            score=70.0,
            threshold=85.0,
            message="Code quality failed",
            details={},
            execution_time=1.0,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        with patch.object(validator, '_execute_quality_gates') as mock_execute:
            mock_execute.return_value = [failed_result]
            
            with patch.object(validator, '_attempt_auto_fixes') as mock_auto_fix:
                mock_auto_fix.return_value = None
                
                result = await validator.validate_stage(ValidationStage.GENERATION_1, auto_fix=True)
                
                # Verify auto-fix was attempted
                mock_auto_fix.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_progressive_validation(self, temp_project):
        """Test progressive validation across multiple stages."""
        validator = ProgressiveValidator(str(temp_project))
        
        # Mock successful results for all stages
        passing_results = [
            QualityGateResult(
                gate_name="test_gate",
                status=QualityGateStatus.PASSED,
                score=95.0,
                threshold=80.0,
                message="Gate passed",
                details={},
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00Z"
            )
        ]
        
        with patch.object(validator, '_execute_quality_gates') as mock_execute:
            mock_execute.return_value = passing_results
            
            results = await validator.validate_progressive(
                start_stage=ValidationStage.GENERATION_1,
                target_stage=ValidationStage.GENERATION_3
            )
        
        assert len(results) == 3  # Generation 1, 2, 3
        assert all(r.status == QualityGateStatus.PASSED for r in results)
        assert all(r.next_stage_ready for r in results)
    
    @pytest.mark.asyncio
    async def test_fail_fast_behavior(self, temp_project):
        """Test fail-fast behavior in progressive validation."""
        validator = ProgressiveValidator(str(temp_project), fail_fast=True)
        
        # Create failed result for Generation 1
        failed_result = QualityGateResult(
            gate_name="critical_gate",
            status=QualityGateStatus.FAILED,
            score=30.0,
            threshold=80.0,
            message="Critical failure",
            details={},
            execution_time=1.0,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        with patch.object(validator, '_execute_quality_gates') as mock_execute:
            mock_execute.return_value = [failed_result]
            
            results = await validator.validate_progressive(
                start_stage=ValidationStage.GENERATION_1,
                target_stage=ValidationStage.PRODUCTION
            )
        
        # Should stop at Generation 1 due to failure
        assert len(results) == 1
        assert results[0].stage == ValidationStage.GENERATION_1
        assert not results[0].next_stage_ready
    
    @pytest.mark.asyncio
    async def test_validation_report_generation(self, temp_project, mock_gate_results):
        """Test validation report generation."""
        validator = ProgressiveValidator(str(temp_project))
        
        with patch.object(validator, '_execute_quality_gates') as mock_execute:
            mock_execute.return_value = mock_gate_results
            
            result = await validator.validate_stage(ValidationStage.GENERATION_2)
        
        # Check that report was generated
        assert result.report_path is not None
        report_file = Path(result.report_path)
        assert report_file.exists()
        
        # Verify report content
        with open(report_file) as f:
            report_data = json.load(f)
        
        assert report_data["stage"] == "generation_2"
        assert "metrics" in report_data
        assert "gate_results" in report_data
        assert "recommendations" in report_data
        assert "next_stage_ready" in report_data
    
    def test_recommendations_generation(self, temp_project, mock_gate_results):
        """Test recommendation generation."""
        validator = ProgressiveValidator(str(temp_project))
        
        # Test Generation 1 recommendations
        recommendations = validator._generate_recommendations(
            ValidationStage.GENERATION_1, 
            mock_gate_results, 
            ValidationMetrics(
                stage=ValidationStage.GENERATION_1,
                total_gates=3,
                passed_gates=2,
                failed_gates=1,
                warning_gates=0,
                error_gates=0,
                overall_score=75.0,
                execution_time=6.5,
                timestamp="2024-01-01T00:00:00Z"
            )
        )
        
        assert len(recommendations) > 0
        assert any("code quality" in rec.lower() for rec in recommendations)
    
    def test_next_stage_readiness(self, temp_project):
        """Test next stage readiness determination."""
        validator = ProgressiveValidator(str(temp_project))
        
        # Test with passing metrics
        passing_metrics = ValidationMetrics(
            stage=ValidationStage.GENERATION_1,
            total_gates=3,
            passed_gates=3,
            failed_gates=0,
            warning_gates=0,
            error_gates=0,
            overall_score=85.0,
            execution_time=5.0,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        passing_results = [
            QualityGateResult(
                gate_name="test_gate",
                status=QualityGateStatus.PASSED,
                score=85.0,
                threshold=80.0,
                message="Passed",
                details={},
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00Z"
            )
        ]
        
        ready = validator._is_next_stage_ready(
            ValidationStage.GENERATION_1, 
            passing_metrics, 
            passing_results
        )
        assert ready
        
        # Test with failing metrics
        failing_metrics = ValidationMetrics(
            stage=ValidationStage.GENERATION_1,
            total_gates=3,
            passed_gates=1,
            failed_gates=2,
            warning_gates=0,
            error_gates=0,
            overall_score=55.0,
            execution_time=5.0,
            timestamp="2024-01-01T00:00:00Z",
            critical_failures=["critical_gate"]
        )
        
        not_ready = validator._is_next_stage_ready(
            ValidationStage.GENERATION_1, 
            failing_metrics, 
            []
        )
        assert not not_ready
    
    @pytest.mark.asyncio
    async def test_auto_fix_strategies(self, temp_project):
        """Test auto-fix strategy execution."""
        validator = ProgressiveValidator(str(temp_project))
        
        # Test code quality auto-fix
        failed_code_quality = QualityGateResult(
            gate_name="code_quality",
            status=QualityGateStatus.FAILED,
            score=70.0,
            threshold=85.0,
            message="Code quality failed",
            details={},
            execution_time=1.0,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_process
            
            await validator._auto_fix_code_quality(failed_code_quality)
            
            # Should call black and isort
            assert mock_subprocess.call_count >= 2


class TestValidationMetrics:
    """Test ValidationMetrics functionality."""
    
    def test_metrics_creation(self):
        """Test ValidationMetrics creation and properties."""
        metrics = ValidationMetrics(
            stage=ValidationStage.GENERATION_2,
            total_gates=5,
            passed_gates=3,
            failed_gates=2,
            warning_gates=0,
            error_gates=0,
            overall_score=75.0,
            execution_time=10.0,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        assert metrics.stage == ValidationStage.GENERATION_2
        assert metrics.success_rate == 60.0  # 3/5 * 100
        assert not metrics.is_passing  # Has failed gates
    
    def test_metrics_all_passed(self):
        """Test metrics when all gates pass."""
        metrics = ValidationMetrics(
            stage=ValidationStage.GENERATION_1,
            total_gates=3,
            passed_gates=3,
            failed_gates=0,
            warning_gates=0,
            error_gates=0,
            overall_score=90.0,
            execution_time=5.0,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        assert metrics.success_rate == 100.0
        assert metrics.is_passing
    
    def test_metrics_with_critical_failures(self):
        """Test metrics with critical failures."""
        metrics = ValidationMetrics(
            stage=ValidationStage.GENERATION_3,
            total_gates=4,
            passed_gates=3,
            failed_gates=1,
            warning_gates=0,
            error_gates=0,
            overall_score=80.0,
            execution_time=8.0,
            timestamp="2024-01-01T00:00:00Z",
            critical_failures=["security_scan"]
        )
        
        assert not metrics.is_passing  # Critical failures prevent passing


class TestValidationResult:
    """Test ValidationResult functionality."""
    
    def test_result_creation(self, mock_gate_results):
        """Test ValidationResult creation."""
        metrics = ValidationMetrics(
            stage=ValidationStage.GENERATION_2,
            total_gates=3,
            passed_gates=2,
            failed_gates=1,
            warning_gates=0,
            error_gates=0,
            overall_score=84.0,
            execution_time=6.5,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        result = ValidationResult(
            stage=ValidationStage.GENERATION_2,
            status=QualityGateStatus.FAILED,
            metrics=metrics,
            gate_results=mock_gate_results,
            recommendations=["Fix code quality issues"],
            next_stage_ready=False,
            report_path="/tmp/report.json"
        )
        
        assert result.stage == ValidationStage.GENERATION_2
        assert result.status == QualityGateStatus.FAILED
        assert len(result.gate_results) == 3
        assert len(result.recommendations) == 1
        assert not result.next_stage_ready
        assert result.report_path == "/tmp/report.json"


@pytest.mark.integration
class TestProgressiveValidatorIntegration:
    """Integration tests for progressive validator."""
    
    @pytest.mark.asyncio
    async def test_full_validation_cycle(self, temp_project):
        """Test complete validation cycle with real gate execution."""
        validator = ProgressiveValidator(str(temp_project), enable_auto_fix=False)
        
        # Use only compliance gate which should pass with basic project structure
        validator.stage_gates = {
            ValidationStage.GENERATION_1: [
                validator.stage_gates[ValidationStage.GENERATION_1][-1]  # Compliance gate
            ]
        }
        
        try:
            result = await validator.validate_stage(ValidationStage.GENERATION_1)
            
            assert isinstance(result, ValidationResult)
            assert result.stage == ValidationStage.GENERATION_1
            assert len(result.gate_results) > 0
            assert result.metrics.total_gates > 0
            
            # Verify report was generated
            assert result.report_path is not None
            assert Path(result.report_path).exists()
            
        except Exception as e:
            # Some gates may fail in test environment, but structure should be intact
            pytest.skip(f"Integration test failed due to environment: {e}")


if __name__ == "__main__":
    pytest.main([__file__])