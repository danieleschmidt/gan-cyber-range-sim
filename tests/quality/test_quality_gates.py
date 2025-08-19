"""Tests for core quality gates functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
import tempfile

from gan_cyber_range.quality.quality_gates import (
    QualityGate,
    QualityGateResult,
    QualityGateStatus,
    QualityGateRunner,
    TestCoverageGate,
    SecurityGate,
    CodeQualityGate,
    PerformanceGate,
    ComplianceGate
)


class MockQualityGate(QualityGate):
    """Mock quality gate for testing."""
    
    def __init__(self, name: str, should_pass: bool = True, execution_time: float = 0.1):
        super().__init__(name, threshold=80.0)
        self.should_pass = should_pass
        self.execution_time = execution_time
    
    async def execute(self, context: dict) -> QualityGateResult:
        """Mock execution."""
        await asyncio.sleep(self.execution_time)
        
        if self.should_pass:
            status = QualityGateStatus.PASSED
            score = 90.0
            message = "Mock gate passed"
        else:
            status = QualityGateStatus.FAILED
            score = 60.0
            message = "Mock gate failed"
        
        return self._create_result(
            status=status,
            score=score,
            message=message,
            execution_time=self.execution_time
        )


@pytest.fixture
def temp_project():
    """Create temporary project directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create basic project structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "LICENSE").write_text("MIT License")
        (project_path / "README.md").write_text("# Test Project")
        (project_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        
        yield project_path


@pytest.fixture
def mock_context(temp_project):
    """Create mock context for testing."""
    return {
        "project_root": str(temp_project),
        "stage": "test",
        "auto_fix_enabled": False
    }


class TestQualityGateRunner:
    """Test QualityGateRunner functionality."""
    
    @pytest.mark.asyncio
    async def test_run_all_parallel(self):
        """Test parallel execution of quality gates."""
        gates = [
            MockQualityGate("gate1", should_pass=True, execution_time=0.1),
            MockQualityGate("gate2", should_pass=True, execution_time=0.2),
            MockQualityGate("gate3", should_pass=False, execution_time=0.1)
        ]
        
        runner = QualityGateRunner(gates)
        
        start_time = time.time()
        results = await runner.run_all(context={}, parallel=True)
        execution_time = time.time() - start_time
        
        # Should complete in roughly the time of the slowest gate (0.2s) plus overhead
        assert execution_time < 0.5
        assert len(results) == 3
        
        # Check individual results
        assert results[0].status == QualityGateStatus.PASSED
        assert results[1].status == QualityGateStatus.PASSED
        assert results[2].status == QualityGateStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_run_all_sequential(self):
        """Test sequential execution of quality gates."""
        gates = [
            MockQualityGate("gate1", should_pass=True, execution_time=0.1),
            MockQualityGate("gate2", should_pass=True, execution_time=0.1)
        ]
        
        runner = QualityGateRunner(gates)
        
        start_time = time.time()
        results = await runner.run_all(context={}, parallel=False)
        execution_time = time.time() - start_time
        
        # Should take roughly the sum of all gate times
        assert execution_time >= 0.2
        assert len(results) == 2
        assert all(r.status == QualityGateStatus.PASSED for r in results)
    
    @pytest.mark.asyncio
    async def test_fail_fast(self):
        """Test fail-fast behavior."""
        gates = [
            MockQualityGate("gate1", should_pass=False, execution_time=0.1),
            MockQualityGate("gate2", should_pass=True, execution_time=0.1),
            MockQualityGate("gate3", should_pass=True, execution_time=0.1)
        ]
        
        # Make first gate required
        gates[0].required = True
        
        runner = QualityGateRunner(gates)
        
        results = await runner.run_all(context={}, fail_fast=True, parallel=False)
        
        # Should stop after first gate failure
        assert len(results) == 1
        assert results[0].status == QualityGateStatus.FAILED


class TestTestCoverageGate:
    """Test TestCoverageGate functionality."""
    
    @pytest.mark.asyncio
    async def test_coverage_gate_success(self, mock_context, temp_project):
        """Test successful coverage gate execution."""
        # Create mock coverage.json
        coverage_data = {
            "totals": {
                "percent_covered": 85.5,
                "covered_lines": 855,
                "num_statements": 1000
            },
            "files": {}
        }
        
        coverage_file = temp_project / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))
        
        gate = TestCoverageGate(threshold=80.0)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful pytest execution
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_process
            
            result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 85.5
        assert "Test coverage: 85.5%" in result.message
    
    @pytest.mark.asyncio
    async def test_coverage_gate_failure(self, mock_context, temp_project):
        """Test failed coverage gate execution."""
        # Create mock coverage.json with low coverage
        coverage_data = {
            "totals": {
                "percent_covered": 65.0,
                "covered_lines": 650,
                "num_statements": 1000
            },
            "files": {}
        }
        
        coverage_file = temp_project / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))
        
        gate = TestCoverageGate(threshold=80.0)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_process
            
            result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 65.0
    
    @pytest.mark.asyncio
    async def test_coverage_gate_timeout(self, mock_context):
        """Test coverage gate timeout handling."""
        gate = TestCoverageGate(threshold=80.0, timeout=1)  # 1 second timeout
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock process that never completes
            mock_process = Mock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_subprocess.return_value = mock_process
            
            result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.ERROR
        assert "timed out" in result.message.lower()


class TestSecurityGate:
    """Test SecurityGate functionality."""
    
    @pytest.mark.asyncio
    async def test_security_gate_success(self, mock_context, temp_project):
        """Test successful security gate execution."""
        # Create mock bandit report
        bandit_data = {
            "results": [
                {"issue_severity": "LOW", "test_name": "test1"},
                {"issue_severity": "MEDIUM", "test_name": "test2"}
            ]
        }
        
        bandit_file = temp_project / "bandit_report.json"
        bandit_file.write_text(json.dumps(bandit_data))
        
        gate = SecurityGate(threshold=90.0)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock bandit execution
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_process
            
            result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score >= 90.0  # Should be high with only low/medium issues
    
    @pytest.mark.asyncio
    async def test_security_gate_high_severity_issues(self, mock_context, temp_project):
        """Test security gate with high severity issues."""
        # Create mock bandit report with high severity issues
        bandit_data = {
            "results": [
                {"issue_severity": "HIGH", "test_name": "critical_issue"},
                {"issue_severity": "HIGH", "test_name": "another_critical"},
                {"issue_severity": "MEDIUM", "test_name": "medium_issue"}
            ]
        }
        
        bandit_file = temp_project / "bandit_report.json"
        bandit_file.write_text(json.dumps(bandit_data))
        
        gate = SecurityGate(threshold=90.0)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_process
            
            result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score < 90.0  # Should be low due to high severity issues


class TestCodeQualityGate:
    """Test CodeQualityGate functionality."""
    
    @pytest.mark.asyncio
    async def test_code_quality_gate_success(self, mock_context):
        """Test successful code quality gate execution."""
        gate = CodeQualityGate(threshold=85.0)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful ruff and mypy execution
            mock_process = Mock()
            mock_process.communicate.return_value = (b"[]", b"")  # Empty JSON array (no issues)
            mock_subprocess.return_value = mock_process
            
            result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score >= 85.0
    
    @pytest.mark.asyncio
    async def test_code_quality_gate_with_issues(self, mock_context):
        """Test code quality gate with linting issues."""
        gate = CodeQualityGate(threshold=85.0)
        
        # Mock ruff output with issues
        ruff_issues = [
            {"severity": "error", "message": "syntax error"},
            {"severity": "warning", "message": "unused import"}
        ]
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (json.dumps(ruff_issues).encode(), b"")
            mock_subprocess.return_value = mock_process
            
            result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score < 85.0


class TestPerformanceGate:
    """Test PerformanceGate functionality."""
    
    @pytest.mark.asyncio
    async def test_performance_gate_success(self, mock_context, temp_project):
        """Test successful performance gate execution."""
        # Create mock benchmark results
        benchmark_data = {
            "benchmarks": [
                {"stats": {"mean": 0.5}},  # Fast benchmark
                {"stats": {"mean": 0.3}},  # Even faster
            ]
        }
        
        benchmark_file = temp_project / "benchmark_results.json"
        benchmark_file.write_text(json.dumps(benchmark_data))
        
        gate = PerformanceGate(threshold=80.0)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_process
            
            result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score >= 80.0
    
    @pytest.mark.asyncio
    async def test_performance_gate_slow_benchmarks(self, mock_context, temp_project):
        """Test performance gate with slow benchmarks."""
        # Create mock benchmark results with slow tests
        benchmark_data = {
            "benchmarks": [
                {"stats": {"mean": 2.0}},  # Slow benchmark (>1s)
                {"stats": {"mean": 1.5}},  # Another slow one
                {"stats": {"mean": 0.3}},  # Fast one
            ]
        }
        
        benchmark_file = temp_project / "benchmark_results.json"
        benchmark_file.write_text(json.dumps(benchmark_data))
        
        gate = PerformanceGate(threshold=80.0)
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_process
            
            result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score < 80.0


class TestComplianceGate:
    """Test ComplianceGate functionality."""
    
    @pytest.mark.asyncio
    async def test_compliance_gate_success(self, mock_context, temp_project):
        """Test successful compliance gate execution."""
        # Project already has required files from fixture
        gate = ComplianceGate(threshold=95.0)
        
        result = await gate.execute(mock_context)
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score >= 95.0
    
    @pytest.mark.asyncio
    async def test_compliance_gate_missing_files(self, mock_context):
        """Test compliance gate with missing required files."""
        # Use a context with an empty directory
        with tempfile.TemporaryDirectory() as temp_dir:
            context = {
                "project_root": temp_dir,
                "stage": "test"
            }
            
            gate = ComplianceGate(threshold=95.0)
            result = await gate.execute(context)
            
            assert result.status == QualityGateStatus.FAILED
            assert result.score < 95.0
            assert "missing_files" in result.details


@pytest.mark.integration
class TestQualityGatesIntegration:
    """Integration tests for quality gates system."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, mock_context):
        """Test execution of complete quality gates pipeline."""
        gates = [
            MockQualityGate("gate1", should_pass=True),
            MockQualityGate("gate2", should_pass=True),
            MockQualityGate("gate3", should_pass=False)
        ]
        
        runner = QualityGateRunner(gates)
        results = await runner.run_all(mock_context, parallel=True)
        
        assert len(results) == 3
        assert sum(1 for r in results if r.status == QualityGateStatus.PASSED) == 2
        assert sum(1 for r in results if r.status == QualityGateStatus.FAILED) == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_context):
        """Test error handling in quality gates."""
        class ErrorGate(QualityGate):
            async def execute(self, context):
                raise Exception("Simulated error")
        
        gate = ErrorGate("error_gate")
        runner = QualityGateRunner([gate])
        
        results = await runner.run_all(mock_context)
        
        assert len(results) == 1
        assert results[0].status == QualityGateStatus.ERROR
        assert "Simulated error" in results[0].message
    
    @pytest.mark.asyncio
    async def test_mixed_gate_types(self, mock_context):
        """Test execution of different gate types together."""
        gates = [
            MockQualityGate("mock_gate", should_pass=True),
            ComplianceGate(threshold=80.0)  # Real gate with lower threshold
        ]
        
        runner = QualityGateRunner(gates)
        results = await runner.run_all(mock_context, parallel=True)
        
        assert len(results) == 2
        # Both should pass with lower compliance threshold
        for result in results:
            assert result.status in [QualityGateStatus.PASSED, QualityGateStatus.WARNING]


class TestQualityGateResult:
    """Test QualityGateResult functionality."""
    
    def test_result_creation(self):
        """Test QualityGateResult creation and properties."""
        result = QualityGateResult(
            gate_name="test_gate",
            status=QualityGateStatus.PASSED,
            score=85.5,
            threshold=80.0,
            message="Test passed",
            details={"key": "value"},
            execution_time=1.5,
            timestamp="2024-01-01T00:00:00Z",
            artifacts=["test.log"]
        )
        
        assert result.gate_name == "test_gate"
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 85.5
        assert result.threshold == 80.0
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}
        assert result.execution_time == 1.5
        assert result.artifacts == ["test.log"]
    
    def test_result_with_defaults(self):
        """Test QualityGateResult with default values."""
        result = QualityGateResult(
            gate_name="test_gate",
            status=QualityGateStatus.PASSED,
            score=85.5,
            threshold=80.0,
            message="Test passed",
            details={},
            execution_time=1.5,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        assert result.artifacts == []  # Default empty list


if __name__ == "__main__":
    pytest.main([__file__])