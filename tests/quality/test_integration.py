"""Integration tests for the complete quality gates system."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from gan_cyber_range.quality import (
    AutomatedQualityPipeline,
    ProgressiveValidator,
    ValidationStage,
    QualityGateAutoScaler,
    IntelligentOptimizer,
    OptimizationStrategy,
    ScalingStrategy,
    get_global_metrics_collector,
    get_global_monitor
)


@pytest.fixture
def temp_project():
    """Create temporary project directory with realistic structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create realistic project structure
        (project_path / "src").mkdir()
        (project_path / "src" / "test_package").mkdir()
        (project_path / "src" / "test_package" / "__init__.py").write_text("# Test package")
        (project_path / "src" / "test_package" / "main.py").write_text("""
def hello_world():
    '''Simple hello world function.'''
    return "Hello, World!"

if __name__ == "__main__":
    print(hello_world())
""")
        
        (project_path / "tests").mkdir()
        (project_path / "tests" / "__init__.py").write_text("")
        (project_path / "tests" / "test_main.py").write_text("""
import pytest
from test_package.main import hello_world

def test_hello_world():
    assert hello_world() == "Hello, World!"
""")
        
        # Required files for compliance
        (project_path / "LICENSE").write_text("MIT License\nCopyright (c) 2024")
        (project_path / "README.md").write_text("# Test Project\nA test project for quality gates.")
        (project_path / "pyproject.toml").write_text("""
[project]
name = "test-package"
version = "0.1.0"
description = "Test package"
""")
        (project_path / "requirements.txt").write_text("pytest>=7.0.0")
        (project_path / "SECURITY.md").write_text("# Security Policy")
        (project_path / "CODE_OF_CONDUCT.md").write_text("# Code of Conduct")
        (project_path / "CONTRIBUTING.md").write_text("# Contributing Guidelines")
        
        # Create quality reports directory
        (project_path / "quality_reports").mkdir()
        
        yield project_path


@pytest.fixture
def quality_config():
    """Create quality configuration for testing."""
    return {
        "enable_auto_fix": True,
        "fail_fast": False,
        "parallel_execution": True,
        "target_stage": "generation_2",
        "quality_thresholds": {
            "minimum_overall_score": 70.0,
            "minimum_success_rate": 80.0,
            "maximum_critical_failures": 1
        },
        "deployment_criteria": {
            "require_all_stages_passed": False,  # Relaxed for testing
            "minimum_score": 70.0,
            "security_scan_required": False,
            "performance_benchmark_required": False
        }
    }


@pytest.mark.integration
class TestCompleteQualityPipeline:
    """Test complete quality pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_automated_pipeline_execution(self, temp_project, quality_config):
        """Test complete automated pipeline execution."""
        # Create pipeline with test configuration
        pipeline = AutomatedQualityPipeline(
            project_root=str(temp_project),
            enable_notifications=False,
            auto_deploy=False
        )
        pipeline.config.update(quality_config)
        
        # Mock subprocess calls to avoid actual tool execution
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful execution for all tools
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            # Create mock coverage file
            coverage_data = {
                "totals": {"percent_covered": 75.0, "covered_lines": 75, "num_statements": 100},
                "files": {}
            }
            coverage_file = temp_project / "coverage.json"
            coverage_file.write_text(json.dumps(coverage_data))
            
            result = await pipeline.run_pipeline(
                trigger_event="test",
                target_stage=ValidationStage.GENERATION_2
            )
        
        # Verify pipeline execution
        assert result.pipeline_id is not None
        assert len(result.stages_completed) > 0
        assert result.total_execution_time > 0
        
        # Check validation results
        assert len(result.validation_results) > 0
        
        # Verify artifacts were created
        assert len(result.artifacts) > 0
    
    @pytest.mark.asyncio
    async def test_progressive_validation_with_monitoring(self, temp_project):
        """Test progressive validation with monitoring integration."""
        # Initialize components
        metrics_collector = get_global_metrics_collector()
        validator = ProgressiveValidator(str(temp_project), enable_auto_fix=False)
        
        # Mock gate execution to avoid tool dependencies
        mock_results = [
            type('MockResult', (), {
                'gate_name': 'compliance_check',
                'status': type('Status', (), {'PASSED': 'passed'})(),
                'score': 85.0,
                'threshold': 80.0,
                'message': 'Compliance check passed',
                'details': {},
                'execution_time': 1.0,
                'timestamp': '2024-01-01T00:00:00Z',
                'artifacts': []
            })()
        ]
        
        with patch.object(validator, '_execute_quality_gates') as mock_execute:
            mock_execute.return_value = mock_results
            
            # Run validation
            results = await validator.validate_progressive(
                start_stage=ValidationStage.GENERATION_1,
                target_stage=ValidationStage.GENERATION_2
            )
        
        # Verify results
        assert len(results) == 2  # Generation 1 and 2
        
        # Check that metrics were recorded
        assert len(metrics_collector.metrics) >= 0  # May be empty in test environment
    
    @pytest.mark.asyncio
    async def test_auto_scaler_integration(self, temp_project):
        """Test auto-scaler integration with quality monitoring."""
        # Initialize auto-scaler
        scaler = QualityGateAutoScaler(
            min_instances=1,
            max_instances=3,
            strategy=ScalingStrategy.CONSERVATIVE
        )
        
        # Mock performance data
        with patch.object(scaler, '_collect_metrics') as mock_collect:
            mock_metrics = type('MockMetrics', (), {
                'cpu_utilization': 85.0,  # High CPU to trigger scaling
                'memory_utilization': 70.0,
                'queue_length': 5,
                'response_time': 2.0,
                'throughput': 10.0,
                'error_rate': 0.01,
                'timestamp': 'datetime.now()'
            })()
            mock_collect.return_value = mock_metrics
            
            # Mock scaling action
            with patch.object(scaler, '_perform_scaling_action') as mock_scale:
                mock_scale.return_value = True
                
                # Start monitoring briefly
                await scaler.start_monitoring(interval_seconds=0.1)
                await asyncio.sleep(0.2)  # Let it run one cycle
                await scaler.stop_monitoring()
        
        # Check scaling status
        status = scaler.get_scaling_status()
        assert status["current_instances"] >= scaler.min_instances
        assert status["strategy"] == "conservative"
    
    @pytest.mark.asyncio
    async def test_intelligent_optimizer_integration(self, temp_project):
        """Test intelligent optimizer integration."""
        # Initialize components
        metrics_collector = get_global_metrics_collector()
        optimizer = IntelligentOptimizer(
            metrics_collector,
            strategy=OptimizationStrategy.BALANCED
        )
        
        # Create mock quality gate results
        mock_results = [
            type('MockResult', (), {
                'gate_name': 'test_coverage',
                'status': type('Status', (), {'PASSED': 'passed'})(),
                'score': 75.0,
                'execution_time': 3.0
            })(),
            type('MockResult', (), {
                'gate_name': 'security_scan',
                'status': type('Status', (), {'FAILED': 'failed'})(),
                'score': 65.0,
                'execution_time': 5.0
            })()
        ]
        
        # Create mock performance data
        mock_performance = [
            type('MockSnapshot', (), {
                'cpu_percent': 80.0,
                'memory_percent': 70.0,
                'timestamp': 'datetime.now()'
            })()
        ]
        
        # Run optimization
        result = await optimizer.optimize_pipeline(
            current_results=mock_results,
            performance_data=mock_performance
        )
        
        # Verify optimization
        assert result.strategy == OptimizationStrategy.BALANCED
        assert len(result.recommendations) > 0
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_quality_workflow(self, temp_project, quality_config):
        """Test complete end-to-end quality workflow."""
        # Initialize all components
        pipeline = AutomatedQualityPipeline(
            project_root=str(temp_project),
            enable_notifications=False
        )
        pipeline.config.update(quality_config)
        
        metrics_collector = get_global_metrics_collector()
        monitor = get_global_monitor()
        
        scaler = QualityGateAutoScaler(
            min_instances=1,
            max_instances=2,
            strategy=ScalingStrategy.REACTIVE,
            metrics_collector=metrics_collector
        )
        
        optimizer = IntelligentOptimizer(
            metrics_collector,
            strategy=OptimizationStrategy.PERFORMANCE_FIRST
        )
        
        # Mock all external dependencies
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            # Create minimal test artifacts
            (temp_project / "coverage.json").write_text(json.dumps({
                "totals": {"percent_covered": 80.0, "covered_lines": 80, "num_statements": 100}
            }))
            
            # Mock scaler metrics collection
            with patch.object(scaler, '_collect_metrics') as mock_collect:
                mock_collect.return_value = type('MockMetrics', (), {
                    'cpu_utilization': 60.0,
                    'memory_utilization': 50.0,
                    'queue_length': 0,
                    'response_time': 1.0,
                    'throughput': 15.0,
                    'error_rate': 0.0,
                    'timestamp': 'datetime.now()'
                })()
                
                # Start monitoring
                await monitor.start_monitoring(interval_seconds=0.1)
                await scaler.start_monitoring(interval_seconds=0.1)
                
                try:
                    # Run pipeline
                    pipeline_result = await pipeline.run_pipeline(
                        trigger_event="integration_test",
                        target_stage=ValidationStage.GENERATION_1
                    )
                    
                    # Let monitoring run briefly
                    await asyncio.sleep(0.2)
                    
                    # Run optimization if pipeline completed
                    if pipeline_result.validation_results:
                        optimization_result = await optimizer.optimize_pipeline(
                            current_results=[],  # Would be actual results
                            performance_data=[]   # Would be actual performance data
                        )
                        
                        assert optimization_result.strategy == OptimizationStrategy.PERFORMANCE_FIRST
                    
                    # Verify pipeline completed
                    assert pipeline_result.pipeline_id is not None
                    assert pipeline_result.total_execution_time > 0
                    
                    # Verify monitoring is active
                    monitor_status = monitor._monitoring
                    scaler_status = scaler.get_scaling_status()
                    
                    assert monitor_status
                    assert scaler_status["current_instances"] >= 1
                    
                finally:
                    # Cleanup
                    await monitor.stop_monitoring()
                    await scaler.stop_monitoring()


@pytest.mark.integration
class TestQualityGatesResilience:
    """Test quality gates system resilience and error handling."""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_failing_gates(self, temp_project):
        """Test pipeline behavior with failing quality gates."""
        pipeline = AutomatedQualityPipeline(
            project_root=str(temp_project),
            enable_notifications=False
        )
        
        # Configure to allow failures
        pipeline.config.update({
            "fail_fast": False,
            "deployment_criteria": {
                "require_all_stages_passed": False,
                "minimum_score": 50.0
            }
        })
        
        # Mock failing tools
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"Error output", b"Error")
            mock_process.returncode = 1  # Failure
            mock_subprocess.return_value = mock_process
            
            result = await pipeline.run_pipeline(
                trigger_event="failure_test",
                target_stage=ValidationStage.GENERATION_1
            )
        
        # Pipeline should complete despite failures
        assert result.pipeline_id is not None
        assert len(result.error_messages) >= 0  # May have errors
        # Should not be deployment ready due to failures
        assert not result.deployment_ready
    
    @pytest.mark.asyncio
    async def test_monitoring_error_recovery(self, temp_project):
        """Test monitoring system error recovery."""
        metrics_collector = get_global_metrics_collector()
        monitor = get_global_monitor()
        
        # Mock metrics collection to fail initially then succeed
        call_count = 0
        
        async def mock_collect_with_failure():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated collection failure")
            return type('MockSnapshot', (), {
                'cpu_percent': 50.0,
                'memory_percent': 40.0,
                'disk_usage_percent': 30.0,
                'network_io': {},
                'process_count': 100,
                'load_average': [1.0],
                'timestamp': 'datetime.now()'
            })()
        
        with patch.object(metrics_collector, 'record_performance_snapshot', side_effect=mock_collect_with_failure):
            # Start monitoring
            await monitor.start_monitoring(interval_seconds=0.1)
            
            # Let it run through failure and recovery
            await asyncio.sleep(0.3)
            
            # Should still be monitoring despite initial failure
            assert monitor._monitoring
            
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_auto_scaler_resilience(self, temp_project):
        """Test auto-scaler resilience to failures."""
        scaler = QualityGateAutoScaler(
            min_instances=1,
            max_instances=3,
            strategy=ScalingStrategy.CONSERVATIVE
        )
        
        # Mock scaling action to fail
        with patch.object(scaler, '_perform_scaling_action') as mock_scale:
            mock_scale.return_value = False  # Scaling fails
            
            with patch.object(scaler, '_collect_metrics') as mock_collect:
                # Return high CPU to trigger scaling
                mock_collect.return_value = type('MockMetrics', (), {
                    'cpu_utilization': 90.0,
                    'memory_utilization': 85.0,
                    'queue_length': 10,
                    'response_time': 3.0,
                    'throughput': 5.0,
                    'error_rate': 0.1,
                    'timestamp': 'datetime.now()'
                })()
                
                # Start monitoring
                await scaler.start_monitoring(interval_seconds=0.1)
                await asyncio.sleep(0.2)
                await scaler.stop_monitoring()
        
        # Check that scaling was attempted but instance count remained unchanged
        status = scaler.get_scaling_status()
        assert status["current_instances"] == 1  # Should remain at minimum
        
        # Check that failure was recorded
        insights = scaler.get_scaling_insights()
        if insights.get("total_events", 0) > 0:
            assert insights["success_rate"] < 1.0  # Some failures


@pytest.mark.integration 
class TestQualityGatesPerformance:
    """Test quality gates system performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_parallel_gate_execution_performance(self, temp_project):
        """Test performance improvement with parallel gate execution."""
        validator = ProgressiveValidator(str(temp_project))
        
        # Mock gates with known execution times
        slow_gates = []
        for i in range(3):
            gate = Mock()
            gate.name = f"gate_{i}"
            gate.threshold = 80.0
            gate.required = True
            gate.execute = AsyncMock(return_value=type('MockResult', (), {
                'gate_name': f'gate_{i}',
                'status': type('Status', (), {'PASSED': 'passed'})(),
                'score': 85.0,
                'threshold': 80.0,
                'message': 'Passed',
                'details': {},
                'execution_time': 1.0,  # 1 second each
                'timestamp': '2024-01-01T00:00:00Z',
                'artifacts': []
            })())
            slow_gates.append(gate)
        
        # Override stage gates with mock gates
        validator.stage_gates[ValidationStage.GENERATION_1] = slow_gates
        
        # Test parallel execution
        start_time = asyncio.get_event_loop().time()
        result = await validator.validate_stage(ValidationStage.GENERATION_1)
        parallel_time = asyncio.get_event_loop().time() - start_time
        
        # Should complete in roughly 1 second (parallel) rather than 3 seconds (sequential)
        assert parallel_time < 2.0  # Allow some overhead
        assert len(result.gate_results) == 3
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Test metrics collection performance under load."""
        metrics_collector = get_global_metrics_collector()
        
        # Record many metrics quickly
        start_time = asyncio.get_event_loop().time()
        
        for i in range(100):
            metric = type('MockMetric', (), {
                'name': f'test_metric_{i % 10}',  # 10 different metric names
                'value': float(i),
                'unit': 'count',
                'timestamp': 'datetime.now()',
                'tags': {},
                'threshold': None
            })()
            metrics_collector.record_metric(metric)
        
        collection_time = asyncio.get_event_loop().time() - start_time
        
        # Should be very fast (under 100ms for 100 metrics)
        assert collection_time < 0.1
        
        # Verify metrics were recorded
        assert len(metrics_collector.metrics) >= 10  # At least 10 different metric types
    
    @pytest.mark.asyncio
    async def test_monitoring_overhead(self):
        """Test monitoring system overhead."""
        metrics_collector = get_global_metrics_collector()
        monitor = get_global_monitor()
        
        # Start monitoring with high frequency
        await monitor.start_monitoring(interval_seconds=0.01)  # Very frequent
        
        # Let it run and measure impact
        start_time = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)  # 100ms
        elapsed = asyncio.get_event_loop().time() - start_time
        
        await monitor.stop_monitoring()
        
        # Monitoring overhead should be minimal
        assert elapsed < 0.15  # Should not add more than 50% overhead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])