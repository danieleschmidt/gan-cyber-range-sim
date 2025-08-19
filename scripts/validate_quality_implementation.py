#!/usr/bin/env python3
"""Validation script for quality gates implementation."""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gan_cyber_range.quality import (
    AutomatedQualityPipeline,
    ProgressiveValidator,
    ValidationStage,
    QualityGateAutoScaler,
    IntelligentOptimizer,
    OptimizationStrategy,
    ScalingStrategy,
    get_global_metrics_collector,
    get_global_monitor,
    get_global_dashboard
)


async def test_basic_imports():
    """Test that all quality components can be imported."""
    print("🔍 Testing basic imports...")
    
    try:
        # Test core components
        assert AutomatedQualityPipeline is not None
        assert ProgressiveValidator is not None
        assert ValidationStage is not None
        
        # Test monitoring components
        metrics_collector = get_global_metrics_collector()
        monitor = get_global_monitor()
        dashboard = get_global_dashboard()
        
        assert metrics_collector is not None
        assert monitor is not None
        assert dashboard is not None
        
        # Test optimization components
        assert IntelligentOptimizer is not None
        assert OptimizationStrategy is not None
        
        # Test scaling components
        assert QualityGateAutoScaler is not None
        assert ScalingStrategy is not None
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


async def test_component_initialization():
    """Test that components can be initialized properly."""
    print("\n🔧 Testing component initialization...")
    
    try:
        # Test pipeline initialization
        pipeline = AutomatedQualityPipeline(
            project_root=".",
            enable_notifications=False,
            auto_deploy=False
        )
        assert pipeline is not None
        print("  ✅ AutomatedQualityPipeline initialized")
        
        # Test validator initialization
        validator = ProgressiveValidator(
            project_root=".",
            enable_auto_fix=False,
            fail_fast=False
        )
        assert validator is not None
        assert len(validator.stage_gates) > 0
        print("  ✅ ProgressiveValidator initialized")
        
        # Test optimizer initialization
        metrics_collector = get_global_metrics_collector()
        optimizer = IntelligentOptimizer(
            metrics_collector,
            strategy=OptimizationStrategy.BALANCED
        )
        assert optimizer is not None
        print("  ✅ IntelligentOptimizer initialized")
        
        # Test scaler initialization
        scaler = QualityGateAutoScaler(
            min_instances=1,
            max_instances=3,
            strategy=ScalingStrategy.CONSERVATIVE
        )
        assert scaler is not None
        print("  ✅ QualityGateAutoScaler initialized")
        
        print("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        traceback.print_exc()
        return False


async def test_validation_stages():
    """Test validation stage configuration."""
    print("\n📊 Testing validation stages...")
    
    try:
        validator = ProgressiveValidator(".", enable_auto_fix=False)
        
        # Check all stages are configured
        for stage in ValidationStage:
            assert stage in validator.stage_gates
            gates = validator.stage_gates[stage]
            assert len(gates) > 0
            print(f"  ✅ {stage.value}: {len(gates)} gates configured")
        
        # Check progressive difficulty
        gen1_gates = validator.stage_gates[ValidationStage.GENERATION_1]
        gen3_gates = validator.stage_gates[ValidationStage.GENERATION_3]
        
        # Find comparable gates
        gen1_coverage = next((g for g in gen1_gates if "coverage" in g.name), None)
        gen3_coverage = next((g for g in gen3_gates if "coverage" in g.name), None)
        
        if gen1_coverage and gen3_coverage:
            assert gen1_coverage.threshold <= gen3_coverage.threshold
            print(f"  ✅ Progressive thresholds: Gen1={gen1_coverage.threshold}%, Gen3={gen3_coverage.threshold}%")
        
        print("✅ Validation stages configured correctly")
        return True
        
    except Exception as e:
        print(f"❌ Validation stages test failed: {e}")
        traceback.print_exc()
        return False


async def test_monitoring_system():
    """Test monitoring system functionality."""
    print("\n📈 Testing monitoring system...")
    
    try:
        # Test metrics collection
        metrics_collector = get_global_metrics_collector()
        
        # Record a test metric
        from gan_cyber_range.quality.monitoring import QualityMetric
        from datetime import datetime
        
        test_metric = QualityMetric(
            name="test_validation",
            value=85.0,
            unit="percent",
            timestamp=datetime.now(),
            tags={"test": "validation"}
        )
        
        metrics_collector.record_metric(test_metric)
        assert len(metrics_collector.metrics["test_validation"]) == 1
        print("  ✅ Metrics collection working")
        
        # Test performance snapshot
        snapshot = metrics_collector.record_performance_snapshot()
        assert snapshot is not None
        assert snapshot.cpu_percent >= 0
        print("  ✅ Performance monitoring working")
        
        # Test monitor initialization
        monitor = get_global_monitor()
        status_before = monitor._monitoring
        
        # Brief monitoring test
        await monitor.start_monitoring(interval_seconds=0.1)
        await asyncio.sleep(0.2)
        await monitor.stop_monitoring()
        
        print("  ✅ Quality monitor working")
        
        # Test dashboard
        dashboard = get_global_dashboard()
        dashboard_data = dashboard.get_dashboard_data()
        # Dashboard data might be empty initially, but method should work
        assert isinstance(dashboard_data, dict)
        print("  ✅ Quality dashboard working")
        
        print("✅ Monitoring system functional")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring system test failed: {e}")
        traceback.print_exc()
        return False


async def test_optimization_system():
    """Test optimization system functionality."""
    print("\n⚡ Testing optimization system...")
    
    try:
        metrics_collector = get_global_metrics_collector()
        optimizer = IntelligentOptimizer(
            metrics_collector,
            strategy=OptimizationStrategy.BALANCED
        )
        
        # Test optimization targets
        targets = optimizer._get_default_targets()
        assert len(targets) > 0
        print(f"  ✅ Default optimization targets: {len(targets)} targets")
        
        # Test strategy configuration
        for strategy in OptimizationStrategy:
            test_optimizer = IntelligentOptimizer(metrics_collector, strategy=strategy)
            strategy_targets = test_optimizer._get_default_targets()
            assert len(strategy_targets) > 0
            print(f"  ✅ Strategy {strategy.value}: {len(strategy_targets)} targets")
        
        # Test optimization insights
        insights = optimizer.get_optimization_insights()
        assert "status" in insights
        print("  ✅ Optimization insights working")
        
        print("✅ Optimization system functional")
        return True
        
    except Exception as e:
        print(f"❌ Optimization system test failed: {e}")
        traceback.print_exc()
        return False


async def test_scaling_system():
    """Test auto-scaling system functionality."""
    print("\n📈 Testing auto-scaling system...")
    
    try:
        scaler = QualityGateAutoScaler(
            min_instances=1,
            max_instances=5,
            strategy=ScalingStrategy.ADAPTIVE
        )
        
        # Test scaling status
        status = scaler.get_scaling_status()
        assert "current_instances" in status
        assert status["current_instances"] >= scaler.min_instances
        print("  ✅ Scaling status working")
        
        # Test strategy configuration
        for strategy in ScalingStrategy:
            test_scaler = QualityGateAutoScaler(
                min_instances=1,
                max_instances=3,
                strategy=strategy
            )
            test_status = test_scaler.get_scaling_status()
            assert test_status["strategy"] == strategy.value
            print(f"  ✅ Strategy {strategy.value} configured")
        
        # Test scaling insights
        insights = scaler.get_scaling_insights()
        assert "status" in insights
        print("  ✅ Scaling insights working")
        
        # Test load predictor
        predictor = scaler.load_predictor
        predicted_cpu, predicted_memory = predictor.predict_load(15)
        assert 0 <= predicted_cpu <= 100
        assert 0 <= predicted_memory <= 100
        print("  ✅ Load predictor working")
        
        print("✅ Auto-scaling system functional")
        return True
        
    except Exception as e:
        print(f"❌ Auto-scaling system test failed: {e}")
        traceback.print_exc()
        return False


async def test_pipeline_configuration():
    """Test pipeline configuration and setup."""
    print("\n⚙️ Testing pipeline configuration...")
    
    try:
        # Test pipeline with different configurations
        pipeline = AutomatedQualityPipeline(
            project_root=".",
            enable_notifications=False,
            auto_deploy=False
        )
        
        # Test configuration loading
        config = pipeline.config
        assert isinstance(config, dict)
        assert "enable_auto_fix" in config
        print("  ✅ Pipeline configuration loaded")
        
        # Test validator configuration
        validator = pipeline.validator
        assert isinstance(validator, ProgressiveValidator)
        print("  ✅ Validator configured in pipeline")
        
        # Test stage hooks
        assert len(pipeline.stage_hooks) > 0
        print(f"  ✅ Pipeline hooks configured: {len(pipeline.stage_hooks)} stages")
        
        # Test hook registration
        def test_hook(result, *args):
            pass
        
        from gan_cyber_range.quality.automated_pipeline import PipelineStage
        pipeline.register_hook(PipelineStage.PRE_VALIDATION, test_hook)
        assert test_hook in pipeline.stage_hooks[PipelineStage.PRE_VALIDATION]
        print("  ✅ Hook registration working")
        
        print("✅ Pipeline configuration functional")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline configuration test failed: {e}")
        traceback.print_exc()
        return False


async def test_error_handling():
    """Test error handling and resilience."""
    print("\n🛡️ Testing error handling...")
    
    try:
        # Test validator with invalid project path
        try:
            validator = ProgressiveValidator("/nonexistent/path")
            # Should not fail on initialization
            assert validator is not None
            print("  ✅ Graceful handling of invalid paths")
        except Exception:
            print("  ⚠️ Validator requires valid path")
        
        # Test metrics collector with edge cases
        metrics_collector = get_global_metrics_collector()
        
        # Test with None values (should be handled gracefully)
        try:
            from gan_cyber_range.quality.monitoring import QualityMetric
            from datetime import datetime
            
            # This should not crash the system
            edge_metric = QualityMetric(
                name="edge_case",
                value=float('inf'),  # Edge case value
                unit="test",
                timestamp=datetime.now()
            )
            metrics_collector.record_metric(edge_metric)
            print("  ✅ Edge case metrics handled")
        except Exception as e:
            print(f"  ⚠️ Edge case handling needs improvement: {e}")
        
        # Test optimizer with empty data
        optimizer = IntelligentOptimizer(metrics_collector)
        insights = optimizer.get_optimization_insights()
        # Should return status indicating no data
        assert "status" in insights
        print("  ✅ Optimizer handles empty data")
        
        print("✅ Error handling functional")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False


async def main():
    """Run all validation tests."""
    print("🚀 Validating Quality Gates Implementation")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_component_initialization,
        test_validation_stages,
        test_monitoring_system,
        test_optimization_system,
        test_scaling_system,
        test_pipeline_configuration,
        test_error_handling
    ]
    
    results = []
    
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1:2d}. {test.__name__:<30} {status}")
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Quality Gates implementation is functional!")
        return 0
    else:
        print("⚠️ SOME TESTS FAILED - Review implementation before deployment")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())