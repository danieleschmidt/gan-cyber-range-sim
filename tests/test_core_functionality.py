"""Comprehensive tests for core cyber range functionality."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Test imports
from gan_cyber_range.core.error_handling import (
    CyberRangeError, AgentError, SecurityError, ErrorHandler, error_recovery
)
from gan_cyber_range.core.health_monitor import (
    HealthStatus, ComponentHealth, SystemResourceCheck, SystemHealthMonitor
)
from gan_cyber_range.core.validation import (
    InputValidator, SecurityValidator, RequiredRule, TypeRule, EmailRule
)
from gan_cyber_range.scaling.auto_scaler import (
    AutoScaler, ScalingPolicy, ScalingMetrics, ScalingDirection
)
from gan_cyber_range.scaling.performance_optimizer import (
    CacheManager, cached, PerformanceOptimizer
)
from gan_cyber_range.scaling.concurrent_executor import (
    TaskPool, TaskPriority, TaskStatus, AsyncTaskManager
)


class TestErrorHandling:
    """Test error handling and recovery systems."""
    
    def test_cyber_range_error_creation(self):
        """Test CyberRangeError creation and serialization."""
        error = CyberRangeError(
            message="Test error",
            error_code="TEST_001",
            context={"component": "test"}
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_001"
        assert error.context["component"] == "test"
        assert isinstance(error.timestamp, datetime)
        
        # Test serialization
        error_dict = error.to_dict()
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["message"] == "Test error"
        assert "timestamp" in error_dict
    
    def test_agent_error_creation(self):
        """Test AgentError specific functionality."""
        error = AgentError(
            message="Agent failed",
            agent_id="agent-123",
            error_code="TIMEOUT"
        )
        
        assert error.agent_id == "agent-123"
        assert error.error_code == "AGENT_TIMEOUT"
    
    def test_security_error_creation(self):
        """Test SecurityError specific functionality."""
        error = SecurityError(
            message="Security violation",
            security_context={"ip": "192.168.1.1"},
            error_code="UNAUTHORIZED"
        )
        
        assert error.security_context["ip"] == "192.168.1.1"
        assert error.error_code == "SEC_UNAUTHORIZED"
    
    def test_error_handler_basic(self):
        """Test basic error handler functionality."""
        handler = ErrorHandler("test_logger")
        
        # Test error tracking
        error = CyberRangeError("Test error", "TEST_001")
        handler.handle_error(error)
        
        stats = handler.get_error_statistics()
        assert stats["error_counts"]["TEST_001"] == 1
        assert stats["total_errors"] == 1
    
    def test_error_recovery_decorator(self):
        """Test error recovery decorator."""
        call_count = 0
        
        @error_recovery(max_retries=2, backoff_seconds=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third try
    
    def test_error_recovery_with_specific_errors(self):
        """Test error recovery with specific error codes."""
        call_count = 0
        
        @error_recovery(error_codes=["TEMP_FAILURE"], max_retries=1, backoff_seconds=0.01)
        def selective_retry():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = CyberRangeError("Temp failure", "TEMP_FAILURE")
                raise error
            return "success"
        
        result = selective_retry()
        assert result == "success"
        assert call_count == 2


class TestHealthMonitoring:
    """Test health monitoring system."""
    
    def test_component_health_creation(self):
        """Test ComponentHealth data structure."""
        health = ComponentHealth(
            component_name="test-service",
            status=HealthStatus.HEALTHY,
            message="All systems operational",
            last_check=datetime.utcnow(),
            metrics={"cpu": 45.2},
            dependencies=["database"]
        )
        
        assert health.component_name == "test-service"
        assert health.status == HealthStatus.HEALTHY
        assert health.metrics["cpu"] == 45.2
        assert "database" in health.dependencies
        
        # Test serialization
        health_dict = health.to_dict()
        assert health_dict["status"] == "healthy"
        assert health_dict["metrics"]["cpu"] == 45.2
    
    @pytest.mark.asyncio
    async def test_system_resource_check(self):
        """Test system resource health check."""
        check = SystemResourceCheck(
            cpu_threshold=80.0,
            memory_threshold=85.0,
            disk_threshold=90.0
        )
        
        # Mock psutil calls
        with patch('gan_cyber_range.core.health_monitor.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 45.0
            mock_psutil.virtual_memory.return_value = Mock(percent=60.0, available=4*1024**3)
            mock_psutil.disk_usage.return_value = Mock(percent=70.0, free=100*1024**3)
            
            health = await check.check_health()
            
            assert health.status == HealthStatus.HEALTHY
            assert health.metrics["cpu_percent"] == 45.0
            assert health.metrics["memory_percent"] == 60.0
            assert health.metrics["disk_percent"] == 70.0
    
    @pytest.mark.asyncio
    async def test_health_monitor_integration(self):
        """Test complete health monitoring system."""
        monitor = SystemHealthMonitor(check_interval=30)
        
        # Add mock health check
        mock_check = Mock()
        mock_check.name = "mock-service"
        mock_check.run_check = AsyncMock()
        
        mock_health = ComponentHealth(
            component_name="mock-service",
            status=HealthStatus.HEALTHY,
            message="Mock service healthy",
            last_check=datetime.utcnow(),
            metrics={},
            dependencies=[]
        )
        mock_check.run_check.return_value = mock_health
        
        monitor.register_health_check(mock_check)
        
        # Run checks
        results = await monitor.run_all_checks()
        
        assert "mock-service" in results
        assert results["mock-service"].status == HealthStatus.HEALTHY
        
        # Test system health summary
        system_health = monitor.get_system_health()
        assert system_health["status"] == "healthy"
        assert "mock-service" in system_health["components"]


class TestValidation:
    """Test input validation framework."""
    
    def test_basic_validation_rules(self):
        """Test basic validation rules."""
        # Required rule
        required = RequiredRule()
        assert required.validate("test") == True
        assert required.validate("") == False
        assert required.validate(None) == False
        
        # Type rule
        string_type = TypeRule(str)
        assert string_type.validate("test") == True
        assert string_type.validate(123) == False
        
        # Email rule
        email = EmailRule()
        assert email.validate("test@example.com") == True
        assert email.validate("invalid-email") == False
    
    def test_input_validator(self):
        """Test InputValidator framework."""
        validator = InputValidator()
        validator.add_rule("name", RequiredRule())
        validator.add_rule("name", TypeRule(str))
        validator.add_rule("email", EmailRule())
        
        # Valid data
        valid_data = {
            "name": "John Doe",
            "email": "john@example.com"
        }
        result = validator.validate(valid_data)
        assert result["name"] == "John Doe"
        assert result["email"] == "john@example.com"
        
        # Invalid data
        invalid_data = {
            "name": "",  # Required field empty
            "email": "invalid-email"
        }
        with pytest.raises(Exception):  # ValidationError
            validator.validate(invalid_data)
    
    def test_security_validator(self):
        """Test security-specific validation."""
        validator = SecurityValidator()
        
        # Test with safe input
        safe_data = {
            "username": "testuser",
            "filename": "document.pdf"
        }
        result = validator.validate(safe_data, strict=False)
        assert result["username"] == "testuser"
        
        # Test with dangerous input
        dangerous_data = {
            "input": "<script>alert('xss')</script>",
            "filename": "../etc/passwd"
        }
        with pytest.raises(Exception):  # ValidationError
            validator.validate(dangerous_data, strict=False)


class TestAutoScaling:
    """Test auto-scaling functionality."""
    
    def test_scaling_policy(self):
        """Test ScalingPolicy configuration."""
        policy = ScalingPolicy(
            min_replicas=2,
            max_replicas=10,
            target_cpu_percent=70.0,
            scale_up_threshold=80.0
        )
        
        assert policy.min_replicas == 2
        assert policy.max_replicas == 10
        assert policy.scale_up_threshold == 80.0
        
        # Test serialization
        policy_dict = policy.to_dict()
        assert policy_dict["min_replicas"] == 2
        assert policy_dict["target_cpu_percent"] == 70.0
    
    def test_scaling_metrics(self):
        """Test ScalingMetrics data structure."""
        metrics = ScalingMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=75.5,
            memory_usage_percent=65.0,
            request_rate_per_second=150.0,
            queue_length=25,
            response_time_ms=250.0,
            active_connections=100,
            custom_metrics={"error_rate": 0.02}
        )
        
        assert metrics.cpu_usage_percent == 75.5
        assert metrics.custom_metrics["error_rate"] == 0.02
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert metrics_dict["cpu_usage_percent"] == 75.5
        assert "timestamp" in metrics_dict
    
    @pytest.mark.asyncio
    async def test_auto_scaler_scale_up(self):
        """Test auto-scaler scale-up logic."""
        policy = ScalingPolicy(
            min_replicas=2,
            max_replicas=10,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
            scale_up_cooldown_seconds=1  # Short for testing
        )
        
        # Mock metrics collector that returns high CPU
        def high_cpu_metrics():
            return ScalingMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage_percent=85.0,  # Above threshold
                memory_usage_percent=60.0,
                request_rate_per_second=100.0,
                queue_length=10,
                response_time_ms=200.0,
                active_connections=50,
                custom_metrics={}
            )
        
        # Mock scaler executor
        scale_calls = []
        def mock_scaler(target_replicas):
            scale_calls.append(target_replicas)
            return True
        
        scaler = AutoScaler(
            component_name="test-service",
            scaling_policy=policy,
            metrics_collector=high_cpu_metrics,
            scaler_executor=mock_scaler
        )
        
        # Add some metrics history to trigger scaling
        for _ in range(5):
            scaler.metrics_history.append(high_cpu_metrics())
        
        # Test scaling decision
        decision = scaler._analyze_scaling_need()
        assert decision["action"] == ScalingDirection.UP
        assert decision["target_replicas"] > policy.min_replicas


class TestPerformanceOptimization:
    """Test performance optimization components."""
    
    def test_cache_manager_basic(self):
        """Test basic cache manager functionality."""
        cache = CacheManager(max_size=100, default_ttl_seconds=60)
        
        # Test set/get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test miss
        assert cache.get("nonexistent") is None
        
        # Test stats
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hit_rate"] > 0
    
    def test_cache_expiration(self):
        """Test cache expiration functionality."""
        cache = CacheManager(default_ttl_seconds=1)
        
        cache.set("expiring_key", "value", ttl_seconds=0.1)
        assert cache.get("expiring_key") == "value"
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("expiring_key") is None
    
    def test_cached_decorator(self):
        """Test cached decorator functionality."""
        call_count = 0
        
        @cached(ttl_seconds=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args (should use cache)
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again
        
        # Different args (should call function)
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_performance_optimizer(self):
        """Test complete performance optimizer."""
        optimizer = PerformanceOptimizer()
        
        # Test performance report
        report = optimizer.get_performance_report()
        assert "cache" in report
        assert "queries" in report
        assert "resources" in report
        assert "timestamp" in report


class TestConcurrentExecution:
    """Test concurrent execution system."""
    
    @pytest.mark.asyncio
    async def test_task_pool_basic(self):
        """Test basic task pool functionality."""
        pool = TaskPool(max_workers=2, worker_type="thread")
        
        try:
            await pool.start(num_workers=2)
            
            # Submit a simple task
            def simple_task(x):
                return x * 2
            
            task_id = await pool.submit_task(
                "test_task",
                simple_task,
                5,
                priority=TaskPriority.NORMAL
            )
            
            # Wait a bit for execution
            await asyncio.sleep(0.5)
            
            # Check result
            result = pool.get_task_status(task_id)
            assert result is not None
            
            # Check stats
            stats = pool.get_stats()
            assert stats["tasks_submitted"] >= 1
            
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_async_task_manager(self):
        """Test async task manager."""
        manager = AsyncTaskManager()
        
        try:
            await manager.start_all_pools()
            
            # Submit async task
            async def async_task(name):
                await asyncio.sleep(0.1)
                return f"Hello {name}"
            
            task_id = await manager.submit(
                async_task,
                "World",
                priority=TaskPriority.HIGH
            )
            
            # Wait for completion
            result = await manager.wait_for_task(task_id, timeout=5.0)
            
            assert result.status == TaskStatus.COMPLETED
            assert result.result == "Hello World"
            assert result.execution_time_seconds > 0
            
        finally:
            await manager.stop_all_pools()
    
    @pytest.mark.asyncio
    async def test_task_priority_ordering(self):
        """Test that high priority tasks are executed first."""
        pool = TaskPool(max_workers=1, worker_type="thread")  # Single worker
        
        try:
            await pool.start(num_workers=1)
            
            execution_order = []
            
            def tracking_task(name):
                execution_order.append(name)
                return name
            
            # Submit tasks in reverse priority order
            await pool.submit_task("low", tracking_task, "low", priority=TaskPriority.LOW)
            await pool.submit_task("high", tracking_task, "high", priority=TaskPriority.HIGH)
            await pool.submit_task("normal", tracking_task, "normal", priority=TaskPriority.NORMAL)
            
            # Wait for all tasks to complete
            await asyncio.sleep(1.0)
            
            # High priority should execute first
            assert execution_order[0] == "high"
            
        finally:
            await pool.stop()


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_error_handling_with_health_monitoring(self):
        """Test error handling integration with health monitoring."""
        # Setup health monitor with error handling
        monitor = SystemHealthMonitor()
        error_handler = ErrorHandler("integration_test")
        
        # Mock a failing health check
        failing_check = Mock()
        failing_check.name = "failing-service"
        failing_check.run_check = AsyncMock()
        failing_check.run_check.side_effect = Exception("Service unavailable")
        
        monitor.register_health_check(failing_check)
        
        # Run health checks (should handle errors gracefully)
        results = await monitor.run_all_checks()
        
        # Should have result for failing service with critical status
        assert "failing-service" in results
        assert results["failing-service"].status == HealthStatus.CRITICAL
        assert "exception" in results["failing-service"].message.lower()
    
    @pytest.mark.asyncio
    async def test_performance_optimization_with_scaling(self):
        """Test performance optimization integration with auto-scaling."""
        # Create performance optimizer
        optimizer = PerformanceOptimizer()
        
        # Create scaling policy that responds to performance metrics
        policy = ScalingPolicy(
            min_replicas=1,
            max_replicas=5,
            scale_up_threshold=80.0
        )
        
        # Mock metrics that would trigger scaling
        def performance_metrics():
            return ScalingMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage_percent=85.0,  # High CPU
                memory_usage_percent=90.0,  # High memory
                request_rate_per_second=500.0,
                queue_length=50,
                response_time_ms=2000.0,  # Slow response
                active_connections=200,
                custom_metrics={}
            )
        
        scaling_executed = []
        def mock_scaler(replicas):
            scaling_executed.append(replicas)
            return True
        
        scaler = AutoScaler(
            component_name="performance-test",
            scaling_policy=policy,
            metrics_collector=performance_metrics,
            scaler_executor=mock_scaler
        )
        
        # Add metrics history
        for _ in range(5):
            scaler.metrics_history.append(performance_metrics())
        
        # Analyze scaling need
        decision = scaler._analyze_scaling_need()
        
        # Should recommend scale up due to high resource usage
        assert decision["action"] == ScalingDirection.UP
        assert len(decision["reasons"]) > 0
    
    def test_security_validation_with_error_handling(self):
        """Test security validation with comprehensive error handling."""
        validator = SecurityValidator()
        error_handler = ErrorHandler("security_test")
        
        # Test malicious input
        malicious_data = {
            "user_input": "<script>alert('xss')</script>",
            "sql_query": "'; DROP TABLE users; --",
            "file_path": "../../etc/passwd"
        }
        
        try:
            validator.validate(malicious_data, strict=False)
            assert False, "Should have raised validation error"
        except Exception as e:
            # Handle the validation error
            security_error = SecurityError(
                message="Malicious input detected",
                security_context=malicious_data,
                cause=e
            )
            
            result = error_handler.handle_error(security_error)
            
            # Should have logged the security violation
            stats = error_handler.get_error_statistics()
            assert stats["total_errors"] >= 1


# Fixtures for testing
@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return ScalingMetrics(
        timestamp=datetime.utcnow(),
        cpu_usage_percent=45.0,
        memory_usage_percent=60.0,
        request_rate_per_second=100.0,
        queue_length=5,
        response_time_ms=150.0,
        active_connections=25,
        custom_metrics={"error_rate": 0.01}
    )


@pytest.fixture
def mock_kubernetes_client():
    """Mock Kubernetes client for testing."""
    mock_client = Mock()
    mock_client.scale_deployment = Mock(return_value=True)
    mock_client.get_pod_metrics = Mock(return_value={
        "cpu_usage": 45.0,
        "memory_usage": 512.0,
        "pod_count": 3
    })
    return mock_client


# Performance benchmarks
@pytest.mark.benchmark(group="cache")
def test_cache_performance(benchmark):
    """Benchmark cache performance."""
    cache = CacheManager(max_size=1000)
    
    def cache_operations():
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
            cache.get(f"key_{i}")
    
    result = benchmark(cache_operations)
    assert result is None  # Operations complete successfully


@pytest.mark.benchmark(group="validation")
def test_validation_performance(benchmark):
    """Benchmark validation performance."""
    validator = SecurityValidator()
    
    test_data = {
        "username": "testuser123",
        "email": "test@example.com",
        "content": "This is safe content for testing validation performance"
    }
    
    def validation_operations():
        for _ in range(50):
            validator.validate(test_data, strict=False)
    
    result = benchmark(validation_operations)
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])