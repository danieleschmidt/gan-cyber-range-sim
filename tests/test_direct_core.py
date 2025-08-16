"""Direct tests for core modules without package imports."""

import pytest
import asyncio
import time
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Add specific core module paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import core modules directly
import gan_cyber_range.core.error_handling as error_handling
import gan_cyber_range.core.health_monitor as health_monitor
import gan_cyber_range.core.validation as validation
import gan_cyber_range.scaling.auto_scaler as auto_scaler
import gan_cyber_range.scaling.performance_optimizer as performance_optimizer
import gan_cyber_range.scaling.concurrent_executor as concurrent_executor


class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_cyber_range_error(self):
        """Test CyberRangeError functionality."""
        error = error_handling.CyberRangeError(
            message="Test error",
            error_code="TEST_001",
            context={"component": "test"}
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_001"
        assert error.context["component"] == "test"
        
        # Test serialization
        error_dict = error.to_dict()
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["message"] == "Test error"
    
    def test_error_handler(self):
        """Test ErrorHandler functionality."""
        handler = error_handling.ErrorHandler("test_logger")
        
        error = error_handling.CyberRangeError("Test", "TEST_001")
        handler.handle_error(error)
        
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 1
        assert stats["error_counts"]["TEST_001"] == 1
    
    def test_error_recovery_decorator(self):
        """Test error recovery decorator."""
        call_count = 0
        
        @error_handling.error_recovery(max_retries=2, backoff_seconds=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3


class TestHealthMonitoring:
    """Test health monitoring functionality."""
    
    def test_component_health(self):
        """Test ComponentHealth data structure."""
        health = health_monitor.ComponentHealth(
            component_name="test-service",
            status=health_monitor.HealthStatus.HEALTHY,
            message="All good",
            last_check=datetime.utcnow(),
            metrics={"cpu": 45.2},
            dependencies=["db"]
        )
        
        assert health.component_name == "test-service"
        assert health.status == health_monitor.HealthStatus.HEALTHY
        assert health.metrics["cpu"] == 45.2
        
        health_dict = health.to_dict()
        assert health_dict["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_system_resource_check(self):
        """Test system resource check with mocked psutil."""
        check = health_monitor.SystemResourceCheck()
        
        with patch.object(health_monitor, 'psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 45.0
            mock_psutil.virtual_memory.return_value = Mock(
                percent=60.0, 
                available=4*1024**3
            )
            mock_psutil.disk_usage.return_value = Mock(
                percent=70.0, 
                free=100*1024**3
            )
            
            health = await check.check_health()
            
            assert health.status == health_monitor.HealthStatus.HEALTHY
            assert health.metrics["cpu_percent"] == 45.0


class TestValidation:
    """Test validation functionality."""
    
    def test_basic_rules(self):
        """Test basic validation rules."""
        # Required rule
        required = validation.RequiredRule()
        assert required.validate("test") == True
        assert required.validate("") == False
        assert required.validate(None) == False
        
        # Type rule
        string_type = validation.TypeRule(str)
        assert string_type.validate("test") == True
        assert string_type.validate(123) == False
        
        # Email rule
        email = validation.EmailRule()
        assert email.validate("test@example.com") == True
        assert email.validate("invalid") == False
    
    def test_input_validator(self):
        """Test InputValidator."""
        validator = validation.InputValidator()
        validator.add_rule("name", validation.RequiredRule())
        validator.add_rule("email", validation.EmailRule())
        
        # Valid data
        result = validator.validate({
            "name": "Test User",
            "email": "test@example.com"
        })
        assert result["name"] == "Test User"
        
        # Invalid data
        with pytest.raises(validation.ValidationError):
            validator.validate({
                "name": "",  # Required but empty
                "email": "invalid-email"
            })
    
    def test_security_validator(self):
        """Test security validation."""
        validator = validation.SecurityValidator()
        
        # Safe input should pass
        safe_data = {"username": "testuser", "content": "safe content"}
        result = validator.validate(safe_data, strict=False)
        assert result["username"] == "testuser"
        
        # Dangerous input should fail
        with pytest.raises(validation.ValidationError):
            validator.validate({
                "input": "<script>alert('xss')</script>"
            }, strict=False)


class TestAutoScaling:
    """Test auto-scaling functionality."""
    
    def test_scaling_policy(self):
        """Test ScalingPolicy."""
        policy = auto_scaler.ScalingPolicy(
            min_replicas=2,
            max_replicas=10,
            scale_up_threshold=80.0
        )
        
        assert policy.min_replicas == 2
        assert policy.max_replicas == 10
        
        policy_dict = policy.to_dict()
        assert policy_dict["min_replicas"] == 2
    
    def test_scaling_metrics(self):
        """Test ScalingMetrics."""
        metrics = auto_scaler.ScalingMetrics(
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
        
        metrics_dict = metrics.to_dict()
        assert metrics_dict["cpu_usage_percent"] == 75.5


class TestPerformanceOptimizer:
    """Test performance optimization."""
    
    def test_cache_manager(self):
        """Test CacheManager basic functionality."""
        cache = performance_optimizer.CacheManager(max_size=100)
        
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
        """Test cache expiration."""
        cache = performance_optimizer.CacheManager()
        cache.set("key", "value", ttl_seconds=0.1)
        
        assert cache.get("key") == "value"
        time.sleep(0.2)
        assert cache.get("key") is None
    
    def test_cached_decorator(self):
        """Test @cached decorator."""
        cache = performance_optimizer.CacheManager()
        call_count = 0
        
        @performance_optimizer.cached(cache_manager=cache)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (should use cache)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not called again


class TestConcurrentExecutor:
    """Test concurrent execution."""
    
    @pytest.mark.asyncio
    async def test_task_pool(self):
        """Test TaskPool basic functionality."""
        pool = concurrent_executor.TaskPool(max_workers=2, worker_type="thread")
        
        try:
            await pool.start(num_workers=1)
            
            def simple_task(x):
                return x * 2
            
            task_id = await pool.submit_task(
                "test_task",
                simple_task,
                5,
                priority=concurrent_executor.TaskPriority.NORMAL
            )
            
            # Wait for execution
            await asyncio.sleep(0.5)
            
            result = pool.get_task_status(task_id)
            assert result is not None
            
            stats = pool.get_stats()
            assert stats["tasks_submitted"] >= 1
            
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_async_task_manager(self):
        """Test AsyncTaskManager."""
        manager = concurrent_executor.AsyncTaskManager()
        
        try:
            await manager.start_all_pools()
            
            async def async_task(name):
                await asyncio.sleep(0.1)
                return f"Hello {name}"
            
            task_id = await manager.submit(
                async_task,
                "World",
                priority=concurrent_executor.TaskPriority.HIGH
            )
            
            result = await manager.wait_for_task(task_id, timeout=2.0)
            
            assert result.status == concurrent_executor.TaskStatus.COMPLETED
            assert result.result == "Hello World"
            
        finally:
            await manager.stop_all_pools()


class TestSecurity:
    """Security-specific tests."""
    
    def test_xss_prevention(self):
        """Test XSS prevention."""
        validator = validation.SecurityValidator()
        
        xss_attacks = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for attack in xss_attacks:
            with pytest.raises(validation.ValidationError):
                validator.validate({"input": attack}, strict=False)
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        validator = validation.SecurityValidator()
        
        sql_injections = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords"
        ]
        
        for injection in sql_injections:
            with pytest.raises(validation.ValidationError):
                validator.validate({"query": injection}, strict=False)


class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.benchmark(group="cache")
    def test_cache_performance(self, benchmark):
        """Benchmark cache operations."""
        cache = performance_optimizer.CacheManager(max_size=1000)
        
        def cache_ops():
            for i in range(100):
                cache.set(f"key_{i}", f"value_{i}")
                cache.get(f"key_{i}")
        
        benchmark(cache_ops)
    
    @pytest.mark.benchmark(group="validation")
    def test_validation_performance(self, benchmark):
        """Benchmark validation operations."""
        validator = validation.InputValidator()
        validator.add_rule("test", validation.RequiredRule())
        
        def validation_ops():
            for _ in range(100):
                try:
                    validator.validate({"test": "value"})
                except:
                    pass
        
        benchmark(validation_ops)


class TestSmokeTests:
    """Quick smoke tests."""
    
    def test_error_handling_smoke(self):
        """Smoke test for error handling."""
        error = error_handling.CyberRangeError("Test", "SMOKE")
        assert error.message == "Test"
    
    def test_validation_smoke(self):
        """Smoke test for validation."""
        rule = validation.RequiredRule()
        assert rule.validate("test") == True
    
    def test_cache_smoke(self):
        """Smoke test for caching."""
        cache = performance_optimizer.CacheManager()
        cache.set("test", "value")
        assert cache.get("test") == "value"
    
    @pytest.mark.asyncio
    async def test_health_smoke(self):
        """Smoke test for health monitoring."""
        health = health_monitor.ComponentHealth(
            component_name="test",
            status=health_monitor.HealthStatus.HEALTHY,
            message="OK",
            last_check=datetime.utcnow(),
            metrics={},
            dependencies=[]
        )
        assert health.status == health_monitor.HealthStatus.HEALTHY


if __name__ == "__main__":
    # Run smoke tests first
    print("Running smoke tests...")
    pytest.main([__file__ + "::TestSmokeTests", "-v", "--tb=short"])
    
    print("\nRunning full test suite...")
    pytest.main([__file__, "-v", "--tb=short"])