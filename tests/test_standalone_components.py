"""Standalone tests for individual components without dependencies."""

import pytest
import asyncio
import time
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import standalone modules
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
        cache = CacheManager()
        call_count = 0
        
        @cached(ttl_seconds=60, cache_manager=cache)
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
            await asyncio.sleep(1.0)
            
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


# Security and performance tests
class TestSecurity:
    """Security-specific tests."""
    
    def test_xss_prevention(self):
        """Test XSS attack prevention."""
        validator = SecurityValidator()
        
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(1)'></iframe>"
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(Exception):
                validator.validate({"input": dangerous_input}, strict=False)
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        validator = SecurityValidator()
        
        sql_injections = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "1; UPDATE users SET admin=1 WHERE id=1; --"
        ]
        
        for injection in sql_injections:
            with pytest.raises(Exception):
                validator.validate({"query": injection}, strict=False)
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention."""
        validator = SecurityValidator()
        
        path_traversals = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "....//....//etc/passwd"
        ]
        
        for path in path_traversals:
            with pytest.raises(Exception):
                validator.validate({"filename": path}, strict=False)


class TestPerformance:
    """Performance and load tests."""
    
    @pytest.mark.benchmark(group="cache")
    def test_cache_performance(self, benchmark):
        """Benchmark cache performance."""
        cache = CacheManager(max_size=1000)
        
        def cache_operations():
            for i in range(100):
                cache.set(f"key_{i}", f"value_{i}")
                cache.get(f"key_{i}")
        
        result = benchmark(cache_operations)
        assert result is None  # Operations complete successfully
    
    @pytest.mark.benchmark(group="validation")
    def test_validation_performance(self, benchmark):
        """Benchmark validation performance."""
        validator = SecurityValidator()
        
        test_data = {
            "username": "testuser123",
            "email": "test@example.com",
            "content": "This is safe content for testing validation performance"
        }
        
        def validation_operations():
            for _ in range(50):
                try:
                    validator.validate(test_data, strict=False)
                except:
                    pass  # Expected for some validations
        
        result = benchmark(validation_operations)
        assert result is None
    
    def test_concurrent_load(self):
        """Test system under concurrent load."""
        cache = CacheManager(max_size=1000)
        
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    cache.set(key, value)
                    retrieved = cache.get(key)
                    if retrieved == value:
                        results.append(True)
                    else:
                        results.append(False)
            except Exception as e:
                errors.append(e)
        
        # Start multiple worker threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 1000  # 10 workers * 100 operations each
        assert all(results), "Some cache operations failed"


# Smoke tests for rapid validation
class TestSmokeTests:
    """Quick smoke tests for basic functionality."""
    
    def test_error_handling_smoke(self):
        """Quick test of error handling."""
        error = CyberRangeError("Test", "SMOKE_001")
        handler = ErrorHandler("smoke")
        handler.handle_error(error)
        stats = handler.get_error_statistics()
        assert stats["total_errors"] > 0
    
    def test_validation_smoke(self):
        """Quick test of validation."""
        validator = InputValidator()
        validator.add_rule("test", RequiredRule())
        result = validator.validate({"test": "value"})
        assert result["test"] == "value"
    
    def test_cache_smoke(self):
        """Quick test of caching."""
        cache = CacheManager()
        cache.set("test", "value")
        assert cache.get("test") == "value"
    
    def test_scaling_smoke(self):
        """Quick test of scaling logic."""
        policy = ScalingPolicy()
        metrics = ScalingMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            request_rate_per_second=100.0,
            queue_length=10,
            response_time_ms=200.0,
            active_connections=50,
            custom_metrics={}
        )
        
        assert policy.min_replicas > 0
        assert metrics.cpu_usage_percent == 50.0


if __name__ == "__main__":
    # Run smoke tests first
    pytest.main([__file__ + "::TestSmokeTests", "-v"])
    
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])