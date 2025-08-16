"""Isolated tests for core modules."""

import pytest
import asyncio
import time
import sys
import os
import importlib.util
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Direct module imports without package resolution
def import_module_from_path(module_name, file_path):
    """Import module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get the source directory path
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')

# Import modules directly
error_handling = import_module_from_path(
    "error_handling", 
    os.path.join(src_dir, "gan_cyber_range", "core", "error_handling.py")
)

validation = import_module_from_path(
    "validation",
    os.path.join(src_dir, "gan_cyber_range", "core", "validation.py")
)

performance_optimizer = import_module_from_path(
    "performance_optimizer",
    os.path.join(src_dir, "gan_cyber_range", "scaling", "performance_optimizer.py")
)

auto_scaler = import_module_from_path(
    "auto_scaler",
    os.path.join(src_dir, "gan_cyber_range", "scaling", "auto_scaler.py")
)


class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_cyber_range_error_creation(self):
        """Test CyberRangeError creation."""
        error = error_handling.CyberRangeError(
            message="Test error",
            error_code="TEST_001",
            context={"component": "test"}
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_001"
        assert error.context["component"] == "test"
        assert isinstance(error.timestamp, datetime)
    
    def test_error_serialization(self):
        """Test error serialization."""
        error = error_handling.CyberRangeError("Test", "TEST_001")
        error_dict = error.to_dict()
        
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["message"] == "Test"
        assert "timestamp" in error_dict
    
    def test_agent_error(self):
        """Test AgentError specific functionality."""
        error = error_handling.AgentError(
            message="Agent failed",
            agent_id="agent-123",
            error_code="TIMEOUT"
        )
        
        assert error.agent_id == "agent-123"
        assert error.error_code == "AGENT_TIMEOUT"
    
    def test_security_error(self):
        """Test SecurityError specific functionality."""
        error = error_handling.SecurityError(
            message="Security violation",
            security_context={"ip": "192.168.1.1"},
            error_code="UNAUTHORIZED"
        )
        
        assert error.security_context["ip"] == "192.168.1.1"
        assert error.error_code == "SEC_UNAUTHORIZED"
    
    def test_error_handler_basic(self):
        """Test basic error handler functionality."""
        handler = error_handling.ErrorHandler("test_logger")
        
        error = error_handling.CyberRangeError("Test error", "TEST_001")
        handler.handle_error(error)
        
        stats = handler.get_error_statistics()
        assert stats["error_counts"]["TEST_001"] == 1
        assert stats["total_errors"] == 1
    
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


class TestValidation:
    """Test validation functionality."""
    
    def test_required_rule(self):
        """Test RequiredRule."""
        rule = validation.RequiredRule()
        
        assert rule.validate("test") == True
        assert rule.validate("") == False
        assert rule.validate(None) == False
        assert rule.validate([]) == False
        assert rule.validate([1, 2]) == True
    
    def test_type_rule(self):
        """Test TypeRule."""
        string_rule = validation.TypeRule(str)
        int_rule = validation.TypeRule(int)
        
        assert string_rule.validate("test") == True
        assert string_rule.validate(123) == False
        
        assert int_rule.validate(123) == True
        assert int_rule.validate("test") == False
    
    def test_range_rule(self):
        """Test RangeRule."""
        rule = validation.RangeRule(min_val=1, max_val=10)
        
        assert rule.validate(5) == True
        assert rule.validate(1) == True
        assert rule.validate(10) == True
        assert rule.validate(0) == False
        assert rule.validate(11) == False
        assert rule.validate("test") == False
    
    def test_email_rule(self):
        """Test EmailRule."""
        rule = validation.EmailRule()
        
        assert rule.validate("test@example.com") == True
        assert rule.validate("user.name@domain.co.uk") == True
        assert rule.validate("invalid-email") == False
        assert rule.validate("@example.com") == False
        assert rule.validate("test@") == False
    
    def test_security_rule_xss(self):
        """Test SecurityRule for XSS prevention."""
        rule = validation.SecurityRule("no_script_tags")
        
        # Safe content
        assert rule.validate("Safe content") == True
        assert rule.validate("Hello world") == True
        
        # Dangerous content
        assert rule.validate("<script>alert('xss')</script>") == False
        assert rule.validate("javascript:alert('xss')") == False
        assert rule.validate("<img src=x onerror=alert(1)>") == False
    
    def test_security_rule_sql_injection(self):
        """Test SecurityRule for SQL injection prevention."""
        rule = validation.SecurityRule("no_sql_injection")
        
        # Safe queries
        assert rule.validate("SELECT * FROM users WHERE id = 1") == True
        assert rule.validate("Normal text content") == True
        
        # SQL injection attempts
        assert rule.validate("'; DROP TABLE users; --") == False
        assert rule.validate("1' OR '1'='1") == False
        assert rule.validate("UNION SELECT password FROM users") == False
    
    def test_input_validator(self):
        """Test InputValidator framework."""
        validator = validation.InputValidator()
        validator.add_rule("name", validation.RequiredRule())
        validator.add_rule("name", validation.TypeRule(str))
        validator.add_rule("age", validation.TypeRule(int))
        validator.add_rule("age", validation.RangeRule(min_val=0, max_val=150))
        
        # Valid data
        valid_data = {
            "name": "John Doe",
            "age": 30
        }
        result = validator.validate(valid_data)
        assert result["name"] == "John Doe"
        assert result["age"] == 30
        
        # Invalid data - missing required field
        with pytest.raises(validation.ValidationError):
            validator.validate({"age": 25})  # Missing name
        
        # Invalid data - wrong type
        with pytest.raises(validation.ValidationError):
            validator.validate({"name": "John", "age": "thirty"})  # Age should be int
        
        # Invalid data - out of range
        with pytest.raises(validation.ValidationError):
            validator.validate({"name": "John", "age": 200})  # Age too high
    
    def test_security_validator(self):
        """Test SecurityValidator."""
        validator = validation.SecurityValidator()
        
        # Safe data should validate
        safe_data = {
            "username": "testuser123",
            "content": "This is safe content"
        }
        result = validator.validate(safe_data, strict=False)
        assert result["username"] == "testuser123"
        
        # Dangerous data should fail
        dangerous_data = {
            "malicious_script": "<script>alert('xss')</script>",
            "sql_injection": "'; DROP TABLE users; --"
        }
        
        with pytest.raises(validation.ValidationError):
            validator.validate(dangerous_data, strict=False)


class TestPerformanceOptimizer:
    """Test performance optimization functionality."""
    
    def test_cache_entry(self):
        """Test CacheEntry functionality."""
        entry = performance_optimizer.CacheEntry(
            value="test_value",
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=0,
            ttl_seconds=60
        )
        
        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert not entry.is_expired
        
        # Test touch
        entry.touch()
        assert entry.access_count == 1
    
    def test_cache_manager_basic(self):
        """Test CacheManager basic operations."""
        cache = performance_optimizer.CacheManager(max_size=100, default_ttl_seconds=60)
        
        # Test set/get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test miss
        assert cache.get("nonexistent_key") is None
        
        # Test stats
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hit_rate"] > 0
        
        # Test delete
        assert cache.delete("key1") == True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") == False
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = performance_optimizer.CacheManager()
        
        # Set with short TTL
        cache.set("expiring_key", "value", ttl_seconds=0.1)
        assert cache.get("expiring_key") == "value"
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("expiring_key") is None
    
    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        cache = performance_optimizer.CacheManager(max_size=3)
        
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add another item (should evict key2, the least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Should still be there
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should still be there
        assert cache.get("key4") == "value4"  # Should be there
    
    def test_cached_decorator(self):
        """Test @cached decorator."""
        cache = performance_optimizer.CacheManager()
        call_count = 0
        
        @performance_optimizer.cached(cache_manager=cache, ttl_seconds=60)
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
        assert call_count == 1  # Function should not be called again
        
        # Call with different args (should call function)
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_performance_optimizer(self):
        """Test PerformanceOptimizer class."""
        optimizer = performance_optimizer.PerformanceOptimizer()
        
        # Test performance report generation
        report = optimizer.get_performance_report()
        
        assert "cache" in report
        assert "queries" in report
        assert "resources" in report
        assert "timestamp" in report
        
        # Cache stats should be present
        assert "size" in report["cache"]
        assert "hit_rate" in report["cache"]


class TestAutoScaler:
    """Test auto-scaling functionality."""
    
    def test_scaling_policy(self):
        """Test ScalingPolicy configuration."""
        policy = auto_scaler.ScalingPolicy(
            min_replicas=2,
            max_replicas=10,
            target_cpu_percent=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0
        )
        
        assert policy.min_replicas == 2
        assert policy.max_replicas == 10
        assert policy.target_cpu_percent == 70.0
        assert policy.scale_up_threshold == 80.0
        assert policy.scale_down_threshold == 30.0
        
        # Test serialization
        policy_dict = policy.to_dict()
        assert policy_dict["min_replicas"] == 2
        assert policy_dict["target_cpu_percent"] == 70.0
    
    def test_scaling_metrics(self):
        """Test ScalingMetrics data structure."""
        timestamp = datetime.utcnow()
        metrics = auto_scaler.ScalingMetrics(
            timestamp=timestamp,
            cpu_usage_percent=75.5,
            memory_usage_percent=65.0,
            request_rate_per_second=150.0,
            queue_length=25,
            response_time_ms=250.0,
            active_connections=100,
            custom_metrics={"error_rate": 0.02}
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage_percent == 75.5
        assert metrics.memory_usage_percent == 65.0
        assert metrics.custom_metrics["error_rate"] == 0.02
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert metrics_dict["cpu_usage_percent"] == 75.5
        assert metrics_dict["custom_metrics"]["error_rate"] == 0.02
        assert "timestamp" in metrics_dict
    
    def test_auto_scaler_scale_up_analysis(self):
        """Test auto-scaler scale-up decision logic."""
        policy = auto_scaler.ScalingPolicy(
            min_replicas=2,
            max_replicas=10,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
            scale_up_cooldown_seconds=1  # Short for testing
        )
        
        # Mock high-CPU metrics
        def high_cpu_metrics():
            return auto_scaler.ScalingMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage_percent=85.0,  # Above scale-up threshold
                memory_usage_percent=60.0,
                request_rate_per_second=100.0,
                queue_length=10,
                response_time_ms=200.0,
                active_connections=50,
                custom_metrics={}
            )
        
        # Mock scaling executor
        scale_calls = []
        def mock_scaler(target_replicas):
            scale_calls.append(target_replicas)
            return True
        
        scaler = auto_scaler.AutoScaler(
            component_name="test-service",
            scaling_policy=policy,
            metrics_collector=high_cpu_metrics,
            scaler_executor=mock_scaler
        )
        
        # Add metrics history to meet window requirements
        for _ in range(10):
            scaler.metrics_history.append(high_cpu_metrics())
        
        # Test scaling analysis
        decision = scaler._analyze_scaling_need()
        
        # Should recommend scale up due to high CPU
        assert decision["action"] == auto_scaler.ScalingDirection.UP
        assert decision["target_replicas"] > policy.min_replicas
        assert "High CPU usage" in str(decision.get("reasons", []))
    
    def test_auto_scaler_scale_down_analysis(self):
        """Test auto-scaler scale-down decision logic."""
        policy = auto_scaler.ScalingPolicy(
            min_replicas=2,
            max_replicas=10,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
            scale_down_cooldown_seconds=1
        )
        
        # Mock low-resource metrics
        def low_resource_metrics():
            return auto_scaler.ScalingMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage_percent=20.0,  # Below scale-down threshold
                memory_usage_percent=25.0,  # Below scale-down threshold
                request_rate_per_second=10.0,
                queue_length=1,
                response_time_ms=100.0,
                active_connections=5,
                custom_metrics={}
            )
        
        scaler = auto_scaler.AutoScaler(
            component_name="test-service",
            scaling_policy=policy,
            metrics_collector=low_resource_metrics,
            scaler_executor=lambda x: True
        )
        
        # Set current replicas above minimum to allow scale-down
        scaler.current_replicas = 5
        
        # Add metrics history
        for _ in range(10):
            scaler.metrics_history.append(low_resource_metrics())
        
        # Test scaling analysis
        decision = scaler._analyze_scaling_need()
        
        # Should recommend scale down due to low resource usage
        assert decision["action"] == auto_scaler.ScalingDirection.DOWN
        assert decision["target_replicas"] < scaler.current_replicas
        assert decision["target_replicas"] >= policy.min_replicas


class TestSmokeTests:
    """Quick smoke tests for basic functionality."""
    
    def test_error_handling_smoke(self):
        """Quick error handling test."""
        error = error_handling.CyberRangeError("Test error", "SMOKE_001")
        assert error.message == "Test error"
        assert error.error_code == "SMOKE_001"
        
        handler = error_handling.ErrorHandler("smoke")
        handler.handle_error(error)
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 1
    
    def test_validation_smoke(self):
        """Quick validation test."""
        validator = validation.InputValidator()
        validator.add_rule("test_field", validation.RequiredRule())
        
        result = validator.validate({"test_field": "test_value"})
        assert result["test_field"] == "test_value"
        
        with pytest.raises(validation.ValidationError):
            validator.validate({"test_field": ""})
    
    def test_cache_smoke(self):
        """Quick cache test."""
        cache = performance_optimizer.CacheManager()
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        stats = cache.get_stats()
        assert stats["size"] == 1
    
    def test_scaling_smoke(self):
        """Quick scaling test."""
        policy = auto_scaler.ScalingPolicy()
        assert policy.min_replicas > 0
        assert policy.max_replicas > policy.min_replicas
        
        metrics = auto_scaler.ScalingMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            request_rate_per_second=100.0,
            queue_length=10,
            response_time_ms=200.0,
            active_connections=50,
            custom_metrics={}
        )
        assert metrics.cpu_usage_percent == 50.0


class TestSecurity:
    """Security-focused tests."""
    
    def test_xss_prevention(self):
        """Test XSS attack prevention."""
        rule = validation.SecurityRule("no_script_tags")
        
        xss_attacks = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<object data='javascript:alert(1)'></object>"
        ]
        
        for attack in xss_attacks:
            assert rule.validate(attack) == False, f"XSS attack should be blocked: {attack}"
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        rule = validation.SecurityRule("no_sql_injection")
        
        sql_injections = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "1; UPDATE users SET admin=1 WHERE id=1; --",
            "admin'--"
        ]
        
        for injection in sql_injections:
            assert rule.validate(injection) == False, f"SQL injection should be blocked: {injection}"
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention."""
        rule = validation.SecurityRule("safe_filename")
        
        path_traversals = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "....//....//etc/passwd",
            "file<script>",
            "file|command"
        ]
        
        for path in path_traversals:
            assert rule.validate(path) == False, f"Path traversal should be blocked: {path}"
        
        # Safe filenames should pass
        safe_filenames = [
            "document.pdf",
            "image.jpg",
            "report-2024.xlsx",
            "backup_20240101.tar.gz"
        ]
        
        for filename in safe_filenames:
            assert rule.validate(filename) == True, f"Safe filename should be allowed: {filename}"


if __name__ == "__main__":
    # Run just the smoke tests for quick validation
    print("🚀 Running smoke tests...")
    test_result = pytest.main([
        __file__ + "::TestSmokeTests", 
        "-v", 
        "--tb=short",
        "-x"  # Stop on first failure
    ])
    
    if test_result == 0:
        print("\n✅ Smoke tests passed! Running full test suite...")
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("\n❌ Smoke tests failed!")
        exit(1)