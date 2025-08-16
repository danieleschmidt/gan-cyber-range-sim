"""Simple validation tests that run independently."""

import pytest
import sys
import os

# Simple test that doesn't import complex modules
def test_basic_python_functionality():
    """Test basic Python functionality works."""
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"
    assert [1, 2, 3][1] == 2


def test_datetime_functionality():
    """Test datetime functionality."""
    from datetime import datetime, timedelta
    
    now = datetime.utcnow()
    later = now + timedelta(hours=1)
    
    assert later > now
    assert (later - now).total_seconds() == 3600


def test_file_system_access():
    """Test file system access."""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    
    try:
        with open(temp_path, 'r') as f:
            content = f.read()
        assert content == "test content"
    finally:
        os.unlink(temp_path)


def test_json_serialization():
    """Test JSON serialization."""
    import json
    
    data = {
        "name": "test",
        "value": 123,
        "items": [1, 2, 3],
        "nested": {"key": "value"}
    }
    
    json_str = json.dumps(data)
    parsed = json.loads(json_str)
    
    assert parsed == data


def test_regex_functionality():
    """Test regex functionality."""
    import re
    
    # Email pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    assert re.match(email_pattern, "test@example.com")
    assert not re.match(email_pattern, "invalid-email")
    
    # XSS detection
    xss_pattern = r'<script.*?>.*?</script>'
    assert re.search(xss_pattern, "<script>alert('xss')</script>", re.IGNORECASE)
    assert not re.search(xss_pattern, "safe content")


def test_async_functionality():
    """Test async/await functionality."""
    import asyncio
    
    async def async_add(a, b):
        await asyncio.sleep(0.01)  # Simulate async work
        return a + b
    
    async def test_async():
        result = await async_add(1, 2)
        assert result == 3
        
        # Test concurrent execution
        tasks = [async_add(i, i+1) for i in range(5)]
        results = await asyncio.gather(*tasks)
        expected = [1, 3, 5, 7, 9]  # 0+1, 1+2, 2+3, 3+4, 4+5
        assert results == expected
    
    # Run the async test
    asyncio.run(test_async())


def test_threading_functionality():
    """Test threading functionality."""
    import threading
    import time
    
    results = []
    
    def worker(worker_id):
        time.sleep(0.01)  # Simulate work
        results.append(worker_id)
    
    # Start multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # All workers should have completed
    assert len(results) == 5
    assert set(results) == {0, 1, 2, 3, 4}


def test_error_handling():
    """Test error handling functionality."""
    
    class CustomError(Exception):
        def __init__(self, message, code=None):
            self.message = message
            self.code = code
            super().__init__(message)
    
    # Test custom exception
    with pytest.raises(CustomError) as exc_info:
        raise CustomError("Test error", code="TEST_001")
    
    assert exc_info.value.message == "Test error"
    assert exc_info.value.code == "TEST_001"
    
    # Test error recovery pattern
    attempt_count = 0
    
    def retry_operation(max_retries=3):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= 2:
            raise ValueError("Temporary failure")
        return "success"
    
    # Should succeed after retries
    for _ in range(3):
        try:
            result = retry_operation()
            break
        except ValueError:
            if attempt_count >= 3:
                raise
    
    assert result == "success"
    assert attempt_count == 3


def test_caching_pattern():
    """Test simple caching pattern."""
    from collections import OrderedDict
    import time
    
    class SimpleCache:
        def __init__(self, max_size=100):
            self.cache = OrderedDict()
            self.max_size = max_size
            self.stats = {"hits": 0, "misses": 0}
        
        def get(self, key):
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return self.cache[key]
            
            self.stats["misses"] += 1
            return None
        
        def set(self, key, value):
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                # Evict oldest if at capacity
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            
            self.cache[key] = value
        
        def hit_rate(self):
            total = self.stats["hits"] + self.stats["misses"]
            return self.stats["hits"] / total if total > 0 else 0
    
    # Test cache functionality
    cache = SimpleCache(max_size=3)
    
    # Test set/get
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None
    
    # Test LRU eviction
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    cache.get("key1")  # Make key1 recently used
    cache.set("key4", "value4")  # Should evict key2
    
    assert cache.get("key1") == "value1"  # Should still be there
    assert cache.get("key2") is None      # Should be evicted
    assert cache.get("key3") == "value3"  # Should still be there
    assert cache.get("key4") == "value4"  # Should be there
    
    # Test hit rate
    assert cache.hit_rate() > 0


def test_validation_patterns():
    """Test validation patterns."""
    import re
    
    def validate_email(email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_safe_string(text):
        """Check for potentially dangerous content."""
        dangerous_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'on\w+\s*=',
            r'(\b(union|select|insert|update|delete|drop)\b)',
            r'[;&|`]'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True
    
    def validate_range(value, min_val=None, max_val=None):
        """Validate numeric range."""
        if not isinstance(value, (int, float)):
            return False
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    
    # Test email validation
    assert validate_email("test@example.com") == True
    assert validate_email("invalid-email") == False
    
    # Test security validation
    assert validate_safe_string("Safe content") == True
    assert validate_safe_string("<script>alert('xss')</script>") == False
    assert validate_safe_string("'; DROP TABLE users; --") == False
    
    # Test range validation
    assert validate_range(5, min_val=1, max_val=10) == True
    assert validate_range(0, min_val=1, max_val=10) == False
    assert validate_range(15, min_val=1, max_val=10) == False


def test_performance_monitoring():
    """Test performance monitoring patterns."""
    import time
    from collections import defaultdict
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = defaultdict(list)
        
        def time_operation(self, name):
            """Context manager for timing operations."""
            class Timer:
                def __init__(self, monitor, operation_name):
                    self.monitor = monitor
                    self.name = operation_name
                    self.start_time = None
                
                def __enter__(self):
                    self.start_time = time.time()
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    end_time = time.time()
                    duration = end_time - self.start_time
                    self.monitor.metrics[self.name].append(duration)
            
            return Timer(self, name)
        
        def get_average_time(self, name):
            times = self.metrics[name]
            return sum(times) / len(times) if times else 0
        
        def get_stats(self):
            stats = {}
            for name, times in self.metrics.items():
                if times:
                    stats[name] = {
                        "count": len(times),
                        "total_time": sum(times),
                        "average_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times)
                    }
            return stats
    
    # Test performance monitoring
    monitor = PerformanceMonitor()
    
    # Time some operations
    with monitor.time_operation("fast_operation"):
        time.sleep(0.01)
    
    with monitor.time_operation("slow_operation"):
        time.sleep(0.05)
    
    with monitor.time_operation("fast_operation"):
        time.sleep(0.01)
    
    # Check stats
    stats = monitor.get_stats()
    assert "fast_operation" in stats
    assert "slow_operation" in stats
    
    fast_stats = stats["fast_operation"]
    slow_stats = stats["slow_operation"]
    
    assert fast_stats["count"] == 2
    assert slow_stats["count"] == 1
    assert fast_stats["average_time"] < slow_stats["average_time"]


if __name__ == "__main__":
    print("🚀 Running simple validation tests...")
    
    # Run tests manually for immediate feedback
    test_functions = [
        test_basic_python_functionality,
        test_datetime_functionality,
        test_file_system_access,
        test_json_serialization,
        test_regex_functionality,
        test_async_functionality,
        test_threading_functionality,
        test_error_handling,
        test_caching_pattern,
        test_validation_patterns,
        test_performance_monitoring
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
        exit(1)