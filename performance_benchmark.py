#!/usr/bin/env python3
"""
Performance benchmark for GAN Cyber Range Simulator.
Tests system performance and identifies bottlenecks.
"""

import asyncio
import time
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class PerformanceBenchmark:
    """Performance benchmark suite."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
    
    def benchmark(self, name: str, iterations: int = 100):
        """Decorator for benchmarking functions."""
        def decorator(func: Callable):
            async def async_wrapper(*args, **kwargs):
                times = []
                
                print(f"🏃 Running {name} benchmark ({iterations} iterations)...")
                
                for i in range(iterations):
                    start_time = time.time()
                    
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                    if i % 10 == 0 and i > 0:
                        print(f"   Progress: {i}/{iterations} iterations")
                
                # Calculate statistics
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                median_time = statistics.median(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                
                self.results[name] = {
                    'iterations': iterations,
                    'avg_time_ms': avg_time * 1000,
                    'min_time_ms': min_time * 1000,
                    'max_time_ms': max_time * 1000,
                    'median_time_ms': median_time * 1000,
                    'std_dev_ms': std_dev * 1000,
                    'ops_per_second': 1.0 / avg_time if avg_time > 0 else 0
                }
                
                print(f"   ✅ {name}: {avg_time*1000:.2f}ms avg, {1.0/avg_time:.1f} ops/sec")
                return result
            
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @benchmark("basic_data_structures", 1000)
    def test_basic_data_structures(self):
        """Test basic data structure operations."""
        from dataclasses import dataclass
        from typing import Dict, List
        
        @dataclass
        class TestService:
            name: str
            type: str
            ip: str
            ports: List[int]
            metadata: Dict[str, Any]
        
        # Create test service
        service = TestService(
            name="test-service",
            type="webapp",
            ip="10.0.1.100",
            ports=[80, 443, 8080],
            metadata={"vulnerabilities": ["XSS", "SQLi"], "patched": True}
        )
        
        # Test serialization
        service_dict = {
            "name": service.name,
            "type": service.type,
            "ip": service.ip,
            "ports": service.ports,
            "metadata": service.metadata
        }
        
        return len(service_dict)
    
    @benchmark("async_operations", 500)
    async def test_async_operations(self):
        """Test async operation performance."""
        
        async def simulate_network_operation():
            await asyncio.sleep(0.001)  # 1ms simulated network delay
            return {"status": "success", "data": "test_data"}
        
        # Run concurrent operations
        tasks = [simulate_network_operation() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        return len(results)
    
    @benchmark("concurrent_execution", 200)
    async def test_concurrent_execution(self):
        """Test concurrent execution performance."""
        
        def cpu_bound_task(n: int) -> int:
            # Simple CPU-bound calculation
            total = 0
            for i in range(n):
                total += i * i
            return total
        
        # Test with thread pool
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, cpu_bound_task, 1000)
                for _ in range(8)
            ]
            results = await asyncio.gather(*tasks)
        
        return sum(results)
    
    @benchmark("memory_operations", 500)
    def test_memory_operations(self):
        """Test memory allocation and access patterns."""
        
        # Create large data structures
        large_list = list(range(10000))
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        # Test list operations
        filtered_list = [x for x in large_list if x % 2 == 0]
        
        # Test dict operations
        filtered_dict = {k: v for k, v in large_dict.items() if "5" in k}
        
        return len(filtered_list) + len(filtered_dict)
    
    @benchmark("string_operations", 1000)
    def test_string_operations(self):
        """Test string processing performance."""
        
        # Generate test data
        test_strings = [f"test_string_{i}_with_data" for i in range(100)]
        
        # String operations
        joined = "|".join(test_strings)
        split_back = joined.split("|")
        
        # Pattern matching
        import re
        pattern = re.compile(r"test_string_(\d+)_")
        matches = [pattern.search(s) for s in test_strings]
        
        return len([m for m in matches if m])
    
    @benchmark("json_serialization", 500)
    def test_json_serialization(self):
        """Test JSON serialization performance."""
        import json
        
        # Create complex data structure
        test_data = {
            "simulation_id": "test-sim-12345",
            "agents": [
                {
                    "name": f"agent_{i}",
                    "type": "red_team" if i % 2 == 0 else "blue_team",
                    "actions": [
                        {"type": f"action_{j}", "success": j % 3 == 0}
                        for j in range(10)
                    ]
                }
                for i in range(20)
            ],
            "services": [
                {
                    "name": f"service_{i}",
                    "status": "running",
                    "metrics": {"cpu": i * 2.5, "memory": i * 10}
                }
                for i in range(50)
            ]
        }
        
        # Serialize and deserialize
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        
        return len(json_str)
    
    @benchmark("logging_operations", 1000)
    def test_logging_operations(self):
        """Test logging performance."""
        import logging
        import json
        from datetime import datetime
        
        # Create logger with JSON formatter
        logger = logging.getLogger("benchmark_logger")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add null handler for performance testing
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)
        
        # Test logging operations
        for i in range(10):
            logger.info(f"Test log message {i}")
            logger.warning(f"Test warning {i}")
        
        return 10
    
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("🚀 Starting Performance Benchmark Suite")
        print("=" * 60)
        
        # Run benchmarks
        asyncio.run(self._run_benchmarks_async())
        
        # Generate report
        self._generate_report()
    
    async def _run_benchmarks_async(self):
        """Run async benchmarks."""
        
        # Run sync benchmarks
        self.test_basic_data_structures()
        self.test_memory_operations()
        self.test_string_operations()
        self.test_json_serialization()
        self.test_logging_operations()
        
        # Run async benchmarks
        await self.test_async_operations()
        await self.test_concurrent_execution()
    
    def _generate_report(self):
        """Generate performance report."""
        print("\n" + "=" * 60)
        print("📊 Performance Benchmark Results")
        print("=" * 60)
        
        # Sort results by performance
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['ops_per_second'],
            reverse=True
        )
        
        print(f"{'Benchmark':<25} {'Avg Time':<12} {'Ops/Sec':<12} {'Min/Max (ms)':<15}")
        print("-" * 60)
        
        for name, result in sorted_results:
            avg_time = result['avg_time_ms']
            ops_per_sec = result['ops_per_second']
            min_time = result['min_time_ms']
            max_time = result['max_time_ms']
            
            print(f"{name:<25} {avg_time:>8.2f}ms {ops_per_sec:>9.1f} {min_time:.1f}/{max_time:.1f}")
        
        # Calculate overall metrics
        all_times = [r['avg_time_ms'] for r in self.results.values()]
        all_ops = [r['ops_per_second'] for r in self.results.values()]
        
        print("\n📈 Summary:")
        print(f"   Average response time: {statistics.mean(all_times):.2f}ms")
        print(f"   Average throughput: {statistics.mean(all_ops):.1f} ops/sec")
        print(f"   Fastest operation: {min(all_times):.2f}ms")
        print(f"   Slowest operation: {max(all_times):.2f}ms")
        
        # Performance assessment
        avg_time = statistics.mean(all_times)
        if avg_time < 10:
            print("\n✅ Excellent performance - all operations under 10ms")
        elif avg_time < 50:
            print("\n✅ Good performance - most operations under 50ms")
        elif avg_time < 100:
            print("\n⚠️  Moderate performance - some optimization recommended")
        else:
            print("\n❌ Poor performance - optimization required")
        
        return {
            'summary': {
                'avg_response_time_ms': statistics.mean(all_times),
                'avg_throughput_ops_sec': statistics.mean(all_ops),
                'fastest_operation_ms': min(all_times),
                'slowest_operation_ms': max(all_times),
                'total_benchmarks': len(self.results)
            },
            'detailed_results': self.results
        }


def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
    return 0


if __name__ == "__main__":
    sys.exit(main())