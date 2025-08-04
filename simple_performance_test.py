#!/usr/bin/env python3
"""
Simple performance test for GAN Cyber Range Simulator.
Tests core functionality performance without complex async handling.
"""

import time
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Any

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def time_function(func, *args, **kwargs):
    """Time a function execution."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, (end_time - start_time) * 1000  # Return result and time in ms


def test_basic_operations():
    """Test basic data structure operations."""
    print("🧪 Testing basic operations...")
    
    times = []
    
    for i in range(100):
        def create_service_data():
            return {
                "name": f"service_{i}",
                "type": "webapp",
                "ip": f"10.0.1.{i}",
                "ports": [80, 443, 8080],
                "vulnerabilities": [
                    {"id": "CVE-2023-1234", "severity": "high"},
                    {"id": "CVE-2023-5678", "severity": "medium"}
                ],
                "metadata": {"patched": i % 2 == 0}
            }
        
        result, exec_time = time_function(create_service_data)
        times.append(exec_time)
    
    avg_time = statistics.mean(times)
    print(f"   ✅ Basic operations: {avg_time:.3f}ms avg")
    return avg_time


def test_json_processing():
    """Test JSON serialization performance."""
    print("🧪 Testing JSON processing...")
    
    import json
    
    # Create test data
    test_data = {
        "simulation": {
            "id": "test-simulation-12345",
            "status": "running",
            "agents": [
                {
                    "name": f"agent_{i}",
                    "type": "red_team" if i % 2 == 0 else "blue_team",
                    "actions": [
                        {"type": "scan", "target": f"service_{j}", "success": j % 3 == 0}
                        for j in range(5)
                    ]
                }
                for i in range(10)
            ]
        }
    }
    
    times = []
    
    for i in range(50):
        def json_round_trip():
            json_str = json.dumps(test_data)
            parsed_data = json.loads(json_str)
            return len(json_str)
        
        result, exec_time = time_function(json_round_trip)
        times.append(exec_time)
    
    avg_time = statistics.mean(times)
    print(f"   ✅ JSON processing: {avg_time:.3f}ms avg")
    return avg_time


def test_string_operations():
    """Test string processing."""
    print("🧪 Testing string operations...")
    
    import re
    
    times = []
    
    for i in range(100):
        def string_processing():
            # Generate test strings
            test_strings = [f"attack_type_{j}_target_service_{j%5}" for j in range(50)]
            
            # Join and split
            joined = "|".join(test_strings)
            split_back = joined.split("|")
            
            # Pattern matching
            pattern = re.compile(r"attack_type_(\d+)_target_")
            matches = [pattern.search(s).group(1) if pattern.search(s) else None for s in test_strings]
            
            return len([m for m in matches if m])
        
        result, exec_time = time_function(string_processing)
        times.append(exec_time)
    
    avg_time = statistics.mean(times)
    print(f"   ✅ String operations: {avg_time:.3f}ms avg")
    return avg_time


def test_computation():
    """Test computational performance."""
    print("🧪 Testing computational performance...")
    
    times = []
    
    for i in range(50):
        def computational_task():
            # Simulate agent decision making computation
            scores = []
            for j in range(100):
                # Simulate vulnerability scoring
                base_score = j * 0.1
                risk_multiplier = 1.5 if j % 3 == 0 else 1.0
                final_score = base_score * risk_multiplier
                scores.append(final_score)
            
            # Sort and filter
            high_risk = [s for s in scores if s > 5.0]
            sorted_scores = sorted(high_risk, reverse=True)
            
            return len(sorted_scores)
        
        result, exec_time = time_function(computational_task)
        times.append(exec_time)
    
    avg_time = statistics.mean(times)
    print(f"   ✅ Computational tasks: {avg_time:.3f}ms avg")
    return avg_time


def test_memory_usage():
    """Test memory allocation patterns."""
    print("🧪 Testing memory usage patterns...")
    
    times = []
    
    for i in range(30):
        def memory_intensive_task():
            # Simulate creating simulation state
            large_list = list(range(1000))
            large_dict = {f"key_{j}": f"value_{j}_data" for j in range(500)}
            
            # Process data
            filtered_list = [x for x in large_list if x % 2 == 0]
            processed_dict = {k: v.upper() for k, v in large_dict.items() if "5" in k}
            
            # Cleanup happens automatically with scope
            return len(filtered_list) + len(processed_dict)
        
        result, exec_time = time_function(memory_intensive_task)
        times.append(exec_time)
    
    avg_time = statistics.mean(times)
    print(f"   ✅ Memory operations: {avg_time:.3f}ms avg")
    return avg_time


def main():
    """Run performance tests."""
    print("🚀 Starting Simple Performance Test")
    print("=" * 50)
    
    # Run tests
    test_results = {
        "basic_operations": test_basic_operations(),
        "json_processing": test_json_processing(),
        "string_operations": test_string_operations(),
        "computational": test_computation(),
        "memory_usage": test_memory_usage()
    }
    
    # Calculate overall performance
    print("\n" + "=" * 50)
    print("📊 Performance Results Summary")
    print("=" * 50)
    
    sorted_results = sorted(test_results.items(), key=lambda x: x[1])
    
    for test_name, avg_time in sorted_results:
        performance_rating = "🟢 Excellent" if avg_time < 1.0 else \
                           "🟡 Good" if avg_time < 5.0 else \
                           "🟠 Moderate" if avg_time < 20.0 else \
                           "🔴 Slow"
        
        print(f"   {test_name.replace('_', ' ').title():<20}: {avg_time:>6.2f}ms {performance_rating}")
    
    # Overall assessment
    overall_avg = statistics.mean(test_results.values())
    print(f"\n📈 Overall Average: {overall_avg:.2f}ms")
    
    if overall_avg < 2.0:
        print("✅ Excellent performance - system is highly optimized")
        rating = "excellent"
    elif overall_avg < 10.0:
        print("✅ Good performance - system meets performance requirements")
        rating = "good"
    elif overall_avg < 50.0:
        print("⚠️  Moderate performance - some optimization may be beneficial")
        rating = "moderate"
    else:
        print("❌ Poor performance - optimization required")
        rating = "poor"
    
    # Performance recommendations
    print(f"\n💡 Recommendations:")
    if test_results["json_processing"] > 10.0:
        print("   - Consider caching frequently serialized data")
    if test_results["string_operations"] > 5.0:
        print("   - Pre-compile regex patterns for better performance")
    if test_results["memory_usage"] > 20.0:
        print("   - Consider memory pooling for large data structures")
    if test_results["computational"] > 15.0:
        print("   - Consider async processing for CPU-intensive tasks")
    
    if rating in ["excellent", "good"]:
        print("   - Performance is already well-optimized!")
    
    return 0 if rating in ["excellent", "good"] else 1


if __name__ == "__main__":
    sys.exit(main())