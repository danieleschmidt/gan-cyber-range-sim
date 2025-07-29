#!/usr/bin/env python3
"""Performance benchmarking script for GAN Cyber Range Simulator."""

import asyncio
import time
import statistics
import json
import sys
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

# Import your modules here (when they exist)
# from gan_cyber_range import CyberRange, RedTeamAgent, BlueTeamAgent


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    test_name: str
    duration: float
    memory_usage: float
    success: bool
    metadata: Dict[str, Any]


class PerformanceBenchmark:
    """Performance benchmarking suite for the cyber range simulator."""
    
    def __init__(self, output_file: str = "benchmark_results.json"):
        self.output_file = Path(output_file)
        self.results: List[BenchmarkResult] = []
    
    def benchmark(self, name: str):
        """Decorator to benchmark a function."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                memory_start = self._get_memory_usage()
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    print(f"Benchmark {name} failed: {e}")
                    result = None
                    success = False
                
                end_time = time.perf_counter()
                memory_end = self._get_memory_usage()
                
                duration = end_time - start_time
                memory_usage = memory_end - memory_start
                
                benchmark_result = BenchmarkResult(
                    test_name=name,
                    duration=duration,
                    memory_usage=memory_usage,
                    success=success,
                    metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
                )
                
                self.results.append(benchmark_result)
                print(f"✓ {name}: {duration:.4f}s, {memory_usage:.2f}MB")
                
                return result
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    @benchmark("Environment Initialization")
    async def benchmark_environment_init(self):
        """Benchmark cyber range environment initialization."""
        # Simulated initialization - replace with actual code
        await asyncio.sleep(0.1)  # Simulate initialization time
        return {"status": "initialized", "services": 5}
    
    @benchmark("Agent Creation")
    async def benchmark_agent_creation(self):
        """Benchmark agent creation and setup."""
        # Simulated agent creation - replace with actual code
        await asyncio.sleep(0.05)  # Simulate agent setup
        return {"red_agents": 1, "blue_agents": 1}
    
    @benchmark("Attack Simulation")
    async def benchmark_attack_simulation(self, duration: int = 1):
        """Benchmark attack simulation performance."""
        # Simulated attack - replace with actual attack logic
        start = time.time()
        while time.time() - start < duration:
            await asyncio.sleep(0.01)  # Simulate attack step
        return {"attacks_executed": duration * 100}
    
    @benchmark("Defense Response")
    async def benchmark_defense_response(self, threat_count: int = 10):
        """Benchmark defense response time."""
        # Simulated defense - replace with actual defense logic
        for _ in range(threat_count):
            await asyncio.sleep(0.001)  # Simulate defense action
        return {"threats_handled": threat_count}
    
    @benchmark("Large Scale Simulation")
    async def benchmark_large_scale(self):
        """Benchmark large scale simulation with multiple agents."""
        # Simulated large scale test - replace with actual implementation
        tasks = []
        for i in range(10):
            tasks.append(asyncio.sleep(0.01))  # Simulate concurrent operations
        
        await asyncio.gather(*tasks)
        return {"concurrent_operations": len(tasks)}
    
    async def run_all_benchmarks(self):
        """Run all benchmark tests."""
        print("🚀 Starting Performance Benchmarks\n")
        
        # Basic benchmarks
        await self.benchmark_environment_init()
        await self.benchmark_agent_creation()
        
        # Performance benchmarks
        await self.benchmark_attack_simulation(duration=2)
        await self.benchmark_defense_response(threat_count=50)
        
        # Stress test
        await self.benchmark_large_scale()
        
        print(f"\n📊 Completed {len(self.results)} benchmarks")
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate statistics."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        durations = [r.duration for r in self.results if r.success]
        memory_usage = [r.memory_usage for r in self.results if r.success]
        
        analysis = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": len([r for r in self.results if r.success]),
                "failed_tests": len([r for r in self.results if not r.success]),
            },
            "performance": {
                "avg_duration": statistics.mean(durations) if durations else 0,
                "median_duration": statistics.median(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
            },
            "memory": {
                "avg_memory_usage": statistics.mean(memory_usage) if memory_usage else 0,
                "max_memory_usage": max(memory_usage) if memory_usage else 0,
                "min_memory_usage": min(memory_usage) if memory_usage else 0,
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "duration": r.duration,
                    "memory_usage": r.memory_usage,
                    "success": r.success,
                    "metadata": r.metadata
                }
                for r in self.results
            ]
        }
        
        return analysis
    
    def save_results(self):
        """Save benchmark results to file."""
        analysis = self.analyze_results()
        
        with open(self.output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n💾 Results saved to {self.output_file}")
        return analysis
    
    def print_summary(self):
        """Print benchmark summary to console."""
        analysis = self.analyze_results()
        
        print("\n" + "="*50)
        print("📊 BENCHMARK SUMMARY")
        print("="*50)
        
        summary = analysis["summary"]
        perf = analysis["performance"]
        memory = analysis["memory"]
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {(summary['successful_tests']/summary['total_tests']*100):.1f}%")
        
        print(f"\nPerformance:")
        print(f"  Average Duration: {perf['avg_duration']:.4f}s")
        print(f"  Median Duration: {perf['median_duration']:.4f}s")
        print(f"  Fastest Test: {perf['min_duration']:.4f}s")
        print(f"  Slowest Test: {perf['max_duration']:.4f}s")
        
        print(f"\nMemory Usage:")
        print(f"  Average: {memory['avg_memory_usage']:.2f}MB")
        print(f"  Peak: {memory['max_memory_usage']:.2f}MB")
        
        # Performance warnings
        if perf['max_duration'] > 5.0:
            print(f"\n⚠️  Warning: Slowest test took {perf['max_duration']:.2f}s")
        
        if memory['max_memory_usage'] > 500:
            print(f"⚠️  Warning: Peak memory usage was {memory['max_memory_usage']:.2f}MB")


async def main():
    """Main benchmark execution."""
    print("GAN Cyber Range Simulator - Performance Benchmarks")
    print("=" * 50)
    
    # Initialize benchmark suite
    benchmark = PerformanceBenchmark("benchmark_results.json")
    
    try:
        # Run all benchmarks
        await benchmark.run_all_benchmarks()
        
        # Analyze and save results
        benchmark.print_summary()
        benchmark.save_results()
        
        # Check for performance regressions
        analysis = benchmark.analyze_results()
        if analysis["performance"]["max_duration"] > 10.0:
            print("\n❌ Performance regression detected!")
            sys.exit(1)
        else:
            print("\n✅ All benchmarks passed performance thresholds")
            
    except Exception as e:
        print(f"\n❌ Benchmark suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Install required packages for memory monitoring
    try:
        import psutil
    except ImportError:
        print("Installing psutil for memory monitoring...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    
    # Run benchmarks
    asyncio.run(main())