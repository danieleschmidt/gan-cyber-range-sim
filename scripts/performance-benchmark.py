#!/usr/bin/env python3

"""
Advanced Performance Benchmarking Suite for GAN Cyber Range Simulator
Comprehensive performance testing with metrics collection and analysis
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import sys

# Third-party imports (install if needed)
try:
    import psutil
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install psutil matplotlib pandas seaborn")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Data class for benchmark results"""
    test_name: str
    duration_seconds: float
    memory_peak_mb: float
    cpu_percent_avg: float
    iterations: int
    success_rate: float
    throughput_ops_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    timestamp: str
    metadata: Dict[str, Any]

class PerformanceBenchmark:
    """Main performance benchmarking class"""
    
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[BenchmarkResult] = []
        
    def monitor_system_resources(self, duration: float = 1.0) -> Dict[str, float]:
        """Monitor system resources during benchmark"""
        start_time = time.time()
        cpu_samples = []
        memory_samples = []
        
        while time.time() - start_time < duration:
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            memory_samples.append(psutil.virtual_memory().percent)
            time.sleep(0.1)
            
        return {
            'cpu_avg': statistics.mean(cpu_samples),
            'cpu_max': max(cpu_samples),
            'memory_avg': statistics.mean(memory_samples),
            'memory_max': max(memory_samples)
        }
    
    async def benchmark_agent_initialization(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark agent initialization performance"""
        logger.info(f"Benchmarking agent initialization ({iterations} iterations)")
        
        start_time = time.time()
        process = psutil.Process()
        memory_start = process.memory_info().rss / 1024 / 1024  # MB
        
        latencies = []
        successful_inits = 0
        
        for i in range(iterations):
            iter_start = time.time()
            
            try:
                # Simulate agent initialization
                # In real implementation, this would initialize actual agents
                await asyncio.sleep(0.001)  # Simulate async initialization
                successful_inits += 1
                
                iter_end = time.time()
                latencies.append((iter_end - iter_start) * 1000)  # Convert to ms
                
            except Exception as e:
                logger.warning(f"Agent initialization failed on iteration {i}: {e}")
                
        end_time = time.time()
        memory_end = process.memory_info().rss / 1024 / 1024  # MB
        total_duration = end_time - start_time
        
        # Calculate metrics
        success_rate = successful_inits / iterations
        throughput = successful_inits / total_duration
        
        return BenchmarkResult(
            test_name="agent_initialization",
            duration_seconds=total_duration,
            memory_peak_mb=memory_end - memory_start,
            cpu_percent_avg=psutil.cpu_percent(),
            iterations=iterations,
            success_rate=success_rate,
            throughput_ops_per_sec=throughput,
            latency_p50_ms=statistics.median(latencies) if latencies else 0,
            latency_p95_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
            latency_p99_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
            timestamp=datetime.now().isoformat(),
            metadata={
                'test_type': 'agent_initialization',
                'iterations': iterations,
                'successful_iterations': successful_inits
            }
        )
    
    async def benchmark_vulnerability_simulation(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark vulnerability simulation performance"""
        logger.info(f"Benchmarking vulnerability simulation ({iterations} iterations)")
        
        start_time = time.time()
        process = psutil.Process()
        memory_start = process.memory_info().rss / 1024 / 1024
        
        latencies = []
        successful_sims = 0
        
        for i in range(iterations):
            iter_start = time.time()
            
            try:
                # Simulate vulnerability creation and injection
                await asyncio.sleep(0.005)  # Simulate vulnerability setup
                successful_sims += 1
                
                iter_end = time.time()
                latencies.append((iter_end - iter_start) * 1000)
                
            except Exception as e:
                logger.warning(f"Vulnerability simulation failed on iteration {i}: {e}")
                
        end_time = time.time()
        memory_end = process.memory_info().rss / 1024 / 1024
        total_duration = end_time - start_time
        
        success_rate = successful_sims / iterations
        throughput = successful_sims / total_duration
        
        return BenchmarkResult(
            test_name="vulnerability_simulation",
            duration_seconds=total_duration,
            memory_peak_mb=memory_end - memory_start,
            cpu_percent_avg=psutil.cpu_percent(),
            iterations=iterations,
            success_rate=success_rate,
            throughput_ops_per_sec=throughput,
            latency_p50_ms=statistics.median(latencies) if latencies else 0,
            latency_p95_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
            latency_p99_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
            timestamp=datetime.now().isoformat(),
            metadata={
                'test_type': 'vulnerability_simulation',
                'iterations': iterations,
                'successful_iterations': successful_sims
            }
        )
    
    async def benchmark_adversarial_training(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark adversarial training loop performance"""
        logger.info(f"Benchmarking adversarial training ({iterations} iterations)")
        
        start_time = time.time()
        process = psutil.Process()
        memory_start = process.memory_info().rss / 1024 / 1024
        
        latencies = []
        successful_episodes = 0
        
        for i in range(iterations):
            iter_start = time.time()
            
            try:
                # Simulate training episode
                await asyncio.sleep(0.02)  # Simulate training computation
                successful_episodes += 1
                
                iter_end = time.time()
                latencies.append((iter_end - iter_start) * 1000)
                
            except Exception as e:
                logger.warning(f"Training episode failed on iteration {i}: {e}")
                
        end_time = time.time()
        memory_end = process.memory_info().rss / 1024 / 1024
        total_duration = end_time - start_time
        
        success_rate = successful_episodes / iterations
        throughput = successful_episodes / total_duration
        
        return BenchmarkResult(
            test_name="adversarial_training",
            duration_seconds=total_duration,
            memory_peak_mb=memory_end - memory_start,
            cpu_percent_avg=psutil.cpu_percent(),
            iterations=iterations,
            success_rate=success_rate,
            throughput_ops_per_sec=throughput,
            latency_p50_ms=statistics.median(latencies) if latencies else 0,
            latency_p95_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
            latency_p99_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
            timestamp=datetime.now().isoformat(),
            metadata={
                'test_type': 'adversarial_training',
                'iterations': iterations,
                'successful_iterations': successful_episodes
            }
        )
    
    def benchmark_memory_usage(self, duration: int = 30) -> BenchmarkResult:
        """Benchmark memory usage patterns"""
        logger.info(f"Benchmarking memory usage for {duration} seconds")
        
        start_time = time.time()
        process = psutil.Process()
        
        memory_samples = []
        cpu_samples = []
        
        while time.time() - start_time < duration:
            memory_info = process.memory_info()
            memory_samples.append(memory_info.rss / 1024 / 1024)  # MB
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            time.sleep(1)
            
        total_duration = time.time() - start_time
        
        return BenchmarkResult(
            test_name="memory_usage_pattern",
            duration_seconds=total_duration,
            memory_peak_mb=max(memory_samples),
            cpu_percent_avg=statistics.mean(cpu_samples),
            iterations=len(memory_samples),
            success_rate=1.0,
            throughput_ops_per_sec=len(memory_samples) / total_duration,
            latency_p50_ms=0,  # Not applicable for this test
            latency_p95_ms=0,
            latency_p99_ms=0,
            timestamp=datetime.now().isoformat(),
            metadata={
                'test_type': 'memory_usage_pattern',
                'memory_samples': memory_samples,
                'cpu_samples': cpu_samples,
                'memory_avg_mb': statistics.mean(memory_samples),
                'memory_std_mb': statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0
            }
        )
    
    def run_load_test(self, concurrent_users: int = 10, duration: int = 60) -> BenchmarkResult:
        """Run load test simulation"""
        logger.info(f"Running load test: {concurrent_users} concurrent users for {duration}s")
        
        start_time = time.time()
        
        # Simulate load test results
        # In real implementation, this would use actual load testing tools
        total_requests = concurrent_users * duration * 2  # Assume 2 requests per second per user
        successful_requests = int(total_requests * 0.95)  # 95% success rate
        
        # Simulate latency distribution
        latencies = [
            50 + (i % 100) + (i % 7) * 10  # Simulated latency pattern
            for i in range(100)
        ]
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        return BenchmarkResult(
            test_name="load_test",
            duration_seconds=total_duration,
            memory_peak_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_percent_avg=psutil.cpu_percent(),
            iterations=total_requests,
            success_rate=successful_requests / total_requests,
            throughput_ops_per_sec=successful_requests / total_duration,
            latency_p50_ms=statistics.median(latencies),
            latency_p95_ms=statistics.quantiles(latencies, n=20)[18],
            latency_p99_ms=statistics.quantiles(latencies, n=100)[98],
            timestamp=datetime.now().isoformat(),
            metadata={
                'test_type': 'load_test',
                'concurrent_users': concurrent_users,
                'total_requests': total_requests,
                'successful_requests': successful_requests
            }
        )
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            logger.warning("No benchmark results available")
            return {}
            
        # Convert results to DataFrame for analysis
        results_data = [asdict(result) for result in self.results]
        df = pd.DataFrame(results_data)
        
        # Calculate summary statistics
        summary = {
            'timestamp': self.timestamp,
            'total_tests': len(self.results),
            'test_overview': {
                'tests_completed': len(self.results),
                'avg_success_rate': df['success_rate'].mean(),
                'total_duration': df['duration_seconds'].sum(),
                'peak_memory_usage_mb': df['memory_peak_mb'].max(),
                'avg_cpu_usage': df['cpu_percent_avg'].mean()
            },
            'performance_metrics': {
                'highest_throughput': {
                    'test': df.loc[df['throughput_ops_per_sec'].idxmax(), 'test_name'],
                    'value': df['throughput_ops_per_sec'].max()
                },
                'lowest_latency_p50': {
                    'test': df.loc[df['latency_p50_ms'].idxmin(), 'test_name'],
                    'value': df['latency_p50_ms'].min()
                },
                'memory_efficiency': {
                    'lowest_memory_test': df.loc[df['memory_peak_mb'].idxmin(), 'test_name'],
                    'value': df['memory_peak_mb'].min()
                }
            },
            'recommendations': self._generate_recommendations(df),
            'detailed_results': results_data
        }
        
        return summary
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        if df['memory_peak_mb'].max() > 1000:
            recommendations.append("High memory usage detected. Consider memory optimization techniques.")
            
        # CPU recommendations  
        if df['cpu_percent_avg'].mean() > 80:
            recommendations.append("High CPU usage detected. Consider performance optimizations.")
            
        # Latency recommendations
        if df['latency_p95_ms'].mean() > 500:
            recommendations.append("High latency detected. Review async operations and caching strategies.")
            
        # Success rate recommendations
        if df['success_rate'].min() < 0.95:
            recommendations.append("Low success rate detected. Review error handling and retry mechanisms.")
            
        # Throughput recommendations
        if df['throughput_ops_per_sec'].mean() < 10:
            recommendations.append("Low throughput detected. Consider parallelization and optimization.")
            
        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges.")
            
        return recommendations
    
    def save_results(self) -> str:
        """Save benchmark results to files"""
        report = self.generate_performance_report()
        
        # Save JSON report
        json_file = self.output_dir / f"performance_report_{self.timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save CSV for detailed analysis
        if self.results:
            csv_file = self.output_dir / f"performance_data_{self.timestamp}.csv"
            results_data = [asdict(result) for result in self.results]
            df = pd.DataFrame(results_data)
            df.to_csv(csv_file, index=False)
            
        logger.info(f"Performance results saved to {self.output_dir}")
        return str(json_file)
    
    def create_visualizations(self):
        """Create performance visualization charts"""
        if not self.results:
            logger.warning("No results available for visualization")
            return
            
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        results_data = [asdict(result) for result in self.results]
        df = pd.DataFrame(results_data)
        
        # Throughput comparison
        ax1.bar(df['test_name'], df['throughput_ops_per_sec'])
        ax1.set_title('Throughput Comparison (ops/sec)')
        ax1.set_xlabel('Test Name')
        ax1.set_ylabel('Operations per Second')
        ax1.tick_params(axis='x', rotation=45)
        
        # Latency distribution
        latency_data = df[['test_name', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms']]
        latency_data.set_index('test_name').plot(kind='bar', ax=ax2)
        ax2.set_title('Latency Distribution (ms)')
        ax2.set_xlabel('Test Name')
        ax2.set_ylabel('Latency (ms)')
        ax2.legend(['P50', 'P95', 'P99'])
        ax2.tick_params(axis='x', rotation=45)
        
        # Memory usage
        ax3.bar(df['test_name'], df['memory_peak_mb'])
        ax3.set_title('Peak Memory Usage (MB)')
        ax3.set_xlabel('Test Name')
        ax3.set_ylabel('Memory (MB)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Success rate
        ax4.bar(df['test_name'], df['success_rate'] * 100)
        ax4.set_title('Success Rate (%)')
        ax4.set_xlabel('Test Name')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_ylim(0, 100)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.output_dir / f"performance_charts_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance charts saved to {plot_file}")
    
    async def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        logger.info("Starting comprehensive performance benchmark suite")
        
        # Run all benchmark tests
        self.results.append(await self.benchmark_agent_initialization(iterations=100))
        self.results.append(await self.benchmark_vulnerability_simulation(iterations=50))
        self.results.append(await self.benchmark_adversarial_training(iterations=20))
        self.results.append(self.benchmark_memory_usage(duration=10))  # Shorter for demo
        self.results.append(self.run_load_test(concurrent_users=5, duration=10))  # Shorter for demo
        
        # Generate reports and visualizations
        report_file = self.save_results()
        self.create_visualizations()
        
        logger.info(f"Benchmark suite completed. Report: {report_file}")
        
        # Print summary
        report = self.generate_performance_report()
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*50)
        print(f"Tests completed: {report['test_overview']['tests_completed']}")
        print(f"Average success rate: {report['test_overview']['avg_success_rate']:.2%}")
        print(f"Peak memory usage: {report['test_overview']['peak_memory_usage_mb']:.1f} MB")
        print(f"Average CPU usage: {report['test_overview']['avg_cpu_usage']:.1f}%")
        print(f"Total duration: {report['test_overview']['total_duration']:.1f} seconds")
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        print("="*50)

async def main():
    """Main benchmark execution function"""
    benchmark = PerformanceBenchmark()
    await benchmark.run_all_benchmarks()

if __name__ == "__main__":
    asyncio.run(main())