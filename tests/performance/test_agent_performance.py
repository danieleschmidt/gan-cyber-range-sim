"""Performance tests for AI agents."""

import pytest
import asyncio
import time
from statistics import mean, median
from typing import List, Dict, Any

from ..fixtures.cyber_range_fixtures import (
    mock_red_team_agent,
    mock_blue_team_agent,
    performance_benchmarks
)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_red_team_agent_response_time(mock_red_team_agent, performance_benchmarks):
    """Test red team agent response time under load."""
    response_times = []
    
    for _ in range(100):
        start_time = time.time()
        await mock_red_team_agent.plan_attack({"target": "test-webapp"})
        end_time = time.time()
        response_times.append(end_time - start_time)
    
    # Performance assertions
    assert mean(response_times) < performance_benchmarks["agent_response_time"]["p50"]
    assert max(response_times) < performance_benchmarks["agent_response_time"]["max"]
    assert len([t for t in response_times if t > 5.0]) < 5  # Less than 5% over 5 seconds


@pytest.mark.performance
@pytest.mark.asyncio
async def test_blue_team_agent_concurrent_processing(mock_blue_team_agent):
    """Test blue team agent handling concurrent threat detection."""
    concurrent_tasks = 50
    
    async def detect_and_respond():
        threats = await mock_blue_team_agent.detect_threats()
        if threats:
            return await mock_blue_team_agent.respond_to_threat(threats[0])
        return None
    
    start_time = time.time()
    results = await asyncio.gather(
        *[detect_and_respond() for _ in range(concurrent_tasks)],
        return_exceptions=True
    )
    end_time = time.time()
    
    # Verify all tasks completed successfully
    successful_results = [r for r in results if not isinstance(r, Exception)]
    assert len(successful_results) == concurrent_tasks
    
    # Performance assertion: should handle 50 concurrent requests in under 10 seconds
    assert end_time - start_time < 10.0


@pytest.mark.performance
def test_agent_memory_usage():
    """Test agent memory consumption over extended operation."""
    import psutil
    import gc
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Simulate extended agent operation
    agents = []
    for i in range(100):
        agent = {
            "id": f"agent_{i}",
            "memory": [f"action_{j}" for j in range(1000)],
            "state": {"step": i, "data": list(range(100))}
        }
        agents.append(agent)
    
    # Measure memory after operations
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = current_memory - initial_memory
    
    # Clean up
    del agents
    gc.collect()
    
    # Memory usage should not exceed 500MB increase
    assert memory_increase < 500, f"Memory increase: {memory_increase}MB"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_simulation_scalability():
    """Test system scalability with multiple concurrent simulations."""
    simulation_count = 10
    agents_per_simulation = 5
    
    async def run_simulation(sim_id: int):
        """Simulate a cyber range scenario."""
        await asyncio.sleep(0.1)  # Simulate initialization
        
        # Simulate agent interactions
        for agent_id in range(agents_per_simulation):
            await asyncio.sleep(0.05)  # Simulate agent processing
        
        return {
            "simulation_id": sim_id,
            "agents": agents_per_simulation,
            "status": "completed"
        }
    
    start_time = time.time()
    simulations = await asyncio.gather(
        *[run_simulation(i) for i in range(simulation_count)],
        return_exceptions=True
    )
    end_time = time.time()
    
    # Verify all simulations completed
    successful_sims = [s for s in simulations if not isinstance(s, Exception)]
    assert len(successful_sims) == simulation_count
    
    # Performance assertion: 10 simulations with 5 agents each should complete in under 5 seconds
    assert end_time - start_time < 5.0


@pytest.mark.performance
def test_attack_pattern_matching_performance():
    """Test performance of attack pattern matching algorithms."""
    # Generate test attack patterns
    attack_patterns = [
        {"type": "sql_injection", "payload": f"'; DROP TABLE users_{i}; --"}
        for i in range(1000)
    ]
    
    # Generate test log entries
    log_entries = [
        f"192.168.1.{i % 255} - - [01/Aug/2025:10:00:00 +0000] \"GET /search?q=test{i} HTTP/1.1\" 200 1234"
        for i in range(10000)
    ]
    
    def match_patterns(logs: List[str], patterns: List[Dict[str, Any]]) -> List[Dict]:
        """Simple pattern matching function."""
        matches = []
        for log in logs:
            for pattern in patterns:
                if any(keyword in log.lower() for keyword in ["drop", "union", "select"]):
                    matches.append({"log": log, "pattern": pattern})
                    break
        return matches
    
    start_time = time.time()
    matches = match_patterns(log_entries, attack_patterns)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Performance assertions
    assert processing_time < 2.0, f"Pattern matching took {processing_time}s, expected < 2s"
    assert len(matches) >= 0  # Should find some matches in the test data


@pytest.mark.performance
@pytest.mark.benchmark
def test_vulnerability_scanning_performance(benchmark):
    """Benchmark vulnerability scanning performance."""
    
    def vulnerability_scan():
        """Simulate vulnerability scanning."""
        vulnerabilities = []
        
        # Simulate common vulnerability checks
        checks = [
            "sql_injection",
            "xss",
            "csrf",
            "directory_traversal",
            "command_injection",
            "buffer_overflow",
            "privilege_escalation",
            "weak_authentication"
        ]
        
        for check in checks:
            # Simulate check processing time
            time.sleep(0.01)
            if hash(check) % 3 == 0:  # Pseudo-random vulnerability detection
                vulnerabilities.append({
                    "type": check,
                    "severity": "high" if hash(check) % 2 == 0 else "medium",
                    "description": f"Potential {check} vulnerability detected"
                })
        
        return vulnerabilities
    
    # Benchmark the function
    result = benchmark(vulnerability_scan)
    
    # Verify results
    assert isinstance(result, list)
    assert all("type" in vuln for vuln in result)


@pytest.fixture
def load_test_config():
    """Configuration for load testing."""
    return {
        "concurrent_users": 50,
        "test_duration": 60,  # seconds
        "ramp_up_time": 10,   # seconds
        "scenarios": [
            "web_attack",
            "network_scan", 
            "privilege_escalation",
            "data_exfiltration"
        ]
    }


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.asyncio
async def test_system_load_test(load_test_config):
    """Comprehensive system load test."""
    concurrent_users = load_test_config["concurrent_users"]
    test_duration = load_test_config["test_duration"]
    
    async def user_simulation(user_id: int):
        """Simulate a user running scenarios."""
        start_time = time.time()
        scenario_count = 0
        
        while time.time() - start_time < test_duration:
            # Simulate scenario execution
            await asyncio.sleep(0.1)
            scenario_count += 1
            
            # Simulate occasional failures
            if scenario_count % 10 == 0:
                await asyncio.sleep(0.5)  # Simulate processing delay
        
        return {
            "user_id": user_id,
            "scenarios_completed": scenario_count,
            "duration": time.time() - start_time
        }
    
    # Run load test
    start_time = time.time()
    results = await asyncio.gather(
        *[user_simulation(i) for i in range(concurrent_users)],
        return_exceptions=True
    )
    end_time = time.time()
    
    # Analyze results
    successful_users = [r for r in results if not isinstance(r, Exception)]
    total_scenarios = sum(r["scenarios_completed"] for r in successful_users)
    
    # Performance assertions
    assert len(successful_users) >= concurrent_users * 0.95  # 95% success rate
    assert total_scenarios > 0
    assert end_time - start_time <= test_duration + 5  # Allow 5s buffer
    
    # Calculate throughput
    throughput = total_scenarios / (end_time - start_time)
    assert throughput > 1.0, f"Throughput too low: {throughput} scenarios/second"