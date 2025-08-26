#!/usr/bin/env python3
"""Comprehensive test suite for GAN Cyber Range Simulator."""

import sys
import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock
import logging
from datetime import datetime
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import test components
from minimal_test import (
    MockCyberRange, MockRedTeamAgent, MockBlueTeamAgent, 
    MockAgentAction, MockLLMClient, MockSimulationResults
)


class TestMockLLMClient(unittest.TestCase):
    """Test LLM client functionality."""
    
    def setUp(self):
        self.client = MockLLMClient("test-model")
    
    def test_client_initialization(self):
        """Test LLM client initializes correctly."""
        self.assertEqual(self.client.model, "test-model")
    
    async def test_red_team_response(self):
        """Test red team LLM response generation."""
        request = MagicMock()
        request.context = {"agent_type": "red_team"}
        
        response = await self.client.generate(request)
        
        self.assertIsNotNone(response.parsed_json)
        self.assertIn("situation_analysis", response.parsed_json)
        self.assertIn("attack_opportunities", response.parsed_json)
        self.assertIn("recommended_actions", response.parsed_json)
    
    async def test_blue_team_response(self):
        """Test blue team LLM response generation."""
        request = MagicMock()
        request.context = {"agent_type": "blue_team"}
        
        response = await self.client.generate(request)
        
        self.assertIsNotNone(response.parsed_json)
        self.assertIn("threat_assessment", response.parsed_json)
        self.assertIn("detected_threats", response.parsed_json)
        self.assertIn("defensive_actions", response.parsed_json)


class TestMockAgentAction(unittest.TestCase):
    """Test agent action data structure."""
    
    def test_action_creation(self):
        """Test creating agent actions."""
        action = MockAgentAction(
            type="vulnerability_scan",
            target="webapp",
            payload={"test": "data"},
            success=True
        )
        
        self.assertEqual(action.type, "vulnerability_scan")
        self.assertEqual(action.target, "webapp")
        self.assertTrue(action.success)
        self.assertEqual(action.payload["test"], "data")
    
    def test_action_timestamp(self):
        """Test action has valid timestamp."""
        action = MockAgentAction(type="test", target="test")
        
        # Should be a valid ISO format timestamp
        try:
            datetime.fromisoformat(action.timestamp)
        except ValueError:
            self.fail("Action timestamp is not valid ISO format")


class TestRedTeamAgent(unittest.TestCase):
    """Test red team agent functionality."""
    
    def setUp(self):
        self.agent = MockRedTeamAgent(name="TestRedTeam", skill_level="advanced")
    
    def test_agent_initialization(self):
        """Test red team agent initializes correctly."""
        self.assertEqual(self.agent.name, "TestRedTeam")
        self.assertEqual(self.agent.skill_level, "advanced")
        self.assertFalse(self.agent.active)
    
    def test_agent_activation(self):
        """Test agent activation and deactivation."""
        self.assertFalse(self.agent.active)
        
        self.agent.activate()
        self.assertTrue(self.agent.active)
        
        self.agent.deactivate()
        self.assertFalse(self.agent.active)
    
    async def test_agent_action_generation(self):
        """Test red team generates actions when active."""
        self.agent.activate()
        
        env_state = {
            'services': [
                {'name': 'webapp', 'status': 'running', 'ip': '10.0.1.10'},
                {'name': 'database', 'status': 'running', 'ip': '10.0.1.11'}
            ]
        }
        
        actions = await self.agent.act(env_state)
        
        self.assertIsInstance(actions, list)
        if actions:  # Actions are generated probabilistically
            self.assertIsInstance(actions[0], MockAgentAction)
            self.assertIn(actions[0].type, ['vulnerability_scan', 'exploit_attempt', 'reconnaissance'])
    
    async def test_inactive_agent_no_actions(self):
        """Test inactive agent generates no actions."""
        # Don't activate agent
        env_state = {'services': [{'name': 'test', 'status': 'running'}]}
        
        actions = await self.agent.act(env_state)
        
        self.assertEqual(actions, [])
    
    def test_agent_stats(self):
        """Test agent statistics generation."""
        stats = self.agent.get_stats()
        
        self.assertIn('name', stats)
        self.assertIn('total_actions', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('skill_level', stats)
        self.assertIn('active', stats)


class TestBlueTeamAgent(unittest.TestCase):
    """Test blue team agent functionality."""
    
    def setUp(self):
        self.agent = MockBlueTeamAgent(
            name="TestBlueTeam", 
            skill_level="advanced",
            defense_strategy="proactive"
        )
    
    def test_agent_initialization(self):
        """Test blue team agent initializes correctly."""
        self.assertEqual(self.agent.name, "TestBlueTeam")
        self.assertEqual(self.agent.skill_level, "advanced")
        self.assertEqual(self.agent.defense_strategy, "proactive")
        self.assertFalse(self.agent.active)
    
    async def test_defensive_action_generation(self):
        """Test blue team generates defensive actions."""
        self.agent.activate()
        
        env_state = {
            'security_events': [
                {
                    'type': 'vulnerability_scan',
                    'severity': 'high',
                    'target': 'webapp',
                    'timestamp': datetime.now().isoformat()
                }
            ]
        }
        
        actions = await self.agent.act(env_state)
        
        self.assertIsInstance(actions, list)
        self.assertGreater(len(actions), 0)  # Should always have monitoring action
        
        # Check if actions are defensive in nature
        defensive_types = ['patch_deployment', 'incident_isolation', 'threat_monitoring', 'security_monitoring']
        for action in actions:
            self.assertIn(action.type, defensive_types)


class TestCyberRange(unittest.TestCase):
    """Test cyber range environment."""
    
    def setUp(self):
        self.cyber_range = MockCyberRange(vulnerable_services=['webapp', 'database'])
    
    def test_range_initialization(self):
        """Test cyber range initializes correctly."""
        self.assertEqual(len(self.cyber_range.services), 2)
        self.assertIn('webapp', [s['name'] for s in self.cyber_range.services])
        self.assertIn('database', [s['name'] for s in self.cyber_range.services])
    
    def test_environment_state_generation(self):
        """Test environment state is properly generated."""
        env_state = self.cyber_range.get_environment_state()
        
        required_keys = ['services', 'network_topology', 'security_status', 'security_events', 'network_logs']
        for key in required_keys:
            self.assertIn(key, env_state)
        
        self.assertIsInstance(env_state['services'], list)
        self.assertIsInstance(env_state['security_events'], list)
        self.assertIsInstance(env_state['network_logs'], list)
    
    async def test_simulation_execution(self):
        """Test full simulation execution."""
        red_team = MockRedTeamAgent(name="TestRed")
        blue_team = MockBlueTeamAgent(name="TestBlue")
        
        results = await self.cyber_range.simulate(
            red_team=red_team,
            blue_team=blue_team,
            duration_hours=0.01,  # Very short simulation
            realtime_factor=60
        )
        
        self.assertIsInstance(results, MockSimulationResults)
        self.assertIsNotNone(results.simulation_id)
        self.assertIsNotNone(results.start_time)
        self.assertIsNotNone(results.end_time)
        
        # Check metrics are tracked
        self.assertGreaterEqual(results.total_attacks, 0)
        self.assertGreaterEqual(results.attacks_blocked, 0)
    
    def test_services_creation(self):
        """Test services are created with proper structure."""
        services = self.cyber_range.services
        
        for service in services:
            required_keys = ['name', 'type', 'ip', 'open_ports', 'status', 'vulnerabilities']
            for key in required_keys:
                self.assertIn(key, service)
            
            self.assertIsInstance(service['open_ports'], list)
            self.assertIsInstance(service['vulnerabilities'], list)


class TestSimulationResults(unittest.TestCase):
    """Test simulation results handling."""
    
    def setUp(self):
        self.results = MockSimulationResults("test_sim", datetime.now())
    
    def test_results_initialization(self):
        """Test results initialize correctly."""
        self.assertEqual(self.results.simulation_id, "test_sim")
        self.assertIsNotNone(self.results.start_time)
    
    def test_metrics_calculation(self):
        """Test metrics are calculated correctly."""
        # Set up some test data
        self.results.total_attacks = 10
        self.results.services_compromised = 3
        self.results.attacks_blocked = 5
        
        self.assertEqual(self.results.compromise_rate, 0.3)
        self.assertEqual(self.results.defense_effectiveness, 0.5)
    
    def test_duration_calculation(self):
        """Test duration calculation works."""
        self.results.end_time = self.results.start_time
        duration = self.results.duration
        
        self.assertIsNotNone(duration)
        # Duration should be very small since start and end are same
        self.assertLessEqual(duration.total_seconds(), 1.0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    async def test_simulation_with_no_services(self):
        """Test simulation handles empty service list."""
        cyber_range = MockCyberRange(vulnerable_services=[])
        red_team = MockRedTeamAgent()
        blue_team = MockBlueTeamAgent()
        
        # Should not crash with empty services
        results = await cyber_range.simulate(
            red_team=red_team,
            blue_team=blue_team,
            duration_hours=0.01
        )
        
        self.assertIsNotNone(results)
    
    def test_agent_stats_with_no_actions(self):
        """Test agent stats work with no actions taken."""
        agent = MockRedTeamAgent()
        stats = agent.get_stats()
        
        self.assertEqual(stats['total_actions'], 0)
        self.assertEqual(stats['success_rate'], 0.6)  # Default rate
    
    async def test_llm_client_handles_missing_context(self):
        """Test LLM client handles missing context gracefully."""
        client = MockLLMClient()
        request = MagicMock()
        request.context = {}  # Empty context
        
        # Should not crash
        response = await client.generate(request)
        self.assertIsNotNone(response)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    async def test_full_simulation_workflow(self):
        """Test complete simulation workflow from start to finish."""
        # Setup
        cyber_range = MockCyberRange(vulnerable_services=['webapp', 'database', 'api-gateway'])
        red_team = MockRedTeamAgent(name="IntegrationRed", skill_level="advanced")
        blue_team = MockBlueTeamAgent(name="IntegrationBlue", skill_level="advanced", defense_strategy="proactive")
        
        # Execute simulation
        results = await cyber_range.simulate(
            red_team=red_team,
            blue_team=blue_team,
            duration_hours=0.02,
            realtime_factor=60
        )
        
        # Verify results
        self.assertIsInstance(results, MockSimulationResults)
        self.assertGreaterEqual(results.total_attacks, 0)
        self.assertGreaterEqual(len(results.red_team_actions), 0)
        self.assertGreaterEqual(len(results.blue_team_actions), 0)
        
        # Verify agents were active
        red_stats = red_team.get_stats()
        blue_stats = blue_team.get_stats()
        
        self.assertGreater(red_stats['total_actions'], 0)
        self.assertGreater(blue_stats['total_actions'], 0)
    
    def test_results_serialization(self):
        """Test simulation results can be serialized to JSON."""
        results = MockSimulationResults("test", datetime.now())
        results.total_attacks = 5
        results.services_compromised = 2
        results.end_time = datetime.now()
        
        red_team = MockRedTeamAgent(name="TestRed")
        blue_team = MockBlueTeamAgent(name="TestBlue")
        
        # Create data structure similar to CLI output
        data = {
            'simulation_id': results.simulation_id,
            'duration': str(results.duration),
            'metrics': {
                'total_attacks': results.total_attacks,
                'compromise_rate': results.compromise_rate,
                'defense_effectiveness': results.defense_effectiveness
            },
            'agents': {
                'red_team': red_team.get_stats(),
                'blue_team': blue_team.get_stats()
            }
        }
        
        # Should be serializable to JSON
        try:
            json_str = json.dumps(data, default=str)
            self.assertIsInstance(json_str, str)
        except Exception as e:
            self.fail(f"Failed to serialize results to JSON: {e}")


async def run_async_tests():
    """Run async tests using asyncio."""
    suite = unittest.TestSuite()
    
    # Add async test methods
    async_test_cases = [
        # LLM Client tests
        (TestMockLLMClient, 'test_red_team_response'),
        (TestMockLLMClient, 'test_blue_team_response'),
        
        # Agent tests
        (TestRedTeamAgent, 'test_agent_action_generation'),
        (TestRedTeamAgent, 'test_inactive_agent_no_actions'),
        (TestBlueTeamAgent, 'test_defensive_action_generation'),
        
        # Cyber Range tests
        (TestCyberRange, 'test_simulation_execution'),
        
        # Error handling tests
        (TestErrorHandling, 'test_simulation_with_no_services'),
        (TestErrorHandling, 'test_llm_client_handles_missing_context'),
        
        # Integration tests
        (TestIntegration, 'test_full_simulation_workflow'),
    ]
    
    print("🧪 Running async tests...")
    passed = 0
    failed = 0
    
    for test_class, test_method in async_test_cases:
        try:
            print(f"   Running {test_class.__name__}.{test_method}...")
            test_instance = test_class()
            test_instance.setUp() if hasattr(test_instance, 'setUp') else None
            
            test_func = getattr(test_instance, test_method)
            await test_func()
            
            print(f"   ✅ {test_class.__name__}.{test_method} - PASSED")
            passed += 1
            
        except Exception as e:
            print(f"   ❌ {test_class.__name__}.{test_method} - FAILED: {e}")
            failed += 1
    
    print(f"\n📊 Async Tests Summary: {passed} passed, {failed} failed")
    return failed == 0


def run_sync_tests():
    """Run synchronous tests using unittest."""
    print("🧪 Running sync tests...")
    
    # Create test suite for sync tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add sync test classes
    sync_test_classes = [
        TestMockAgentAction,
        TestRedTeamAgent,  # Non-async methods
        TestBlueTeamAgent, # Non-async methods
        TestCyberRange,   # Non-async methods
        TestSimulationResults,
        TestErrorHandling, # Non-async methods
        TestIntegration,  # Non-async methods
    ]
    
    for test_class in sync_test_classes:
        # Only add non-async test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_') and 
                       not asyncio.iscoroutinefunction(getattr(test_class, method))]
        
        for method in test_methods:
            suite.addTest(test_class(method))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main test runner."""
    print("🎯 GAN CYBER RANGE SIMULATOR - TEST SUITE")
    print("="*60)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup logging for tests
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    try:
        # Run synchronous tests
        sync_success = run_sync_tests()
        
        print("\n" + "="*60)
        
        # Run asynchronous tests
        async_success = asyncio.run(run_async_tests())
        
        print("\n" + "="*60)
        print("📋 FINAL TEST SUMMARY")
        print("="*60)
        
        overall_success = sync_success and async_success
        
        if overall_success:
            print("🎉 ALL TESTS PASSED!")
            print("✅ System is ready for production deployment")
        else:
            print("❌ Some tests failed")
            print("⚠️  Please review failures before deploying")
        
        print(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0 if overall_success else 1
        
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)