"""
Locust performance testing configuration for GAN Cyber Range Simulator.

This file defines load testing scenarios using Locust to validate system
performance under various load conditions.

Usage:
    locust -f tests/performance/locustfile.py --host=http://localhost:8080
    locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 60s
"""

from locust import HttpUser, task, between
import json
import random
import time


class CyberRangeUser(HttpUser):
    """Simulate a typical user interacting with the cyber range."""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        self.auth_token = None
        self.simulation_id = None
        self.agent_ids = []
        
        # Authenticate user
        self.authenticate()
    
    def authenticate(self):
        """Authenticate the user and get access token."""
        response = self.client.post("/api/v1/auth/login", json={
            "username": f"test_user_{random.randint(1000, 9999)}",
            "password": "test_password"
        })
        
        if response.status_code == 200:
            self.auth_token = response.json().get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.auth_token}"
            })
    
    @task(3)
    def view_dashboard(self):
        """View the main dashboard."""
        self.client.get("/api/v1/dashboard")
    
    @task(2)
    def list_scenarios(self):
        """List available scenarios."""
        self.client.get("/api/v1/scenarios")
    
    @task(2)
    def get_agent_status(self):
        """Check agent status."""
        self.client.get("/api/v1/agents/status")
    
    @task(1)
    def create_simulation(self):
        """Create a new simulation."""
        scenario_data = {
            "name": f"Load_Test_Simulation_{random.randint(1000, 9999)}",
            "scenario_type": "basic_web_attack",
            "red_team_model": "mock",
            "blue_team_model": "mock",
            "duration": 300,
            "difficulty": "beginner"
        }
        
        response = self.client.post("/api/v1/simulations", json=scenario_data)
        
        if response.status_code == 201:
            self.simulation_id = response.json().get("id")
    
    @task(1)
    def start_simulation(self):
        """Start a simulation if one exists."""
        if self.simulation_id:
            response = self.client.post(f"/api/v1/simulations/{self.simulation_id}/start")
            
            if response.status_code != 200:
                self.simulation_id = None
    
    @task(2)
    def get_simulation_status(self):
        """Check simulation status."""
        if self.simulation_id:
            response = self.client.get(f"/api/v1/simulations/{self.simulation_id}/status")
            
            if response.status_code == 404:
                self.simulation_id = None
    
    @task(1)
    def get_metrics(self):
        """Retrieve system metrics."""
        self.client.get("/api/v1/metrics/system")
    
    @task(1)
    def create_agent(self):
        """Create a new AI agent."""
        agent_data = {
            "name": f"Test_Agent_{random.randint(1000, 9999)}",
            "type": random.choice(["red_team", "blue_team"]),
            "model": "mock",
            "configuration": {
                "skill_level": random.choice(["beginner", "intermediate", "advanced"]),
                "tools": ["mock_tool_1", "mock_tool_2"]
            }
        }
        
        response = self.client.post("/api/v1/agents", json=agent_data)
        
        if response.status_code == 201:
            agent_id = response.json().get("id")
            self.agent_ids.append(agent_id)
    
    @task(1)
    def get_agent_details(self):
        """Get details for an agent."""
        if self.agent_ids:
            agent_id = random.choice(self.agent_ids)
            response = self.client.get(f"/api/v1/agents/{agent_id}")
            
            if response.status_code == 404:
                self.agent_ids.remove(agent_id)
    
    def on_stop(self):
        """Cleanup when user session ends."""
        # Stop simulation if running
        if self.simulation_id:
            self.client.post(f"/api/v1/simulations/{self.simulation_id}/stop")
        
        # Delete created agents
        for agent_id in self.agent_ids:
            self.client.delete(f"/api/v1/agents/{agent_id}")


class AdminUser(HttpUser):
    """Simulate administrator operations."""
    
    wait_time = between(2, 8)
    weight = 1  # Lower weight = fewer admin users
    
    def on_start(self):
        """Initialize admin session."""
        self.auth_token = None
        self.authenticate_admin()
    
    def authenticate_admin(self):
        """Authenticate as admin user."""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin_password"
        })
        
        if response.status_code == 200:
            self.auth_token = response.json().get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.auth_token}"
            })
    
    @task(3)
    def view_admin_dashboard(self):
        """View admin dashboard."""
        self.client.get("/api/v1/admin/dashboard")
    
    @task(2)
    def get_system_health(self):
        """Check system health metrics."""
        self.client.get("/api/v1/admin/health")
    
    @task(2)
    def list_all_simulations(self):
        """List all simulations."""
        self.client.get("/api/v1/admin/simulations")
    
    @task(1)
    def get_resource_usage(self):
        """Check resource usage."""
        self.client.get("/api/v1/admin/resources")
    
    @task(1)
    def get_security_logs(self):
        """View security audit logs."""
        self.client.get("/api/v1/admin/security/logs")
    
    @task(1)
    def update_system_config(self):
        """Update system configuration."""
        config_data = {
            "max_concurrent_simulations": random.randint(5, 20),
            "default_timeout": random.randint(300, 3600),
            "log_level": random.choice(["INFO", "DEBUG", "WARNING"])
        }
        
        self.client.put("/api/v1/admin/config", json=config_data)


class ResearcherUser(HttpUser):
    """Simulate researcher/academic user operations."""
    
    wait_time = between(5, 15)
    weight = 2  # Medium weight
    
    def on_start(self):
        """Initialize researcher session."""
        self.auth_token = None
        self.experiment_id = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate researcher."""
        response = self.client.post("/api/v1/auth/login", json={
            "username": f"researcher_{random.randint(100, 999)}",
            "password": "research_password"
        })
        
        if response.status_code == 200:
            self.auth_token = response.json().get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.auth_token}"
            })
    
    @task(3)
    def browse_research_data(self):
        """Browse available research datasets."""
        self.client.get("/api/v1/research/datasets")
    
    @task(2)
    def create_experiment(self):
        """Create a research experiment."""
        experiment_data = {
            "name": f"Research_Experiment_{random.randint(1000, 9999)}",
            "description": "Load testing experiment",
            "parameters": {
                "duration": random.randint(600, 3600),
                "agents": random.randint(2, 10),
                "scenarios": random.randint(1, 5)
            }
        }
        
        response = self.client.post("/api/v1/research/experiments", json=experiment_data)
        
        if response.status_code == 201:
            self.experiment_id = response.json().get("id")
    
    @task(2)
    def get_experiment_results(self):
        """Retrieve experiment results."""
        if self.experiment_id:
            response = self.client.get(f"/api/v1/research/experiments/{self.experiment_id}/results")
            
            if response.status_code == 404:
                self.experiment_id = None
    
    @task(1)
    def download_dataset(self):
        """Download research dataset."""
        self.client.get("/api/v1/research/datasets/sample/download")
    
    @task(1)
    def export_results(self):
        """Export experiment results."""
        if self.experiment_id:
            export_params = {
                "format": random.choice(["json", "csv", "xml"]),
                "include_raw_data": random.choice([True, False])
            }
            
            self.client.post(f"/api/v1/research/experiments/{self.experiment_id}/export", json=export_params)


class WebSocketUser(HttpUser):
    """Test WebSocket connections for real-time updates."""
    
    wait_time = between(10, 30)
    weight = 1  # Lower weight for WebSocket users
    
    def on_start(self):
        """Initialize WebSocket connections."""
        # Note: Locust doesn't natively support WebSocket testing
        # This would require additional libraries like websocket-client
        pass
    
    @task
    def simulate_websocket_connection(self):
        """Simulate WebSocket connection load."""
        # Simulate WebSocket handshake
        response = self.client.get("/ws/dashboard", headers={
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Key": "test-key",
            "Sec-WebSocket-Version": "13"
        })
        
        # Check if upgrade is successful
        if response.status_code == 101:
            # Simulate maintaining connection
            time.sleep(random.uniform(5, 20))


# Custom test scenarios for specific load patterns
class BurstLoadUser(HttpUser):
    """Simulate burst load patterns."""
    
    wait_time = between(0.1, 1)  # Very short wait times
    weight = 1
    
    @task
    def burst_requests(self):
        """Generate burst of requests."""
        endpoints = [
            "/api/v1/health",
            "/api/v1/status", 
            "/api/v1/metrics/quick",
            "/api/v1/agents/count"
        ]
        
        # Make several rapid requests
        for _ in range(random.randint(3, 8)):
            endpoint = random.choice(endpoints)
            self.client.get(endpoint)
            time.sleep(0.1)  # Very short delay between requests


# Configuration for different test scenarios
def create_load_test_config():
    """Create configuration for load testing."""
    return {
        "normal_load": {
            "users": 10,
            "spawn_rate": 2,
            "duration": "5m"
        },
        "peak_load": {
            "users": 50,
            "spawn_rate": 5,
            "duration": "10m"
        },
        "stress_test": {
            "users": 100,
            "spawn_rate": 10,
            "duration": "15m"
        },
        "spike_test": {
            "users": 200,
            "spawn_rate": 50,
            "duration": "2m"
        }
    }


if __name__ == "__main__":
    # Print usage information
    print("Locust Performance Testing for GAN Cyber Range")
    print("=" * 50)
    print("Usage examples:")
    print("  locust -f locustfile.py --host=http://localhost:8080")
    print("  locust -f locustfile.py --headless -u 10 -r 2 -t 60s")
    print("  locust -f locustfile.py --headless -u 50 -r 5 -t 300s --host=http://localhost:8080")
    print("\nTest scenarios:")
    config = create_load_test_config()
    for scenario, params in config.items():
        print(f"  {scenario}: {params}")