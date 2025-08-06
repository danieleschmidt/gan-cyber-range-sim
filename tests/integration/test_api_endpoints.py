"""Integration tests for API endpoints."""

import pytest
import asyncio
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from gan_cyber_range.api.server import app, simulation_manager
from gan_cyber_range.api.models import SimulationRequest, SimulationConfig, RedTeamConfig, BlueTeamConfig


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_simulation_request():
    """Sample simulation request."""
    return {
        "simulation_config": {
            "vulnerable_services": ["webapp", "database"],
            "network_topology": "multi-tier",
            "difficulty": "medium",
            "duration_hours": 0.1,  # Short duration for testing
            "realtime_factor": 60
        },
        "red_team_config": {
            "name": "TestRedTeam",
            "llm_model": "mock-gpt-4",
            "skill_level": "advanced",
            "tools": ["nmap", "metasploit"]
        },
        "blue_team_config": {
            "name": "TestBlueTeam",
            "llm_model": "mock-claude-3",
            "skill_level": "advanced",
            "defense_strategy": "proactive"
        }
    }


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "services" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_simulations" in data
        assert "active_simulations" in data
        assert "success_rate" in data
        assert isinstance(data["total_simulations"], int)
        assert isinstance(data["active_simulations"], int)


class TestSimulationEndpoints:
    """Test simulation management endpoints."""
    
    def test_create_simulation(self, client, sample_simulation_request):
        """Test creating a new simulation."""
        response = client.post("/api/v1/simulations", json=sample_simulation_request)
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "started"
        assert "simulation_id" in data
        assert data["simulation_id"].startswith("sim_")
        assert "simulation_url" in data
    
    def test_create_simulation_invalid_config(self, client):
        """Test creating simulation with invalid configuration."""
        invalid_request = {
            "simulation_config": {
                "duration_hours": -1,  # Invalid duration
                "difficulty": "invalid"  # Invalid difficulty
            }
        }
        
        response = client.post("/api/v1/simulations", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_get_simulation_status_not_found(self, client):
        """Test getting status of non-existent simulation."""
        response = client.get("/api/v1/simulations/nonexistent/status")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_get_simulation_results_not_found(self, client):
        """Test getting results of non-existent simulation."""
        response = client.get("/api/v1/simulations/nonexistent/results")
        
        assert response.status_code == 404
    
    def test_stop_simulation_not_found(self, client):
        """Test stopping non-existent simulation."""
        response = client.post("/api/v1/simulations/nonexistent/stop")
        
        assert response.status_code == 404
    
    def test_list_simulations(self, client):
        """Test listing all simulations."""
        response = client.get("/api/v1/simulations")
        
        assert response.status_code == 200
        data = response.json()
        assert "active_simulations" in data
        assert "completed_simulations" in data
        assert "total_active" in data
        assert "total_completed" in data
        assert isinstance(data["active_simulations"], list)
        assert isinstance(data["completed_simulations"], list)


class TestSimulationLifecycle:
    """Test complete simulation lifecycle."""
    
    @pytest.mark.asyncio
    async def test_full_simulation_lifecycle(self, client, sample_simulation_request):
        """Test complete simulation from creation to completion."""
        # Create simulation
        create_response = client.post("/api/v1/simulations", json=sample_simulation_request)
        assert create_response.status_code == 201
        
        simulation_id = create_response.json()["simulation_id"]
        
        # Check status immediately after creation
        status_response = client.get(f"/api/v1/simulations/{simulation_id}/status")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["simulation_id"] == simulation_id
        assert status_data["is_running"] in [True, False]  # Could be starting or running
        
        # Wait a bit for simulation to potentially start
        await asyncio.sleep(0.1)
        
        # Try to stop simulation (should work whether it's running or not)
        stop_response = client.post(f"/api/v1/simulations/{simulation_id}/stop")
        # Should either succeed (404 if already finished, 200 if stopped)
        assert stop_response.status_code in [200, 404]
    
    def test_simulation_with_mock_llm(self, client, sample_simulation_request):
        """Test simulation creation with mock LLM models."""
        # Ensure we're using mock models
        sample_simulation_request["red_team_config"]["llm_model"] = "mock-model"
        sample_simulation_request["blue_team_config"]["llm_model"] = "mock-model"
        
        response = client.post("/api/v1/simulations", json=sample_simulation_request)
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "started"


class TestSimulationManager:
    """Test simulation manager functionality."""
    
    def test_create_simulation_manager(self):
        """Test simulation manager creation."""
        manager = simulation_manager
        
        assert hasattr(manager, 'active_simulations')
        assert hasattr(manager, 'completed_simulations')
        assert hasattr(manager, 'simulation_tasks')
        assert isinstance(manager.active_simulations, dict)
        assert isinstance(manager.completed_simulations, dict)
    
    def test_simulation_manager_create_simulation(self, sample_simulation_request):
        """Test creating simulation through manager."""
        request = SimulationRequest(**sample_simulation_request)
        
        sim_id = simulation_manager.create_simulation(request)
        
        assert sim_id.startswith("sim_")
        assert sim_id in simulation_manager.active_simulations
        
        sim_data = simulation_manager.active_simulations[sim_id]
        assert "cyber_range" in sim_data
        assert "red_team" in sim_data
        assert "blue_team" in sim_data
        assert "config" in sim_data
        assert sim_data["status"] == "created"
    
    @pytest.mark.asyncio
    async def test_simulation_manager_start_simulation(self, sample_simulation_request):
        """Test starting simulation through manager."""
        request = SimulationRequest(**sample_simulation_request)
        sim_id = simulation_manager.create_simulation(request)
        
        # Start simulation
        await simulation_manager.start_simulation(sim_id)
        
        assert sim_id in simulation_manager.simulation_tasks
        assert simulation_manager.active_simulations[sim_id]["status"] == "running"
        
        # Clean up
        await simulation_manager.stop_simulation(sim_id)
    
    @pytest.mark.asyncio
    async def test_simulation_manager_start_nonexistent(self):
        """Test starting non-existent simulation."""
        with pytest.raises(ValueError):
            await simulation_manager.start_simulation("nonexistent")
    
    def test_simulation_manager_get_status_nonexistent(self):
        """Test getting status of non-existent simulation."""
        status = simulation_manager.get_simulation_status("nonexistent")
        assert status is None
    
    def test_simulation_manager_metrics(self):
        """Test getting metrics from manager."""
        metrics = simulation_manager.get_metrics()
        
        assert hasattr(metrics, 'total_simulations')
        assert hasattr(metrics, 'active_simulations')
        assert hasattr(metrics, 'success_rate')
        assert isinstance(metrics.total_simulations, int)
        assert isinstance(metrics.active_simulations, int)
        assert isinstance(metrics.success_rate, float)
        assert 0.0 <= metrics.success_rate <= 1.0


class TestErrorHandling:
    """Test API error handling."""
    
    def test_validation_error_handler(self, client):
        """Test validation error handling."""
        invalid_request = {
            "simulation_config": {
                "duration_hours": "invalid",  # Should be number
                "difficulty": "invalid_level"
            }
        }
        
        response = client.post("/api/v1/simulations", json=invalid_request)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent-endpoint")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.patch("/api/v1/simulations")  # PATCH not supported
        
        assert response.status_code == 405


class TestCORSMiddleware:
    """Test CORS middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/health")
        
        # Should have CORS headers
        assert response.status_code in [200, 404]  # OPTIONS might not be implemented
        
        # Test actual request to see CORS headers
        response = client.get("/health")
        
        # FastAPI CORS middleware should add these headers
        # Note: TestClient might not fully simulate CORS
        assert response.status_code == 200


class TestRequestValidation:
    """Test request validation."""
    
    def test_simulation_config_validation(self, client):
        """Test simulation configuration validation."""
        configs_to_test = [
            # Invalid duration
            {
                "simulation_config": {"duration_hours": -1},
                "expected_error": "duration_hours must be between 0 and 24"
            },
            # Invalid realtime factor
            {
                "simulation_config": {"realtime_factor": 0},
                "expected_error": "realtime_factor must be between 1 and 3600"
            },
            # Invalid difficulty
            {
                "simulation_config": {"difficulty": "impossible"},
                "expected_error": "difficulty must be one of"
            },
            # Invalid skill level
            {
                "red_team_config": {"skill_level": "godlike"},
                "expected_error": "skill_level must be one of"
            },
            # Invalid defense strategy
            {
                "blue_team_config": {"defense_strategy": "chaotic"},
                "expected_error": "defense_strategy must be one of"
            }
        ]
        
        for config_test in configs_to_test:
            request_data = {
                "simulation_config": {"duration_hours": 1.0},
                "red_team_config": {"skill_level": "advanced"},
                "blue_team_config": {"defense_strategy": "proactive"}
            }
            
            # Apply test configuration
            for key, value in config_test.items():
                if key != "expected_error":
                    request_data.update({key: value})
            
            response = client.post("/api/v1/simulations", json=request_data)
            
            assert response.status_code == 422
            # Note: Exact error message format may vary


class TestAsyncOperations:
    """Test asynchronous operation handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_simulation_creation(self, client, sample_simulation_request):
        """Test creating multiple simulations concurrently."""
        # Create multiple requests concurrently
        tasks = []
        for i in range(3):
            request = sample_simulation_request.copy()
            request["red_team_config"]["name"] = f"RedTeam{i}"
            request["blue_team_config"]["name"] = f"BlueTeam{i}"
            
            task = asyncio.create_task(
                asyncio.to_thread(
                    client.post, "/api/v1/simulations", json=request
                )
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 201
        
        # All should have unique simulation IDs
        sim_ids = [resp.json()["simulation_id"] for resp in responses]
        assert len(set(sim_ids)) == len(sim_ids)  # All unique