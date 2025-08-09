"""
Python examples for GAN Cyber Range API
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"  # Replace with your actual API key

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def create_simulation():
    """Create a new cyber range simulation."""
    
    simulation_request = {
        "simulation_config": {
            "vulnerable_services": ["webapp", "database"],
            "network_topology": "multi-tier", 
            "difficulty": "medium",
            "duration_hours": 1.0,
            "realtime_factor": 60
        },
        "red_team_config": {
            "name": "PythonRedTeam",
            "llm_model": "gpt-4",
            "skill_level": "advanced",
            "tools": ["nmap", "metasploit"]
        },
        "blue_team_config": {
            "name": "PythonBlueTeam", 
            "llm_model": "claude-3",
            "skill_level": "advanced",
            "defense_strategy": "proactive"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/simulations",
        headers=headers,
        json=simulation_request
    )
    
    if response.status_code == 201:
        simulation_data = response.json()
        print(f"✅ Simulation created: {simulation_data['simulation_id']}")
        return simulation_data['simulation_id']
    else:
        print(f"❌ Failed to create simulation: {response.text}")
        return None

def get_simulation_status(simulation_id: str):
    """Get simulation status and metrics."""
    
    response = requests.get(
        f"{BASE_URL}/api/v1/simulations/{simulation_id}/status",
        headers=headers
    )
    
    if response.status_code == 200:
        status_data = response.json()
        print(f"📊 Simulation Status: {status_data['status']}")
        print(f"📈 Progress: {status_data.get('progress', {})}")
        return status_data
    else:
        print(f"❌ Failed to get status: {response.text}")
        return None

def monitor_simulation(simulation_id: str, duration_minutes: int = 10):
    """Monitor simulation for specified duration."""
    import time
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    print(f"🔍 Monitoring simulation {simulation_id} for {duration_minutes} minutes...")
    
    while time.time() < end_time:
        status = get_simulation_status(simulation_id)
        
        if status and not status.get('is_running', False):
            print("✅ Simulation completed!")
            break
        
        time.sleep(30)  # Check every 30 seconds
    
    # Get final results
    get_simulation_results(simulation_id)

def get_simulation_results(simulation_id: str):
    """Get final simulation results."""
    
    response = requests.get(
        f"{BASE_URL}/api/v1/simulations/{simulation_id}/results",
        headers=headers
    )
    
    if response.status_code == 200:
        results = response.json()
        print("🏆 Final Results:")
        print(f"   Compromise Rate: {results.get('compromise_rate', 0):.2%}")
        print(f"   Detection Rate: {results.get('detection_rate', 0):.2%}")
        print(f"   Mean Time to Detection: {results.get('avg_detection_time', 0):.1f}s")
        return results
    else:
        print(f"❌ Failed to get results: {response.text}")
        return None

def list_simulations():
    """List all simulations."""
    
    response = requests.get(f"{BASE_URL}/api/v1/simulations", headers=headers)
    
    if response.status_code == 200:
        simulations = response.json()
        print(f"📋 Active Simulations: {simulations['total_active']}")
        print(f"📋 Completed Simulations: {simulations['total_completed']}")
        return simulations
    else:
        print(f"❌ Failed to list simulations: {response.text}")
        return None

if __name__ == "__main__":
    # Example workflow
    print("🚀 Starting GAN Cyber Range API Example")
    
    # Create simulation
    sim_id = create_simulation()
    
    if sim_id:
        # Monitor simulation
        monitor_simulation(sim_id, duration_minutes=5)
        
        # List all simulations
        list_simulations()
