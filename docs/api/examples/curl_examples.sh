#!/bin/bash
# cURL examples for GAN Cyber Range API

BASE_URL="http://localhost:8000"
API_KEY="your-api-key-here"  # Replace with your actual API key

# Health check
echo "🏥 Health Check"
curl -X GET "$BASE_URL/health" \
  -H "Authorization: Bearer $API_KEY"

echo -e "\n\n"

# Create simulation
echo "🚀 Creating Simulation"
SIMULATION_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/simulations" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_config": {
      "vulnerable_services": ["webapp", "database"],
      "network_topology": "multi-tier",
      "difficulty": "medium", 
      "duration_hours": 1.0,
      "realtime_factor": 60
    },
    "red_team_config": {
      "name": "CurlRedTeam",
      "llm_model": "gpt-4",
      "skill_level": "advanced",
      "tools": ["nmap", "metasploit"]
    },
    "blue_team_config": {
      "name": "CurlBlueTeam",
      "llm_model": "claude-3", 
      "skill_level": "advanced",
      "defense_strategy": "proactive"
    }
  }')

echo $SIMULATION_RESPONSE | jq .

# Extract simulation ID
SIMULATION_ID=$(echo $SIMULATION_RESPONSE | jq -r '.simulation_id')
echo "📝 Simulation ID: $SIMULATION_ID"

echo -e "\n\n"

# Get simulation status
echo "📊 Getting Simulation Status"
curl -X GET "$BASE_URL/api/v1/simulations/$SIMULATION_ID/status" \
  -H "Authorization: Bearer $API_KEY" | jq .

echo -e "\n\n"

# List simulations
echo "📋 Listing All Simulations"
curl -X GET "$BASE_URL/api/v1/simulations" \
  -H "Authorization: Bearer $API_KEY" | jq .

echo -e "\n\n"

# Get metrics
echo "📈 Getting System Metrics"
curl -X GET "$BASE_URL/metrics" \
  -H "Authorization: Bearer $API_KEY" | jq .

echo -e "\n\n"

# Stop simulation (optional)
echo "🛑 Stopping Simulation"
curl -X POST "$BASE_URL/api/v1/simulations/$SIMULATION_ID/stop" \
  -H "Authorization: Bearer $API_KEY" | jq .
