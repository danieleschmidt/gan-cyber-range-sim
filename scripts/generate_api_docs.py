#!/usr/bin/env python3
"""
Automated API documentation generator for GAN Cyber Range.

This script automatically generates comprehensive API documentation including:
- OpenAPI/Swagger documentation
- Endpoint documentation with examples
- Schema definitions
- Authentication guides
- Usage examples
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.openapi.utils import get_openapi
from gan_cyber_range.api.server import app


def generate_openapi_spec(output_dir: Path) -> dict:
    """Generate OpenAPI specification."""
    openapi_spec = get_openapi(
        title="GAN Cyber Range API",
        version="1.0.0",
        description="""
        # GAN Cyber Range API

        A comprehensive API for managing adversarial cybersecurity simulations using AI agents.

        ## Features

        - **Simulation Management**: Create, monitor, and control cyber range simulations
        - **Agent Configuration**: Configure red team (attacker) and blue team (defender) AI agents
        - **Real-time Monitoring**: Track simulation progress and metrics
        - **Security**: Enterprise-grade authentication and authorization
        - **Performance**: Optimized for high-throughput simulations

        ## Authentication

        This API uses JWT tokens for authentication. Include your token in the Authorization header:

        ```
        Authorization: Bearer <your-jwt-token>
        ```

        ## Rate Limiting

        API calls are rate limited to prevent abuse:
        - 1000 requests per hour for authenticated users
        - 100 requests per hour for unauthenticated users

        ## Response Format

        All responses follow a consistent format:

        ```json
        {
            "status": "success|error",
            "data": {},
            "message": "Human readable message",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        ```

        ## Error Handling

        Standard HTTP status codes are used:
        - 200: Success
        - 201: Created
        - 400: Bad Request
        - 401: Unauthorized
        - 404: Not Found
        - 422: Validation Error
        - 500: Internal Server Error
        """,
        routes=app.routes
    )
    
    # Add additional metadata
    openapi_spec["info"]["x-logo"] = {
        "url": "https://github.com/yourusername/gan-cyber-range-sim/raw/main/docs/logo.png"
    }
    
    openapi_spec["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.gan-cyber-range.org",
            "description": "Production server"
        }
    ]
    
    # Add security schemes
    openapi_spec["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "apiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    # Add examples to schemas
    openapi_spec = add_examples_to_schemas(openapi_spec)
    
    return openapi_spec


def add_examples_to_schemas(openapi_spec: dict) -> dict:
    """Add examples to OpenAPI schemas."""
    if "components" not in openapi_spec:
        openapi_spec["components"] = {}
    
    if "schemas" not in openapi_spec["components"]:
        openapi_spec["components"]["schemas"] = {}
    
    # Add examples for key schemas
    schema_examples = {
        "SimulationRequest": {
            "example": {
                "simulation_config": {
                    "vulnerable_services": ["webapp", "database", "api-gateway"],
                    "network_topology": "multi-tier",
                    "difficulty": "medium",
                    "duration_hours": 2.0,
                    "realtime_factor": 60
                },
                "red_team_config": {
                    "name": "AdvancedRedTeam",
                    "llm_model": "gpt-4",
                    "skill_level": "advanced",
                    "tools": ["nmap", "metasploit", "custom_exploits"]
                },
                "blue_team_config": {
                    "name": "ProactiveBlueTeam",
                    "llm_model": "claude-3",
                    "skill_level": "advanced",
                    "defense_strategy": "proactive"
                }
            }
        },
        "SimulationResponse": {
            "example": {
                "status": "started",
                "simulation_id": "sim_abc123def456",
                "simulation_url": "/api/v1/simulations/sim_abc123def456",
                "estimated_duration_minutes": 120,
                "created_at": "2024-01-01T12:00:00Z"
            }
        },
        "SimulationStatus": {
            "example": {
                "simulation_id": "sim_abc123def456",
                "status": "running",
                "is_running": True,
                "start_time": "2024-01-01T12:00:00Z",
                "elapsed_time_seconds": 1800,
                "progress": {
                    "red_team_actions": 15,
                    "blue_team_responses": 12,
                    "services_compromised": 2,
                    "services_defended": 3,
                    "current_phase": "exploitation"
                },
                "metrics": {
                    "compromise_rate": 0.4,
                    "detection_rate": 0.8,
                    "mean_time_to_detection": 45.0,
                    "mean_time_to_response": 30.0
                }
            }
        }
    }
    
    for schema_name, example_data in schema_examples.items():
        if schema_name in openapi_spec["components"]["schemas"]:
            openapi_spec["components"]["schemas"][schema_name].update(example_data)
    
    return openapi_spec


def generate_api_documentation(output_dir: Path):
    """Generate comprehensive API documentation."""
    
    # Generate OpenAPI spec
    openapi_spec = generate_openapi_spec(output_dir)
    
    # Save OpenAPI spec in JSON and YAML formats
    with open(output_dir / "openapi.json", "w") as f:
        json.dump(openapi_spec, f, indent=2)
    
    with open(output_dir / "openapi.yaml", "w") as f:
        yaml.dump(openapi_spec, f, default_flow_style=False, indent=2)
    
    # Generate markdown documentation
    generate_markdown_docs(openapi_spec, output_dir)
    
    # Generate HTML documentation
    generate_html_docs(openapi_spec, output_dir)
    
    print(f"✅ API documentation generated in {output_dir}")


def generate_markdown_docs(openapi_spec: dict, output_dir: Path):
    """Generate markdown documentation."""
    
    markdown_content = f"""# GAN Cyber Range API Documentation

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{openapi_spec['info']['description']}

## Base URLs

"""
    
    for server in openapi_spec.get('servers', []):
        markdown_content += f"- `{server['url']}` - {server['description']}\n"
    
    markdown_content += "\n## Endpoints\n\n"
    
    # Group endpoints by tags
    endpoints_by_tag = {}
    for path, methods in openapi_spec.get('paths', {}).items():
        for method, details in methods.items():
            if method in ['get', 'post', 'put', 'delete', 'patch']:
                tags = details.get('tags', ['Untagged'])
                for tag in tags:
                    if tag not in endpoints_by_tag:
                        endpoints_by_tag[tag] = []
                    endpoints_by_tag[tag].append({
                        'path': path,
                        'method': method.upper(),
                        'summary': details.get('summary', ''),
                        'description': details.get('description', ''),
                        'operationId': details.get('operationId', '')
                    })
    
    for tag, endpoints in endpoints_by_tag.items():
        markdown_content += f"### {tag}\n\n"
        
        for endpoint in endpoints:
            markdown_content += f"#### {endpoint['method']} {endpoint['path']}\n\n"
            markdown_content += f"**Summary:** {endpoint['summary']}\n\n"
            
            if endpoint['description']:
                markdown_content += f"{endpoint['description']}\n\n"
            
            markdown_content += "---\n\n"
    
    # Add schemas section
    markdown_content += "## Data Models\n\n"
    
    for schema_name, schema_details in openapi_spec.get('components', {}).get('schemas', {}).items():
        markdown_content += f"### {schema_name}\n\n"
        
        if 'description' in schema_details:
            markdown_content += f"{schema_details['description']}\n\n"
        
        if 'example' in schema_details:
            markdown_content += "**Example:**\n\n```json\n"
            markdown_content += json.dumps(schema_details['example'], indent=2)
            markdown_content += "\n```\n\n"
        
        markdown_content += "---\n\n"
    
    with open(output_dir / "API.md", "w") as f:
        f.write(markdown_content)


def generate_html_docs(openapi_spec: dict, output_dir: Path):
    """Generate HTML documentation using Swagger UI."""
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GAN Cyber Range API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.0.0/swagger-ui.css" />
    <style>
        html {
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }
        *, *:before, *:after {
            box-sizing: inherit;
        }
        body {
            margin:0;
            background: #fafafa;
        }
        .swagger-ui .topbar {
            background-color: #1f2937;
        }
        .swagger-ui .topbar .download-url-wrapper .download-url-button {
            background-color: #3b82f6;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    
    <script src="https://unpkg.com/swagger-ui-dist@5.0.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.0.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: './openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                tryItOutEnabled: true,
                requestInterceptor: function(request) {
                    console.log('Request:', request);
                    return request;
                },
                responseInterceptor: function(response) {
                    console.log('Response:', response);
                    return response;
                }
            });
        };
    </script>
</body>
</html>"""
    
    with open(output_dir / "index.html", "w") as f:
        f.write(html_template)


def generate_usage_examples(output_dir: Path):
    """Generate usage examples in multiple languages."""
    
    examples = {
        "python": {
            "filename": "python_examples.py",
            "content": '''"""
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
'''
        },
        "curl": {
            "filename": "curl_examples.sh",
            "content": '''#!/bin/bash
# cURL examples for GAN Cyber Range API

BASE_URL="http://localhost:8000"
API_KEY="your-api-key-here"  # Replace with your actual API key

# Health check
echo "🏥 Health Check"
curl -X GET "$BASE_URL/health" \\
  -H "Authorization: Bearer $API_KEY"

echo -e "\\n\\n"

# Create simulation
echo "🚀 Creating Simulation"
SIMULATION_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/simulations" \\
  -H "Authorization: Bearer $API_KEY" \\
  -H "Content-Type: application/json" \\
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

echo -e "\\n\\n"

# Get simulation status
echo "📊 Getting Simulation Status"
curl -X GET "$BASE_URL/api/v1/simulations/$SIMULATION_ID/status" \\
  -H "Authorization: Bearer $API_KEY" | jq .

echo -e "\\n\\n"

# List simulations
echo "📋 Listing All Simulations"
curl -X GET "$BASE_URL/api/v1/simulations" \\
  -H "Authorization: Bearer $API_KEY" | jq .

echo -e "\\n\\n"

# Get metrics
echo "📈 Getting System Metrics"
curl -X GET "$BASE_URL/metrics" \\
  -H "Authorization: Bearer $API_KEY" | jq .

echo -e "\\n\\n"

# Stop simulation (optional)
echo "🛑 Stopping Simulation"
curl -X POST "$BASE_URL/api/v1/simulations/$SIMULATION_ID/stop" \\
  -H "Authorization: Bearer $API_KEY" | jq .
'''
        },
        "javascript": {
            "filename": "javascript_examples.js",
            "content": '''/**
 * JavaScript examples for GAN Cyber Range API
 */

const BASE_URL = "http://localhost:8000";
const API_KEY = "your-api-key-here"; // Replace with your actual API key

const headers = {
    "Authorization": `Bearer ${API_KEY}`,
    "Content-Type": "application/json"
};

/**
 * Create a new cyber range simulation
 */
async function createSimulation() {
    const simulationRequest = {
        simulation_config: {
            vulnerable_services: ["webapp", "database"],
            network_topology: "multi-tier",
            difficulty: "medium",
            duration_hours: 1.0,
            realtime_factor: 60
        },
        red_team_config: {
            name: "JavaScriptRedTeam",
            llm_model: "gpt-4",
            skill_level: "advanced",
            tools: ["nmap", "metasploit"]
        },
        blue_team_config: {
            name: "JavaScriptBlueTeam",
            llm_model: "claude-3",
            skill_level: "advanced",
            defense_strategy: "proactive"
        }
    };

    try {
        const response = await fetch(`${BASE_URL}/api/v1/simulations`, {
            method: "POST",
            headers: headers,
            body: JSON.stringify(simulationRequest)
        });

        if (response.ok) {
            const data = await response.json();
            console.log("✅ Simulation created:", data.simulation_id);
            return data.simulation_id;
        } else {
            console.error("❌ Failed to create simulation:", await response.text());
            return null;
        }
    } catch (error) {
        console.error("❌ Error creating simulation:", error);
        return null;
    }
}

/**
 * Get simulation status and metrics
 */
async function getSimulationStatus(simulationId) {
    try {
        const response = await fetch(`${BASE_URL}/api/v1/simulations/${simulationId}/status`, {
            headers: headers
        });

        if (response.ok) {
            const data = await response.json();
            console.log("📊 Simulation Status:", data.status);
            console.log("📈 Progress:", data.progress);
            return data;
        } else {
            console.error("❌ Failed to get status:", await response.text());
            return null;
        }
    } catch (error) {
        console.error("❌ Error getting status:", error);
        return null;
    }
}

/**
 * Monitor simulation with WebSocket (if available)
 */
async function monitorSimulationWebSocket(simulationId) {
    try {
        const wsUrl = `ws://localhost:8000/api/v1/simulations/${simulationId}/ws`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("🔗 WebSocket connected for real-time monitoring");
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("📡 Real-time update:", data);
        };

        ws.onclose = () => {
            console.log("🔌 WebSocket connection closed");
        };

        ws.onerror = (error) => {
            console.error("❌ WebSocket error:", error);
        };

        return ws;
    } catch (error) {
        console.error("❌ Failed to connect WebSocket:", error);
        return null;
    }
}

/**
 * Get final simulation results
 */
async function getSimulationResults(simulationId) {
    try {
        const response = await fetch(`${BASE_URL}/api/v1/simulations/${simulationId}/results`, {
            headers: headers
        });

        if (response.ok) {
            const results = await response.json();
            console.log("🏆 Final Results:");
            console.log(`   Compromise Rate: ${(results.compromise_rate * 100).toFixed(1)}%`);
            console.log(`   Detection Rate: ${(results.detection_rate * 100).toFixed(1)}%`);
            console.log(`   Mean Time to Detection: ${results.avg_detection_time.toFixed(1)}s`);
            return results;
        } else {
            console.error("❌ Failed to get results:", await response.text());
            return null;
        }
    } catch (error) {
        console.error("❌ Error getting results:", error);
        return null;
    }
}

/**
 * Example workflow
 */
async function runExample() {
    console.log("🚀 Starting GAN Cyber Range API Example");

    // Create simulation
    const simulationId = await createSimulation();

    if (simulationId) {
        // Monitor with WebSocket
        const ws = await monitorSimulationWebSocket(simulationId);

        // Poll status every 30 seconds
        const statusInterval = setInterval(async () => {
            const status = await getSimulationStatus(simulationId);
            
            if (status && !status.is_running) {
                console.log("✅ Simulation completed!");
                clearInterval(statusInterval);
                
                if (ws) {
                    ws.close();
                }
                
                // Get final results
                await getSimulationResults(simulationId);
            }
        }, 30000);
    }
}

// Run example if this is the main module
if (typeof window !== "undefined") {
    // Browser environment
    console.log("Running in browser - call runExample() to start");
} else if (typeof module !== "undefined" && module.exports) {
    // Node.js environment
    runExample().catch(console.error);
}
'''
        }
    }
    
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    for lang, example in examples.items():
        with open(examples_dir / example["filename"], "w") as f:
            f.write(example["content"])
    
    print(f"✅ Usage examples generated in {examples_dir}")


def main():
    """Main function to generate all documentation."""
    
    # Ensure output directory exists
    docs_dir = Path(__file__).parent.parent / "docs" / "api"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Generating API documentation...")
    
    try:
        # Generate API documentation
        generate_api_documentation(docs_dir)
        
        # Generate usage examples
        generate_usage_examples(docs_dir)
        
        print("✅ Documentation generation completed successfully!")
        print(f"📁 Documentation available at: {docs_dir}")
        print(f"🌐 Open {docs_dir}/index.html in your browser to view interactive docs")
        
    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()