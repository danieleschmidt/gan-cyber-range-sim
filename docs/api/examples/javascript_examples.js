/**
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
