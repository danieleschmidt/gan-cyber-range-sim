"""FastAPI server for GAN Cyber Range simulation."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..environment.cyber_range import CyberRange
from ..agents.red_team import RedTeamAgent
from ..agents.blue_team import BlueTeamAgent
from .models import (
    SimulationRequest, SimulationResponse, SimulationStatus, 
    SimulationResults, ErrorResponse, HealthCheck, MetricsResponse
)


# Global state management
class SimulationManager:
    """Manages active simulations."""
    
    def __init__(self):
        self.active_simulations: Dict[str, Dict] = {}
        self.completed_simulations: Dict[str, SimulationResults] = {}
        self.simulation_tasks: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger("SimulationManager")
        
        # Metrics
        self.total_simulations_started = 0
        self.total_simulations_completed = 0
        self.total_agents_deployed = 0
    
    def create_simulation(self, request: SimulationRequest) -> str:
        """Create a new simulation."""
        simulation_id = f"sim_{uuid.uuid4()}"
        
        # Create cyber range
        cyber_range = CyberRange(
            vulnerable_services=request.simulation_config.vulnerable_services,
            network_topology=request.simulation_config.network_topology,
            difficulty=request.simulation_config.difficulty,
            isolation_enabled=request.simulation_config.isolation_enabled
        )
        
        # Create agents
        red_team = RedTeamAgent(
            name=request.red_team_config.name,
            llm_model=request.red_team_config.llm_model,
            skill_level=request.red_team_config.skill_level,
            tools=request.red_team_config.tools,
            api_key=request.red_team_config.api_key
        )
        
        blue_team = BlueTeamAgent(
            name=request.blue_team_config.name,
            llm_model=request.blue_team_config.llm_model,
            skill_level=request.blue_team_config.skill_level,
            defense_strategy=request.blue_team_config.defense_strategy,
            tools=request.blue_team_config.tools,
            threat_intelligence_feeds=request.blue_team_config.threat_intelligence_feeds,
            auto_response_enabled=request.blue_team_config.auto_response_enabled,
            api_key=request.blue_team_config.api_key
        )
        
        # Store simulation data
        self.active_simulations[simulation_id] = {
            "cyber_range": cyber_range,
            "red_team": red_team,
            "blue_team": blue_team,
            "config": request,
            "start_time": datetime.now(),
            "status": "created"
        }
        
        self.total_simulations_started += 1
        self.total_agents_deployed += 2
        
        return simulation_id
    
    async def start_simulation(self, simulation_id: str) -> None:
        """Start a simulation in the background."""
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        sim_data = self.active_simulations[simulation_id]
        sim_data["status"] = "running"
        
        # Create background task
        task = asyncio.create_task(self._run_simulation(simulation_id))
        self.simulation_tasks[simulation_id] = task
    
    async def _run_simulation(self, simulation_id: str) -> None:
        """Run a simulation to completion."""
        try:
            sim_data = self.active_simulations[simulation_id]
            cyber_range = sim_data["cyber_range"]
            red_team = sim_data["red_team"]
            blue_team = sim_data["blue_team"]
            config = sim_data["config"]
            
            self.logger.info(f"Starting simulation {simulation_id}")
            
            # Run the simulation
            results = await cyber_range.simulate(
                red_team=red_team,
                blue_team=blue_team,
                duration_hours=config.simulation_config.duration_hours,
                realtime_factor=config.simulation_config.realtime_factor
            )
            
            # Convert to API response format
            api_results = SimulationResults(
                simulation_id=simulation_id,
                start_time=results.start_time,
                end_time=results.end_time,
                red_team_actions=[
                    self._convert_action(action) for action in results.red_team_actions
                ],
                blue_team_actions=[
                    self._convert_action(action) for action in results.blue_team_actions
                ],
                services_compromised=results.services_compromised,
                patches_deployed=results.patches_deployed,
                attacks_blocked=results.attacks_blocked,
                total_attacks=results.total_attacks,
                final_status=self._get_simulation_status(simulation_id)
            )
            
            # Move to completed simulations
            self.completed_simulations[simulation_id] = api_results
            
            if simulation_id in self.active_simulations:
                del self.active_simulations[simulation_id]
            
            if simulation_id in self.simulation_tasks:
                del self.simulation_tasks[simulation_id]
            
            self.total_simulations_completed += 1
            self.logger.info(f"Completed simulation {simulation_id}")
            
        except Exception as e:
            self.logger.error(f"Simulation {simulation_id} failed: {e}")
            # Mark as failed
            if simulation_id in self.active_simulations:
                self.active_simulations[simulation_id]["status"] = "failed"
                self.active_simulations[simulation_id]["error"] = str(e)
    
    def _convert_action(self, action) -> Dict:
        """Convert internal action to API format."""
        from .models import AgentAction
        
        return AgentAction(
            type=action.type,
            target=action.target,
            timestamp=action.timestamp,
            success=action.success,
            payload=action.payload,
            metadata=action.metadata
        ).dict()
    
    def get_simulation_status(self, simulation_id: str) -> Optional[SimulationStatus]:
        """Get current status of a simulation."""
        if simulation_id in self.active_simulations:
            return self._get_simulation_status(simulation_id)
        elif simulation_id in self.completed_simulations:
            return self.completed_simulations[simulation_id].final_status
        return None
    
    def _get_simulation_status(self, simulation_id: str) -> SimulationStatus:
        """Get status from active simulation."""
        sim_data = self.active_simulations[simulation_id]
        cyber_range = sim_data["cyber_range"]
        status_dict = cyber_range.get_simulation_status()
        
        return SimulationStatus(**status_dict)
    
    async def stop_simulation(self, simulation_id: str) -> bool:
        """Stop a running simulation."""
        if simulation_id in self.simulation_tasks:
            task = self.simulation_tasks[simulation_id]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Clean up
            if simulation_id in self.active_simulations:
                self.active_simulations[simulation_id]["status"] = "stopped"
            
            return True
        return False
    
    def get_metrics(self) -> MetricsResponse:
        """Get system metrics."""
        active_count = len(self.active_simulations)
        
        # Calculate average duration from completed simulations
        if self.completed_simulations:
            total_duration = sum(
                sim.duration_seconds for sim in self.completed_simulations.values()
            )
            avg_duration_minutes = (total_duration / len(self.completed_simulations)) / 60
        else:
            avg_duration_minutes = 0.0
        
        # Calculate success rate
        success_rate = (
            self.total_simulations_completed / self.total_simulations_started
            if self.total_simulations_started > 0 else 1.0
        )
        
        return MetricsResponse(
            total_simulations=self.total_simulations_started,
            active_simulations=active_count,
            average_duration_minutes=avg_duration_minutes,
            success_rate=success_rate,
            total_agents_deployed=self.total_agents_deployed
        )


# Initialize simulation manager
simulation_manager = SimulationManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("GAN-CyberRange")
    logger.info("Starting GAN Cyber Range API server")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GAN Cyber Range API server")
    
    # Stop all active simulations
    for sim_id in list(simulation_manager.simulation_tasks.keys()):
        await simulation_manager.stop_simulation(sim_id)


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="GAN Cyber Range Simulator",
        description="API for adversarial cybersecurity training with AI agents",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# Create app instance
app = create_app()


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred"
        ).dict()
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Check service health."""
    return HealthCheck(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now(),
        services={
            "api": "healthy",
            "simulation_engine": "healthy",
            "agent_framework": "healthy"
        }
    )


# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get system metrics."""
    return simulation_manager.get_metrics()


# Simulation endpoints
@app.post(
    "/api/v1/simulations", 
    response_model=SimulationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Simulations"]
)
async def create_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """Create and start a new cyber range simulation."""
    try:
        # Create simulation
        simulation_id = simulation_manager.create_simulation(request)
        
        # Start simulation in background
        background_tasks.add_task(simulation_manager.start_simulation, simulation_id)
        
        return SimulationResponse(
            simulation_id=simulation_id,
            status="started",
            message="Simulation created and started successfully",
            simulation_url=f"/api/v1/simulations/{simulation_id}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create simulation: {str(e)}"
        )


@app.get(
    "/api/v1/simulations/{simulation_id}/status",
    response_model=SimulationStatus,
    tags=["Simulations"]
)
async def get_simulation_status(simulation_id: str):
    """Get the current status of a simulation."""
    status = simulation_manager.get_simulation_status(simulation_id)
    
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found"
        )
    
    return status


@app.get(
    "/api/v1/simulations/{simulation_id}/results",
    response_model=SimulationResults,
    tags=["Simulations"]
)
async def get_simulation_results(simulation_id: str):
    """Get the results of a completed simulation."""
    if simulation_id in simulation_manager.completed_simulations:
        return simulation_manager.completed_simulations[simulation_id]
    
    if simulation_id in simulation_manager.active_simulations:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Simulation {simulation_id} is still running"
        )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Simulation {simulation_id} not found"
    )


@app.post(
    "/api/v1/simulations/{simulation_id}/stop",
    tags=["Simulations"]
)
async def stop_simulation(simulation_id: str):
    """Stop a running simulation."""
    success = await simulation_manager.stop_simulation(simulation_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found or not running"
        )
    
    return {"message": f"Simulation {simulation_id} stopped successfully"}


@app.get(
    "/api/v1/simulations",
    tags=["Simulations"]
)
async def list_simulations():
    """List all simulations."""
    active = [
        {
            "simulation_id": sim_id,
            "status": sim_data["status"],
            "start_time": sim_data["start_time"].isoformat()
        }
        for sim_id, sim_data in simulation_manager.active_simulations.items()
    ]
    
    completed = [
        {
            "simulation_id": sim_id,
            "start_time": results.start_time.isoformat(),
            "end_time": results.end_time.isoformat() if results.end_time else None,
            "duration_seconds": results.duration_seconds
        }
        for sim_id, results in simulation_manager.completed_simulations.items()
    ]
    
    return {
        "active_simulations": active,
        "completed_simulations": completed,
        "total_active": len(active),
        "total_completed": len(completed)
    }


# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )