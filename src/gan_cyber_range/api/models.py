"""Pydantic models for API requests and responses."""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    
    name: str = Field(..., description="Agent name")
    llm_model: str = Field(default="gpt-4", description="LLM model to use")
    skill_level: str = Field(default="intermediate", description="Agent skill level")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    api_key: Optional[str] = Field(default=None, description="API key for LLM service")
    
    @validator("skill_level")
    def validate_skill_level(cls, v):
        allowed_levels = ["beginner", "intermediate", "advanced", "expert"]
        if v not in allowed_levels:
            raise ValueError(f"skill_level must be one of {allowed_levels}")
        return v


class RedTeamConfig(AgentConfig):
    """Configuration for red team agent."""
    
    name: str = Field(default="RedTeam", description="Red team agent name")
    attack_objectives: List[str] = Field(
        default_factory=lambda: ["reconnaissance", "exploitation", "persistence"],
        description="Attack objectives"
    )


class BlueTeamConfig(AgentConfig):
    """Configuration for blue team agent."""
    
    name: str = Field(default="BlueTeam", description="Blue team agent name")
    defense_strategy: str = Field(default="proactive", description="Defense strategy")
    threat_intelligence_feeds: List[str] = Field(
        default_factory=lambda: ["mitre_attack", "cti_feeds"],
        description="Threat intelligence feeds"
    )
    auto_response_enabled: bool = Field(default=True, description="Enable automated responses")
    
    @validator("defense_strategy")
    def validate_defense_strategy(cls, v):
        allowed_strategies = ["reactive", "proactive", "adaptive", "predictive"]
        if v not in allowed_strategies:
            raise ValueError(f"defense_strategy must be one of {allowed_strategies}")
        return v


class SimulationConfig(BaseModel):
    """Configuration for a simulation."""
    
    vulnerable_services: List[str] = Field(
        default_factory=lambda: ["webapp", "database", "api-gateway"],
        description="Services to deploy in the range"
    )
    network_topology: str = Field(default="multi-tier", description="Network topology")
    difficulty: str = Field(default="medium", description="Simulation difficulty")
    duration_hours: float = Field(default=1.0, description="Simulation duration in hours")
    realtime_factor: int = Field(default=60, description="Time acceleration factor")
    isolation_enabled: bool = Field(default=True, description="Enable network isolation")
    
    @validator("difficulty")
    def validate_difficulty(cls, v):
        allowed_difficulties = ["easy", "medium", "hard", "expert"]
        if v not in allowed_difficulties:
            raise ValueError(f"difficulty must be one of {allowed_difficulties}")
        return v
    
    @validator("duration_hours")
    def validate_duration(cls, v):
        if v <= 0 or v > 24:
            raise ValueError("duration_hours must be between 0 and 24")
        return v
    
    @validator("realtime_factor")
    def validate_realtime_factor(cls, v):
        if v < 1 or v > 3600:
            raise ValueError("realtime_factor must be between 1 and 3600")
        return v


class SimulationRequest(BaseModel):
    """Request to start a new simulation."""
    
    simulation_config: SimulationConfig = Field(default_factory=SimulationConfig)
    red_team_config: RedTeamConfig = Field(default_factory=RedTeamConfig)
    blue_team_config: BlueTeamConfig = Field(default_factory=BlueTeamConfig)
    
    class Config:
        schema_extra = {
            "example": {
                "simulation_config": {
                    "vulnerable_services": ["webapp", "database"],
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
                    "llm_model": "claude-3-sonnet-20240229",
                    "skill_level": "advanced",
                    "defense_strategy": "proactive"
                }
            }
        }


class AgentAction(BaseModel):
    """Represents an action taken by an agent."""
    
    type: str = Field(..., description="Action type")
    target: str = Field(..., description="Action target")
    timestamp: datetime = Field(..., description="When the action was taken")
    success: bool = Field(..., description="Whether the action succeeded")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Action payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Action metadata")


class SimulationStatus(BaseModel):
    """Current status of a simulation."""
    
    simulation_id: str = Field(..., description="Simulation identifier")
    is_running: bool = Field(..., description="Whether simulation is active")
    current_round: int = Field(..., description="Current round number")
    services_total: int = Field(..., description="Total number of services")
    services_compromised: int = Field(..., description="Number of compromised services")
    services_isolated: int = Field(..., description="Number of isolated services")
    total_attacks: int = Field(..., description="Total number of attacks")
    attacks_blocked: int = Field(..., description="Number of blocked attacks")
    patches_deployed: int = Field(..., description="Number of patches deployed")
    duration: str = Field(..., description="Simulation duration")
    
    @property
    def compromise_rate(self) -> float:
        """Calculate compromise rate."""
        if self.services_total == 0:
            return 0.0
        return self.services_compromised / self.services_total
    
    @property
    def defense_effectiveness(self) -> float:
        """Calculate defense effectiveness."""
        if self.total_attacks == 0:
            return 1.0
        return self.attacks_blocked / self.total_attacks


class SimulationResults(BaseModel):
    """Results from a completed simulation."""
    
    simulation_id: str = Field(..., description="Simulation identifier")
    start_time: datetime = Field(..., description="Simulation start time")
    end_time: Optional[datetime] = Field(None, description="Simulation end time")
    red_team_actions: List[AgentAction] = Field(default_factory=list)
    blue_team_actions: List[AgentAction] = Field(default_factory=list)
    services_compromised: int = Field(default=0)
    patches_deployed: int = Field(default=0)
    attacks_blocked: int = Field(default=0)
    total_attacks: int = Field(default=0)
    final_status: SimulationStatus
    
    @property
    def duration_seconds(self) -> float:
        """Get simulation duration in seconds."""
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def compromise_rate(self) -> float:
        """Calculate compromise rate."""
        if self.total_attacks == 0:
            return 0.0
        return self.services_compromised / self.total_attacks
    
    @property
    def defense_effectiveness(self) -> float:
        """Calculate defense effectiveness."""
        if self.total_attacks == 0:
            return 1.0
        return self.attacks_blocked / self.total_attacks


class SimulationResponse(BaseModel):
    """Response from starting a simulation."""
    
    simulation_id: str = Field(..., description="Unique simulation identifier")
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    simulation_url: Optional[str] = Field(None, description="URL to monitor simulation")
    
    class Config:
        schema_extra = {
            "example": {
                "simulation_id": "sim_12345678-1234-1234-1234-123456789012",
                "status": "started",
                "message": "Simulation started successfully",
                "simulation_url": "/api/v1/simulations/sim_12345678-1234-1234-1234-123456789012"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid configuration provided",
                "details": {"field": "skill_level", "issue": "must be one of ['beginner', 'intermediate', 'advanced']"}
            }
        }


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(..., description="Check timestamp")
    services: Dict[str, str] = Field(..., description="Service statuses")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2024-01-01T12:00:00Z",
                "services": {
                    "database": "healthy",
                    "llm_services": "healthy",
                    "kubernetes": "healthy"
                }
            }
        }


class MetricsResponse(BaseModel):
    """Metrics response model."""
    
    total_simulations: int = Field(..., description="Total simulations run")
    active_simulations: int = Field(..., description="Currently active simulations")
    average_duration_minutes: float = Field(..., description="Average simulation duration")
    success_rate: float = Field(..., description="Simulation success rate")
    total_agents_deployed: int = Field(..., description="Total agents deployed")
    
    class Config:
        schema_extra = {
            "example": {
                "total_simulations": 150,
                "active_simulations": 3,
                "average_duration_minutes": 45.2,
                "success_rate": 0.94,
                "total_agents_deployed": 300
            }
        }