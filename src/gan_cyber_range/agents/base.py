"""Base agent class for cyber range simulation."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """Represents an action taken by an agent."""
    
    type: str = Field(..., description="Type of action")
    target: str = Field(..., description="Target of the action")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Action payload")
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = Field(default=False, description="Whether action succeeded")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMemory(BaseModel):
    """Agent's memory system for learning and adaptation."""
    
    actions: List[AgentAction] = Field(default_factory=list)
    successes: List[AgentAction] = Field(default_factory=list)
    failures: List[AgentAction] = Field(default_factory=list)
    patterns: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all cyber range agents."""
    
    def __init__(
        self,
        name: str,
        llm_model: str = "gpt-4",
        skill_level: str = "intermediate",
        max_actions_per_round: int = 5
    ):
        self.name = name
        self.llm_model = llm_model
        self.skill_level = skill_level
        self.max_actions_per_round = max_actions_per_round
        self.memory = AgentMemory()
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self.active = False
        self._round_counter = 0
    
    @abstractmethod
    async def analyze_environment(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the current environment state."""
        pass
    
    @abstractmethod
    async def plan_actions(self, analysis: Dict[str, Any]) -> List[AgentAction]:
        """Plan actions based on environment analysis."""
        pass
    
    @abstractmethod
    async def execute_action(self, action: AgentAction) -> AgentAction:
        """Execute a single action."""
        pass
    
    async def act(self, environment_state: Dict[str, Any]) -> List[AgentAction]:
        """Main action loop for the agent."""
        if not self.active:
            return []
        
        self._round_counter += 1
        self.logger.info(f"Starting round {self._round_counter}")
        
        try:
            # Analyze environment
            analysis = await self.analyze_environment(environment_state)
            
            # Plan actions
            planned_actions = await self.plan_actions(analysis)
            
            # Limit actions per round
            actions_to_execute = planned_actions[:self.max_actions_per_round]
            
            # Execute actions
            executed_actions = []
            for action in actions_to_execute:
                try:
                    result = await self.execute_action(action)
                    executed_actions.append(result)
                    self.memory.actions.append(result)
                    
                    if result.success:
                        self.memory.successes.append(result)
                    else:
                        self.memory.failures.append(result)
                        
                except Exception as e:
                    self.logger.error(f"Action execution failed: {e}")
                    action.success = False
                    action.metadata["error"] = str(e)
                    executed_actions.append(action)
                    self.memory.failures.append(action)
            
            # Learn from actions
            await self.learn_from_actions(executed_actions)
            
            return executed_actions
            
        except Exception as e:
            self.logger.error(f"Agent action cycle failed: {e}")
            return []
    
    async def learn_from_actions(self, actions: List[AgentAction]) -> None:
        """Learn from executed actions to improve future performance."""
        success_rate = sum(1 for a in actions if a.success) / len(actions) if actions else 0
        
        # Update patterns based on success rate
        self.memory.patterns["success_rate"] = success_rate
        self.memory.patterns["last_round"] = self._round_counter
        
        # Simple pattern recognition
        successful_types = [a.type for a in actions if a.success]
        if successful_types:
            self.memory.patterns["successful_actions"] = successful_types
    
    def activate(self) -> None:
        """Activate the agent."""
        self.active = True
        self.logger.info(f"Agent {self.name} activated")
    
    def deactivate(self) -> None:
        """Deactivate the agent."""
        self.active = False
        self.logger.info(f"Agent {self.name} deactivated")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        total_actions = len(self.memory.actions)
        successes = len(self.memory.successes)
        failures = len(self.memory.failures)
        
        return {
            "name": self.name,
            "total_actions": total_actions,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total_actions if total_actions > 0 else 0,
            "rounds_completed": self._round_counter,
            "skill_level": self.skill_level,
            "active": self.active
        }