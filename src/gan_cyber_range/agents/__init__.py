"""Agent framework for red and blue team simulation."""

from .red_team import RedTeamAgent
from .blue_team import BlueTeamAgent
from .base import BaseAgent

__all__ = ["RedTeamAgent", "BlueTeamAgent", "BaseAgent"]