"""GAN Cyber Range Simulator.

A generative adversarial cyber-range where attacker LLMs spin up exploits
while defender LLMs patch in real time.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "info@gan-cyber-range.org"
__license__ = "MIT"

from .environment import CyberRange
from .agents import RedTeamAgent, BlueTeamAgent

__all__ = ["CyberRange", "RedTeamAgent", "BlueTeamAgent"]