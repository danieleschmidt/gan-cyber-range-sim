"""GAN Cyber Range Simulator — synthetic network attack traffic for security training."""

from .models import TrafficGAN
from .flows import NetworkFlowGenerator, AttackType
from .simulator import CyberRangeSimulator
from .evaluator import FlowEvaluator

__all__ = [
    "TrafficGAN",
    "NetworkFlowGenerator",
    "AttackType",
    "CyberRangeSimulator",
    "FlowEvaluator",
]

__version__ = "1.0.0"
