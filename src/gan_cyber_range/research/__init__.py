"""
Research module for novel cybersecurity AI algorithms.

This module contains cutting-edge research implementations for:
- Adversarial training algorithms
- Multi-modal attack detection
- Zero-shot vulnerability discovery
- Self-healing security systems
"""

from .adversarial_training import CoevolutionaryGANTrainer
from .multimodal_detection import MultiModalThreatDetector
from .zero_shot_vuln import ZeroShotVulnerabilityDetector
from .self_healing import SelfHealingSecuritySystem

__all__ = [
    "CoevolutionaryGANTrainer",
    "MultiModalThreatDetector", 
    "ZeroShotVulnerabilityDetector",
    "SelfHealingSecuritySystem"
]