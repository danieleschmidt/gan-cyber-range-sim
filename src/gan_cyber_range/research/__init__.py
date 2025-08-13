"""
Research module for novel cybersecurity AI algorithms.

This module contains cutting-edge research implementations for:
- Adversarial training algorithms
- Multi-modal attack detection
- Zero-shot vulnerability discovery
- Self-healing security systems
- Quantum-enhanced adversarial training
- Neuromorphic computing for adaptive security
"""

from .adversarial_training import CoevolutionaryGANTrainer
from .multimodal_detection import MultiModalThreatDetector
from .zero_shot_vuln import ZeroShotVulnerabilityDetector
from .self_healing import SelfHealingSecuritySystem
from .quantum_adversarial import QuantumAdversarialTrainer
from .neuromorphic_security import NeuromorphicThreatDetector, NeuromorphicSecuritySystem

__all__ = [
    "CoevolutionaryGANTrainer",
    "MultiModalThreatDetector", 
    "ZeroShotVulnerabilityDetector",
    "SelfHealingSecuritySystem",
    "QuantumAdversarialTrainer",
    "NeuromorphicThreatDetector",
    "NeuromorphicSecuritySystem"
]