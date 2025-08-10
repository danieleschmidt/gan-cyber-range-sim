"""GAN Cyber Range Simulator with Advanced Research Capabilities.

A generative adversarial cyber-range where attacker LLMs spin up exploits
while defender LLMs patch in real time. Now enhanced with breakthrough
research in adversarial AI, multi-modal detection, zero-shot vulnerability
discovery, and self-healing security infrastructure.

Research Contributions:
- Coevolutionary adversarial training algorithms
- Multi-modal threat detection with fusion learning
- Zero-shot vulnerability detection using meta-learning
- Self-healing security systems with AI-driven adaptation
- Comprehensive validation framework for cybersecurity research
"""

__version__ = "1.0.0"  # Major version increment for research enhancements
__author__ = "Daniel Schmidt"
__email__ = "info@gan-cyber-range.org"
__license__ = "MIT"

# Core platform components
from .environment import CyberRange
from .agents import RedTeamAgent, BlueTeamAgent

# Research modules (import conditionally to handle optional dependencies)
try:
    from .research import (
        CoevolutionaryGANTrainer,
        MultiModalThreatDetector, 
        ZeroShotVulnerabilityDetector,
        SelfHealingSecuritySystem
    )
    RESEARCH_AVAILABLE = True
    
    __all__ = [
        # Core platform
        "CyberRange", "RedTeamAgent", "BlueTeamAgent",
        
        # Research components
        "CoevolutionaryGANTrainer",
        "MultiModalThreatDetector", 
        "ZeroShotVulnerabilityDetector",
        "SelfHealingSecuritySystem"
    ]
    
except ImportError as e:
    # Research dependencies not available
    RESEARCH_AVAILABLE = False
    
    __all__ = ["CyberRange", "RedTeamAgent", "BlueTeamAgent"]
    
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Research modules not available due to missing dependencies: {e}")

# Version and capability information
__research_version__ = "1.0.0"
__capabilities__ = {
    "core_platform": True,
    "adversarial_training": RESEARCH_AVAILABLE,
    "multimodal_detection": RESEARCH_AVAILABLE, 
    "zero_shot_vulnerability": RESEARCH_AVAILABLE,
    "self_healing_systems": RESEARCH_AVAILABLE,
    "research_validation": RESEARCH_AVAILABLE
}