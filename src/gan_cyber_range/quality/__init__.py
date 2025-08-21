"""Enhanced Quality Framework with Next-Generation Capabilities.

This module provides a comprehensive quality assurance framework with:
- Real-time quality monitoring with live feedback
- Adaptive threshold management with ML optimization
- Predictive quality assurance with anomaly detection
- Progressive validation systems for autonomous SDLC
"""

__version__ = "2.0.0"
__author__ = "Terragon Labs"

# Core quality gate system
from .quality_gates import (
    QualityGate,
    QualityGateResult,
    QualityGateStatus,
    QualityGateRunner,
    SecurityGate,
    PerformanceGate,
    TestCoverageGate,
    CodeQualityGate,
    ComplianceGate
)

# Enhanced quality systems
try:
    from .realtime_monitor import (
        RealTimeQualityMonitor,
        QualityMetric,
        QualityAlert,
        QualityTrend,
        MonitoringLevel
    )
    from .adaptive_thresholds import (
        AdaptiveThresholdManager as EnhancedAdaptiveThresholdManager,
        AdaptationStrategy,
        ThresholdHistory,
        StatisticalAnalyzer,
        ContextAnalyzer
    )
    from .ml_optimizer import (
        MLQualityOptimizer,
        MLPrediction,
        MLFeature,
        MLModelType,
        FeatureExtractor,
        QualityMLModel
    )
    from .predictive_qa import (
        PredictiveQualityAssurance,
        QualityPrediction,
        AnomalyDetection,
        QualityRisk,
        PredictionHorizon,
        AnomalyType,
        StatisticalAnomalyDetector,
        PatternAnomalyDetector,
        CorrelationAnomalyDetector
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    # Fallback if enhanced dependencies not available
    ENHANCED_FEATURES_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced quality features not available: {e}")

from .progressive_validator import (
    ProgressiveValidator,
    ValidationStage,
    ValidationResult,
    ValidationMetrics
)

from .automated_pipeline import (
    AutomatedQualityPipeline,
    PipelineStage,
    PipelineResult,
    QualityReport
)

from .monitoring import (
    QualityMetric,
    MetricsCollector,
    QualityMonitor,
    QualityDashboard,
    QualityTrendAnalyzer,
    get_global_metrics_collector,
    get_global_monitor,
    get_global_dashboard
)

from .validation_framework import (
    ValidationFramework,
    ValidationLevel,
    SecurityValidator,
    PerformanceValidator
)

from .intelligent_optimizer import (
    IntelligentOptimizer,
    OptimizationStrategy,
    OptimizationTarget,
    OptimizationResult,
    AdaptiveThresholdManager
)

from .auto_scaler import (
    QualityGateAutoScaler,
    ScalingStrategy,
    ScalingDirection,
    ScalingMetrics,
    LoadPredictor
)

# Build __all__ dynamically based on available features
__all__ = [
    # Core quality gates
    "QualityGate",
    "QualityGateResult", 
    "QualityGateStatus",
    "QualityGateRunner",
    "SecurityGate",
    "PerformanceGate",
    "TestCoverageGate",
    "CodeQualityGate",
    "ComplianceGate",
    
    # Progressive validation
    "ProgressiveValidator",
    "ValidationStage",
    "ValidationResult",
    "ValidationMetrics",
    
    # Automated pipeline
    "AutomatedQualityPipeline",
    "PipelineStage",
    "PipelineResult",
    "QualityReport",
    
    # Monitoring and observability
    "QualityMetric",
    "MetricsCollector",
    "QualityMonitor",
    "QualityDashboard",
    "QualityTrendAnalyzer",
    "get_global_metrics_collector",
    "get_global_monitor",
    "get_global_dashboard",
    
    # Advanced validation
    "ValidationFramework",
    "ValidationLevel",
    "SecurityValidator",
    "PerformanceValidator",
    
    # Intelligent optimization
    "IntelligentOptimizer",
    "OptimizationStrategy",
    "OptimizationTarget",
    "OptimizationResult",
    "AdaptiveThresholdManager",
    
    # Auto-scaling
    "QualityGateAutoScaler",
    "ScalingStrategy",
    "ScalingDirection",
    "ScalingMetrics",
    "LoadPredictor"
]

# Add enhanced features if available
if ENHANCED_FEATURES_AVAILABLE:
    __all__.extend([
        # Real-time monitoring
        "RealTimeQualityMonitor",
        "QualityAlert",
        "QualityTrend", 
        "MonitoringLevel",
        
        # Enhanced adaptive thresholds
        "EnhancedAdaptiveThresholdManager",
        "AdaptationStrategy",
        "ThresholdHistory",
        "StatisticalAnalyzer",
        "ContextAnalyzer",
        
        # ML optimization
        "MLQualityOptimizer",
        "MLPrediction",
        "MLFeature",
        "MLModelType",
        "FeatureExtractor",
        "QualityMLModel",
        
        # Predictive QA
        "PredictiveQualityAssurance",
        "QualityPrediction",
        "AnomalyDetection",
        "QualityRisk",
        "PredictionHorizon",
        "AnomalyType",
        "StatisticalAnomalyDetector",
        "PatternAnomalyDetector",
        "CorrelationAnomalyDetector"
    ])

# Framework capabilities
ENHANCED_CAPABILITIES = {
    "real_time_monitoring": ENHANCED_FEATURES_AVAILABLE,
    "adaptive_thresholds": True,
    "ml_optimization": ENHANCED_FEATURES_AVAILABLE,
    "predictive_qa": ENHANCED_FEATURES_AVAILABLE,
    "anomaly_detection": ENHANCED_FEATURES_AVAILABLE,
    "progressive_validation": True,
    "statistical_analysis": ENHANCED_FEATURES_AVAILABLE,
    "pattern_recognition": ENHANCED_FEATURES_AVAILABLE,
    "correlation_analysis": ENHANCED_FEATURES_AVAILABLE,
    "risk_assessment": ENHANCED_FEATURES_AVAILABLE
}

def get_framework_info():
    """Get enhanced quality framework information."""
    return {
        "version": __version__,
        "author": __author__,
        "enhanced_features_available": ENHANCED_FEATURES_AVAILABLE,
        "capabilities": ENHANCED_CAPABILITIES,
        "components": len(__all__),
        "description": "Next-generation quality assurance framework with AI-powered insights"
    }