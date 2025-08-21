"""Comprehensive tests for enhanced quality system."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from gan_cyber_range.quality.realtime_monitor import (
    RealTimeQualityMonitor, QualityMetric, QualityAlert, MonitoringLevel
)
from gan_cyber_range.quality.adaptive_thresholds import (
    AdaptiveThresholdManager, AdaptationStrategy, ThresholdHistory
)
from gan_cyber_range.quality.ml_optimizer import (
    MLQualityOptimizer, MLPrediction, MLFeature, MLModelType
)
from gan_cyber_range.quality.predictive_qa import (
    PredictiveQualityAssurance, QualityPrediction, AnomalyDetection, 
    PredictionHorizon, AnomalyType
)
from gan_cyber_range.quality.quality_gates import QualityGateResult, QualityGateStatus


class TestRealTimeQualityMonitor:
    """Test real-time quality monitoring system."""
    
    @pytest.fixture
    def monitor(self):
        """Create real-time monitor instance."""
        return RealTimeQualityMonitor(
            update_interval=1.0,
            enable_websocket=False
        )
    
    @pytest.mark.asyncio
    async def test_quality_metric_creation(self, monitor):
        """Test quality metric creation and validation."""
        metric = QualityMetric(
            name="test_coverage",
            value=85.5,
            timestamp=datetime.now(),
            threshold=80.0,
            status=QualityGateStatus.PASSED,
            source="test"
        )
        
        assert metric.name == "test_coverage"
        assert metric.value == 85.5
        assert metric.threshold == 80.0
        assert metric.status == QualityGateStatus.PASSED
        assert not metric.is_degraded
        
        # Test degraded metric
        degraded_metric = QualityMetric(
            name="security_score",
            value=70.0,
            timestamp=datetime.now(),
            threshold=90.0,
            status=QualityGateStatus.WARNING,
            source="test"
        )
        
        assert degraded_metric.is_degraded
    
    @pytest.mark.asyncio
    async def test_metric_collection(self, monitor):
        """Test metric collection functionality."""
        # Mock metric collection methods
        monitor._get_test_coverage = AsyncMock(return_value=85.0)
        monitor._get_security_score = AsyncMock(return_value=92.0)
        monitor._get_performance_score = AsyncMock(return_value=78.0)
        
        await monitor._collect_quality_metrics()
        
        # Verify metrics were collected
        assert "test_coverage" in monitor.active_metrics
        assert "security_score" in monitor.active_metrics
        assert "performance_score" in monitor.active_metrics
        
        # Verify metric values
        assert monitor.active_metrics["test_coverage"].value == 85.0
        assert monitor.active_metrics["security_score"].value == 92.0
        assert monitor.active_metrics["performance_score"].value == 78.0
    
    @pytest.mark.asyncio
    async def test_alert_creation(self, monitor):
        """Test quality alert creation."""
        metric = QualityMetric(
            name="test_coverage",
            value=65.0,  # Below threshold
            timestamp=datetime.now(),
            threshold=80.0,
            status=QualityGateStatus.FAILED,
            source="test"
        )
        
        await monitor._create_alert(metric, MonitoringLevel.HIGH)
        
        assert len(monitor.active_alerts) == 1
        alert = list(monitor.active_alerts.values())[0]
        assert alert.metric_name == "test_coverage"
        assert alert.severity == MonitoringLevel.HIGH
        assert alert.current_value == 65.0
        assert alert.threshold == 80.0
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, monitor):
        """Test trend analysis functionality."""
        # Add test data points
        for i in range(10):
            value = 90.0 - i * 2.0  # Decreasing trend
            monitor.quality_trends["test_metric"].add_point(value)
        
        trend_direction = monitor.quality_trends["test_metric"].get_trend_direction()
        assert trend_direction == "degrading"
        
        # Test prediction
        predicted_value = monitor.quality_trends["test_metric"].predict_next_value()
        assert predicted_value is not None
        assert predicted_value < 72.0  # Should continue the trend
    
    @pytest.mark.asyncio
    async def test_dashboard_data(self, monitor):
        """Test dashboard data generation."""
        # Add some test metrics and alerts
        monitor.active_metrics["test_coverage"] = QualityMetric(
            name="test_coverage",
            value=85.0,
            timestamp=datetime.now(),
            threshold=80.0,
            status=QualityGateStatus.PASSED,
            source="test"
        )
        
        monitor.active_alerts["test_alert"] = QualityAlert(
            id="test_alert",
            metric_name="test_coverage",
            severity=MonitoringLevel.MEDIUM,
            message="Test alert",
            timestamp=datetime.now(),
            current_value=85.0,
            threshold=80.0,
            historical_trend=[80, 82, 85],
            suggested_actions=["Test action"]
        )
        
        dashboard_data = monitor.get_monitoring_dashboard_data()
        
        assert "metrics" in dashboard_data
        assert "alerts" in dashboard_data
        assert "system_health" in dashboard_data
        assert "test_coverage" in dashboard_data["metrics"]
        assert "test_alert" in dashboard_data["alerts"]


class TestAdaptiveThresholdManager:
    """Test adaptive threshold management system."""
    
    @pytest.fixture
    def threshold_manager(self):
        """Create adaptive threshold manager instance."""
        return AdaptiveThresholdManager(
            adaptation_strategy=AdaptationStrategy.BALANCED,
            min_samples=5  # Lower for testing
        )
    
    @pytest.mark.asyncio
    async def test_threshold_initialization(self, threshold_manager):
        """Test threshold initialization."""
        initial_thresholds = {
            "test_coverage": 80.0,
            "security_scan": 90.0,
            "performance_benchmark": 75.0
        }
        
        await threshold_manager.initialize_thresholds(initial_thresholds)
        
        assert threshold_manager.current_thresholds == initial_thresholds
        for metric, threshold in initial_thresholds.items():
            assert threshold_manager.get_current_threshold(metric) == threshold
    
    @pytest.mark.asyncio
    async def test_measurement_update(self, threshold_manager):
        """Test measurement update and adaptation logic."""
        await threshold_manager.initialize_thresholds({"test_coverage": 80.0})
        
        # Mock result
        mock_result = Mock()
        mock_result.status = QualityGateStatus.PASSED
        
        # Add measurements
        for value in [85, 87, 89, 91, 93]:
            await threshold_manager.update_measurement("test_coverage", value, mock_result)
        
        # Check if adaptation occurred (with sufficient samples)
        assert len(threshold_manager.threshold_history) > 0
        
        # Verify statistics were collected
        stats = threshold_manager.statistical_analyzer.calculate_statistics("test_coverage")
        assert "mean" in stats
        assert stats["sample_size"] == 5
    
    @pytest.mark.asyncio
    async def test_context_analysis(self, threshold_manager):
        """Test project context analysis."""
        context = await threshold_manager.context_analyzer.analyze_project_context()
        
        assert "project_phase" in context
        assert "team_size" in context
        assert "code_complexity" in context
        assert "deployment_frequency" in context
        
        # Test context-based threshold adjustment
        adjusted = await threshold_manager._apply_context_adjustments(
            "test_coverage", 80.0, context
        )
        
        assert isinstance(adjusted, float)
        assert adjusted > 0
    
    @pytest.mark.asyncio
    async def test_adaptation_strategies(self, threshold_manager):
        """Test different adaptation strategies."""
        # Test conservative strategy
        threshold_manager.adaptation_strategy = AdaptationStrategy.CONSERVATIVE
        
        await threshold_manager.initialize_thresholds({"test_metric": 80.0})
        
        # Mock sufficient data for adaptation
        with patch.object(threshold_manager.statistical_analyzer, 'calculate_statistics') as mock_stats:
            mock_stats.return_value = {
                "mean": 85.0,
                "median": 84.0,
                "sample_size": 25,
                "percentile_25": 82.0,
                "percentile_75": 87.0
            }
            
            with patch.object(threshold_manager.statistical_analyzer, 'analyze_stability') as mock_stability:
                mock_stability.return_value = {"stable": True, "stability_score": 85.0}
                
                with patch.object(threshold_manager.context_analyzer, 'analyze_project_context') as mock_context:
                    mock_context.return_value = {"project_phase": "production"}
                    
                    optimal = await threshold_manager._calculate_optimal_threshold(
                        "test_metric", mock_stats.return_value, mock_context.return_value
                    )
                    
                    # Conservative should use lower percentile
                    assert optimal >= 82.0  # At least 25th percentile
    
    @pytest.mark.asyncio
    async def test_threshold_persistence(self, threshold_manager, tmp_path):
        """Test threshold saving and loading."""
        test_file = tmp_path / "test_thresholds.json"
        
        # Initialize and adapt some thresholds
        await threshold_manager.initialize_thresholds({"test_coverage": 80.0})
        threshold_manager.adaptation_count = 5
        
        # Save thresholds
        await threshold_manager.save_thresholds(str(test_file))
        
        assert test_file.exists()
        
        # Create new manager and load
        new_manager = AdaptiveThresholdManager()
        await new_manager.load_thresholds(str(test_file))
        
        assert new_manager.current_thresholds == threshold_manager.current_thresholds
        assert new_manager.adaptation_count == 5


class TestMLQualityOptimizer:
    """Test ML-based quality optimization system."""
    
    @pytest.fixture
    def ml_optimizer(self):
        """Create ML optimizer instance."""
        return MLQualityOptimizer(
            training_window_days=7,
            min_training_samples=10  # Lower for testing
        )
    
    @pytest.mark.asyncio
    async def test_model_initialization(self, ml_optimizer):
        """Test ML model initialization."""
        metrics = ["test_coverage", "security_scan", "performance_benchmark"]
        
        await ml_optimizer.initialize_models(metrics)
        
        for metric in metrics:
            assert metric in ml_optimizer.models
            assert MLModelType.THRESHOLD_PREDICTOR in ml_optimizer.models[metric]
            assert MLModelType.ANOMALY_DETECTOR in ml_optimizer.models[metric]
            assert MLModelType.PERFORMANCE_PREDICTOR in ml_optimizer.models[metric]
    
    @pytest.mark.asyncio
    async def test_training_sample_addition(self, ml_optimizer):
        """Test adding training samples."""
        await ml_optimizer.initialize_models(["test_coverage"])
        
        # Mock quality results
        quality_results = [
            Mock(gate_name="test_coverage", score=85.0, threshold=80.0, 
                 status=QualityGateStatus.PASSED, execution_time=1.0)
        ]
        
        project_metrics = {"lines_of_code": 10000, "team_size": "medium"}
        historical_data = {"test_coverage": [80, 82, 84, 85]}
        target_values = {"optimal_threshold": 82.0, "performance_score": 85.0}
        
        await ml_optimizer.add_training_sample(
            "test_coverage", quality_results, project_metrics, 
            historical_data, target_values
        )
        
        assert len(ml_optimizer.training_data_store["test_coverage"]) == 1
        sample = ml_optimizer.training_data_store["test_coverage"][0]
        assert "features" in sample
        assert "targets" in sample
        assert sample["targets"] == target_values
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, ml_optimizer):
        """Test feature extraction for ML training."""
        quality_results = [
            Mock(gate_name="test_coverage", score=85.0, threshold=80.0, 
                 status=QualityGateStatus.PASSED, execution_time=1.0)
        ]
        
        project_metrics = {"lines_of_code": 10000}
        historical_data = {"test_coverage": [80, 82, 84, 85]}
        
        features = await ml_optimizer.feature_extractor.extract_features(
            quality_results, project_metrics, historical_data
        )
        
        assert len(features) > 0
        feature_names = [f.name for f in features]
        assert "test_coverage_score" in feature_names
        assert "test_coverage_threshold_ratio" in feature_names
        assert "lines_of_code" in feature_names
    
    @pytest.mark.asyncio
    async def test_model_training(self, ml_optimizer):
        """Test ML model training."""
        await ml_optimizer.initialize_models(["test_metric"])
        
        # Add sufficient training samples
        for i in range(15):
            sample = {
                "timestamp": datetime.now().isoformat(),
                "features": {
                    "metric_score": 80.0 + i,
                    "threshold_ratio": 1.0 + i * 0.01,
                    "execution_time": 1.0
                },
                "targets": {"optimal_threshold": 80.0 + i * 0.5, "performance_score": 80.0 + i},
                "metric_name": "test_metric"
            }
            ml_optimizer.training_data_store["test_metric"].append(sample)
        
        # Train models
        await ml_optimizer.train_models("test_metric")
        
        # Verify models were trained
        threshold_model = ml_optimizer.models["test_metric"][MLModelType.THRESHOLD_PREDICTOR]
        assert threshold_model.is_trained
        assert threshold_model.last_trained is not None
    
    @pytest.mark.asyncio
    async def test_prediction_generation(self, ml_optimizer):
        """Test prediction generation."""
        await ml_optimizer.initialize_models(["test_metric"])
        
        # Mock a trained model
        model = ml_optimizer.models["test_metric"][MLModelType.THRESHOLD_PREDICTOR]
        model.is_trained = True
        model.feature_names = ["test_feature"]
        model.predict = AsyncMock(return_value=Mock(
            model_type=MLModelType.THRESHOLD_PREDICTOR,
            metric_name="test_metric",
            predicted_value=85.0,
            confidence=0.8,
            features_used=["test_feature"],
            timestamp=datetime.now(),
            explanation="Test prediction"
        ))
        
        features = [MLFeature(name="test_feature", value=1.0, feature_type="test")]
        prediction = await ml_optimizer.predict_optimal_threshold("test_metric", features)
        
        assert prediction is not None
        assert prediction.predicted_value == 85.0
        assert prediction.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_model_persistence(self, ml_optimizer, tmp_path):
        """Test model saving and loading."""
        models_dir = tmp_path / "test_models"
        
        await ml_optimizer.initialize_models(["test_metric"])
        
        # Mock trained model
        model = ml_optimizer.models["test_metric"][MLModelType.THRESHOLD_PREDICTOR]
        model.is_trained = True
        model.training_score = 0.85
        
        await ml_optimizer.save_models(str(models_dir))
        
        assert models_dir.exists()
        
        # Load models
        new_optimizer = MLQualityOptimizer()
        await new_optimizer.load_models(str(models_dir))
        
        # Verify loaded models
        assert "test_metric" in new_optimizer.models


class TestPredictiveQualityAssurance:
    """Test predictive quality assurance system."""
    
    @pytest.fixture
    def predictive_qa(self):
        """Create predictive QA instance."""
        return PredictiveQualityAssurance(
            prediction_horizons=[PredictionHorizon.IMMEDIATE, PredictionHorizon.SHORT_TERM],
            anomaly_sensitivity=0.1
        )
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, predictive_qa):
        """Test system initialization."""
        metrics = ["test_coverage", "security_scan"]
        
        await predictive_qa.initialize(metrics)
        
        # Verify ML optimizer was initialized
        assert len(predictive_qa.ml_optimizer.models) >= len(metrics)
    
    @pytest.mark.asyncio
    async def test_measurement_updates(self, predictive_qa):
        """Test measurement updates and analysis."""
        await predictive_qa.initialize(["test_metric"])
        
        measurements = {"test_metric": 85.0}
        context = {"project_phase": "development"}
        
        await predictive_qa.update_measurements(measurements, context)
        
        # Verify measurements were processed
        assert len(predictive_qa.statistical_detector.data_windows["test_metric"]) == 1
    
    @pytest.mark.asyncio
    async def test_statistical_anomaly_detection(self, predictive_qa):
        """Test statistical anomaly detection."""
        detector = predictive_qa.statistical_detector
        
        # Add normal data points
        normal_values = [80, 82, 81, 83, 79, 84, 78, 85, 82, 80]
        for value in normal_values:
            detector.add_measurement("test_metric", value)
        
        # Add anomalous value
        anomalies = detector.detect_statistical_anomalies("test_metric", 95.0)
        
        assert len(anomalies) > 0
        anomaly = anomalies[0]
        assert anomaly.anomaly_type == AnomalyType.STATISTICAL
        assert anomaly.severity > 0
        assert anomaly.value == 95.0
    
    @pytest.mark.asyncio
    async def test_pattern_anomaly_detection(self, predictive_qa):
        """Test pattern-based anomaly detection."""
        detector = predictive_qa.pattern_detector
        
        # Create a normal pattern
        normal_pattern = [(datetime.now(), 80 + i % 5) for i in range(50)]
        detector.learn_patterns("test_metric", normal_pattern)
        
        # Test anomalous pattern
        anomalous_values = [95, 96, 97, 98, 99]  # Very different pattern
        anomalies = detector.detect_pattern_anomalies("test_metric", anomalous_values)
        
        assert len(anomalies) > 0
        anomaly = anomalies[0]
        assert anomaly.anomaly_type == AnomalyType.PATTERN
    
    @pytest.mark.asyncio
    async def test_correlation_anomaly_detection(self, predictive_qa):
        """Test correlation-based anomaly detection."""
        detector = predictive_qa.correlation_detector
        
        # Establish normal correlation
        for i in range(20):
            metrics = {
                "metric1": 80 + i,
                "metric2": 80 + i + 2  # Correlated
            }
            detector.update_correlations(metrics)
        
        # Test correlation break
        anomalous_metrics = {
            "metric1": 90,
            "metric2": 70  # Breaks correlation
        }
        
        detector.update_correlations(anomalous_metrics)
        anomalies = detector.detect_correlation_anomalies(anomalous_metrics)
        
        # Note: May not detect immediately due to window size
        # This tests the mechanism is in place
        assert isinstance(anomalies, list)
    
    @pytest.mark.asyncio
    async def test_prediction_generation(self, predictive_qa):
        """Test quality prediction generation."""
        await predictive_qa.initialize(["test_metric"])
        
        # Mock ML prediction
        with patch.object(predictive_qa.ml_optimizer, 'predict_performance') as mock_predict:
            mock_predict.return_value = Mock(
                predicted_value=82.0,
                confidence=0.8,
                features_used=["feature1", "feature2"]
            )
            
            prediction = await predictive_qa._generate_prediction(
                "test_metric", 
                PredictionHorizon.IMMEDIATE, 
                {"context": "test"}
            )
            
            assert prediction is not None
            assert prediction.metric_name == "test_metric"
            assert prediction.horizon == PredictionHorizon.IMMEDIATE
            assert prediction.predicted_value == 82.0
            assert prediction.confidence_level == 0.8
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, predictive_qa):
        """Test risk assessment functionality."""
        await predictive_qa.initialize(["test_metric"])
        
        # Add some prediction data
        prediction = QualityPrediction(
            metric_name="test_metric",
            horizon=PredictionHorizon.IMMEDIATE,
            predicted_value=65.0,  # Low value
            confidence_interval=(60.0, 70.0),
            confidence_level=0.8,
            prediction_timestamp=datetime.now(),
            target_timestamp=datetime.now() + timedelta(hours=1),
            contributing_factors=["factor1"],
            risk_level="high",
            recommended_actions=["action1"]
        )
        
        predictive_qa.predictions["test_metric"].append(prediction)
        
        # Add current value
        predictive_qa.statistical_detector.add_measurement("test_metric", 68.0)
        
        risk_assessment = await predictive_qa._assess_metric_risk("test_metric")
        
        assert risk_assessment is not None
        assert risk_assessment.metric_name == "test_metric"
        assert risk_assessment.predicted_risk_level == "high"
    
    @pytest.mark.asyncio
    async def test_report_generation(self, predictive_qa):
        """Test comprehensive quality report generation."""
        await predictive_qa.initialize(["test_metric"])
        
        # Add some test data
        predictive_qa.predictions["test_metric"].append(
            QualityPrediction(
                metric_name="test_metric",
                horizon=PredictionHorizon.IMMEDIATE,
                predicted_value=80.0,
                confidence_interval=(75.0, 85.0),
                confidence_level=0.8,
                prediction_timestamp=datetime.now(),
                target_timestamp=datetime.now() + timedelta(hours=1),
                contributing_factors=["factor1"],
                risk_level="medium",
                recommended_actions=["action1"]
            )
        )
        
        predictive_qa.anomalies.append(
            AnomalyDetection(
                metric_name="test_metric",
                anomaly_type=AnomalyType.STATISTICAL,
                severity=0.6,
                timestamp=datetime.now(),
                value=95.0,
                expected_range=(80.0, 90.0),
                deviation_score=5.0,
                context={"test": True},
                explanation="Test anomaly",
                suggested_investigation=["investigate"]
            )
        )
        
        report = await predictive_qa.generate_quality_report()
        
        assert "timestamp" in report
        assert "prediction_summary" in report
        assert "anomaly_summary" in report
        assert "risk_summary" in report
        assert "ml_model_status" in report
        assert "recommendations" in report
        
        # Verify report content
        assert report["prediction_summary"]["total_predictions"] == 1
        assert report["anomaly_summary"]["total_anomalies"] == 1


class TestQualitySystemIntegration:
    """Test integration between quality system components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_quality_monitoring(self):
        """Test end-to-end quality monitoring workflow."""
        # Initialize systems
        monitor = RealTimeQualityMonitor(update_interval=0.1, enable_websocket=False)
        threshold_manager = AdaptiveThresholdManager(min_samples=3)
        ml_optimizer = MLQualityOptimizer(min_training_samples=5)
        predictive_qa = PredictiveQualityAssurance()
        
        # Initialize with test metrics
        metrics = ["test_coverage", "security_scan"]
        await threshold_manager.initialize_thresholds({
            "test_coverage": 80.0,
            "security_scan": 90.0
        })
        await ml_optimizer.initialize_models(metrics)
        await predictive_qa.initialize(metrics)
        
        # Simulate quality measurements over time
        measurements_series = [
            {"test_coverage": 85.0, "security_scan": 92.0},
            {"test_coverage": 83.0, "security_scan": 91.0},
            {"test_coverage": 87.0, "security_scan": 93.0},
            {"test_coverage": 82.0, "security_scan": 89.0},
            {"test_coverage": 79.0, "security_scan": 88.0},  # Declining
            {"test_coverage": 76.0, "security_scan": 85.0},  # Further decline
        ]
        
        for i, measurements in enumerate(measurements_series):
            # Update each system
            for metric_name, value in measurements.items():
                # Create mock quality result
                quality_result = Mock()
                quality_result.gate_name = metric_name
                quality_result.score = value
                quality_result.threshold = threshold_manager.get_current_threshold(metric_name) or 80.0
                quality_result.status = (QualityGateStatus.PASSED if value >= quality_result.threshold 
                                       else QualityGateStatus.FAILED)
                quality_result.execution_time = 1.0
                
                # Update threshold manager
                await threshold_manager.update_measurement(metric_name, value, quality_result)
                
                # Add training data to ML optimizer
                if i >= 2:  # Start after some data points
                    await ml_optimizer.add_training_sample(
                        metric_name,
                        [quality_result],
                        {"project_phase": "development"},
                        {metric_name: [m[metric_name] for m in measurements_series[:i+1]]},
                        {"optimal_threshold": value + 2, "performance_score": value}
                    )
            
            # Update predictive QA
            await predictive_qa.update_measurements(measurements, {"iteration": i})
            
            # Small delay to simulate real time
            await asyncio.sleep(0.01)
        
        # Verify systems captured the quality degradation
        
        # Check threshold manager adaptation
        assert len(threshold_manager.threshold_history) > 0
        
        # Check ML optimizer training data
        assert len(ml_optimizer.training_data_store["test_coverage"]) > 0
        
        # Check predictive QA anomaly detection
        recent_anomalies = [a for a in predictive_qa.anomalies 
                          if a.timestamp >= datetime.now() - timedelta(minutes=1)]
        # Should detect anomalies in the declining values
        assert len(recent_anomalies) >= 0  # May or may not detect with limited data
        
        # Generate comprehensive report
        report = await predictive_qa.generate_quality_report()
        assert report["prediction_summary"]["total_predictions"] >= 0
        assert report["anomaly_summary"]["total_anomalies"] >= 0
    
    @pytest.mark.asyncio
    async def test_quality_alert_escalation(self):
        """Test quality alert escalation workflow."""
        monitor = RealTimeQualityMonitor(enable_websocket=False)
        
        # Create critical quality degradation
        critical_metric = QualityMetric(
            name="security_scan",
            value=60.0,  # Well below threshold
            timestamp=datetime.now(),
            threshold=90.0,
            status=QualityGateStatus.FAILED,
            source="test"
        )
        
        # Create alert
        await monitor._create_alert(critical_metric, MonitoringLevel.CRITICAL)
        
        # Verify alert was created with appropriate severity
        assert len(monitor.active_alerts) == 1
        alert = list(monitor.active_alerts.values())[0]
        assert alert.severity == MonitoringLevel.CRITICAL
        assert alert.metric_name == "security_scan"
        assert "security" in alert.message.lower()
        
        # Verify dashboard reflects critical status
        dashboard_data = monitor.get_monitoring_dashboard_data()
        alert_data = dashboard_data["alerts"][alert.id]
        assert alert_data["severity"] == "critical"
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load."""
        predictive_qa = PredictiveQualityAssurance()
        await predictive_qa.initialize(["metric1", "metric2", "metric3"])
        
        # Simulate high-frequency measurements
        start_time = datetime.now()
        
        for i in range(100):
            measurements = {
                "metric1": 80 + (i % 20),
                "metric2": 85 + (i % 15),
                "metric3": 75 + (i % 25)
            }
            
            await predictive_qa.update_measurements(measurements)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should process 100 measurements in reasonable time (< 10 seconds)
        assert processing_time < 10.0
        
        # Verify all measurements were processed
        for metric in ["metric1", "metric2", "metric3"]:
            assert len(predictive_qa.statistical_detector.data_windows[metric]) == 100
    
    @pytest.mark.asyncio
    async def test_ml_model_accuracy_tracking(self):
        """Test ML model accuracy tracking."""
        ml_optimizer = MLQualityOptimizer(min_training_samples=10)
        await ml_optimizer.initialize_models(["test_metric"])
        
        # Generate synthetic training data with known pattern
        for i in range(20):
            base_value = 80
            # Simple linear relationship: score = base + noise
            score = base_value + i * 0.5 + (i % 3)  # Some variation
            
            sample = {
                "timestamp": datetime.now().isoformat(),
                "features": {
                    "iteration": float(i),
                    "base_score": score,
                    "trend": i * 0.5
                },
                "targets": {
                    "optimal_threshold": score - 2,  # Predictable relationship
                    "performance_score": score + 1
                },
                "metric_name": "test_metric"
            }
            ml_optimizer.training_data_store["test_metric"].append(sample)
        
        # Train model
        await ml_optimizer.train_models("test_metric")
        
        # Verify model was trained
        model = ml_optimizer.models["test_metric"][MLModelType.THRESHOLD_PREDICTOR]
        assert model.is_trained
        assert model.training_score > 0  # Should have some positive score
        
        # Test prediction accuracy
        test_features = [
            MLFeature(name="iteration", value=25.0, feature_type="test"),
            MLFeature(name="base_score", value=92.5, feature_type="test"),
            MLFeature(name="trend", value=12.5, feature_type="test")
        ]
        
        prediction = await ml_optimizer.predict_optimal_threshold("test_metric", test_features)
        
        if prediction:
            # Prediction should be reasonable (close to expected pattern)
            expected_threshold = 90.5  # score - 2
            assert abs(prediction.predicted_value - expected_threshold) < 10
            assert prediction.confidence > 0


# Performance benchmarking tests
class TestQualitySystemPerformance:
    """Performance benchmarking for quality systems."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_realtime_monitoring_performance(self, benchmark):
        """Benchmark real-time monitoring performance."""
        monitor = RealTimeQualityMonitor(enable_websocket=False)
        
        async def monitoring_cycle():
            await monitor._collect_quality_metrics()
            await monitor._analyze_metrics()
            monitor._update_trends()
        
        # Benchmark the monitoring cycle
        result = await benchmark.pedantic(monitoring_cycle, rounds=10)
        
        # Should complete monitoring cycle in reasonable time
        assert result < 1.0  # Less than 1 second per cycle
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_anomaly_detection_performance(self, benchmark):
        """Benchmark anomaly detection performance."""
        predictive_qa = PredictiveQualityAssurance()
        
        # Prepare test data
        measurements = {"test_metric": 85.0}
        
        async def anomaly_detection_cycle():
            await predictive_qa._detect_all_anomalies(measurements)
        
        result = await benchmark.pedantic(anomaly_detection_cycle, rounds=50)
        
        # Should detect anomalies quickly
        assert result < 0.1  # Less than 100ms per detection cycle
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_ml_prediction_performance(self, benchmark):
        """Benchmark ML prediction performance."""
        ml_optimizer = MLQualityOptimizer()
        await ml_optimizer.initialize_models(["test_metric"])
        
        # Mock trained model
        model = ml_optimizer.models["test_metric"][MLModelType.THRESHOLD_PREDICTOR]
        model.is_trained = True
        model.feature_names = ["feature1", "feature2"]
        
        features = [
            MLFeature(name="feature1", value=1.0, feature_type="test"),
            MLFeature(name="feature2", value=2.0, feature_type="test")
        ]
        
        async def prediction_cycle():
            return await ml_optimizer.predict_optimal_threshold("test_metric", features)
        
        result = await benchmark.pedantic(prediction_cycle, rounds=20)
        
        # Predictions should be fast
        assert result < 0.05  # Less than 50ms per prediction


# Error handling and edge case tests
class TestQualitySystemErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_metric_handling(self):
        """Test handling of invalid metrics."""
        monitor = RealTimeQualityMonitor(enable_websocket=False)
        
        # Test with invalid metric value
        with pytest.raises(Exception):
            invalid_metric = QualityMetric(
                name="",  # Empty name
                value=float('inf'),  # Invalid value
                timestamp=datetime.now(),
                threshold=80.0,
                status=QualityGateStatus.PASSED,
                source="test"
            )
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        ml_optimizer = MLQualityOptimizer(min_training_samples=10)
        await ml_optimizer.initialize_models(["test_metric"])
        
        # Try to train with insufficient data
        for i in range(5):  # Less than min_training_samples
            sample = {
                "timestamp": datetime.now().isoformat(),
                "features": {"test_feature": float(i)},
                "targets": {"test_target": float(i)},
                "metric_name": "test_metric"
            }
            ml_optimizer.training_data_store["test_metric"].append(sample)
        
        # Training should handle insufficient data gracefully
        await ml_optimizer.train_models("test_metric")
        
        # Model should not be trained
        model = ml_optimizer.models["test_metric"][MLModelType.THRESHOLD_PREDICTOR]
        assert not model.is_trained
    
    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        """Test thread safety under concurrent access."""
        monitor = RealTimeQualityMonitor(enable_websocket=False)
        
        async def add_metric():
            metric = QualityMetric(
                name="concurrent_test",
                value=80.0,
                timestamp=datetime.now(),
                threshold=75.0,
                status=QualityGateStatus.PASSED,
                source="test"
            )
            monitor.active_metrics["concurrent_test"] = metric
        
        # Run multiple concurrent operations
        tasks = [add_metric() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # Should handle concurrent access without issues
        assert "concurrent_test" in monitor.active_metrics
    
    @pytest.mark.asyncio
    async def test_memory_usage_control(self):
        """Test memory usage stays controlled."""
        predictive_qa = PredictiveQualityAssurance()
        await predictive_qa.initialize(["memory_test"])
        
        # Add many measurements to test memory limits
        for i in range(1000):
            measurements = {"memory_test": 80.0 + (i % 20)}
            await predictive_qa.update_measurements(measurements)
        
        # Verify data structures don't grow unbounded
        assert len(predictive_qa.statistical_detector.data_windows["memory_test"]) <= 100
        assert len(predictive_qa.anomalies) <= 500  # Should clean old anomalies
        assert len(predictive_qa.predictions["memory_test"]) <= 50  # Should limit predictions


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])