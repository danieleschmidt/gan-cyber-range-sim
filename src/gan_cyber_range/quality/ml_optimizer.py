"""Machine Learning-based quality optimization system."""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from .quality_gates import QualityGateResult, QualityGateStatus
from .adaptive_thresholds import AdaptiveThresholdManager
from ..core.error_handling import CyberRangeError, ErrorSeverity


class MLModelType(Enum):
    """Types of ML models for quality optimization."""
    THRESHOLD_PREDICTOR = "threshold_predictor"
    ANOMALY_DETECTOR = "anomaly_detector"
    PERFORMANCE_PREDICTOR = "performance_predictor"
    RISK_ASSESSOR = "risk_assessor"


@dataclass
class MLFeature:
    """Feature for ML model training."""
    name: str
    value: float
    feature_type: str
    importance: float = 0.0
    
    
@dataclass
class MLPrediction:
    """ML model prediction result."""
    model_type: MLModelType
    metric_name: str
    predicted_value: float
    confidence: float
    features_used: List[str]
    timestamp: datetime
    explanation: str = ""


@dataclass
class TrainingData:
    """Training data for ML models."""
    features: List[List[float]]
    targets: List[float]
    feature_names: List[str]
    timestamps: List[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractor:
    """Extracts features for ML model training."""
    
    def __init__(self):
        self.logger = logging.getLogger("feature_extractor")
        
    async def extract_features(
        self, 
        quality_results: List[QualityGateResult],
        project_metrics: Dict[str, Any],
        historical_data: Dict[str, List[float]]
    ) -> List[MLFeature]:
        """Extract comprehensive features for ML training."""
        features = []
        
        # Quality gate features
        features.extend(await self._extract_quality_features(quality_results))
        
        # Project context features
        features.extend(await self._extract_project_features(project_metrics))
        
        # Historical trend features
        features.extend(await self._extract_trend_features(historical_data))
        
        # Time-based features
        features.extend(await self._extract_temporal_features())
        
        return features
    
    async def _extract_quality_features(self, results: List[QualityGateResult]) -> List[MLFeature]:
        """Extract features from quality gate results."""
        features = []
        
        for result in results:
            # Basic metrics
            features.append(MLFeature(
                name=f"{result.gate_name}_score",
                value=result.score,
                feature_type="quality_metric"
            ))
            
            features.append(MLFeature(
                name=f"{result.gate_name}_threshold_ratio",
                value=result.score / result.threshold if result.threshold > 0 else 0,
                feature_type="quality_ratio"
            ))
            
            features.append(MLFeature(
                name=f"{result.gate_name}_execution_time",
                value=result.execution_time,
                feature_type="performance_metric"
            ))
            
            # Status encoding
            status_value = {
                QualityGateStatus.PASSED: 1.0,
                QualityGateStatus.WARNING: 0.5,
                QualityGateStatus.FAILED: 0.0,
                QualityGateStatus.ERROR: -1.0,
                QualityGateStatus.SKIPPED: -0.5
            }.get(result.status, 0.0)
            
            features.append(MLFeature(
                name=f"{result.gate_name}_status",
                value=status_value,
                feature_type="categorical"
            ))
        
        # Aggregate features
        if results:
            scores = [r.score for r in results]
            features.append(MLFeature(
                name="overall_quality_score",
                value=np.mean(scores),
                feature_type="aggregate"
            ))
            
            features.append(MLFeature(
                name="quality_score_variance",
                value=np.var(scores),
                feature_type="aggregate"
            ))
            
            passed_ratio = len([r for r in results if r.status == QualityGateStatus.PASSED]) / len(results)
            features.append(MLFeature(
                name="passed_ratio",
                value=passed_ratio,
                feature_type="aggregate"
            ))
        
        return features
    
    async def _extract_project_features(self, project_metrics: Dict[str, Any]) -> List[MLFeature]:
        """Extract project context features."""
        features = []
        
        # Code metrics
        if "lines_of_code" in project_metrics:
            features.append(MLFeature(
                name="lines_of_code",
                value=float(project_metrics["lines_of_code"]),
                feature_type="project_metric"
            ))
        
        if "file_count" in project_metrics:
            features.append(MLFeature(
                name="file_count",
                value=float(project_metrics["file_count"]),
                feature_type="project_metric"
            ))
        
        # Team metrics
        if "team_size" in project_metrics:
            team_size_map = {"small": 1.0, "medium": 2.0, "large": 3.0}
            features.append(MLFeature(
                name="team_size",
                value=team_size_map.get(project_metrics["team_size"], 2.0),
                feature_type="categorical"
            ))
        
        # Project phase
        if "project_phase" in project_metrics:
            phase_map = {"initial": 1.0, "development": 2.0, "testing": 3.0, "production": 4.0}
            features.append(MLFeature(
                name="project_phase",
                value=phase_map.get(project_metrics["project_phase"], 2.0),
                feature_type="categorical"
            ))
        
        # Complexity metrics
        if "cyclomatic_complexity" in project_metrics:
            features.append(MLFeature(
                name="cyclomatic_complexity",
                value=float(project_metrics["cyclomatic_complexity"]),
                feature_type="complexity_metric"
            ))
        
        return features
    
    async def _extract_trend_features(self, historical_data: Dict[str, List[float]]) -> List[MLFeature]:
        """Extract trend-based features from historical data."""
        features = []
        
        for metric_name, values in historical_data.items():
            if len(values) < 5:
                continue
            
            # Trend direction
            recent_avg = np.mean(values[-5:])
            older_avg = np.mean(values[:-5]) if len(values) > 5 else recent_avg
            trend = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
            
            features.append(MLFeature(
                name=f"{metric_name}_trend",
                value=trend,
                feature_type="trend"
            ))
            
            # Volatility
            volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            features.append(MLFeature(
                name=f"{metric_name}_volatility",
                value=volatility,
                feature_type="volatility"
            ))
            
            # Recent performance
            features.append(MLFeature(
                name=f"{metric_name}_recent_avg",
                value=recent_avg,
                feature_type="recent_performance"
            ))
        
        return features
    
    async def _extract_temporal_features(self) -> List[MLFeature]:
        """Extract time-based features."""
        now = datetime.now()
        
        features = [
            MLFeature(
                name="hour_of_day",
                value=float(now.hour),
                feature_type="temporal"
            ),
            MLFeature(
                name="day_of_week",
                value=float(now.weekday()),
                feature_type="temporal"
            ),
            MLFeature(
                name="is_weekend",
                value=1.0 if now.weekday() >= 5 else 0.0,
                feature_type="temporal"
            )
        ]
        
        return features


class QualityMLModel:
    """Machine learning model for quality optimization."""
    
    def __init__(
        self,
        model_type: MLModelType,
        metric_name: str,
        model_class: Any = None
    ):
        self.model_type = model_type
        self.metric_name = metric_name
        self.model = model_class or self._get_default_model()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names: List[str] = []
        self.training_score = 0.0
        self.last_trained = None
        
        self.logger = logging.getLogger(f"ml_model_{model_type.value}")
    
    def _get_default_model(self):
        """Get default model based on type."""
        if self.model_type == MLModelType.THRESHOLD_PREDICTOR:
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == MLModelType.ANOMALY_DETECTOR:
            return IsolationForest(contamination=0.1, random_state=42)
        elif self.model_type == MLModelType.PERFORMANCE_PREDICTOR:
            return RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            return LinearRegression()
    
    async def train(self, training_data: TrainingData) -> Dict[str, float]:
        """Train the ML model."""
        if len(training_data.features) < 10:
            raise CyberRangeError(
                f"Insufficient training data: {len(training_data.features)} samples",
                severity=ErrorSeverity.MEDIUM
            )
        
        # Prepare data
        X = np.array(training_data.features)
        y = np.array(training_data.targets)
        self.feature_names = training_data.feature_names.copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        if self.model_type == MLModelType.ANOMALY_DETECTOR:
            self.model.fit(X_train)
            # For anomaly detection, we don't have traditional accuracy
            training_metrics = {"status": "trained"}
        else:
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            training_metrics = {
                "mae": mae,
                "r2_score": r2,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            self.training_score = r2
        
        self.is_trained = True
        self.last_trained = datetime.now()
        
        self.logger.info(f"Model trained for {self.metric_name}: {training_metrics}")
        
        return training_metrics
    
    async def predict(self, features: List[MLFeature]) -> MLPrediction:
        """Make prediction using trained model."""
        if not self.is_trained:
            raise CyberRangeError(
                f"Model not trained for {self.metric_name}",
                severity=ErrorSeverity.HIGH
            )
        
        # Prepare feature vector
        feature_dict = {f.name: f.value for f in features}
        feature_vector = [feature_dict.get(name, 0.0) for name in self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform([feature_vector])
        
        # Make prediction
        if self.model_type == MLModelType.ANOMALY_DETECTOR:
            prediction = self.model.predict(X_scaled)[0]
            confidence = abs(self.model.score_samples(X_scaled)[0])
            explanation = "Anomaly detected" if prediction == -1 else "Normal behavior"
        else:
            prediction = self.model.predict(X_scaled)[0]
            
            # Calculate confidence based on feature importance and model certainty
            if hasattr(self.model, 'feature_importances_'):
                feature_importance_sum = sum(
                    self.model.feature_importances_[i] * abs(feature_vector[i])
                    for i in range(len(feature_vector))
                )
                confidence = min(1.0, feature_importance_sum / np.sum(self.model.feature_importances_))
            else:
                confidence = 0.8  # Default confidence
            
            explanation = f"Predicted based on {len(self.feature_names)} features"
        
        return MLPrediction(
            model_type=self.model_type,
            metric_name=self.metric_name,
            predicted_value=float(prediction),
            confidence=float(confidence),
            features_used=self.feature_names,
            timestamp=datetime.now(),
            explanation=explanation
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    async def save_model(self, file_path: str):
        """Save trained model to file."""
        model_data = {
            "model_type": self.model_type.value,
            "metric_name": self.metric_name,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "training_score": self.training_score,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {file_path}")
    
    async def load_model(self, file_path: str):
        """Load trained model from file."""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self.is_trained = model_data["is_trained"]
            self.training_score = model_data["training_score"]
            
            if model_data["last_trained"]:
                self.last_trained = datetime.fromisoformat(model_data["last_trained"])
            
            self.logger.info(f"Model loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {file_path}: {e}")
            raise


class MLQualityOptimizer:
    """ML-based quality optimization system."""
    
    def __init__(
        self,
        training_window_days: int = 30,
        retrain_interval_hours: int = 24,
        min_training_samples: int = 50
    ):
        self.training_window_days = training_window_days
        self.retrain_interval_hours = retrain_interval_hours
        self.min_training_samples = min_training_samples
        
        self.logger = logging.getLogger("ml_quality_optimizer")
        self.feature_extractor = FeatureExtractor()
        
        # ML models for different metrics
        self.models: Dict[str, Dict[MLModelType, QualityMLModel]] = defaultdict(dict)
        
        # Training data collection
        self.training_data_store: Dict[str, List[Dict]] = defaultdict(list)
        
        # Performance tracking
        self.prediction_history: List[MLPrediction] = []
        
    async def initialize_models(self, metrics: List[str]):
        """Initialize ML models for quality metrics."""
        for metric in metrics:
            # Threshold predictor
            self.models[metric][MLModelType.THRESHOLD_PREDICTOR] = QualityMLModel(
                MLModelType.THRESHOLD_PREDICTOR,
                metric
            )
            
            # Anomaly detector
            self.models[metric][MLModelType.ANOMALY_DETECTOR] = QualityMLModel(
                MLModelType.ANOMALY_DETECTOR,
                metric
            )
            
            # Performance predictor
            self.models[metric][MLModelType.PERFORMANCE_PREDICTOR] = QualityMLModel(
                MLModelType.PERFORMANCE_PREDICTOR,
                metric
            )
        
        self.logger.info(f"Initialized ML models for {len(metrics)} metrics")
    
    async def add_training_sample(
        self,
        metric_name: str,
        quality_results: List[QualityGateResult],
        project_metrics: Dict[str, Any],
        historical_data: Dict[str, List[float]],
        target_values: Dict[str, float]
    ):
        """Add training sample for ML models."""
        # Extract features
        features = await self.feature_extractor.extract_features(
            quality_results, project_metrics, historical_data
        )
        
        # Store training sample
        sample = {
            "timestamp": datetime.now().isoformat(),
            "features": {f.name: f.value for f in features},
            "targets": target_values,
            "metric_name": metric_name
        }
        
        self.training_data_store[metric_name].append(sample)
        
        # Limit storage to training window
        cutoff_time = datetime.now() - timedelta(days=self.training_window_days)
        self.training_data_store[metric_name] = [
            s for s in self.training_data_store[metric_name]
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_time
        ]
        
        self.logger.debug(f"Added training sample for {metric_name}")
    
    async def train_models(self, metric_name: str = None):
        """Train ML models for specified metric or all metrics."""
        metrics_to_train = [metric_name] if metric_name else list(self.models.keys())
        
        for metric in metrics_to_train:
            if metric not in self.training_data_store:
                continue
            
            samples = self.training_data_store[metric]
            if len(samples) < self.min_training_samples:
                self.logger.warning(
                    f"Insufficient training data for {metric}: {len(samples)} samples"
                )
                continue
            
            await self._train_metric_models(metric, samples)
    
    async def _train_metric_models(self, metric_name: str, samples: List[Dict]):
        """Train all models for a specific metric."""
        # Prepare training data
        all_features = []
        feature_names = []
        
        # Get consistent feature names from first sample
        if samples:
            feature_names = list(samples[0]["features"].keys())
        
        # Prepare feature matrix
        for sample in samples:
            feature_vector = [sample["features"].get(name, 0.0) for name in feature_names]
            all_features.append(feature_vector)
        
        # Train different model types
        for model_type in [MLModelType.THRESHOLD_PREDICTOR, MLModelType.PERFORMANCE_PREDICTOR]:
            if model_type not in self.models[metric_name]:
                continue
            
            # Prepare targets based on model type
            if model_type == MLModelType.THRESHOLD_PREDICTOR:
                targets = [sample["targets"].get("optimal_threshold", 85.0) for sample in samples]
            elif model_type == MLModelType.PERFORMANCE_PREDICTOR:
                targets = [sample["targets"].get("performance_score", 80.0) for sample in samples]
            else:
                continue
            
            training_data = TrainingData(
                features=all_features,
                targets=targets,
                feature_names=feature_names,
                timestamps=[datetime.fromisoformat(s["timestamp"]) for s in samples]
            )
            
            try:
                model = self.models[metric_name][model_type]
                training_metrics = await model.train(training_data)
                
                self.logger.info(
                    f"Trained {model_type.value} for {metric_name}: {training_metrics}"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_type.value} for {metric_name}: {e}")
        
        # Train anomaly detector (unsupervised)
        if MLModelType.ANOMALY_DETECTOR in self.models[metric_name]:
            try:
                training_data = TrainingData(
                    features=all_features,
                    targets=[0] * len(all_features),  # Dummy targets for anomaly detection
                    feature_names=feature_names,
                    timestamps=[datetime.fromisoformat(s["timestamp"]) for s in samples]
                )
                
                model = self.models[metric_name][MLModelType.ANOMALY_DETECTOR]
                training_metrics = await model.train(training_data)
                
                self.logger.info(f"Trained anomaly detector for {metric_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to train anomaly detector for {metric_name}: {e}")
    
    async def predict_optimal_threshold(
        self,
        metric_name: str,
        current_features: List[MLFeature]
    ) -> Optional[MLPrediction]:
        """Predict optimal threshold for metric."""
        if (metric_name not in self.models or 
            MLModelType.THRESHOLD_PREDICTOR not in self.models[metric_name]):
            return None
        
        model = self.models[metric_name][MLModelType.THRESHOLD_PREDICTOR]
        if not model.is_trained:
            return None
        
        try:
            prediction = await model.predict(current_features)
            self.prediction_history.append(prediction)
            return prediction
            
        except Exception as e:
            self.logger.error(f"Threshold prediction failed for {metric_name}: {e}")
            return None
    
    async def detect_anomaly(
        self,
        metric_name: str,
        current_features: List[MLFeature]
    ) -> Optional[MLPrediction]:
        """Detect anomalies in quality metrics."""
        if (metric_name not in self.models or 
            MLModelType.ANOMALY_DETECTOR not in self.models[metric_name]):
            return None
        
        model = self.models[metric_name][MLModelType.ANOMALY_DETECTOR]
        if not model.is_trained:
            return None
        
        try:
            prediction = await model.predict(current_features)
            if prediction.predicted_value == -1:  # Anomaly detected
                self.logger.warning(f"Anomaly detected in {metric_name}")
            
            self.prediction_history.append(prediction)
            return prediction
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed for {metric_name}: {e}")
            return None
    
    async def predict_performance(
        self,
        metric_name: str,
        current_features: List[MLFeature]
    ) -> Optional[MLPrediction]:
        """Predict performance for metric."""
        if (metric_name not in self.models or 
            MLModelType.PERFORMANCE_PREDICTOR not in self.models[metric_name]):
            return None
        
        model = self.models[metric_name][MLModelType.PERFORMANCE_PREDICTOR]
        if not model.is_trained:
            return None
        
        try:
            prediction = await model.predict(current_features)
            self.prediction_history.append(prediction)
            return prediction
            
        except Exception as e:
            self.logger.error(f"Performance prediction failed for {metric_name}: {e}")
            return None
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all ML models."""
        status = {
            "total_metrics": len(self.models),
            "trained_models": 0,
            "total_predictions": len(self.prediction_history),
            "training_data_samples": sum(len(samples) for samples in self.training_data_store.values()),
            "models": {}
        }
        
        for metric_name, metric_models in self.models.items():
            metric_status = {
                "model_types": list(metric_models.keys()),
                "trained_count": sum(1 for model in metric_models.values() if model.is_trained),
                "training_samples": len(self.training_data_store.get(metric_name, []))
            }
            
            for model_type, model in metric_models.items():
                metric_status[model_type.value] = {
                    "trained": model.is_trained,
                    "training_score": model.training_score,
                    "last_trained": model.last_trained.isoformat() if model.last_trained else None,
                    "feature_count": len(model.feature_names)
                }
            
            status["models"][metric_name] = metric_status
            if any(model.is_trained for model in metric_models.values()):
                status["trained_models"] += 1
        
        return status
    
    async def save_models(self, directory: str = "ml_models"):
        """Save all trained models."""
        models_dir = Path(directory)
        models_dir.mkdir(exist_ok=True)
        
        for metric_name, metric_models in self.models.items():
            for model_type, model in metric_models.items():
                if model.is_trained:
                    file_path = models_dir / f"{metric_name}_{model_type.value}.pkl"
                    await model.save_model(str(file_path))
        
        # Save training data
        training_data_file = models_dir / "training_data.json"
        with open(training_data_file, 'w') as f:
            json.dump(self.training_data_store, f, indent=2)
        
        self.logger.info(f"Saved ML models to {directory}")
    
    async def load_models(self, directory: str = "ml_models"):
        """Load trained models."""
        models_dir = Path(directory)
        if not models_dir.exists():
            self.logger.info(f"No models directory found at {directory}")
            return
        
        # Load training data
        training_data_file = models_dir / "training_data.json"
        if training_data_file.exists():
            with open(training_data_file) as f:
                self.training_data_store = defaultdict(list, json.load(f))
        
        # Load models
        for model_file in models_dir.glob("*.pkl"):
            try:
                # Parse filename to get metric and model type
                name_parts = model_file.stem.split("_")
                if len(name_parts) >= 2:
                    metric_name = "_".join(name_parts[:-1])
                    model_type_str = name_parts[-1]
                    
                    try:
                        model_type = MLModelType(model_type_str)
                        
                        if metric_name not in self.models:
                            self.models[metric_name] = {}
                        
                        self.models[metric_name][model_type] = QualityMLModel(
                            model_type, metric_name
                        )
                        
                        await self.models[metric_name][model_type].load_model(str(model_file))
                        
                    except ValueError:
                        self.logger.warning(f"Unknown model type in {model_file}")
                        
            except Exception as e:
                self.logger.error(f"Failed to load model from {model_file}: {e}")
        
        self.logger.info(f"Loaded ML models from {directory}")
    
    async def start_background_training(self):
        """Start background training process."""
        self.logger.info("Starting background ML training")
        
        while True:
            try:
                await asyncio.sleep(self.retrain_interval_hours * 3600)
                
                self.logger.info("Starting scheduled model retraining")
                await self.train_models()
                await self.save_models()
                
            except Exception as e:
                self.logger.error(f"Background training error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry