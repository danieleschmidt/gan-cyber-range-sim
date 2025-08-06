"""Machine learning-based threat detection and anomaly analysis."""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
import joblib
import pandas as pd


class MLModelType(Enum):
    """Types of ML models for threat detection."""
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"


class ThreatCategory(Enum):
    """Categories of threats detected by ML models."""
    MALWARE = "malware"
    INSIDER_THREAT = "insider_threat"
    APT = "advanced_persistent_threat"
    DDoS = "ddos_attack"
    DATA_EXFILTRATION = "data_exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNKNOWN = "unknown"


@dataclass
class MLDetectionResult:
    """Result from ML-based threat detection."""
    model_id: str
    model_type: MLModelType
    threat_category: ThreatCategory
    confidence: float
    anomaly_score: float
    features_analyzed: List[str]
    timestamp: datetime
    affected_entities: List[str]
    raw_prediction: Any
    feature_importance: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class TrainingData:
    """Training data for ML models."""
    features: np.ndarray
    labels: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractor:
    """Extract features from security events for ML analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger("FeatureExtractor")
        self.categorical_encoders = {}
        self.feature_cache = deque(maxlen=10000)
    
    async def extract_features(self, events: List[Dict[str, Any]]) -> TrainingData:
        """Extract features from security events."""
        if not events:
            return TrainingData(features=np.array([]), feature_names=[])
        
        feature_vectors = []
        feature_names = []
        timestamps = []
        
        for event in events:
            features = await self._extract_event_features(event)
            if features:
                feature_vectors.append(features)
                timestamps.append(
                    datetime.fromisoformat(event.get("timestamp", datetime.now().isoformat()))
                )
        
        if not feature_vectors:
            return TrainingData(features=np.array([]), feature_names=[])
        
        # Convert to numpy array
        features_array = np.array(feature_vectors)
        
        # Define feature names
        feature_names = [
            "hour_of_day", "day_of_week", "event_frequency",
            "data_volume", "connection_count", "failed_auth_count",
            "privilege_level", "process_count", "network_entropy",
            "file_access_count", "registry_changes", "dns_queries",
            "external_connections", "suspicious_domains", "file_entropy",
            "command_length", "script_execution", "admin_activity",
            "off_hours_activity", "geographic_anomaly"
        ]
        
        return TrainingData(
            features=features_array,
            feature_names=feature_names,
            timestamps=timestamps
        )
    
    async def _extract_event_features(self, event: Dict[str, Any]) -> Optional[List[float]]:
        """Extract numerical features from a single event."""
        try:
            timestamp = datetime.fromisoformat(event.get("timestamp", datetime.now().isoformat()))
            
            features = [
                # Temporal features
                timestamp.hour,  # 0-23
                timestamp.weekday(),  # 0-6
                
                # Event frequency (events per hour)
                self._calculate_event_frequency(event),
                
                # Data volume
                float(event.get("data_size", 0)) / 1024 / 1024,  # MB
                
                # Network features
                float(event.get("connection_count", 0)),
                float(event.get("failed_auth_count", 0)),
                
                # User/Process features
                self._encode_privilege_level(event.get("privilege_level", "user")),
                float(event.get("process_count", 0)),
                
                # Entropy measures
                self._calculate_network_entropy(event),
                
                # Activity counts
                float(event.get("file_access_count", 0)),
                float(event.get("registry_changes", 0)),
                float(event.get("dns_queries", 0)),
                
                # Suspicious indicators
                float(event.get("external_connections", 0)),
                float(event.get("suspicious_domains", 0)),
                self._calculate_file_entropy(event),
                
                # Command analysis
                len(event.get("command", "")),
                1.0 if self._is_script_execution(event) else 0.0,
                1.0 if event.get("admin_activity", False) else 0.0,
                
                # Behavioral indicators
                1.0 if self._is_off_hours(timestamp) else 0.0,
                1.0 if event.get("geographic_anomaly", False) else 0.0
            ]
            
            return features
        
        except Exception as e:
            self.logger.error(f"Error extracting features from event: {e}")
            return None
    
    def _calculate_event_frequency(self, event: Dict[str, Any]) -> float:
        """Calculate event frequency for the source."""
        source = event.get("source_ip", "unknown")
        
        # Count recent events from same source
        recent_count = len([
            e for e in self.feature_cache
            if e.get("source_ip") == source and
            datetime.fromisoformat(e.get("timestamp", "")) >
            datetime.now() - timedelta(hours=1)
        ])
        
        return float(recent_count)
    
    def _encode_privilege_level(self, privilege: str) -> float:
        """Encode privilege level as numerical value."""
        privilege_map = {
            "system": 4.0,
            "administrator": 3.0,
            "power_user": 2.0,
            "user": 1.0,
            "guest": 0.0
        }
        return privilege_map.get(privilege.lower(), 1.0)
    
    def _calculate_network_entropy(self, event: Dict[str, Any]) -> float:
        """Calculate entropy of network connections."""
        connections = event.get("network_connections", [])
        if not connections:
            return 0.0
        
        # Simple entropy calculation based on destination diversity
        destinations = [conn.get("destination") for conn in connections]
        unique_destinations = len(set(destinations))
        total_connections = len(destinations)
        
        if total_connections <= 1:
            return 0.0
        
        # Shannon entropy approximation
        entropy = unique_destinations / total_connections
        return entropy
    
    def _calculate_file_entropy(self, event: Dict[str, Any]) -> float:
        """Calculate file entropy score."""
        file_path = event.get("file_path", "")
        if not file_path:
            return 0.0
        
        # Simple entropy based on file path characteristics
        entropy = 0.0
        
        # Random-looking file names (high entropy)
        if any(c.isdigit() for c in file_path) and any(c.isalpha() for c in file_path):
            entropy += 0.3
        
        # Executable files in temp directories
        if "temp" in file_path.lower() and file_path.endswith((".exe", ".bat", ".ps1")):
            entropy += 0.4
        
        # Hidden files
        if "/." in file_path or "\\." in file_path:
            entropy += 0.2
        
        return min(entropy, 1.0)
    
    def _is_script_execution(self, event: Dict[str, Any]) -> bool:
        """Check if event represents script execution."""
        process = event.get("process", "").lower()
        command = event.get("command", "").lower()
        
        script_indicators = [
            "powershell", "cmd.exe", "bash", "python", "perl",
            "wscript", "cscript", "regsvr32"
        ]
        
        return any(indicator in process or indicator in command for indicator in script_indicators)
    
    def _is_off_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is outside business hours."""
        # Business hours: 7 AM to 7 PM, Monday to Friday
        if timestamp.weekday() >= 5:  # Weekend
            return True
        
        return timestamp.hour < 7 or timestamp.hour > 19


class MLThreatDetector:
    """Machine learning-based threat detection system."""
    
    def __init__(self):
        self.models = {}
        self.feature_extractor = FeatureExtractor()
        self.training_data = {}
        self.model_performance = {}
        self.logger = logging.getLogger("MLThreatDetector")
        
        # Initialize default models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize default ML models."""
        # Anomaly detection model
        self.models["anomaly_detector"] = {
            "model": IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            ),
            "type": MLModelType.ANOMALY_DETECTION,
            "scaler": StandardScaler(),
            "trained": False,
            "threat_categories": [ThreatCategory.UNKNOWN]
        }
        
        # Malware detection model
        self.models["malware_detector"] = {
            "model": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ),
            "type": MLModelType.CLASSIFICATION,
            "scaler": StandardScaler(),
            "label_encoder": LabelEncoder(),
            "trained": False,
            "threat_categories": [ThreatCategory.MALWARE]
        }
        
        # Insider threat detection
        self.models["insider_threat_detector"] = {
            "model": IsolationForest(
                contamination=0.05,
                random_state=42
            ),
            "type": MLModelType.ANOMALY_DETECTION,
            "scaler": StandardScaler(),
            "trained": False,
            "threat_categories": [ThreatCategory.INSIDER_THREAT]
        }
        
        # Network clustering for APT detection
        self.models["apt_detector"] = {
            "model": DBSCAN(eps=0.5, min_samples=5),
            "type": MLModelType.CLUSTERING,
            "scaler": StandardScaler(),
            "trained": False,
            "threat_categories": [ThreatCategory.APT]
        }
    
    async def train_models(self, training_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train ML models with security event data."""
        training_results = {}
        
        # Extract features from training data
        training_data = await self.feature_extractor.extract_features(training_events)
        
        if training_data.features.size == 0:
            self.logger.warning("No features extracted from training data")
            return {"error": "No valid training data"}
        
        self.logger.info(f"Training models with {len(training_events)} events and {len(training_data.feature_names)} features")
        
        # Train each model
        for model_id, model_config in self.models.items():
            try:
                result = await self._train_single_model(model_id, model_config, training_data)
                training_results[model_id] = result
            except Exception as e:
                self.logger.error(f"Error training model {model_id}: {e}")
                training_results[model_id] = {"error": str(e)}
        
        return training_results
    
    async def _train_single_model(
        self,
        model_id: str,
        model_config: Dict[str, Any],
        training_data: TrainingData
    ) -> Dict[str, Any]:
        """Train a single ML model."""
        model = model_config["model"]
        scaler = model_config["scaler"]
        model_type = model_config["type"]
        
        # Scale features
        features_scaled = scaler.fit_transform(training_data.features)
        
        if model_type == MLModelType.ANOMALY_DETECTION:
            # Unsupervised learning
            model.fit(features_scaled)
            
            # Evaluate on training data (for baseline)
            anomaly_scores = model.decision_function(features_scaled)
            anomalies = model.predict(features_scaled)
            anomaly_rate = np.sum(anomalies == -1) / len(anomalies)
            
            result = {
                "status": "trained",
                "model_type": "anomaly_detection",
                "anomaly_rate": anomaly_rate,
                "training_samples": len(training_data.features)
            }
        
        elif model_type == MLModelType.CLASSIFICATION:
            # Need labeled data for classification
            if training_data.labels is None:
                # Create synthetic labels based on anomaly detection
                anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
                synthetic_labels = anomaly_detector.fit_predict(features_scaled)
                training_data.labels = synthetic_labels
            
            # Train classifier
            label_encoder = model_config.get("label_encoder")
            if label_encoder:
                encoded_labels = label_encoder.fit_transform(training_data.labels)
            else:
                encoded_labels = training_data.labels
            
            model.fit(features_scaled, encoded_labels)
            
            # Get feature importance
            feature_importance = {}
            if hasattr(model, "feature_importances_"):
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(training_data.feature_names):
                        feature_importance[training_data.feature_names[i]] = float(importance)
            
            result = {
                "status": "trained",
                "model_type": "classification",
                "training_samples": len(training_data.features),
                "feature_importance": feature_importance
            }
        
        elif model_type == MLModelType.CLUSTERING:
            # Unsupervised clustering
            cluster_labels = model.fit_predict(features_scaled)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            noise_points = np.sum(cluster_labels == -1)
            
            result = {
                "status": "trained",
                "model_type": "clustering",
                "clusters_found": n_clusters,
                "noise_points": noise_points,
                "training_samples": len(training_data.features)
            }
        
        # Mark model as trained
        model_config["trained"] = True
        self.training_data[model_id] = training_data
        
        self.logger.info(f"Successfully trained model {model_id}")
        return result
    
    async def detect_threats(self, events: List[Dict[str, Any]]) -> List[MLDetectionResult]:
        """Detect threats in security events using trained ML models."""
        if not events:
            return []
        
        detections = []
        
        # Extract features
        event_data = await self.feature_extractor.extract_features(events)
        if event_data.features.size == 0:
            return detections
        
        # Run detection with each trained model
        for model_id, model_config in self.models.items():
            if not model_config["trained"]:
                continue
            
            try:
                model_detections = await self._detect_with_model(
                    model_id, model_config, event_data, events
                )
                detections.extend(model_detections)
            except Exception as e:
                self.logger.error(f"Error in threat detection with model {model_id}: {e}")
        
        return detections
    
    async def _detect_with_model(
        self,
        model_id: str,
        model_config: Dict[str, Any],
        event_data: TrainingData,
        original_events: List[Dict[str, Any]]
    ) -> List[MLDetectionResult]:
        """Detect threats using a specific model."""
        detections = []
        
        model = model_config["model"]
        scaler = model_config["scaler"]
        model_type = model_config["type"]
        threat_categories = model_config["threat_categories"]
        
        # Scale features
        features_scaled = scaler.transform(event_data.features)
        
        if model_type == MLModelType.ANOMALY_DETECTION:
            # Anomaly detection
            anomaly_scores = model.decision_function(features_scaled)
            predictions = model.predict(features_scaled)
            
            for i, (score, prediction) in enumerate(zip(anomaly_scores, predictions)):
                if prediction == -1:  # Anomaly detected
                    confidence = min(abs(score) / 2.0, 1.0)  # Normalize score to confidence
                    
                    detection = MLDetectionResult(
                        model_id=model_id,
                        model_type=model_type,
                        threat_category=threat_categories[0],
                        confidence=confidence,
                        anomaly_score=float(score),
                        features_analyzed=event_data.feature_names,
                        timestamp=datetime.now(),
                        affected_entities=self._extract_affected_entities(original_events[i]),
                        raw_prediction=prediction,
                        explanation=f"Anomaly detected with score {score:.3f}"
                    )
                    
                    detections.append(detection)
        
        elif model_type == MLModelType.CLASSIFICATION:
            # Classification
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled) if hasattr(model, "predict_proba") else None
            
            for i, prediction in enumerate(predictions):
                # Get confidence from probabilities
                if probabilities is not None:
                    confidence = float(np.max(probabilities[i]))
                else:
                    confidence = 0.8  # Default confidence
                
                # Only report high-confidence threats
                if confidence > 0.7:
                    # Get feature importance for explanation
                    feature_importance = {}
                    if hasattr(model, "feature_importances_"):
                        top_features = np.argsort(model.feature_importances_)[-3:][::-1]
                        for idx in top_features:
                            if idx < len(event_data.feature_names):
                                feature_importance[event_data.feature_names[idx]] = float(
                                    model.feature_importances_[idx]
                                )
                    
                    detection = MLDetectionResult(
                        model_id=model_id,
                        model_type=model_type,
                        threat_category=threat_categories[0],
                        confidence=confidence,
                        anomaly_score=0.0,
                        features_analyzed=event_data.feature_names,
                        timestamp=datetime.now(),
                        affected_entities=self._extract_affected_entities(original_events[i]),
                        raw_prediction=prediction,
                        feature_importance=feature_importance,
                        explanation=f"Classification prediction: {prediction} (confidence: {confidence:.3f})"
                    )
                    
                    detections.append(detection)
        
        elif model_type == MLModelType.CLUSTERING:
            # Clustering - detect outliers
            cluster_labels = model.fit_predict(features_scaled)
            
            for i, cluster_label in enumerate(cluster_labels):
                if cluster_label == -1:  # Noise/outlier point
                    detection = MLDetectionResult(
                        model_id=model_id,
                        model_type=model_type,
                        threat_category=threat_categories[0],
                        confidence=0.6,  # Medium confidence for clustering outliers
                        anomaly_score=0.0,
                        features_analyzed=event_data.feature_names,
                        timestamp=datetime.now(),
                        affected_entities=self._extract_affected_entities(original_events[i]),
                        raw_prediction=cluster_label,
                        explanation="Outlier detected in clustering analysis"
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def _extract_affected_entities(self, event: Dict[str, Any]) -> List[str]:
        """Extract affected entities from an event."""
        entities = []
        
        # Add common entity fields
        entity_fields = ["hostname", "source_ip", "username", "process", "file_path"]
        for field in entity_fields:
            value = event.get(field)
            if value:
                entities.append(f"{field}:{value}")
        
        return entities
    
    async def update_model_performance(self, detections: List[MLDetectionResult], feedback: List[bool]) -> None:
        """Update model performance metrics based on feedback."""
        if len(detections) != len(feedback):
            self.logger.warning("Mismatch between detections and feedback length")
            return
        
        # Group by model ID
        model_feedback = defaultdict(list)
        for detection, is_true_positive in zip(detections, feedback):
            model_feedback[detection.model_id].append({
                "detection": detection,
                "is_true_positive": is_true_positive
            })
        
        # Update performance metrics for each model
        for model_id, feedback_data in model_feedback.items():
            true_positives = sum(1 for f in feedback_data if f["is_true_positive"])
            false_positives = len(feedback_data) - true_positives
            
            precision = true_positives / len(feedback_data) if feedback_data else 0.0
            
            # Store performance metrics
            if model_id not in self.model_performance:
                self.model_performance[model_id] = {
                    "total_detections": 0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "precision": 0.0
                }
            
            perf = self.model_performance[model_id]
            perf["total_detections"] += len(feedback_data)
            perf["true_positives"] += true_positives
            perf["false_positives"] += false_positives
            perf["precision"] = perf["true_positives"] / perf["total_detections"]
            
            self.logger.info(
                f"Updated performance for model {model_id}: "
                f"Precision: {perf['precision']:.3f} "
                f"({perf['true_positives']}/{perf['total_detections']})"
            )
    
    async def save_models(self, file_path: str) -> bool:
        """Save trained models to disk."""
        try:
            model_data = {
                "models": {},
                "training_data": {},
                "performance": self.model_performance
            }
            
            # Save each model
            for model_id, model_config in self.models.items():
                if model_config["trained"]:
                    model_data["models"][model_id] = {
                        "type": model_config["type"].value,
                        "threat_categories": [cat.value for cat in model_config["threat_categories"]],
                        "trained": True
                    }
                    
                    # Save model using joblib
                    model_file = f"{file_path}_{model_id}_model.pkl"
                    scaler_file = f"{file_path}_{model_id}_scaler.pkl"
                    
                    joblib.dump(model_config["model"], model_file)
                    joblib.dump(model_config["scaler"], scaler_file)
                    
                    if "label_encoder" in model_config:
                        encoder_file = f"{file_path}_{model_id}_encoder.pkl"
                        joblib.dump(model_config["label_encoder"], encoder_file)
            
            # Save metadata
            with open(f"{file_path}_metadata.json", "w") as f:
                json.dump(model_data, f, indent=2)
            
            self.logger.info(f"Saved {len(model_data['models'])} trained models to {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return False
    
    async def load_models(self, file_path: str) -> bool:
        """Load trained models from disk."""
        try:
            # Load metadata
            with open(f"{file_path}_metadata.json", "r") as f:
                model_data = json.load(f)
            
            # Load each model
            for model_id, model_info in model_data["models"].items():
                if model_id in self.models:
                    # Load model components
                    model_file = f"{file_path}_{model_id}_model.pkl"
                    scaler_file = f"{file_path}_{model_id}_scaler.pkl"
                    
                    self.models[model_id]["model"] = joblib.load(model_file)
                    self.models[model_id]["scaler"] = joblib.load(scaler_file)
                    self.models[model_id]["trained"] = True
                    
                    # Load label encoder if exists
                    try:
                        encoder_file = f"{file_path}_{model_id}_encoder.pkl"
                        self.models[model_id]["label_encoder"] = joblib.load(encoder_file)
                    except FileNotFoundError:
                        pass  # Not all models have encoders
            
            # Load performance metrics
            self.model_performance = model_data.get("performance", {})
            
            self.logger.info(f"Loaded {len(model_data['models'])} trained models from {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models and their performance."""
        summary = {
            "total_models": len(self.models),
            "trained_models": sum(1 for config in self.models.values() if config["trained"]),
            "models": {},
            "overall_performance": {}
        }
        
        # Model details
        for model_id, model_config in self.models.items():
            summary["models"][model_id] = {
                "type": model_config["type"].value,
                "trained": model_config["trained"],
                "threat_categories": [cat.value for cat in model_config["threat_categories"]]
            }
            
            # Add performance if available
            if model_id in self.model_performance:
                summary["models"][model_id]["performance"] = self.model_performance[model_id]
        
        # Overall performance
        if self.model_performance:
            total_detections = sum(perf["total_detections"] for perf in self.model_performance.values())
            total_true_positives = sum(perf["true_positives"] for perf in self.model_performance.values())
            
            if total_detections > 0:
                summary["overall_performance"] = {
                    "total_detections": total_detections,
                    "overall_precision": total_true_positives / total_detections,
                    "models_with_performance_data": len(self.model_performance)
                }
        
        return summary


class AutomatedMLPipeline:
    """Automated ML pipeline for continuous threat detection improvement."""
    
    def __init__(self, ml_detector: MLThreatDetector):
        self.ml_detector = ml_detector
        self.training_queue = deque(maxlen=50000)  # Keep recent events for retraining
        self.feedback_queue = deque(maxlen=10000)
        self.retraining_threshold = 1000  # Retrain after 1000 new events
        self.performance_threshold = 0.7  # Minimum precision threshold
        self.logger = logging.getLogger("AutomatedMLPipeline")
    
    async def add_training_data(self, events: List[Dict[str, Any]]) -> None:
        """Add new events to the training queue."""
        for event in events:
            self.training_queue.append(event)
        
        # Check if retraining is needed
        if len(self.training_queue) >= self.retraining_threshold:
            await self.trigger_retraining()
    
    async def add_feedback(self, detections: List[MLDetectionResult], feedback: List[bool]) -> None:
        """Add feedback for detections."""
        for detection, is_correct in zip(detections, feedback):
            self.feedback_queue.append({
                "detection": detection,
                "is_correct": is_correct,
                "timestamp": datetime.now()
            })
        
        # Update model performance
        await self.ml_detector.update_model_performance(detections, feedback)
    
    async def trigger_retraining(self) -> Dict[str, Any]:
        """Trigger automatic model retraining."""
        self.logger.info(f"Starting automatic retraining with {len(self.training_queue)} events")
        
        # Convert training queue to list
        training_events = list(self.training_queue)
        
        # Retrain models
        results = await self.ml_detector.train_models(training_events)
        
        # Clear training queue after successful training
        if not any("error" in result for result in results.values()):
            self.training_queue.clear()
            self.logger.info("Automatic retraining completed successfully")
        else:
            self.logger.warning("Some models failed retraining, keeping training data")
        
        return results
    
    async def monitor_performance(self) -> Dict[str, Any]:
        """Monitor model performance and trigger retraining if needed."""
        performance_summary = {"models_needing_retraining": []}
        
        for model_id, performance in self.ml_detector.model_performance.items():
            if performance["precision"] < self.performance_threshold:
                performance_summary["models_needing_retraining"].append({
                    "model_id": model_id,
                    "current_precision": performance["precision"],
                    "threshold": self.performance_threshold
                })
        
        # Trigger retraining for underperforming models if we have enough data
        if (performance_summary["models_needing_retraining"] and 
            len(self.training_queue) >= self.retraining_threshold / 2):
            
            self.logger.info("Triggering retraining due to poor model performance")
            retraining_results = await self.trigger_retraining()
            performance_summary["retraining_triggered"] = True
            performance_summary["retraining_results"] = retraining_results
        
        return performance_summary
    
    async def run_continuous_improvement(self) -> None:
        """Run continuous improvement loop."""
        while True:
            try:
                # Monitor performance every hour
                await asyncio.sleep(3600)
                
                performance_results = await self.monitor_performance()
                
                if performance_results.get("retraining_triggered"):
                    self.logger.info("Automatic model improvement cycle completed")
                
            except Exception as e:
                self.logger.error(f"Error in continuous improvement loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry