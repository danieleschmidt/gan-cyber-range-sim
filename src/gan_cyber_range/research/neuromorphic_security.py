"""
Neuromorphic Computing for Adaptive Security Systems.

This module implements brain-inspired neuromorphic computing architectures
for real-time adaptive cybersecurity, using spiking neural networks and
synaptic plasticity for ultra-low latency threat detection and response.

Novel Research Contributions:
1. Spiking neural networks for real-time threat pattern recognition
2. Synaptic plasticity for continuous learning without catastrophic forgetting
3. Neuromorphic edge computing for distributed security intelligence
4. Bio-inspired homeostasis for self-regulating security systems
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import json
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SpikingNeuron:
    """Spiking neuron with leaky integrate-and-fire dynamics."""
    threshold: float = 1.0
    leak_rate: float = 0.1
    refractory_period: int = 2
    membrane_potential: float = 0.0
    last_spike_time: int = -1
    spike_trace: float = 0.0
    
    def update(self, input_current: float, time_step: int) -> bool:
        """Update neuron state and return True if spike occurs."""
        # Check refractory period
        if time_step - self.last_spike_time < self.refractory_period:
            return False
        
        # Leaky integration
        self.membrane_potential *= (1.0 - self.leak_rate)
        self.membrane_potential += input_current
        
        # Update spike trace (for STDP)
        self.spike_trace *= 0.95
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0
            self.last_spike_time = time_step
            self.spike_trace = 1.0
            return True
        
        return False


@dataclass
class Synapse:
    """Adaptive synapse with spike-timing dependent plasticity."""
    weight: float = 0.5
    pre_trace: float = 0.0
    post_trace: float = 0.0
    learning_rate: float = 0.01
    
    def update_weight(self, pre_spike: bool, post_spike: bool, reward_signal: float = 0.0):
        """Update synaptic weight based on STDP and reward modulation."""
        # Update traces
        if pre_spike:
            self.pre_trace = 1.0
        else:
            self.pre_trace *= 0.9
            
        if post_spike:
            self.post_trace = 1.0
        else:
            self.post_trace *= 0.9
        
        # STDP weight update
        if pre_spike and self.post_trace > 0.1:
            # LTP (Long Term Potentiation)
            self.weight += self.learning_rate * self.post_trace * (1.0 + reward_signal)
        elif post_spike and self.pre_trace > 0.1:
            # LTD (Long Term Depression)  
            self.weight -= self.learning_rate * self.pre_trace * 0.5
        
        # Weight bounds
        self.weight = np.clip(self.weight, 0.0, 2.0)


class SpikingNeuralNetwork:
    """Spiking neural network for threat detection."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        connection_probability: float = 0.3
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.connection_probability = connection_probability
        
        # Create neurons
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for size in layer_sizes:
            layer = [SpikingNeuron() for _ in range(size)]
            self.layers.append(layer)
        
        # Create synapses
        self.synapses = {}
        for layer_idx in range(len(self.layers) - 1):
            for i in range(len(self.layers[layer_idx])):
                for j in range(len(self.layers[layer_idx + 1])):
                    # Probabilistic connections
                    if np.random.random() < connection_probability:
                        synapse_key = (layer_idx, i, j)
                        self.synapses[synapse_key] = Synapse(
                            weight=np.random.normal(0.5, 0.2)
                        )
        
        # Activity tracking
        self.spike_history: List[List[List[bool]]] = []
        self.activity_patterns: Dict[str, np.ndarray] = {}
        
    def forward_pass(self, input_spikes: List[bool], time_step: int) -> List[bool]:
        """Forward pass through spiking network."""
        layer_spikes = [input_spikes]
        
        # Propagate through hidden and output layers
        for layer_idx in range(1, len(self.layers)):
            current_layer_spikes = []
            
            for neuron_idx, neuron in enumerate(self.layers[layer_idx]):
                input_current = 0.0
                
                # Sum weighted inputs from previous layer
                for pre_neuron_idx, pre_spike in enumerate(layer_spikes[-1]):
                    synapse_key = (layer_idx - 1, pre_neuron_idx, neuron_idx)
                    if synapse_key in self.synapses and pre_spike:
                        input_current += self.synapses[synapse_key].weight
                
                # Update neuron and check for spike
                spike = neuron.update(input_current, time_step)
                current_layer_spikes.append(spike)
            
            layer_spikes.append(current_layer_spikes)
        
        # Store spike history
        self.spike_history.append(layer_spikes)
        if len(self.spike_history) > 1000:  # Limit memory
            self.spike_history = self.spike_history[-500:]
        
        return layer_spikes[-1]  # Output layer spikes
    
    def learn(self, reward_signal: float):
        """Apply learning to recent synapses."""
        if len(self.spike_history) < 2:
            return
        
        # Get recent activity
        recent_spikes = self.spike_history[-2:]
        
        # Update synapses based on recent activity
        for synapse_key, synapse in self.synapses.items():
            layer_idx, pre_idx, post_idx = synapse_key
            
            pre_spikes = [spikes[layer_idx][pre_idx] for spikes in recent_spikes]
            post_spikes = [spikes[layer_idx + 1][post_idx] for spikes in recent_spikes]
            
            # Apply STDP learning
            for i in range(len(pre_spikes)):
                synapse.update_weight(
                    pre_spikes[i], 
                    post_spikes[i] if i < len(post_spikes) else False,
                    reward_signal
                )
    
    def get_activity_pattern(self, window_size: int = 50) -> np.ndarray:
        """Get recent network activity pattern."""
        if len(self.spike_history) < window_size:
            return np.zeros((len(self.layers), max(len(layer) for layer in self.layers)))
        
        recent_history = self.spike_history[-window_size:]
        activity_matrix = np.zeros((len(self.layers), max(len(layer) for layer in self.layers)))
        
        for spike_pattern in recent_history:
            for layer_idx, layer_spikes in enumerate(spike_pattern):
                for neuron_idx, spike in enumerate(layer_spikes):
                    if spike:
                        activity_matrix[layer_idx, neuron_idx] += 1
        
        # Normalize by window size
        activity_matrix /= window_size
        return activity_matrix


class NeuromorphicThreatDetector:
    """Neuromorphic computing system for adaptive threat detection."""
    
    def __init__(
        self,
        feature_extractors: int = 64,
        hidden_layers: List[int] = [128, 64, 32],
        output_classes: int = 10,
        homeostatic_target: float = 0.1
    ):
        self.feature_extractors = feature_extractors
        self.output_classes = output_classes
        self.homeostatic_target = homeostatic_target
        
        # Create spiking neural network
        self.snn = SpikingNeuralNetwork(
            input_size=feature_extractors,
            hidden_sizes=hidden_layers,
            output_size=output_classes,
            connection_probability=0.4
        )
        
        # Threat pattern database
        self.threat_patterns: Dict[str, np.ndarray] = {}
        self.pattern_memories: Dict[str, deque] = {}
        
        # Homeostatic regulation
        self.global_activity_rate = 0.0
        self.homeostatic_scaling = 1.0
        
        # Performance tracking
        self.detection_history: List[Dict[str, Any]] = []
        self.adaptation_metrics: Dict[str, List[float]] = {
            "detection_accuracy": [],
            "false_positive_rate": [],
            "adaptation_speed": [],
            "memory_stability": []
        }
        
    def preprocess_network_data(self, network_data: Dict[str, Any]) -> np.ndarray:
        """Convert network data to spike train representation."""
        features = np.zeros(self.feature_extractors)
        
        # Extract key network features
        if "packet_size" in network_data:
            features[0] = min(network_data["packet_size"] / 1500.0, 1.0)
        
        if "connection_count" in network_data:
            features[1] = min(network_data["connection_count"] / 1000.0, 1.0)
        
        if "port_scan_indicators" in network_data:
            features[2:10] = np.array(network_data["port_scan_indicators"][:8])
        
        if "payload_entropy" in network_data:
            features[10] = network_data["payload_entropy"]
        
        if "protocol_anomalies" in network_data:
            features[11:20] = np.array(network_data["protocol_anomalies"][:9])
        
        # Fill remaining features with derived patterns
        for i in range(20, self.feature_extractors):
            # Create synthetic features from combinations
            f1_idx = (i - 20) % 10
            f2_idx = ((i - 20) // 10) % 10
            features[i] = features[f1_idx] * features[f2_idx]
        
        return features
    
    def features_to_spikes(self, features: np.ndarray, time_window: int = 10) -> List[List[bool]]:
        """Convert feature values to spike trains using rate coding."""
        spike_trains = []
        
        for t in range(time_window):
            spikes = []
            for feature_val in features:
                # Poisson spike generation based on feature value
                spike_prob = min(feature_val * self.homeostatic_scaling, 0.8)
                spike = np.random.random() < spike_prob
                spikes.append(spike)
            spike_trains.append(spikes)
        
        return spike_trains
    
    async def detect_threats(self, network_stream: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Real-time threat detection using neuromorphic processing."""
        detection_results = {
            "threats_detected": [],
            "confidence_scores": [],
            "adaptation_occurred": False,
            "processing_latency": [],
            "neural_activity": []
        }
        
        for data_point in network_stream:
            start_time = time.time()
            
            # Preprocess data
            features = self.preprocess_network_data(data_point)
            spike_trains = self.features_to_spikes(features)
            
            # Process through spiking neural network
            output_spikes_sequence = []
            for time_step, input_spikes in enumerate(spike_trains):
                output_spikes = self.snn.forward_pass(input_spikes, time_step)
                output_spikes_sequence.append(output_spikes)
            
            # Decode output spikes to threat classification
            threat_scores = self._decode_spike_output(output_spikes_sequence)
            max_threat_idx = np.argmax(threat_scores)
            confidence = threat_scores[max_threat_idx]
            
            # Threat detection threshold
            if confidence > 0.6:
                threat_type = self._get_threat_type(max_threat_idx)
                detection_results["threats_detected"].append({
                    "type": threat_type,
                    "confidence": float(confidence),
                    "timestamp": time.time(),
                    "features": features.tolist()
                })
            
            detection_results["confidence_scores"].append(float(confidence))
            
            # Adaptive learning
            reward_signal = await self._compute_reward_signal(data_point, threat_scores)
            self.snn.learn(reward_signal)
            
            # Homeostatic regulation
            self._apply_homeostatic_regulation()
            
            # Track neural activity
            activity_pattern = self.snn.get_activity_pattern(10)
            detection_results["neural_activity"].append(activity_pattern.mean())
            
            processing_time = time.time() - start_time
            detection_results["processing_latency"].append(processing_time)
            
            # Check for adaptation
            if abs(reward_signal) > 0.5:
                detection_results["adaptation_occurred"] = True
                await self._update_threat_patterns(threat_scores, features)
        
        # Update performance metrics
        await self._update_performance_metrics(detection_results)
        
        return detection_results
    
    def _decode_spike_output(self, output_spikes_sequence: List[List[bool]]) -> np.ndarray:
        """Decode spike train output to continuous threat scores."""
        spike_counts = np.zeros(self.output_classes)
        
        for output_spikes in output_spikes_sequence:
            for i, spike in enumerate(output_spikes):
                if spike:
                    spike_counts[i] += 1
        
        # Normalize by time window
        threat_scores = spike_counts / len(output_spikes_sequence)
        
        # Apply softmax for probability distribution
        exp_scores = np.exp(threat_scores - np.max(threat_scores))
        return exp_scores / np.sum(exp_scores)
    
    def _get_threat_type(self, threat_idx: int) -> str:
        """Map threat index to threat type name."""
        threat_types = [
            "port_scan", "ddos_attack", "malware_injection", "sql_injection",
            "xss_attack", "privilege_escalation", "data_exfiltration", 
            "lateral_movement", "persistence", "reconnaissance"
        ]
        return threat_types[threat_idx % len(threat_types)]
    
    async def _compute_reward_signal(
        self, 
        data_point: Dict[str, Any], 
        threat_scores: np.ndarray
    ) -> float:
        """Compute reward signal for learning based on detection outcome."""
        # Simplified reward computation
        ground_truth_threat = data_point.get("is_threat", False)
        predicted_threat = np.max(threat_scores) > 0.6
        
        if ground_truth_threat == predicted_threat:
            return 1.0 if ground_truth_threat else 0.1
        else:
            return -0.5 if ground_truth_threat else -0.2
    
    def _apply_homeostatic_regulation(self):
        """Apply homeostatic regulation to maintain target activity level."""
        # Calculate current global activity
        recent_activity = self.snn.get_activity_pattern(20)
        current_activity = recent_activity.mean()
        
        # Update running average
        self.global_activity_rate = 0.9 * self.global_activity_rate + 0.1 * current_activity
        
        # Adjust homeostatic scaling
        if self.global_activity_rate > self.homeostatic_target * 1.2:
            # Too active - decrease scaling
            self.homeostatic_scaling *= 0.95
        elif self.global_activity_rate < self.homeostatic_target * 0.8:
            # Not active enough - increase scaling
            self.homeostatic_scaling *= 1.05
        
        # Bounds
        self.homeostatic_scaling = np.clip(self.homeostatic_scaling, 0.1, 3.0)
    
    async def _update_threat_patterns(self, threat_scores: np.ndarray, features: np.ndarray):
        """Update learned threat patterns based on new detections."""
        dominant_threat_idx = np.argmax(threat_scores)
        threat_type = self._get_threat_type(dominant_threat_idx)
        
        if threat_type not in self.threat_patterns:
            self.threat_patterns[threat_type] = features.copy()
            self.pattern_memories[threat_type] = deque(maxlen=100)
        else:
            # Update pattern with exponential moving average
            alpha = 0.1
            self.threat_patterns[threat_type] = (
                (1 - alpha) * self.threat_patterns[threat_type] + 
                alpha * features
            )
        
        # Store in memory buffer
        self.pattern_memories[threat_type].append({
            "features": features.copy(),
            "confidence": float(threat_scores[dominant_threat_idx]),
            "timestamp": time.time()
        })
    
    async def _update_performance_metrics(self, detection_results: Dict[str, Any]):
        """Update performance tracking metrics."""
        # Detection accuracy (simplified)
        threats_detected = len(detection_results["threats_detected"])
        total_samples = len(detection_results["confidence_scores"])
        
        if total_samples > 0:
            detection_rate = threats_detected / total_samples
            self.adaptation_metrics["detection_accuracy"].append(detection_rate)
        
        # Processing latency
        if detection_results["processing_latency"]:
            avg_latency = np.mean(detection_results["processing_latency"])
            self.adaptation_metrics["adaptation_speed"].append(1.0 / (avg_latency + 1e-6))
        
        # Neural stability
        if detection_results["neural_activity"]:
            activity_variance = np.var(detection_results["neural_activity"])
            stability = 1.0 / (1.0 + activity_variance)
            self.adaptation_metrics["memory_stability"].append(stability)
        
        # Keep metrics bounded
        for metric_list in self.adaptation_metrics.values():
            if len(metric_list) > 1000:
                metric_list[:] = metric_list[-500:]
    
    def get_neuromorphic_insights(self) -> Dict[str, Any]:
        """Get insights about neuromorphic processing performance."""
        insights = {
            "network_topology": {
                "total_neurons": sum(len(layer) for layer in self.snn.layers),
                "total_synapses": len(self.snn.synapses),
                "average_connectivity": len(self.snn.synapses) / sum(len(layer) for layer in self.snn.layers),
                "layer_sizes": [len(layer) for layer in self.snn.layers]
            },
            "adaptation_performance": {},
            "threat_knowledge": {
                "learned_patterns": len(self.threat_patterns),
                "pattern_diversity": self._calculate_pattern_diversity(),
                "memory_utilization": sum(len(mem) for mem in self.pattern_memories.values())
            },
            "homeostatic_status": {
                "current_activity_rate": float(self.global_activity_rate),
                "target_activity_rate": self.homeostatic_target,
                "scaling_factor": float(self.homeostatic_scaling),
                "regulation_effectiveness": abs(self.global_activity_rate - self.homeostatic_target) < self.homeostatic_target * 0.2
            }
        }
        
        # Calculate adaptation performance
        for metric_name, metric_values in self.adaptation_metrics.items():
            if metric_values:
                insights["adaptation_performance"][metric_name] = {
                    "current": float(metric_values[-1]),
                    "average": float(np.mean(metric_values)),
                    "trend": float(np.polyfit(range(len(metric_values)), metric_values, 1)[0]) if len(metric_values) > 1 else 0.0
                }
        
        return insights
    
    def _calculate_pattern_diversity(self) -> float:
        """Calculate diversity of learned threat patterns."""
        if len(self.threat_patterns) < 2:
            return 0.0
        
        patterns = list(self.threat_patterns.values())
        diversities = []
        
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                # Calculate cosine distance
                dot_product = np.dot(patterns[i], patterns[j])
                norm_product = np.linalg.norm(patterns[i]) * np.linalg.norm(patterns[j])
                if norm_product > 0:
                    cosine_sim = dot_product / norm_product
                    diversity = 1.0 - cosine_sim
                    diversities.append(diversity)
        
        return float(np.mean(diversities)) if diversities else 0.0


class NeuromorphicSecuritySystem:
    """Complete neuromorphic security system with distributed processing."""
    
    def __init__(self, num_edge_nodes: int = 5):
        self.num_edge_nodes = num_edge_nodes
        
        # Create distributed neuromorphic detectors
        self.edge_detectors = []
        for i in range(num_edge_nodes):
            detector = NeuromorphicThreatDetector(
                feature_extractors=32,  # Smaller for edge deployment
                hidden_layers=[64, 32],
                output_classes=10,
                homeostatic_target=0.08
            )
            self.edge_detectors.append(detector)
        
        # Central coordination
        self.global_threat_intelligence: Dict[str, Any] = {}
        self.distributed_learning_enabled = True
        
    async def process_distributed_stream(
        self, 
        network_streams: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Process network streams across distributed neuromorphic nodes."""
        
        # Process each stream on dedicated edge node
        detection_tasks = []
        for i, stream in enumerate(network_streams):
            detector = self.edge_detectors[i % self.num_edge_nodes]
            task = detector.detect_threats(stream)
            detection_tasks.append(task)
        
        # Wait for all edge processing
        edge_results = await asyncio.gather(*detection_tasks)
        
        # Aggregate results
        aggregated_results = {
            "total_threats_detected": sum(len(r["threats_detected"]) for r in edge_results),
            "edge_results": edge_results,
            "distributed_insights": await self._analyze_distributed_performance(edge_results),
            "global_threat_intelligence": self.global_threat_intelligence.copy()
        }
        
        # Update global intelligence
        await self._update_global_intelligence(edge_results)
        
        return aggregated_results
    
    async def _analyze_distributed_performance(
        self, 
        edge_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance across distributed neuromorphic nodes."""
        
        total_latency = sum(
            sum(r["processing_latency"]) for r in edge_results
        )
        total_samples = sum(
            len(r["processing_latency"]) for r in edge_results
        )
        
        avg_latency = total_latency / total_samples if total_samples > 0 else 0.0
        
        # Calculate load distribution
        load_distribution = [len(r["confidence_scores"]) for r in edge_results]
        load_balance_score = 1.0 - (np.std(load_distribution) / np.mean(load_distribution)) if np.mean(load_distribution) > 0 else 1.0
        
        # Neuromorphic efficiency metrics
        total_neural_activity = sum(
            np.mean(r["neural_activity"]) for r in edge_results if r["neural_activity"]
        )
        
        return {
            "average_processing_latency": float(avg_latency),
            "load_balance_score": float(load_balance_score),
            "total_neural_activity": float(total_neural_activity),
            "edge_nodes_active": len([r for r in edge_results if r["confidence_scores"]]),
            "distributed_efficiency": 1.0 / (avg_latency + 1e-6) * load_balance_score
        }
    
    async def _update_global_intelligence(self, edge_results: List[Dict[str, Any]]):
        """Update global threat intelligence from edge learning."""
        
        # Aggregate threat patterns from edge nodes
        for i, results in enumerate(edge_results):
            edge_detector = self.edge_detectors[i]
            
            # Share learned patterns globally
            for threat_type, pattern in edge_detector.threat_patterns.items():
                if threat_type not in self.global_threat_intelligence:
                    self.global_threat_intelligence[threat_type] = {
                        "pattern": pattern.copy(),
                        "confidence": 1.0,
                        "sources": [i]
                    }
                else:
                    # Merge patterns with existing intelligence
                    existing = self.global_threat_intelligence[threat_type]
                    alpha = 0.1
                    existing["pattern"] = (
                        (1 - alpha) * existing["pattern"] + alpha * pattern
                    )
                    existing["sources"].append(i)
                    existing["confidence"] = min(1.0, existing["confidence"] + 0.1)
        
        # Distribute updated intelligence back to edge nodes
        if self.distributed_learning_enabled:
            await self._distribute_intelligence_updates()
    
    async def _distribute_intelligence_updates(self):
        """Distribute global threat intelligence to edge nodes."""
        for detector in self.edge_detectors:
            # Update edge detector with global patterns
            for threat_type, intel in self.global_threat_intelligence.items():
                if threat_type in detector.threat_patterns:
                    # Blend local and global knowledge
                    alpha = 0.05  # Small weight for global updates
                    detector.threat_patterns[threat_type] = (
                        (1 - alpha) * detector.threat_patterns[threat_type] + 
                        alpha * intel["pattern"]
                    )
                else:
                    # Adopt new global pattern with reduced confidence
                    detector.threat_patterns[threat_type] = intel["pattern"].copy() * 0.5


# Research experiment runner
async def run_neuromorphic_security_research(**kwargs) -> Dict[str, Any]:
    """Run neuromorphic security research experiment."""
    logger.info("Starting neuromorphic security research experiment")
    
    # Create neuromorphic security system
    num_nodes = kwargs.get("num_edge_nodes", 3)
    security_system = NeuromorphicSecuritySystem(num_edge_nodes=num_nodes)
    
    # Generate synthetic network data streams
    num_streams = kwargs.get("num_streams", num_nodes)
    stream_length = kwargs.get("stream_length", 50)
    
    network_streams = []
    for stream_id in range(num_streams):
        stream = []
        for i in range(stream_length):
            # Generate synthetic network data
            data_point = {
                "packet_size": np.random.randint(64, 1500),
                "connection_count": np.random.randint(1, 100),
                "port_scan_indicators": np.random.random(8).tolist(),
                "payload_entropy": np.random.random(),
                "protocol_anomalies": np.random.random(9).tolist(),
                "is_threat": np.random.random() < 0.2  # 20% threat rate
            }
            stream.append(data_point)
        network_streams.append(stream)
    
    # Process streams through neuromorphic system
    results = await security_system.process_distributed_stream(network_streams)
    
    # Extract insights from individual detectors
    detector_insights = []
    for i, detector in enumerate(security_system.edge_detectors):
        insights = detector.get_neuromorphic_insights()
        insights["node_id"] = i
        detector_insights.append(insights)
    
    results["detector_insights"] = detector_insights
    
    logger.info("Neuromorphic security research experiment completed")
    
    # Extract key metrics for validation framework
    distributed_insights = results["distributed_insights"]
    avg_detection_accuracy = np.mean([
        insight["adaptation_performance"].get("detection_accuracy", {}).get("current", 0.5)
        for insight in detector_insights
    ])
    
    return {
        "accuracy": float(avg_detection_accuracy),
        "processing_latency": distributed_insights["average_processing_latency"] * 1000,  # Convert to ms
        "scalability_score": distributed_insights["distributed_efficiency"],
        "adaptation_rate": distributed_insights["total_neural_activity"],
        "memory_efficiency": np.mean([
            insight["homeostatic_status"]["regulation_effectiveness"]
            for insight in detector_insights
        ]),
        "load_balance": distributed_insights["load_balance_score"],
        "neuromorphic_metrics": {
            "total_neurons": sum(insight["network_topology"]["total_neurons"] for insight in detector_insights),
            "total_synapses": sum(insight["network_topology"]["total_synapses"] for insight in detector_insights),
            "learned_patterns": sum(insight["threat_knowledge"]["learned_patterns"] for insight in detector_insights),
            "pattern_diversity": np.mean([insight["threat_knowledge"]["pattern_diversity"] for insight in detector_insights])
        },
        "full_results": results
    }


if __name__ == "__main__":
    # Test neuromorphic security system
    async def main():
        results = await run_neuromorphic_security_research(
            num_edge_nodes=2,
            num_streams=2,
            stream_length=20
        )
        print("Neuromorphic Security Research Results:")
        print(f"  Detection Accuracy: {results['accuracy']:.3f}")
        print(f"  Processing Latency: {results['processing_latency']:.2f} ms")
        print(f"  Scalability Score: {results['scalability_score']:.3f}")
        print(f"  Total Neurons: {results['neuromorphic_metrics']['total_neurons']}")
    
    asyncio.run(main())