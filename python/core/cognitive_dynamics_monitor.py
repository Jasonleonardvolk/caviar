#!/usr/bin/env python3
"""
Cognitive Dynamics Monitor
Detects and resolves chaotic reasoning patterns using Lyapunov analysis
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
from enum import Enum
from collections import deque
import json

from python.core.unified_metacognitive_integration import (
    CognitiveDynamicsSystem, CognitiveStateManager,
    MetacognitiveState
)
from python.core.reasoning_traversal import ReasoningPath

logger = logging.getLogger(__name__)

# ========== Dynamics States ==========

class DynamicsState(Enum):
    """States of cognitive dynamics"""
    STABLE = "stable"           # Convergent, predictable
    PERIODIC = "periodic"       # Oscillating between states
    CHAOTIC = "chaotic"        # Unpredictable, sensitive
    EDGE_OF_CHAOS = "edge_of_chaos"  # Critical transition
    FROZEN = "frozen"          # No change, stuck

@dataclass
class DynamicsMetrics:
    """Metrics for cognitive dynamics analysis"""
    lyapunov_exponents: np.ndarray
    energy: float
    entropy: float
    fractal_dimension: float
    phase_space_volume: float
    attractor_type: str
    stability_margin: float
    
@dataclass
class StabilizationStrategy:
    """Strategy for stabilizing chaotic dynamics"""
    name: str
    strength: float
    target_state: Optional[np.ndarray]
    parameters: Dict[str, Any] = field(default_factory=dict)

# ========== Advanced Dynamics Analysis ==========

class AdvancedDynamicsAnalyzer:
    """Advanced analysis of cognitive dynamics"""
    
    def __init__(self, state_dim: int = 100):
        self.state_dim = state_dim
        self.phase_space_history = deque(maxlen=1000)
        self.attractor_library = {}
    
    def analyze_phase_space(self, trajectory: List[np.ndarray]) -> DynamicsMetrics:
        """Comprehensive phase space analysis"""
        
        if len(trajectory) < 3:
            return self._default_metrics()
        
        # Compute Lyapunov spectrum
        lyapunov = self._compute_full_lyapunov_spectrum(trajectory)
        
        # Compute energy and entropy
        energy = self._compute_trajectory_energy(trajectory)
        entropy = self._compute_trajectory_entropy(trajectory)
        
        # Estimate fractal dimension
        fractal_dim = self._estimate_fractal_dimension(trajectory)
        
        # Compute phase space volume
        phase_volume = self._compute_phase_space_volume(trajectory)
        
        # Identify attractor type
        attractor = self._identify_attractor_type(trajectory, lyapunov)
        
        # Calculate stability margin
        stability_margin = self._calculate_stability_margin(lyapunov)
        
        return DynamicsMetrics(
            lyapunov_exponents=lyapunov,
            energy=energy,
            entropy=entropy,
            fractal_dimension=fractal_dim,
            phase_space_volume=phase_volume,
            attractor_type=attractor,
            stability_margin=stability_margin
        )
    
    def _compute_full_lyapunov_spectrum(self, trajectory: List[np.ndarray]) -> np.ndarray:
        """Compute full Lyapunov exponent spectrum"""
        n_points = len(trajectory)
        if n_points < 10:
            return np.zeros(3)
        
        # Simplified Lyapunov calculation
        n_exponents = min(3, self.state_dim)
        exponents = []
        
        for k in range(n_exponents):
            # Measure divergence rate in different directions
            divergences = []
            for i in range(n_points - 2):
                if i + k + 1 < n_points:
                    d1 = np.linalg.norm(trajectory[i+k+1] - trajectory[i])
                    d0 = np.linalg.norm(trajectory[i+1] - trajectory[i])
                    if d0 > 0 and d1 > 0:
                        divergences.append(np.log(d1/d0))
            
            if divergences:
                exponents.append(np.mean(divergences))
            else:
                exponents.append(0.0)
        
        return np.array(exponents)
    
    def _compute_trajectory_energy(self, trajectory: List[np.ndarray]) -> float:
        """Compute total energy of trajectory"""
        if not trajectory:
            return 0.0
        
        # Kinetic energy (velocity)
        kinetic = 0.0
        for i in range(len(trajectory) - 1):
            velocity = trajectory[i+1] - trajectory[i]
            kinetic += np.linalg.norm(velocity) ** 2
        
        # Potential energy (distance from origin)
        potential = sum(np.linalg.norm(state) ** 2 for state in trajectory)
        
        return (kinetic + potential) / len(trajectory)
    
    def _compute_trajectory_entropy(self, trajectory: List[np.ndarray]) -> float:
        """Compute entropy of trajectory"""
        if len(trajectory) < 2:
            return 0.0
        
        # Discretize phase space
        n_bins = 10
        bins = np.linspace(-10, 10, n_bins)
        
        # Count occupancy
        occupancy = np.zeros((n_bins, n_bins))
        for state in trajectory:
            if len(state) >= 2:
                i = np.digitize(state[0], bins) - 1
                j = np.digitize(state[1], bins) - 1
                if 0 <= i < n_bins and 0 <= j < n_bins:
                    occupancy[i, j] += 1
        
        # Compute entropy
        occupancy = occupancy / len(trajectory)
        occupancy = occupancy[occupancy > 0]
        entropy = -np.sum(occupancy * np.log(occupancy + 1e-10))
        
        return entropy
    
    def _estimate_fractal_dimension(self, trajectory: List[np.ndarray]) -> float:
        """Estimate fractal dimension using box-counting"""
        if len(trajectory) < 10:
            return 1.0
        
        # Simplified box-counting
        dimensions = []
        for scale in [0.1, 0.5, 1.0, 2.0]:
            boxes = set()
            for state in trajectory:
                if len(state) >= 2:
                    box = (int(state[0]/scale), int(state[1]/scale))
                    boxes.add(box)
            
            if len(boxes) > 0:
                dimensions.append(np.log(len(boxes)) / np.log(1/scale))
        
        return np.mean(dimensions) if dimensions else 1.0
    
    def _compute_phase_space_volume(self, trajectory: List[np.ndarray]) -> float:
        """Compute volume occupied in phase space"""
        if len(trajectory) < 2:
            return 0.0
        
        # Compute convex hull volume (simplified)
        trajectory_array = np.array(trajectory)
        if trajectory_array.shape[0] < 3:
            return 0.0
        
        # Use standard deviation as proxy for volume
        volume = np.prod(np.std(trajectory_array, axis=0)[:3] + 1e-6)
        
        return float(volume)
    
    def _identify_attractor_type(self, trajectory: List[np.ndarray], 
                               lyapunov: np.ndarray) -> str:
        """Identify type of attractor"""
        
        max_lyapunov = np.max(lyapunov)
        sum_lyapunov = np.sum(lyapunov)
        
        # Classification based on Lyapunov exponents
        if max_lyapunov < -0.1:
            return "fixed_point"
        elif max_lyapunov < 0.01 and sum_lyapunov < 0:
            return "limit_cycle"
        elif max_lyapunov > 0.1:
            return "strange_attractor"
        else:
            return "quasi_periodic"
    
    def _calculate_stability_margin(self, lyapunov: np.ndarray) -> float:
        """Calculate margin of stability"""
        max_exp = np.max(lyapunov)
        
        if max_exp < 0:
            # Stable - margin is distance from 0
            return abs(max_exp)
        else:
            # Unstable - negative margin
            return -max_exp
    
    def _default_metrics(self) -> DynamicsMetrics:
        """Default metrics for insufficient data"""
        return DynamicsMetrics(
            lyapunov_exponents=np.zeros(3),
            energy=0.0,
            entropy=0.0,
            fractal_dimension=1.0,
            phase_space_volume=0.0,
            attractor_type="unknown",
            stability_margin=0.0
        )

# ========== Stabilization Strategies ==========

class StabilizationController:
    """Controller for applying stabilization strategies"""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.strategy_history = []
    
    def _initialize_strategies(self) -> Dict[str, StabilizationStrategy]:
        """Initialize available stabilization strategies"""
        return {
            "damping": StabilizationStrategy(
                name="damping",
                strength=0.1,
                target_state=None,
                parameters={"damping_factor": 0.95}
            ),
            "attractor_injection": StabilizationStrategy(
                name="attractor_injection",
                strength=0.3,
                target_state=np.zeros(100),
                parameters={"injection_rate": 0.2}
            ),
            "noise_reduction": StabilizationStrategy(
                name="noise_reduction",
                strength=0.2,
                target_state=None,
                parameters={"filter_cutoff": 0.8}
            ),
            "phase_reset": StabilizationStrategy(
                name="phase_reset",
                strength=0.5,
                target_state=None,
                parameters={"reset_threshold": 0.9}
            ),
            "bifurcation_control": StabilizationStrategy(
                name="bifurcation_control",
                strength=0.4,
                target_state=None,
                parameters={"control_parameter": 0.5}
            )
        }
    
    def select_strategy(self, metrics: DynamicsMetrics, 
                       state: DynamicsState) -> StabilizationStrategy:
        """Select appropriate stabilization strategy"""
        
        if state == DynamicsState.CHAOTIC:
            if metrics.energy > 100:
                return self.strategies["damping"]
            elif metrics.fractal_dimension > 2.5:
                return self.strategies["phase_reset"]
            else:
                return self.strategies["attractor_injection"]
        
        elif state == DynamicsState.PERIODIC:
            if metrics.stability_margin < 0.1:
                return self.strategies["bifurcation_control"]
            else:
                return self.strategies["noise_reduction"]
        
        elif state == DynamicsState.EDGE_OF_CHAOS:
            return self.strategies["bifurcation_control"]
        
        elif state == DynamicsState.FROZEN:
            # Add noise to unfreeze
            strategy = StabilizationStrategy(
                name="noise_reduction",
                strength=0.2,
                target_state=None,
                parameters={"filter_cutoff": 0.8, "add_noise": True}
            )
            return strategy
        
        else:  # STABLE
            return self.strategies["noise_reduction"]
    
    def apply_strategy(self, state: np.ndarray, 
                      strategy: StabilizationStrategy) -> np.ndarray:
        """Apply stabilization strategy to state"""
        
        stabilized = state.copy()
        
        if strategy.name == "damping":
            # Apply damping
            factor = strategy.parameters["damping_factor"]
            stabilized *= factor
        
        elif strategy.name == "attractor_injection":
            # Pull toward target attractor
            if strategy.target_state is not None:
                rate = strategy.parameters["injection_rate"]
                stabilized += rate * (strategy.target_state[:len(state)] - state)
        
        elif strategy.name == "noise_reduction":
            # Apply low-pass filter
            if strategy.parameters.get("add_noise", False):
                # Add controlled noise instead
                stabilized += np.random.randn(*state.shape) * 0.1
            else:
                # Smooth the state
                cutoff = strategy.parameters["filter_cutoff"]
                stabilized = cutoff * state + (1 - cutoff) * np.mean(state)
        
        elif strategy.name == "phase_reset":
            # Reset if magnitude exceeds threshold
            threshold = strategy.parameters["reset_threshold"]
            if np.linalg.norm(state) > threshold * 10:
                stabilized = state / np.linalg.norm(state)
        
        elif strategy.name == "bifurcation_control":
            # Adjust control parameter
            param = strategy.parameters["control_parameter"]
            stabilized = state * (1 - param) + np.mean(state) * param
        
        # Record application
        self.strategy_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy.name,
            "input_norm": float(np.linalg.norm(state)),
            "output_norm": float(np.linalg.norm(stabilized))
        })
        
        return stabilized

# ========== Cognitive Dynamics Monitor ==========

class CognitiveDynamicsMonitor:
    """Main monitor for cognitive dynamics"""
    
    def __init__(self, state_manager: CognitiveStateManager):
        self.state_manager = state_manager
        self.dynamics_system = CognitiveDynamicsSystem(state_manager)
        self.analyzer = AdvancedDynamicsAnalyzer(state_manager.state_dim)
        self.controller = StabilizationController()
        
        # Monitoring state
        self.current_dynamics_state = DynamicsState.STABLE
        self.metrics_history = deque(maxlen=100)
        self.intervention_log = []
        
        # Thresholds
        self.chaos_threshold = 0.5
        self.intervention_threshold = 0.7
        self.emergency_threshold = 1.0
    
    def monitor_and_stabilize(self, window_size: int = 20) -> Dict[str, Any]:
        """Monitor dynamics and apply stabilization if needed"""
        
        # Get recent trajectory
        trajectory = self.state_manager.get_trajectory(window_size)
        
        if len(trajectory) < 3:
            return {
                "status": "insufficient_data",
                "dynamics_state": DynamicsState.STABLE.value,
                "intervention": None
            }
        
        # Analyze dynamics
        metrics = self.analyzer.analyze_phase_space(trajectory)
        self.metrics_history.append(metrics)
        
        # Determine dynamics state
        dynamics_state = self._classify_dynamics_state(metrics)
        self.current_dynamics_state = dynamics_state
        
        # Check if intervention needed
        intervention_score = self._calculate_intervention_score(metrics, dynamics_state)
        
        intervention_result = None
        if intervention_score > self.intervention_threshold:
            intervention_result = self._perform_intervention(metrics, dynamics_state)
        
        # Generate report
        report = {
            "status": "monitored",
            "dynamics_state": dynamics_state.value,
            "metrics": {
                "max_lyapunov": float(np.max(metrics.lyapunov_exponents)),
                "energy": metrics.energy,
                "entropy": metrics.entropy,
                "fractal_dimension": metrics.fractal_dimension,
                "attractor_type": metrics.attractor_type,
                "stability_margin": metrics.stability_margin
            },
            "intervention_score": intervention_score,
            "intervention": intervention_result
        }
        
        return report
    
    def _classify_dynamics_state(self, metrics: DynamicsMetrics) -> DynamicsState:
        """Classify current dynamics state"""
        
        max_lyapunov = np.max(metrics.lyapunov_exponents)
        
        # Check for chaos
        if max_lyapunov > self.chaos_threshold:
            return DynamicsState.CHAOTIC
        
        # Check for edge of chaos
        elif 0 < max_lyapunov < self.chaos_threshold:
            return DynamicsState.EDGE_OF_CHAOS
        
        # Check for periodic behavior
        elif metrics.attractor_type == "limit_cycle":
            return DynamicsState.PERIODIC
        
        # Check for frozen state
        elif metrics.energy < 0.01 and metrics.entropy < 0.1:
            return DynamicsState.FROZEN
        
        else:
            return DynamicsState.STABLE
    
    def _calculate_intervention_score(self, metrics: DynamicsMetrics,
                                    state: DynamicsState) -> float:
        """Calculate need for intervention"""
        
        score = 0.0
        
        # Chaos contribution
        if state == DynamicsState.CHAOTIC:
            score += 0.5
            if np.max(metrics.lyapunov_exponents) > self.emergency_threshold:
                score += 0.5
        
        # Energy contribution
        if metrics.energy > 50:
            score += 0.3
        
        # Stability margin contribution
        if metrics.stability_margin < 0:
            score += abs(metrics.stability_margin)
        
        # Fractal dimension contribution
        if metrics.fractal_dimension > 2.5:
            score += 0.2
        
        return min(1.0, score)
    
    def _perform_intervention(self, metrics: DynamicsMetrics,
                            dynamics_state: DynamicsState) -> Dict[str, Any]:
        """Perform stabilization intervention"""
        
        # Select strategy
        strategy = self.controller.select_strategy(metrics, dynamics_state)
        
        # Get current state
        current_state = self.state_manager.get_state()
        
        # Apply stabilization
        stabilized_state = self.controller.apply_strategy(current_state, strategy)
        
        # Update state
        self.state_manager.update_state(
            new_state=stabilized_state,
            metadata={
                "intervention": strategy.name,
                "pre_intervention_state": dynamics_state.value
            }
        )
        
        # Log intervention
        intervention_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dynamics_state": dynamics_state.value,
            "strategy": strategy.name,
            "metrics_before": {
                "max_lyapunov": float(np.max(metrics.lyapunov_exponents)),
                "energy": metrics.energy
            }
        }
        
        self.intervention_log.append(intervention_log)
        
        return {
            "strategy": strategy.name,
            "strength": strategy.strength,
            "parameters": strategy.parameters
        }
    
    def predict_future_dynamics(self, steps: int = 10) -> List[DynamicsState]:
        """Predict future dynamics states"""
        
        if not self.metrics_history:
            return [DynamicsState.STABLE] * steps
        
        predictions = []
        
        # Simple prediction based on trend
        recent_metrics = list(self.metrics_history)[-5:]
        if recent_metrics:
            # Check trend in Lyapunov exponents
            lyapunov_trend = []
            for m in recent_metrics:
                lyapunov_trend.append(np.max(m.lyapunov_exponents))
            
            if len(lyapunov_trend) > 1:
                trend_slope = (lyapunov_trend[-1] - lyapunov_trend[0]) / len(lyapunov_trend)
                
                for i in range(steps):
                    predicted_lyapunov = lyapunov_trend[-1] + trend_slope * (i + 1)
                    
                    if predicted_lyapunov > self.chaos_threshold:
                        predictions.append(DynamicsState.CHAOTIC)
                    elif predicted_lyapunov > 0:
                        predictions.append(DynamicsState.EDGE_OF_CHAOS)
                    else:
                        predictions.append(DynamicsState.STABLE)
            else:
                predictions = [self.current_dynamics_state] * steps
        
        return predictions
    
    def get_dynamics_report(self) -> Dict[str, Any]:
        """Generate comprehensive dynamics report"""
        
        # Recent metrics summary
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]
            avg_lyapunov = np.mean([np.max(m.lyapunov_exponents) for m in recent_metrics])
            avg_energy = np.mean([m.energy for m in recent_metrics])
            avg_entropy = np.mean([m.entropy for m in recent_metrics])
        else:
            avg_lyapunov = avg_energy = avg_entropy = 0.0
        
        # Intervention summary
        intervention_count = len(self.intervention_log)
        if intervention_count > 0:
            recent_interventions = self.intervention_log[-5:]
            strategy_counts = {}
            for interv in recent_interventions:
                strategy = interv["strategy"]
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            most_common_strategy = max(strategy_counts, key=strategy_counts.get)
        else:
            recent_interventions = []
            most_common_strategy = "none"
        
        # Future prediction
        future_predictions = self.predict_future_dynamics(steps=5)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            self.current_dynamics_state,
            avg_lyapunov,
            intervention_count
        )
        
        return {
            "current_state": self.current_dynamics_state.value,
            "metrics_summary": {
                "average_lyapunov": avg_lyapunov,
                "average_energy": avg_energy,
                "average_entropy": avg_entropy
            },
            "intervention_summary": {
                "total_interventions": intervention_count,
                "recent_count": len(recent_interventions),
                "most_common_strategy": most_common_strategy
            },
            "future_predictions": [s.value for s in future_predictions],
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, current_state: DynamicsState,
                                avg_lyapunov: float,
                                intervention_count: int) -> List[str]:
        """Generate recommendations based on dynamics"""
        
        recommendations = []
        
        if current_state == DynamicsState.CHAOTIC:
            recommendations.append("System is chaotic - apply strong stabilization")
            recommendations.append("Consider reducing input complexity")
        
        elif current_state == DynamicsState.EDGE_OF_CHAOS:
            recommendations.append("System at critical point - monitor closely")
            recommendations.append("Small perturbations may have large effects")
        
        elif current_state == DynamicsState.FROZEN:
            recommendations.append("System is frozen - inject controlled noise")
            recommendations.append("Consider diversifying inputs")
        
        elif current_state == DynamicsState.PERIODIC:
            recommendations.append("System in periodic cycle - stable but limited")
            recommendations.append("Consider breaking periodicity for innovation")
        
        # Intervention frequency
        if intervention_count > 10:
            recommendations.append("Frequent interventions needed - review system parameters")
        
        # Lyapunov trend
        if avg_lyapunov > 0.3:
            recommendations.append("Average Lyapunov positive - increase damping")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations

# ========== Integration with Reasoning ==========

class ReasoningDynamicsIntegration:
    """Integrate dynamics monitoring with reasoning system"""
    
    def __init__(self, monitor: CognitiveDynamicsMonitor):
        self.monitor = monitor
        self.reasoning_metrics = []
    
    def analyze_reasoning_dynamics(self, reasoning_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """Analyze dynamics of reasoning process"""
        
        # Convert reasoning to state trajectory
        trajectory = self._reasoning_to_trajectory(reasoning_paths)
        
        # Update state manager with trajectory
        for state in trajectory:
            self.monitor.state_manager.update_state(
                new_state=state,
                metadata={"source": "reasoning_path"}
            )
        
        # Monitor and potentially stabilize
        monitoring_result = self.monitor.monitor_and_stabilize()
        
        # Generate reasoning-specific metrics
        reasoning_metrics = {
            "path_count": len(reasoning_paths),
            "average_confidence": np.mean([p.confidence for p in reasoning_paths]),
            "path_diversity": self._calculate_path_diversity(reasoning_paths),
            "convergence_rate": self._estimate_convergence_rate(reasoning_paths)
        }
        
        # Combine with dynamics monitoring
        result = {
            "dynamics": monitoring_result,
            "reasoning_metrics": reasoning_metrics,
            "stability_assessment": self._assess_reasoning_stability(
                monitoring_result,
                reasoning_metrics
            )
        }
        
        self.reasoning_metrics.append(result)
        return result
    
    def _reasoning_to_trajectory(self, paths: List[ReasoningPath]) -> List[np.ndarray]:
        """Convert reasoning paths to state trajectory"""
        
        trajectory = []
        state_dim = self.monitor.state_manager.state_dim
        
        for path in paths:
            # Create state vector from path properties
            state = np.zeros(state_dim)
            
            # Encode path properties
            state[0] = path.score
            state[1] = path.confidence
            state[2] = len(path.chain)
            
            # Encode node information
            for i, node in enumerate(path.chain[:10]):
                if i * 3 + 5 < state_dim:
                    # Simple hash encoding
                    state[i * 3 + 3] = hash(node.id) % 100 / 100.0
                    state[i * 3 + 4] = len(node.sources) / 10.0
                    state[i * 3 + 5] = len(node.description) / 100.0
            
            trajectory.append(state)
        
        return trajectory
    
    def _calculate_path_diversity(self, paths: List[ReasoningPath]) -> float:
        """Calculate diversity of reasoning paths"""
        
        if len(paths) < 2:
            return 0.0
        
        # Calculate pairwise differences
        diversities = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                # Compare paths
                shared_nodes = set(n.id for n in paths[i].chain) & \
                              set(n.id for n in paths[j].chain)
                total_nodes = set(n.id for n in paths[i].chain) | \
                             set(n.id for n in paths[j].chain)
                
                if total_nodes:
                    diversity = 1.0 - len(shared_nodes) / len(total_nodes)
                    diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _estimate_convergence_rate(self, paths: List[ReasoningPath]) -> float:
        """Estimate how quickly reasoning converges"""
        
        if len(paths) < 2:
            return 1.0
        
        # Check if paths converge to similar conclusions
        final_nodes = [path.chain[-1].id if path.chain else None for path in paths]
        unique_finals = len(set(final_nodes))
        
        convergence = 1.0 - (unique_finals - 1) / len(paths)
        return convergence
    
    def _assess_reasoning_stability(self, dynamics: Dict[str, Any],
                                  reasoning: Dict[str, Any]) -> str:
        """Assess overall stability of reasoning process"""
        
        dynamics_state = dynamics["dynamics_state"]
        path_diversity = reasoning["path_diversity"]
        convergence_rate = reasoning["convergence_rate"]
        
        if dynamics_state == "chaotic":
            return "Unstable - reasoning is chaotic and unpredictable"
        elif dynamics_state == "edge_of_chaos":
            return "Critical - reasoning at transition point"
        elif path_diversity > 0.8 and convergence_rate < 0.3:
            return "Divergent - multiple incompatible reasoning paths"
        elif path_diversity < 0.2:
            return "Overfitted - reasoning lacks diversity"
        elif convergence_rate > 0.8:
            return "Convergent - reasoning reaches consistent conclusions"
        else:
            return "Balanced - healthy reasoning dynamics"

# ========== Demo and Testing ==========

def demonstrate_cognitive_dynamics_monitor():
    """Demonstrate cognitive dynamics monitoring"""
    
    print("🎢 Cognitive Dynamics Monitor Demo")
    print("=" * 60)
    
    # Initialize systems
    state_manager = CognitiveStateManager()
    monitor = CognitiveDynamicsMonitor(state_manager)
    reasoning_integration = ReasoningDynamicsIntegration(monitor)
    
    # Simulate different dynamics scenarios
    scenarios = [
        {
            "name": "Stable Reasoning",
            "states": [np.random.randn(100) * 0.1 for _ in range(20)]
        },
        {
            "name": "Chaotic Reasoning",
            "states": [np.random.randn(100) * (i * 0.5) for i in range(20)]
        },
        {
            "name": "Periodic Reasoning",
            "states": [np.sin(i * 0.5) * np.ones(100) + np.random.randn(100) * 0.1 
                      for i in range(20)]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print("-" * 60)
        
        # Reset state
        state_manager.state_history.clear()
        
        # Feed states
        for state in scenario["states"]:
            state_manager.update_state(new_state=state)
        
        # Monitor
        result = monitor.monitor_and_stabilize()
        
        print(f"Dynamics State: {result['dynamics_state']}")
        print(f"Max Lyapunov: {result['metrics']['max_lyapunov']:.3f}")
        print(f"Energy: {result['metrics']['energy']:.3f}")
        print(f"Attractor Type: {result['metrics']['attractor_type']}")
        
        if result['intervention']:
            print(f"\nIntervention Applied: {result['intervention']['strategy']}")
            print(f"Parameters: {result['intervention']['parameters']}")
    
    # Test reasoning dynamics
    print(f"\n{'='*60}")
    print("Testing Reasoning Dynamics Integration")
    print("-" * 60)
    
    # Create test reasoning paths
    from python.core.reasoning_traversal import ConceptNode
    
    test_paths = [
        ReasoningPath(
            chain=[
                ConceptNode(f"node_{i}", f"Node {i}", f"Description {i}")
                for i in range(3)
            ],
            edge_justifications=["implies", "supports"],
            score=0.8 - i * 0.1,
            path_type="inference",
            confidence=0.9 - i * 0.05
        )
        for i in range(5)
    ]
    
    # Analyze reasoning dynamics
    reasoning_result = reasoning_integration.analyze_reasoning_dynamics(test_paths)
    
    print(f"Path Diversity: {reasoning_result['reasoning_metrics']['path_diversity']:.3f}")
    print(f"Convergence Rate: {reasoning_result['reasoning_metrics']['convergence_rate']:.3f}")
    print(f"Stability Assessment: {reasoning_result['stability_assessment']}")
    
    # Generate final report
    print(f"\n{'='*60}")
    print("📊 Dynamics Report")
    print("-" * 60)
    
    report = monitor.get_dynamics_report()
    print(f"Current State: {report['current_state']}")
    print(f"Average Lyapunov: {report['metrics_summary']['average_lyapunov']:.3f}")
    print(f"Total Interventions: {report['intervention_summary']['total_interventions']}")
    print(f"Future Predictions: {report['future_predictions']}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    print("\n✅ Cognitive dynamics monitoring complete!")

if __name__ == "__main__":
    demonstrate_cognitive_dynamics_monitor()
