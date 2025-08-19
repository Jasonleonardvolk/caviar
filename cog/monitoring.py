from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\monitoring.py

"""
Consciousness and Stability Monitoring
====================================

Implements monitoring systems for consciousness preservation
and dynamic stability in the cognitive framework.
"""

import numpy as np
from typing import Optional, Callable, Dict, List, Tuple
import warnings
from collections import deque
from .utils import compute_iit_phi, compute_free_energy


class ConsciousnessMonitor:
    """
    Monitor consciousness levels using IIT and other measures.
    
    Tracks integrated information and alerts when consciousness
    metrics fall below specified thresholds.
    
    Attributes:
        phi_threshold: Minimum acceptable Φ value
        iit_func: Function to compute integrated information
        history_size: Size of monitoring history buffer
        alert_callbacks: Functions to call on threshold violations
    """
    
    def __init__(self, 
                 phi_threshold: float = 0.5,
                 iit_func: Optional[Callable] = None,
                 history_size: int = 100,
                 additional_metrics: Optional[Dict[str, Callable]] = None):
        """
        Initialize consciousness monitor.
        
        Args:
            phi_threshold: Minimum Φ for consciousness preservation
            iit_func: Custom IIT computation function
            history_size: Number of historical values to track
            additional_metrics: Extra metrics to monitor
        """
        self.threshold = phi_threshold
        self.iit = iit_func or compute_iit_phi
        self.history_size = history_size
        self.additional_metrics = additional_metrics or {}
        
        # History tracking
        self.phi_history = deque(maxlen=history_size)
        self.state_history = deque(maxlen=history_size)
        self.metric_history = {name: deque(maxlen=history_size) 
                              for name in self.additional_metrics}
        
        # Alert system
        self.alert_callbacks = []
        self.alert_history = []
        
        # Statistics
        self.total_checks = 0
        self.violations = 0

    def check_preservation(self, 
                          s_before: np.ndarray,
                          s_after: np.ndarray,
                          connectivity: Optional[np.ndarray] = None) -> bool:
        """
        Check if consciousness is preserved across state transition.
        
        Args:
            s_before: State before transition
            s_after: State after transition
            connectivity: Optional connectivity matrix
            
        Returns:
            True if consciousness preserved, False otherwise
        """
        # Compute Φ values
        phi_before = self.iit(s_before, connectivity)
        phi_after = self.iit(s_after, connectivity)
        
        # Update history
        self.phi_history.append(phi_after)
        self.state_history.append(s_after.copy())
        
        # Compute additional metrics
        for name, func in self.additional_metrics.items():
            value = func(s_after)
            self.metric_history[name].append(value)
        
        # Check threshold
        preserved = phi_after >= self.threshold
        
        # Update statistics
        self.total_checks += 1
        if not preserved:
            self.violations += 1
            
            # Trigger alerts
            alert_info = {
                'phi_before': phi_before,
                'phi_after': phi_after,
                'threshold': self.threshold,
                'state_before': s_before,
                'state_after': s_after,
                'violation_number': self.violations
            }
            
            self._trigger_alerts(alert_info)
            self.alert_history.append(alert_info)
        
        return preserved

    def _trigger_alerts(self, alert_info: Dict):
        """
        Trigger alert callbacks.
        
        Args:
            alert_info: Information about the violation
        """
        for callback in self.alert_callbacks:
            try:
                callback(alert_info)
            except Exception as e:
                warnings.warn(f"Alert callback failed: {e}")

    def add_alert_callback(self, callback: Callable):
        """
        Add function to call on consciousness violations.
        
        Args:
            callback: Function accepting alert_info dict
        """
        self.alert_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, float]:
        """
        Get monitoring statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_checks': self.total_checks,
            'violations': self.violations,
            'violation_rate': self.violations / max(1, self.total_checks),
            'current_phi': self.phi_history[-1] if self.phi_history else 0.0,
            'average_phi': np.mean(self.phi_history) if self.phi_history else 0.0,
            'min_phi': np.min(self.phi_history) if self.phi_history else 0.0,
            'max_phi': np.max(self.phi_history) if self.phi_history else 0.0,
            'phi_std': np.std(self.phi_history) if self.phi_history else 0.0
        }
        
        # Add additional metric statistics
        for name, history in self.metric_history.items():
            if history:
                stats[f'{name}_mean'] = np.mean(history)
                stats[f'{name}_std'] = np.std(history)
                stats[f'{name}_current'] = history[-1]
        
        return stats

    def get_trend(self, window_size: int = 10) -> float:
        """
        Compute recent trend in consciousness.
        
        Args:
            window_size: Number of recent values to consider
            
        Returns:
            Trend slope (positive = increasing consciousness)
        """
        if len(self.phi_history) < 2:
            return 0.0
        
        # Get recent values
        recent = list(self.phi_history)[-window_size:]
        x = np.arange(len(recent))
        
        # Linear regression
        coeffs = np.polyfit(x, recent, 1)
        
        return coeffs[0]  # Slope

    def suggest_intervention(self) -> Optional[Dict[str, any]]:
        """
        Suggest intervention based on consciousness trends.
        
        Returns:
            Intervention suggestion or None
        """
        if len(self.phi_history) < 5:
            return None
        
        stats = self.get_statistics()
        trend = self.get_trend()
        
        # Check for critical situations
        if stats['current_phi'] < self.threshold * 0.8:
            # Very low consciousness
            return {
                'type': 'critical',
                'action': 'increase_iit_weight',
                'urgency': 'high',
                'reason': f"Φ = {stats['current_phi']:.3f} is critically low"
            }
        
        elif trend < -0.01 and stats['current_phi'] < self.threshold * 1.2:
            # Declining consciousness
            return {
                'type': 'preventive',
                'action': 'stabilize',
                'urgency': 'medium',
                'reason': f"Consciousness declining (trend = {trend:.3f})"
            }
        
        elif stats['violation_rate'] > 0.1:
            # Frequent violations
            return {
                'type': 'systematic',
                'action': 'adjust_dynamics',
                'urgency': 'medium',
                'reason': f"High violation rate ({stats['violation_rate']:.1%})"
            }
        
        return None


class LyapunovStabilizer:
    """
    Monitor and stabilize cognitive dynamics using Lyapunov functions.
    
    Ensures cognitive trajectories remain bounded and converge
    to desirable regions of the state space.
    
    Attributes:
        free_energy_func: Function computing free energy
        stability_margin: Required decrease rate for stability
        intervention_threshold: When to intervene
    """
    
    def __init__(self,
                 free_energy_func: Optional[Callable] = None,
                 stability_margin: float = 0.01,
                 intervention_threshold: float = 0.1,
                 history_size: int = 100):
        """
        Initialize Lyapunov stabilizer.
        
        Args:
            free_energy_func: Function to use as Lyapunov function
            stability_margin: Required decrease in V for stability
            intervention_threshold: Threshold for intervention
            history_size: Size of history buffer
        """
        self.free_energy = free_energy_func or compute_free_energy
        self.margin = stability_margin
        self.intervention_threshold = intervention_threshold
        
        # History tracking
        self.lyapunov_history = deque(maxlen=history_size)
        self.trajectory_history = deque(maxlen=history_size)
        self.intervention_history = []
        
        # Control parameters
        self.control_gain = 0.1
        self.adaptive_gain = True

    def compute_lyapunov(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute Lyapunov function along trajectory.
        
        V(s) = ||s||² + F(s)
        
        Args:
            trajectory: Cognitive trajectory (n_steps × n_features)
            
        Returns:
            Lyapunov values along trajectory
        """
        V = []
        
        for s in trajectory:
            # Energy term
            energy = np.linalg.norm(s)**2
            
            # Free energy term
            free_energy = self.free_energy(s)
            
            # Combined Lyapunov function
            v = energy + free_energy
            V.append(v)
        
        return np.array(V)

    def check_stability(self, trajectory: np.ndarray) -> Dict[str, any]:
        """
        Check stability of cognitive trajectory.
        
        Args:
            trajectory: Cognitive trajectory
            
        Returns:
            Stability analysis results
        """
        if len(trajectory) < 2:
            return {'stable': True, 'reason': 'Trajectory too short'}
        
        # Compute Lyapunov values
        V = self.compute_lyapunov(trajectory)
        
        # Update history
        self.lyapunov_history.extend(V)
        self.trajectory_history.append(trajectory)
        
        # Check for decrease
        dV = np.diff(V)
        avg_dV = np.mean(dV)
        max_increase = np.max(dV)
        
        # Stability criteria
        is_stable = avg_dV < self.margin
        is_bounded = np.all(V < np.max(V[0], 100))
        needs_intervention = max_increase > self.intervention_threshold
        
        # Compute stability margin
        stability_score = -avg_dV / (np.mean(np.abs(V)) + 1e-10)
        
        return {
            'stable': is_stable and is_bounded,
            'average_dV': avg_dV,
            'max_increase': max_increase,
            'needs_intervention': needs_intervention,
            'stability_score': stability_score,
            'lyapunov_values': V,
            'bounded': is_bounded
        }

    def compute_control(self, 
                       current_state: np.ndarray,
                       target_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute stabilizing control input.
        
        Uses Control Lyapunov Function (CLF) approach.
        
        Args:
            current_state: Current cognitive state
            target_state: Optional target state (default: origin)
            
        Returns:
            Control vector u
        """
        if target_state is None:
            target_state = np.zeros_like(current_state)
        
        # Error state
        error = current_state - target_state
        
        # Gradient of Lyapunov function
        def V(s):
            return np.linalg.norm(s - target_state)**2 + self.free_energy(s)
        
        from .utils import numeric_gradient
        grad_V = numeric_gradient(V, current_state)
        
        # Control law: u = -k * ∇V
        control = -self.control_gain * grad_V
        
        # Adaptive gain
        if self.adaptive_gain and len(self.lyapunov_history) > 10:
            recent_V = list(self.lyapunov_history)[-10:]
            if np.mean(np.diff(recent_V)) > 0:
                # Increase gain if Lyapunov increasing
                self.control_gain *= 1.1
                self.control_gain = min(self.control_gain, 1.0)
            else:
                # Decrease gain if stable
                self.control_gain *= 0.95
                self.control_gain = max(self.control_gain, 0.01)
        
        # Saturate control
        max_control = 1.0
        control_norm = np.linalg.norm(control)
        if control_norm > max_control:
            control = control * max_control / control_norm
        
        # Record intervention
        self.intervention_history.append({
            'state': current_state.copy(),
            'control': control.copy(),
            'gain': self.control_gain,
            'time': len(self.intervention_history)
        })
        
        return control

    def design_barrier_function(self, 
                               safe_set_center: np.ndarray,
                               safe_set_radius: float) -> Callable:
        """
        Design control barrier function for safety.
        
        Args:
            safe_set_center: Center of safe region
            safe_set_radius: Radius of safe region
            
        Returns:
            Barrier function h(s)
        """
        def barrier(s: np.ndarray) -> float:
            # h(s) = R² - ||s - center||²
            # h(s) > 0 implies s is in safe set
            distance_sq = np.sum((s - safe_set_center)**2)
            return safe_set_radius**2 - distance_sq
        
        return barrier

    def get_phase_portrait(self, 
                          bounds: Tuple[float, float] = (-2, 2),
                          resolution: int = 20) -> Dict[str, np.ndarray]:
        """
        Compute phase portrait for 2D visualization.
        
        Args:
            bounds: (min, max) for each axis
            resolution: Grid resolution
            
        Returns:
            Dictionary with grid and vector field
        """
        if hasattr(self, 'trajectory_history') and self.trajectory_history:
            dim = len(self.trajectory_history[-1][0])
            if dim != 2:
                warnings.warn("Phase portrait only available for 2D systems")
                return {}
        
        # Create grid
        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[0], bounds[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(resolution):
            for j in range(resolution):
                state = np.array([X[i, j], Y[i, j]])
                control = self.compute_control(state)
                U[i, j] = control[0]
                V[i, j] = control[1]
        
        return {
            'X': X,
            'Y': Y,
            'U': U,
            'V': V,
            'x': x,
            'y': y
        }