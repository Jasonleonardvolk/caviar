from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\dynamics.py

"""
Cognitive Dynamics Implementation
================================

Implements stochastic differential equations for cognitive state evolution,
including deterministic drift from reflection and stochastic exploration.
"""

import numpy as np
from typing import Optional, Callable, Dict, List, Tuple
from .manifold import MetaCognitiveManifold
from .reflective import ReflectiveOperator
from .curiosity import CuriosityFunctional


class CognitiveDynamics:
    """
    Evolves cognitive states via SDE: ds = (R(s)-s)dt + σ dW.
    
    Combines deterministic reflection with stochastic exploration
    to model cognitive dynamics on the manifold.
    
    Attributes:
        manifold: Cognitive manifold
        reflective: Reflective operator for deterministic drift
        curiosity: Optional curiosity functional for adaptive noise
        sigma: Base noise standard deviation
        dt: Time step for integration
    """
    
    def __init__(self,
                 manifold: MetaCognitiveManifold,
                 reflective_op: ReflectiveOperator,
                 curiosity_func: Optional[CuriosityFunctional] = None,
                 noise_sigma: float = 0.1,
                 dt: float = 0.01):
        """
        Initialize cognitive dynamics.
        
        Args:
            manifold: Cognitive manifold
            reflective_op: Reflective operator R
            curiosity_func: Optional curiosity functional
            noise_sigma: Standard deviation of noise
            dt: Default time step
        """
        self.manifold = manifold
        self.reflective = reflective_op
        self.curiosity = curiosity_func
        self.sigma = noise_sigma
        self.dt = dt
        self.trajectory_history = []

    def evolve(self, 
               s0: np.ndarray,
               t_span: float,
               dt: Optional[float] = None,
               return_times: bool = False) -> np.ndarray:
        """
        Evolve cognitive state over time using Euler-Maruyama method.
        
        Args:
            s0: Initial cognitive state
            t_span: Total time to evolve
            dt: Time step (uses self.dt if None)
            return_times: Whether to return time points
            
        Returns:
            Trajectory array (n_steps × n_features) or tuple with times
        """
        dt = dt or self.dt
        steps = int(t_span / dt)
        
        # Initialize trajectory
        trajectory = np.zeros((steps + 1, len(s0)))
        trajectory[0] = s0
        times = np.linspace(0, t_span, steps + 1)
        
        # Current state
        s = s0.copy()
        
        # Evolve system
        for i in range(1, steps + 1):
            # Deterministic drift: R(s) - s
            s_reflected = self.reflective.apply(s)
            drift = s_reflected - s
            
            # Adaptive noise based on curiosity
            if self.curiosity is not None:
                # Use curiosity to modulate exploration
                curiosity_score = self.curiosity.compute(s, s_reflected)
                noise_scale = self.sigma * (1.0 + curiosity_score)
            else:
                noise_scale = self.sigma
            
            # Stochastic term: σ dW
            noise = noise_scale * np.sqrt(dt) * np.random.randn(*s.shape)
            
            # Euler-Maruyama update
            s = s + drift * dt + noise
            
            # Store in trajectory
            trajectory[i] = s
        
        # Store for analysis
        self.trajectory_history.append({
            'trajectory': trajectory,
            'times': times,
            'initial_state': s0,
            'final_state': s,
            't_span': t_span
        })
        
        if return_times:
            return trajectory, times
        return trajectory

    def evolve_adaptive(self,
                       s0: np.ndarray,
                       t_span: float,
                       dt_min: float = 1e-4,
                       dt_max: float = 0.1,
                       tol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve with adaptive time stepping based on local dynamics.
        
        Args:
            s0: Initial state
            t_span: Total time
            dt_min: Minimum time step
            dt_max: Maximum time step
            tol: Error tolerance for step size control
            
        Returns:
            Tuple of (trajectory, times)
        """
        trajectory = [s0.copy()]
        times = [0.0]
        
        s = s0.copy()
        t = 0.0
        dt = self.dt
        
        while t < t_span:
            # Ensure we don't overshoot
            dt = min(dt, t_span - t)
            
            # Trial step with current dt
            s_reflected = self.reflective.apply(s)
            drift = s_reflected - s
            noise = self.sigma * np.sqrt(dt) * np.random.randn(*s.shape)
            s_trial = s + drift * dt + noise
            
            # Estimate error with half-step
            dt_half = dt / 2
            s_half = s + drift * dt_half + self.sigma * np.sqrt(dt_half) * np.random.randn(*s.shape)
            s_reflected_half = self.reflective.apply(s_half)
            drift_half = s_reflected_half - s_half
            noise_half = self.sigma * np.sqrt(dt_half) * np.random.randn(*s.shape)
            s_double = s_half + drift_half * dt_half + noise_half
            
            # Error estimate
            error = np.linalg.norm(s_trial - s_double) / (np.linalg.norm(s) + 1e-10)
            
            # Accept or reject step
            if error < tol:
                # Accept step
                s = s_trial
                t += dt
                trajectory.append(s.copy())
                times.append(t)
                
                # Increase step size
                dt = min(dt * 1.5, dt_max)
            else:
                # Reject step and decrease step size
                dt = max(dt * 0.5, dt_min)
        
        return np.array(trajectory), np.array(times)

    def simulate_with_control(self,
                            s0: np.ndarray,
                            t_span: float,
                            control_func: Callable[[np.ndarray, float], np.ndarray],
                            dt: Optional[float] = None) -> np.ndarray:
        """
        Evolve with external control input.
        
        Args:
            s0: Initial state
            t_span: Total time
            control_func: Control function u(s, t)
            dt: Time step
            
        Returns:
            Controlled trajectory
        """
        dt = dt or self.dt
        steps = int(t_span / dt)
        
        trajectory = np.zeros((steps + 1, len(s0)))
        trajectory[0] = s0
        
        s = s0.copy()
        
        for i in range(1, steps + 1):
            t = i * dt
            
            # Reflective drift
            s_reflected = self.reflective.apply(s)
            drift = s_reflected - s
            
            # Control input
            control = control_func(s, t)
            
            # Noise
            noise = self.sigma * np.sqrt(dt) * np.random.randn(*s.shape)
            
            # Controlled dynamics
            s = s + (drift + control) * dt + noise
            trajectory[i] = s
        
        return trajectory

    def compute_lyapunov_exponents(self,
                                  s0: np.ndarray,
                                  t_span: float,
                                  n_exponents: int = None,
                                  dt: Optional[float] = None) -> np.ndarray:
        """
        Estimate Lyapunov exponents of the dynamics.
        
        Args:
            s0: Initial state
            t_span: Time span for estimation
            n_exponents: Number of exponents to compute (default: all)
            dt: Time step
            
        Returns:
            Array of Lyapunov exponents
        """
        dt = dt or self.dt
        n_exponents = n_exponents or len(s0)
        steps = int(t_span / dt)
        
        # Initialize tangent vectors (orthonormal)
        Q = np.eye(len(s0), n_exponents)
        
        # Accumulate stretching
        lyap_sum = np.zeros(n_exponents)
        
        s = s0.copy()
        
        for i in range(steps):
            # Evolve state
            s_reflected = self.reflective.apply(s)
            drift = s_reflected - s
            noise = self.sigma * np.sqrt(dt) * np.random.randn(*s.shape)
            s_next = s + drift * dt + noise
            
            # Linearized dynamics (finite difference)
            J = np.zeros((len(s), len(s)))
            eps = 1e-6
            for j in range(len(s)):
                s_pert = s.copy()
                s_pert[j] += eps
                s_pert_reflected = self.reflective.apply(s_pert)
                drift_pert = s_pert_reflected - s_pert
                J[:, j] = (drift_pert - drift) / eps
            
            # Evolve tangent vectors
            Q_next = Q + J @ Q * dt
            
            # QR decomposition for orthonormalization
            Q_next, R = np.linalg.qr(Q_next)
            
            # Accumulate stretching factors
            for j in range(n_exponents):
                lyap_sum[j] += np.log(abs(R[j, j]))
            
            Q = Q_next
            s = s_next
        
        # Average over time
        lyapunov_exponents = lyap_sum / t_span
        
        return lyapunov_exponents

    def analyze_stability(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Analyze stability properties of a trajectory.
        
        Args:
            trajectory: Cognitive trajectory
            
        Returns:
            Dictionary of stability metrics
        """
        if len(trajectory) < 2:
            return {}
        
        # Compute various stability metrics
        velocities = np.diff(trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Average speed
        avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
        
        # Speed variance (measure of stability)
        speed_var = np.var(np.linalg.norm(velocities, axis=1))
        
        # Maximum acceleration (measure of smoothness)
        if len(accelerations) > 0:
            max_accel = np.max(np.linalg.norm(accelerations, axis=1))
        else:
            max_accel = 0.0
        
        # Trajectory curvature
        curvatures = []
        for i in range(1, len(velocities)):
            v1 = velocities[i-1]
            v2 = velocities[i]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                curvatures.append(np.arccos(cos_angle))
        
        avg_curvature = np.mean(curvatures) if curvatures else 0.0
        
        # Convergence measure (distance between consecutive states)
        convergence_rate = []
        for i in range(1, len(trajectory)):
            dist = self.manifold.distance(trajectory[i], trajectory[i-1])
            convergence_rate.append(dist)
        
        final_convergence = convergence_rate[-1] if convergence_rate else 0.0
        
        return {
            'average_speed': avg_speed,
            'speed_variance': speed_var,
            'max_acceleration': max_accel,
            'average_curvature': avg_curvature,
            'final_convergence': final_convergence,
            'trajectory_length': len(trajectory)
        }