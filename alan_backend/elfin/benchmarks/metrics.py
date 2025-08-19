"""
Benchmark Metrics

This module provides metrics for evaluating barrier functions and controllers.
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple, Callable, Optional
import logging
from scipy.integrate import solve_ivp

from .benchmark import BenchmarkSystem, BenchmarkMetric

logger = logging.getLogger(__name__)


class ValidationSuccessRate(BenchmarkMetric):
    """
    Metric that measures the success rate of barrier function validation.
    
    This metric evaluates what percentage of sampled states in the safe set
    have positive barrier values and satisfy the barrier derivative condition.
    """
    
    def __init__(self, samples: int = 1000):
        """
        Initialize the validation success rate metric.
        
        Args:
            samples: Number of states to sample
        """
        super().__init__(
            name="ValidationSuccessRate",
            description="Percentage of states that validate barrier conditions"
        )
        self.samples = samples
    
    def evaluate(
        self,
        system: BenchmarkSystem,
        **kwargs
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the validation success rate.
        
        Args:
            system: System to evaluate
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (success_rate, details)
        """
        # Get state bounds
        state_bounds = system.get_state_bounds()
        
        # Sample states uniformly from the state space
        states = np.random.uniform(
            state_bounds[:, 0],
            state_bounds[:, 1],
            size=(self.samples, system.state_dim)
        )
        
        # Check barrier function values
        barrier_values = np.array([
            system.barrier_function(state) for state in states
        ])
        
        # Check which states are safe (according to is_safe)
        safe_states = np.array([
            system.is_safe(state) for state in states
        ])
        
        # Check if barrier values match safety (positive in safe set)
        barrier_condition = barrier_values > 0
        
        # Count correct barrier values
        correct_barriers = np.sum(
            np.logical_or(
                np.logical_and(safe_states, barrier_condition),  # True positives
                np.logical_and(~safe_states, ~barrier_condition)  # True negatives
            )
        )
        
        # Check barrier derivative condition
        # We'll check this only for safe states
        safe_indices = np.where(safe_states)[0]
        
        derivative_failures = 0
        
        for idx in safe_indices:
            state = states[idx]
            
            # Get random control input
            if system.input_dim > 0:
                input_bounds = system.get_input_bounds()
                input_vec = np.random.uniform(
                    input_bounds[:, 0],
                    input_bounds[:, 1]
                )
            else:
                input_vec = np.array([])
            
            # Compute gradient and dynamics
            gradient = system.barrier_derivative(state)
            dynamics = system.dynamics(state, input_vec)
            
            # Compute Lie derivative
            lie_derivative = np.dot(gradient, dynamics)
            
            # Check Lie derivative condition
            # For a simple class-K function α(B(x)) = B(x)
            # The condition is: ∇B(x) · f(x, u) ≥ -B(x)
            barrier_value = barrier_values[idx]
            
            if barrier_value <= 0:
                continue  # Skip unsafe states
            
            if lie_derivative < -barrier_value:
                derivative_failures += 1
        
        # Compute success rates
        total_states = len(states)
        barrier_success_rate = correct_barriers / total_states
        
        if len(safe_indices) > 0:
            derivative_success_rate = 1.0 - derivative_failures / len(safe_indices)
        else:
            derivative_success_rate = float('nan')
        
        # Combined success rate (weighted average)
        if np.isnan(derivative_success_rate):
            combined_rate = barrier_success_rate
        else:
            combined_rate = 0.5 * barrier_success_rate + 0.5 * derivative_success_rate
        
        # Return results
        details = {
            "samples": self.samples,
            "barrier_success_rate": barrier_success_rate,
            "derivative_success_rate": derivative_success_rate,
            "safe_states_count": int(np.sum(safe_states)),
            "barrier_failures": int(total_states - correct_barriers),
            "derivative_failures": derivative_failures
        }
        
        return combined_rate, details


class ComputationTime(BenchmarkMetric):
    """
    Metric that measures the computation time for barrier functions and controllers.
    """
    
    def __init__(self, samples: int = 100, repetitions: int = 10):
        """
        Initialize the computation time metric.
        
        Args:
            samples: Number of states to sample
            repetitions: Number of repetitions for timing
        """
        super().__init__(
            name="ComputationTime",
            description="Average computation time for barrier functions (ms)"
        )
        self.samples = samples
        self.repetitions = repetitions
    
    def evaluate(
        self,
        system: BenchmarkSystem,
        **kwargs
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the computation time.
        
        Args:
            system: System to evaluate
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (computation_time_ms, details)
        """
        # Get state bounds
        state_bounds = system.get_state_bounds()
        
        # Sample states uniformly from the state space
        states = np.random.uniform(
            state_bounds[:, 0],
            state_bounds[:, 1],
            size=(self.samples, system.state_dim)
        )
        
        # Measure barrier function computation time
        barrier_times = []
        
        for _ in range(self.repetitions):
            start_time = time.time()
            
            for state in states:
                _ = system.barrier_function(state)
            
            end_time = time.time()
            barrier_times.append((end_time - start_time) * 1000 / self.samples)  # ms per call
        
        barrier_avg_time = np.mean(barrier_times)
        
        # Measure barrier derivative computation time
        derivative_times = []
        
        for _ in range(self.repetitions):
            start_time = time.time()
            
            for state in states:
                _ = system.barrier_derivative(state)
            
            end_time = time.time()
            derivative_times.append((end_time - start_time) * 1000 / self.samples)  # ms per call
        
        derivative_avg_time = np.mean(derivative_times)
        
        # Return results (use barrier function time as primary metric)
        details = {
            "barrier_function_time_ms": barrier_avg_time,
            "barrier_derivative_time_ms": derivative_avg_time,
            "samples": self.samples,
            "repetitions": self.repetitions
        }
        
        return barrier_avg_time, details


class Conservativeness(BenchmarkMetric):
    """
    Metric that measures how conservative a barrier function is.
    
    This metric evaluates the ratio of the verified safe set to the actual safe set.
    A lower ratio means the barrier function is more conservative (missing safe states).
    """
    
    def __init__(self, samples: int = 1000):
        """
        Initialize the conservativeness metric.
        
        Args:
            samples: Number of states to sample
        """
        super().__init__(
            name="Conservativeness",
            description="Ratio of verified safe set to actual safe set"
        )
        self.samples = samples
    
    def evaluate(
        self,
        system: BenchmarkSystem,
        **kwargs
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the conservativeness.
        
        Args:
            system: System to evaluate
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (ratio, details)
        """
        # Get state bounds
        state_bounds = system.get_state_bounds()
        
        # Sample states uniformly from the state space
        states = np.random.uniform(
            state_bounds[:, 0],
            state_bounds[:, 1],
            size=(self.samples, system.state_dim)
        )
        
        # Check actual safety
        actual_safe = np.array([
            system.is_safe(state) for state in states
        ])
        
        # Check barrier function values
        barrier_safe = np.array([
            system.barrier_function(state) > 0 for state in states
        ])
        
        # Count true positives, false negatives, etc.
        true_positives = np.sum(np.logical_and(actual_safe, barrier_safe))
        false_negatives = np.sum(np.logical_and(actual_safe, ~barrier_safe))
        false_positives = np.sum(np.logical_and(~actual_safe, barrier_safe))
        
        # Compute metrics
        if np.sum(actual_safe) > 0:
            recall = true_positives / np.sum(actual_safe)  # True positive rate
        else:
            recall = float('nan')
        
        if np.sum(barrier_safe) > 0:
            precision = true_positives / np.sum(barrier_safe)  # Positive predictive value
        else:
            precision = float('nan')
        
        # Compute F1 score (harmonic mean of precision and recall)
        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        # Compute the ratio of verified safe set to actual safe set
        if np.sum(actual_safe) > 0:
            ratio = true_positives / np.sum(actual_safe)
        else:
            ratio = float('nan')
        
        # Return results
        details = {
            "samples": self.samples,
            "actual_safe_count": int(np.sum(actual_safe)),
            "barrier_safe_count": int(np.sum(barrier_safe)),
            "true_positives": int(true_positives),
            "false_negatives": int(false_negatives),
            "false_positives": int(false_positives),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
        return ratio, details


class DisturbanceRobustness(BenchmarkMetric):
    """
    Metric that measures the robustness of a barrier function to disturbances.
    
    This metric evaluates how well the barrier function maintains safety under
    various disturbance levels.
    """
    
    def __init__(
        self,
        num_trajectories: int = 20,
        T: float = 10.0,
        disturbance_levels: Optional[List[float]] = None
    ):
        """
        Initialize the disturbance robustness metric.
        
        Args:
            num_trajectories: Number of trajectories to simulate
            T: Simulation time horizon
            disturbance_levels: Disturbance levels to test (relative to state scale)
        """
        super().__init__(
            name="DisturbanceRobustness",
            description="Maximum disturbance level that maintains safety"
        )
        self.num_trajectories = num_trajectories
        self.T = T
        self.disturbance_levels = disturbance_levels or [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    def evaluate(
        self,
        system: BenchmarkSystem,
        **kwargs
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the disturbance robustness.
        
        Args:
            system: System to evaluate
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (max_safe_disturbance, details)
        """
        # Get initial states
        initial_states = system.get_initial_states(self.num_trajectories)
        
        # Estimate state scale for disturbance scaling
        state_bounds = system.get_state_bounds()
        state_scale = np.mean(state_bounds[:, 1] - state_bounds[:, 0])
        
        # Define controller (if any)
        controller = kwargs.get('controller', lambda x: np.zeros(system.input_dim))
        
        # Define disturbed dynamics
        def disturbed_dynamics(t, x, disturbance_scale):
            # Compute control input
            u = controller(x)
            
            # Compute nominal dynamics
            dx = system.dynamics(x, u)
            
            # Add disturbance
            disturbance = disturbance_scale * state_scale * np.random.normal(
                0, 1, size=system.state_dim
            )
            
            return dx + disturbance
        
        # Track safety for each disturbance level
        safety_results = {}
        
        for level in self.disturbance_levels:
            safety_counts = 0
            barrier_violations = 0
            
            for init_state in initial_states:
                # Check if initial state is safe
                if not system.is_safe(init_state):
                    continue
                
                # Simulate trajectory with disturbance
                dynamics_fn = lambda t, x: disturbed_dynamics(t, x, level)
                
                result = solve_ivp(
                    dynamics_fn,
                    (0, self.T),
                    init_state,
                    method='RK45',
                    t_eval=np.linspace(0, self.T, 100)
                )
                
                # Check if trajectory remains safe
                safety_violations = 0
                for state in result.y.T:
                    if not system.is_safe(state):
                        safety_violations += 1
                
                # Check if barrier function remains positive
                barrier_values = np.array([
                    system.barrier_function(state) for state in result.y.T
                ])
                
                barrier_violations += int(np.any(barrier_values <= 0))
                
                # Count as safe if less than 5% of trajectory points are unsafe
                if safety_violations <= 5:
                    safety_counts += 1
            
            # Store results
            safety_ratio = safety_counts / self.num_trajectories
            safety_results[str(level)] = {
                "safety_ratio": safety_ratio,
                "barrier_violations": barrier_violations
            }
            
            logger.debug(f"Disturbance level {level}: safety ratio = {safety_ratio}")
        
        # Find the highest disturbance level with safety ratio > 0.8
        max_safe_disturbance = 0.0
        for level in sorted(self.disturbance_levels):
            level_str = str(level)
            if level_str in safety_results and safety_results[level_str]["safety_ratio"] > 0.8:
                max_safe_disturbance = level
        
        # Return results
        details = {
            "num_trajectories": self.num_trajectories,
            "T": self.T,
            "state_scale": float(state_scale),
            "disturbance_levels": self.disturbance_levels,
            "safety_results": safety_results
        }
        
        return max_safe_disturbance, details
