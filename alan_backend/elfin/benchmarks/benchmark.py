"""
Core Benchmark Framework

This module provides the base classes for defining benchmarks, systems, and metrics.
"""

import numpy as np
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

logger = logging.getLogger(__name__)


class BenchmarkSystem(ABC):
    """
    Base class for benchmark systems.
    
    A benchmark system represents a dynamical system with specific safety properties
    that can be verified using barrier functions.
    """
    
    def __init__(
        self,
        name: str,
        state_dim: int,
        input_dim: int,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a benchmark system.
        
        Args:
            name: Name of the system
            state_dim: Dimension of the state space
            input_dim: Dimension of the input space
            params: System parameters
        """
        self.name = name
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.params = params or {}
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        """
        Compute the system dynamics dx/dt = f(x, u).
        
        Args:
            state: Current state
            input_vec: Control input
        
        Returns:
            State derivative
        """
        pass
    
    @abstractmethod
    def barrier_function(self, state: np.ndarray) -> float:
        """
        Compute the barrier function B(x).
        
        Args:
            state: Current state
        
        Returns:
            Barrier function value
        """
        pass
    
    @abstractmethod
    def is_safe(self, state: np.ndarray) -> bool:
        """
        Check if a state is safe.
        
        Args:
            state: State to check
        
        Returns:
            True if the state is safe, False otherwise
        """
        pass
    
    def barrier_derivative(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the barrier function at a given state.
        
        By default, uses numeric differentiation. Override for analytical gradients.
        
        Args:
            state: Current state
        
        Returns:
            Gradient of the barrier function
        """
        # Compute gradient numerically using central difference
        h = 1e-6  # Step size for finite difference
        gradient = np.zeros(self.state_dim)
        
        for i in range(self.state_dim):
            # Create perturbation vectors
            state_plus = state.copy()
            state_plus[i] += h
            
            state_minus = state.copy()
            state_minus[i] -= h
            
            # Compute central difference
            b_plus = self.barrier_function(state_plus)
            b_minus = self.barrier_function(state_minus)
            
            gradient[i] = (b_plus - b_minus) / (2 * h)
        
        return gradient
    
    def lyapunov_function(self, state: np.ndarray) -> float:
        """
        Compute the Lyapunov function V(x), if available.
        
        Args:
            state: Current state
        
        Returns:
            Lyapunov function value
        """
        # Default implementation (not available)
        return 0.0
    
    def get_state_bounds(self) -> np.ndarray:
        """
        Get state space bounds for this system.
        
        Returns:
            Array of state bounds, shape (state_dim, 2)
        """
        # Default implementation, override for specific systems
        return np.array([[-10.0, 10.0]] * self.state_dim)
    
    def get_input_bounds(self) -> np.ndarray:
        """
        Get input space bounds for this system.
        
        Returns:
            Array of input bounds, shape (input_dim, 2)
        """
        # Default implementation, override for specific systems
        return np.array([[-10.0, 10.0]] * self.input_dim) if self.input_dim > 0 else np.array([])
    
    def get_initial_states(self, num_states: int = 10) -> np.ndarray:
        """
        Get representative initial states for simulation.
        
        Args:
            num_states: Number of initial states to generate
        
        Returns:
            Array of initial states, shape (num_states, state_dim)
        """
        # Default implementation, override for specific systems
        bounds = self.get_state_bounds()
        states = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(num_states, self.state_dim)
        )
        return states
    
    def get_elfin_spec(self) -> str:
        """
        Get the ELFIN specification for this system.
        
        Returns:
            ELFIN specification as a string
        """
        # Default implementation (not available)
        return f"# ELFIN specification for {self.name}\n\n# Not implemented"
    
    def __str__(self) -> str:
        return f"{self.name} (state_dim={self.state_dim}, input_dim={self.input_dim})"


class BenchmarkMetric(ABC):
    """
    Base class for benchmark metrics.
    
    A benchmark metric evaluates the performance of a barrier function or controller
    on a specific system.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a benchmark metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def evaluate(
        self,
        system: BenchmarkSystem,
        **kwargs
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the metric on a system.
        
        Args:
            system: System to evaluate
            **kwargs: Additional parameters for the evaluation
        
        Returns:
            Tuple of (metric_value, details)
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark evaluation.
    """
    system_name: str
    system_params: Dict[str, Any]
    metrics: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "system_name": self.system_name,
            "system_params": self.system_params,
            "metrics": self.metrics,
            "details": self.details,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(
            system_name=data["system_name"],
            system_params=data["system_params"],
            metrics=data["metrics"],
            details=data["details"],
            timestamp=data["timestamp"]
        )
    
    def to_json(self, filepath: str = None):
        """Save to JSON file."""
        data = self.to_dict()
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return filepath
        else:
            return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, filepath_or_string: str) -> 'BenchmarkResult':
        """Load from JSON file or string."""
        try:
            # Try to open as file
            with open(filepath_or_string, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, IsADirectoryError):
            # Try to parse as string
            data = json.loads(filepath_or_string)
        
        return cls.from_dict(data)


class Benchmark:
    """
    A benchmark for evaluating barrier functions on a specific system.
    """
    
    def __init__(
        self,
        system: BenchmarkSystem,
        metrics: List[BenchmarkMetric],
        name: Optional[str] = None
    ):
        """
        Initialize a benchmark.
        
        Args:
            system: System to benchmark
            metrics: Metrics to evaluate
            name: Name of the benchmark
        """
        self.system = system
        self.metrics = metrics
        self.name = name or f"Benchmark_{system.name}"
    
    def run(self, **kwargs) -> BenchmarkResult:
        """
        Run the benchmark.
        
        Args:
            **kwargs: Additional parameters for metrics
        
        Returns:
            Benchmark result
        """
        logger.info(f"Running benchmark: {self.name}")
        
        # Evaluate all metrics
        metrics_result = {}
        details = {}
        
        for metric in self.metrics:
            logger.info(f"Evaluating metric: {metric.name}")
            try:
                value, metric_details = metric.evaluate(self.system, **kwargs)
                metrics_result[metric.name] = value
                details[metric.name] = metric_details
                logger.info(f"Metric {metric.name}: {value}")
            except Exception as e:
                logger.exception(f"Error evaluating metric {metric.name}")
                metrics_result[metric.name] = float('nan')
                details[metric.name] = {"error": str(e)}
        
        # Create result
        result = BenchmarkResult(
            system_name=self.system.name,
            system_params=self.system.params.copy(),
            metrics=metrics_result,
            details=details
        )
        
        return result
    
    def __str__(self) -> str:
        metrics_str = ", ".join(m.name for m in self.metrics)
        return f"{self.name}: System={self.system.name}, Metrics=[{metrics_str}]"
