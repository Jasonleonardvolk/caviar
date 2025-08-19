"""
Trajectory sampling for Lyapunov function training.

This module provides the TrajectorySampler class, which generates state samples
for training neural Lyapunov functions with a focus on adaptive sampling.
"""

from typing import Callable, Iterable, List, Tuple, Optional
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class TrajectorySampler:
    """
    Generates state samples for training neural Lyapunov functions.
    
    For continuous systems, generates (x, f(x)) pairs. For discrete systems,
    generates (x_k, x_{k+1}) pairs. Supports adaptive sampling via
    counterexample incorporation.
    
    Attributes:
        f: System dynamics function that maps states to their derivatives/next states
        dim: Dimension of the state space
        low: Lower bounds of the sampling domain
        high: Upper bounds of the sampling domain
        batch_size: Number of samples to generate per batch
        _counterexample_bank: Storage for counterexamples found during verification
    """

    def __init__(
        self,
        dynamics_fn: Callable[[np.ndarray], np.ndarray],
        dim: int,
        domain: Tuple[np.ndarray, np.ndarray],
        batch_size: int = 1024,
        discrete: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize the trajectory sampler.
        
        Args:
            dynamics_fn: Function that maps states to their derivatives (for continuous systems)
                        or next states (for discrete systems)
            dim: Dimension of the state space
            domain: Tuple of (lower_bounds, upper_bounds) for the sampling domain
            batch_size: Number of samples to generate per batch
            discrete: Whether the system is discrete-time (True) or continuous-time (False)
            seed: Random seed for reproducibility
        """
        self.f = dynamics_fn
        self.dim = dim
        self.low, self.high = domain
        self.batch_size = batch_size
        self.discrete = discrete
        self._counterexample_bank: List[np.ndarray] = []
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Validate inputs
        if not callable(dynamics_fn):
            raise TypeError("dynamics_fn must be callable")
        
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
            
        if not isinstance(self.low, np.ndarray) or not isinstance(self.high, np.ndarray):
            raise TypeError("domain bounds must be numpy arrays")
            
        if self.low.shape != (dim,) or self.high.shape != (dim,):
            raise ValueError(f"domain bounds must have shape ({dim},)")
            
        if np.any(self.low >= self.high):
            raise ValueError("low bounds must be less than high bounds")
            
        logger.info(f"Initialized {self.__class__.__name__} for {'discrete' if discrete else 'continuous'} "
                    f"system with dimension {dim}")

    def random_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of random samples from the domain.
        
        Returns:
            Tuple of (x, f(x)) where:
                x: Batch of state vectors, shape (batch_size, dim)
                f(x): Batch of state derivatives/next states, shape (batch_size, dim)
        """
        # Generate random states uniformly within domain bounds
        x = np.random.uniform(
            self.low, self.high, 
            size=(self.batch_size, self.dim)
        )
        
        # Compute derivatives or next states
        fx = self.f(x)
        
        # Validate output dimensions
        if fx.shape != (self.batch_size, self.dim):
            raise ValueError(
                f"dynamics_fn returned incorrect shape: expected {(self.batch_size, self.dim)}, "
                f"got {fx.shape}"
            )
            
        return x, fx

    def add_counterexamples(self, xs: Iterable[np.ndarray]) -> None:
        """
        Add counterexamples to the sampling bank.
        
        These counterexamples will be included in future balanced batches,
        enabling adaptive sampling focused on problematic regions.
        
        Args:
            xs: Iterable of counterexample state vectors
        """
        count_before = len(self._counterexample_bank)
        
        for x in xs:
            # Validate counterexample
            if not isinstance(x, np.ndarray):
                logger.warning(f"Skipping non-ndarray counterexample: {type(x)}")
                continue
                
            if x.shape != (self.dim,):
                logger.warning(f"Skipping counterexample with incorrect shape: {x.shape}, expected ({self.dim},)")
                continue
                
            # Check if it's within bounds
            if np.any(x < self.low) or np.any(x > self.high):
                logger.warning(f"Counterexample outside domain bounds: {x}")
                
            # Add to bank
            self._counterexample_bank.append(x)
            
        count_added = len(self._counterexample_bank) - count_before
        logger.info(f"Added {count_added} new counterexamples (total: {len(self._counterexample_bank)})")

    def balanced_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a balanced batch combining random samples and counterexamples.
        
        The batch will contain approximately 50% random samples and 50% 
        counterexamples (if available), helping to focus training on 
        problematic regions while maintaining exploration.
        
        Returns:
            Tuple of (x, f(x)) where:
                x: Batch of state vectors, shape (batch_size, dim)
                f(x): Batch of state derivatives/next states, shape (batch_size, dim)
        """
        # Count available counterexamples
        k = len(self._counterexample_bank)
        
        if k == 0:
            # No counterexamples available, return random batch
            return self.random_batch()
            
        # Determine how many counterexamples to include (up to half the batch)
        m = min(k, self.batch_size // 2)
        
        # Randomly select counterexamples
        idx = np.random.choice(k, m, replace=False)
        ce_x = np.array([self._counterexample_bank[i] for i in idx])
        
        # Generate random samples for the rest of the batch
        rand_needed = self.batch_size - m
        rand_x, _ = self.random_batch()
        rand_x = rand_x[:rand_needed]
        
        # Combine counterexamples and random samples
        x = np.vstack([ce_x, rand_x])
        
        # Compute derivatives/next states for the combined batch
        fx = self.f(x)
        
        return x, fx
        
    def simulate_trajectory(
        self, 
        x0: np.ndarray, 
        steps: int, 
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a trajectory starting from the given initial state.
        
        For continuous systems, uses Euler integration with the given time step.
        For discrete systems, directly applies the dynamics function.
        
        Args:
            x0: Initial state vector
            steps: Number of steps to simulate
            dt: Time step for continuous systems (ignored for discrete systems)
            
        Returns:
            Tuple of (states, derivatives) where:
                states: Array of states along the trajectory, shape (steps+1, dim)
                derivatives: Array of derivatives/next_states, shape (steps+1, dim)
        """
        if x0.shape != (self.dim,):
            raise ValueError(f"Initial state must have shape ({self.dim},), got {x0.shape}")
            
        # Initialize arrays
        states = np.zeros((steps + 1, self.dim))
        derivatives = np.zeros((steps + 1, self.dim))
        
        # Set initial state
        states[0] = x0
        derivatives[0] = self.f(np.array([x0]))[0]
        
        # Simulate trajectory
        for i in range(steps):
            if self.discrete:
                # Discrete-time: x[k+1] = f(x[k])
                states[i+1] = self.f(np.array([states[i]]))[0]
            else:
                # Continuous-time: dx/dt = f(x) -> x[t+dt] ≈ x[t] + dt*f(x[t])
                states[i+1] = states[i] + dt * derivatives[i]
            
            # Compute derivative/next-state for the new state
            derivatives[i+1] = self.f(np.array([states[i+1]]))[0]
            
        return states, derivatives
        
    def clear_counterexamples(self) -> None:
        """Clear all stored counterexamples."""
        count = len(self._counterexample_bank)
        self._counterexample_bank = []
        logger.info(f"Cleared {count} counterexamples")
        
    def get_counterexample_count(self) -> int:
        """Return the number of stored counterexamples."""
        return len(self._counterexample_bank)
