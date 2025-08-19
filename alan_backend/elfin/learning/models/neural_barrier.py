"""
Neural Barrier Network Base Class

This module provides the base class for neural barrier networks, which can be
implemented using different frameworks (PyTorch, JAX).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import numpy as np


class NeuralBarrierNetwork(ABC):
    """
    Abstract base class for neural barrier networks.
    
    A neural barrier network computes a barrier function B(x), where B(x) > 0 in the safe set
    and B(x) <= 0 in the unsafe set. The network also provides methods for computing gradients
    of B with respect to inputs, which is necessary for verifying the barrier certificate condition:
        ∇B(x) · f(x, u) >= -α(B(x)) for all x in the safe set
    """
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute barrier function value for given inputs.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Barrier function values, shape (batch_size, 1)
        """
        pass
    
    @abstractmethod
    def get_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the barrier function with respect to the input.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Gradient array, shape (batch_size, state_dim)
        """
        pass
    
    @abstractmethod
    def verify_condition(
        self,
        x: np.ndarray,
        dynamics_fn: Callable,
        u: Optional[np.ndarray] = None,
        alpha_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Verify the barrier certificate condition: ∇B(x) · f(x, u) >= -α(B(x))
        
        Args:
            x: State array, shape (batch_size, state_dim)
            dynamics_fn: Function mapping (x, u) to state derivatives
            u: Control inputs (optional), shape (batch_size, input_dim)
            alpha_fn: Class-K function (optional)
            
        Returns:
            Boolean array indicating whether the condition is satisfied for each state
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str, **kwargs):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model
        """
        pass
    
    def is_safe(self, x: np.ndarray) -> np.ndarray:
        """
        Determine if states are in the safe set.
        
        A state is in the safe set if B(x) > 0.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Boolean array indicating whether each state is in the safe set
        """
        barrier_values = self(x)
        return barrier_values > 0
    
    def is_unsafe(self, x: np.ndarray) -> np.ndarray:
        """
        Determine if states are in the unsafe set.
        
        A state is in the unsafe set if B(x) <= 0.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Boolean array indicating whether each state is in the unsafe set
        """
        return ~self.is_safe(x)
    
    def safety_margin(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the safety margin for each state.
        
        This is simply the barrier function value, which indicates how "deep" a state is
        in the safe or unsafe set.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Safety margin, shape (batch_size, 1)
        """
        return self(x)


class NeuralLyapunovNetwork(NeuralBarrierNetwork):
    """
    Abstract base class for neural Lyapunov networks.
    
    A neural Lyapunov network computes a Lyapunov function V(x), where:
    1. V(x) > 0 for all x != 0
    2. V(0) = 0 (at the equilibrium point)
    3. ∇V(x) · f(x, u) < 0 for all x != 0 (decreasing along trajectories)
    """
    
    @abstractmethod
    def get_origin(self) -> np.ndarray:
        """
        Get the origin (equilibrium point) of the Lyapunov function.
        
        Returns:
            Origin vector, shape (state_dim,)
        """
        pass
    
    @abstractmethod
    def set_origin(self, origin: np.ndarray):
        """
        Set the origin (equilibrium point) of the Lyapunov function.
        
        Args:
            origin: Origin vector, shape (state_dim,)
        """
        pass
    
    def is_stable(self, x: np.ndarray, dynamics_fn: Callable, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Determine if the system is stable at the given states.
        
        The system is stable if the Lyapunov function decreases along trajectories.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            dynamics_fn: Function mapping (x, u) to state derivatives
            u: Control inputs (optional), shape (batch_size, input_dim)
            
        Returns:
            Boolean array indicating whether the system is stable at each state
        """
        return self.verify_condition(x, dynamics_fn, u)
    
    def region_of_attraction(
        self,
        dynamics_fn: Callable,
        level_set_value: float,
        controller_fn: Optional[Callable] = None,
        num_samples: int = 1000,
        state_bounds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate the region of attraction (ROA) for the controlled system.
        
        The ROA is the set of states that converge to the equilibrium point.
        It is estimated as the largest level set of the Lyapunov function
        where the Lyapunov derivative is negative.
        
        Args:
            dynamics_fn: Function mapping (x, u) to state derivatives
            level_set_value: Value defining the level set V(x) <= level_set_value
            controller_fn: Function mapping state to control input (optional)
            num_samples: Number of samples for estimation
            state_bounds: Bounds of the state space, shape (state_dim, 2)
            
        Returns:
            Array of states in the estimated ROA, shape (n, state_dim)
        """
        # If state bounds not provided, use default large bounds
        if state_bounds is None:
            state_bounds = np.array([[-10.0, 10.0]] * x.shape[1])
        
        # Sample states uniformly from the state space
        states = np.random.uniform(
            state_bounds[:, 0],
            state_bounds[:, 1],
            size=(num_samples, state_bounds.shape[0])
        )
        
        # Compute Lyapunov function values for all states
        v_values = self(states)
        
        # Filter states within the level set
        level_set_mask = v_values <= level_set_value
        
        # If controller provided, compute control inputs
        if controller_fn is not None:
            control_inputs = np.array([controller_fn(state) for state in states])
        else:
            control_inputs = None
        
        # Check stability condition for all states in the level set
        stability_mask = self.is_stable(states[level_set_mask.flatten()], dynamics_fn, control_inputs)
        
        # Return states in the level set that satisfy the stability condition
        return states[level_set_mask.flatten()][stability_mask.flatten()]
