"""
Neural Lyapunov Network Base Class

This module provides the base class for neural Lyapunov networks, which can be
implemented using different frameworks (PyTorch, JAX).
"""

from .neural_barrier import NeuralLyapunovNetwork
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import numpy as np


# Note: The abstract base class is already defined in neural_barrier.py
# This file provides concrete implementations and specialized methods


def quadratic_lyapunov(x: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Compute a quadratic Lyapunov function V(x) = x^T P x.
    
    Args:
        x: State array, shape (batch_size, state_dim)
        P: Positive definite matrix, shape (state_dim, state_dim)
        
    Returns:
        Lyapunov function values, shape (batch_size, 1)
    """
    batch_size = x.shape[0]
    v = np.zeros((batch_size, 1))
    
    for i in range(batch_size):
        v[i, 0] = x[i].dot(P).dot(x[i])
    
    return v


def quadratic_lyapunov_gradient(x: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of a quadratic Lyapunov function.
    
    The gradient of V(x) = x^T P x is grad V(x) = 2 P x.
    
    Args:
        x: State array, shape (batch_size, state_dim)
        P: Positive definite matrix, shape (state_dim, state_dim)
        
    Returns:
        Gradient array, shape (batch_size, state_dim)
    """
    batch_size = x.shape[0]
    state_dim = x.shape[1]
    grad = np.zeros((batch_size, state_dim))
    
    for i in range(batch_size):
        grad[i] = 2 * P.dot(x[i])
    
    return grad


class QuadraticLyapunovNetwork(NeuralLyapunovNetwork):
    """
    Quadratic Lyapunov function V(x) = (x - x_0)^T P (x - x_0).
    
    This class implements a simple quadratic Lyapunov function, which is 
    suitable for linear systems or systems that can be approximated as 
    linear near their equilibrium points.
    """
    
    def __init__(self, state_dim: int, P: Optional[np.ndarray] = None, origin: Optional[np.ndarray] = None):
        """
        Initialize a quadratic Lyapunov function.
        
        Args:
            state_dim: Dimension of the state space
            P: Positive definite matrix, shape (state_dim, state_dim)
            origin: Equilibrium point, shape (state_dim,)
        """
        self.state_dim = state_dim
        
        # Initialize P as identity matrix if not provided
        if P is None:
            self.P = np.eye(state_dim)
        else:
            # Check if P is positive definite
            eigvals = np.linalg.eigvals(P)
            if np.any(eigvals <= 0):
                raise ValueError("P must be positive definite")
            self.P = P
        
        # Initialize origin as zero vector if not provided
        if origin is None:
            self._origin = np.zeros(state_dim)
        else:
            self._origin = origin
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Lyapunov function value for given inputs.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Lyapunov function values, shape (batch_size, 1)
        """
        # Shift by origin
        x_shifted = x - self._origin
        
        # Compute quadratic form
        return quadratic_lyapunov(x_shifted, self.P)
    
    def get_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Lyapunov function.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Gradient array, shape (batch_size, state_dim)
        """
        # Shift by origin
        x_shifted = x - self._origin
        
        # Compute gradient
        return quadratic_lyapunov_gradient(x_shifted, self.P)
    
    def verify_condition(
        self,
        x: np.ndarray,
        dynamics_fn: Callable,
        u: Optional[np.ndarray] = None,
        alpha_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Verify the Lyapunov condition: ∇V(x) · f(x, u) < 0 for all x != 0
        
        Args:
            x: State array, shape (batch_size, state_dim)
            dynamics_fn: Function mapping (x, u) to state derivatives
            u: Control inputs (optional), shape (batch_size, input_dim)
            alpha_fn: Class-K function (optional), defaults to small fraction of norm
            
        Returns:
            Boolean array indicating whether the condition is satisfied for each state
        """
        # Default alpha function (small fraction of norm)
        if alpha_fn is None:
            alpha_fn = lambda x_norm: 0.1 * x_norm
        
        # Compute Lyapunov value
        v = self(x)
        
        # Compute gradient
        grad_v = self.get_gradient(x)
        
        # Compute dynamics
        if u is None:
            # If no control input is provided, assume autonomous system
            f = dynamics_fn(x, None)
        else:
            f = dynamics_fn(x, u)
        
        # Compute Lie derivative: ∇V(x) · f(x, u)
        batch_size = x.shape[0]
        lie_derivatives = np.zeros((batch_size, 1))
        for i in range(batch_size):
            lie_derivatives[i, 0] = np.dot(grad_v[i], f[i])
        
        # Compute norm of x for the alpha function
        x_shifted = x - self._origin
        x_norm = np.linalg.norm(x_shifted, axis=1, keepdims=True)
        
        # Compute α(||x||)
        alpha_x = alpha_fn(x_norm)
        
        # Check condition: ∇V(x) · f(x, u) <= -α(||x||)
        # For x very close to origin, we relax the condition to avoid numerical issues
        near_origin = x_norm < 1e-6
        condition = (near_origin) | (lie_derivatives <= -alpha_x)
        
        return condition
    
    def get_origin(self) -> np.ndarray:
        """
        Get the origin (equilibrium point) of the Lyapunov function.
        
        Returns:
            Origin vector, shape (state_dim,)
        """
        return self._origin
    
    def set_origin(self, origin: np.ndarray):
        """
        Set the origin (equilibrium point) of the Lyapunov function.
        
        Args:
            origin: Origin vector, shape (state_dim,)
        """
        if origin.shape != (self.state_dim,):
            raise ValueError(f"Origin shape must be ({self.state_dim},)")
        self._origin = origin
    
    def save(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        np.savez(
            filepath,
            P=self.P,
            origin=self._origin,
            state_dim=self.state_dim
        )
    
    @classmethod
    def load(cls, filepath: str, **kwargs):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model
        """
        data = np.load(filepath)
        return cls(
            state_dim=int(data['state_dim']),
            P=data['P'],
            origin=data['origin']
        )


class SumOfSquaresLyapunovNetwork(NeuralLyapunovNetwork):
    """
    Sum of squares (SOS) Lyapunov function.
    
    This class implements a Lyapunov function that is a sum of squared terms,
    which is guaranteed to be positive definite.
    
    V(x) = Σ_i (ϕ_i(x))²
    
    where ϕ_i are basis functions (e.g., monomials).
    """
    
    def __init__(
        self,
        state_dim: int,
        basis_functions: List[Callable],
        weights: Optional[np.ndarray] = None,
        origin: Optional[np.ndarray] = None
    ):
        """
        Initialize a sum of squares Lyapunov function.
        
        Args:
            state_dim: Dimension of the state space
            basis_functions: List of basis functions ϕ_i(x)
            weights: Weights for the basis functions, shape (num_basis,)
            origin: Equilibrium point, shape (state_dim,)
        """
        self.state_dim = state_dim
        self.basis_functions = basis_functions
        self.num_basis = len(basis_functions)
        
        # Initialize weights if not provided
        if weights is None:
            self.weights = np.ones(self.num_basis)
        else:
            if weights.shape != (self.num_basis,):
                raise ValueError(f"Weights shape must be ({self.num_basis},)")
            self.weights = weights
        
        # Initialize origin as zero vector if not provided
        if origin is None:
            self._origin = np.zeros(state_dim)
        else:
            self._origin = origin
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Lyapunov function value for given inputs.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Lyapunov function values, shape (batch_size, 1)
        """
        # Shift by origin
        x_shifted = x - self._origin
        
        # Compute basis function values
        batch_size = x.shape[0]
        basis_values = np.zeros((batch_size, self.num_basis))
        
        for i, basis_fn in enumerate(self.basis_functions):
            basis_values[:, i] = basis_fn(x_shifted)
        
        # Apply weights
        weighted_basis = basis_values * self.weights
        
        # Compute sum of squares
        v = np.sum(weighted_basis ** 2, axis=1, keepdims=True)
        
        return v
    
    def get_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Lyapunov function.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Gradient array, shape (batch_size, state_dim)
        """
        # This implementation uses finite differences
        # A more efficient implementation would compute analytical gradients
        eps = 1e-6
        batch_size, state_dim = x.shape
        gradient = np.zeros((batch_size, state_dim))
        
        for i in range(batch_size):
            for j in range(state_dim):
                # Create perturbation vectors
                x_plus = x[i].copy()
                x_plus[j] += eps
                
                x_minus = x[i].copy()
                x_minus[j] -= eps
                
                # Compute central difference
                v_plus = self(x_plus.reshape(1, -1))[0, 0]
                v_minus = self(x_minus.reshape(1, -1))[0, 0]
                
                gradient[i, j] = (v_plus - v_minus) / (2 * eps)
        
        return gradient
    
    def verify_condition(
        self,
        x: np.ndarray,
        dynamics_fn: Callable,
        u: Optional[np.ndarray] = None,
        alpha_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Verify the Lyapunov condition: ∇V(x) · f(x, u) < 0 for all x != 0
        
        Args:
            x: State array, shape (batch_size, state_dim)
            dynamics_fn: Function mapping (x, u) to state derivatives
            u: Control inputs (optional), shape (batch_size, input_dim)
            alpha_fn: Class-K function (optional), defaults to small fraction of norm
            
        Returns:
            Boolean array indicating whether the condition is satisfied for each state
        """
        # Default alpha function (small fraction of norm)
        if alpha_fn is None:
            alpha_fn = lambda x_norm: 0.1 * x_norm
        
        # Compute Lyapunov value
        v = self(x)
        
        # Compute gradient
        grad_v = self.get_gradient(x)
        
        # Compute dynamics
        if u is None:
            # If no control input is provided, assume autonomous system
            f = dynamics_fn(x, None)
        else:
            f = dynamics_fn(x, u)
        
        # Compute Lie derivative: ∇V(x) · f(x, u)
        batch_size = x.shape[0]
        lie_derivatives = np.zeros((batch_size, 1))
        for i in range(batch_size):
            lie_derivatives[i, 0] = np.dot(grad_v[i], f[i])
        
        # Compute norm of x for the alpha function
        x_shifted = x - self._origin
        x_norm = np.linalg.norm(x_shifted, axis=1, keepdims=True)
        
        # Compute α(||x||)
        alpha_x = alpha_fn(x_norm)
        
        # Check condition: ∇V(x) · f(x, u) <= -α(||x||)
        # For x very close to origin, we relax the condition to avoid numerical issues
        near_origin = x_norm < 1e-6
        condition = (near_origin) | (lie_derivatives <= -alpha_x)
        
        return condition
    
    def get_origin(self) -> np.ndarray:
        """
        Get the origin (equilibrium point) of the Lyapunov function.
        
        Returns:
            Origin vector, shape (state_dim,)
        """
        return self._origin
    
    def set_origin(self, origin: np.ndarray):
        """
        Set the origin (equilibrium point) of the Lyapunov function.
        
        Args:
            origin: Origin vector, shape (state_dim,)
        """
        if origin.shape != (self.state_dim,):
            raise ValueError(f"Origin shape must be ({self.state_dim},)")
        self._origin = origin
    
    def save(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        # Note: Saving basis functions is challenging
        # This implementation only saves the weights and origin
        np.savez(
            filepath,
            weights=self.weights,
            origin=self._origin,
            state_dim=self.state_dim,
            num_basis=self.num_basis
        )
    
    @classmethod
    def load(cls, filepath: str, basis_functions: List[Callable], **kwargs):
        """
        Load the model from a file.
        
        Args:
            filepath: Path to the saved model
            basis_functions: List of basis functions (must be provided)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model
        """
        data = np.load(filepath)
        return cls(
            state_dim=int(data['state_dim']),
            basis_functions=basis_functions,
            weights=data['weights'],
            origin=data['origin']
        )
