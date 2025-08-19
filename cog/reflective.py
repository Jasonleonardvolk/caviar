from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\reflective.py

"""
Reflective Operator Implementation
=================================

Implements the reflective operator R: M → M using natural gradient ascent
on a log posterior function. This enables metacognitive reflection and
belief updating on the cognitive manifold.
"""

import numpy as np
from typing import Callable, Optional
from .utils import numeric_gradient
from .manifold import MetaCognitiveManifold


class ReflectiveOperator:
    """
    Implements R: M → M with natural gradient.
    
    The reflective operator updates cognitive states by following the
    natural gradient of a log posterior function, respecting the
    Riemannian geometry of the cognitive manifold.
    
    Attributes:
        manifold: The cognitive manifold
        log_posterior: Function computing log posterior probability
        step_size: Learning rate for gradient ascent
        momentum: Momentum coefficient for accelerated convergence
    """
    
    def __init__(self, 
                 manifold: MetaCognitiveManifold, 
                 log_posterior_func: Callable[[np.ndarray], float],
                 step_size: float = 0.01,
                 momentum: float = 0.0):
        """
        Initialize the reflective operator.
        
        Args:
            manifold: MetaCognitive manifold for geometric operations
            log_posterior_func: Function computing log P(s|evidence)
            step_size: Step size for gradient ascent
            momentum: Momentum coefficient (0 = no momentum)
        """
        self.manifold = manifold
        self.log_posterior = log_posterior_func
        self.step_size = step_size
        self.momentum = momentum
        self.velocity = None

    def natural_gradient(self, s: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Convert Euclidean gradient to natural gradient.
        
        The natural gradient follows the steepest ascent direction
        with respect to the manifold's Riemannian metric.
        
        Args:
            s: Current state
            grad: Euclidean gradient
            
        Returns:
            Natural gradient vector
        """
        if self.manifold.metric == "fisher_rao":
            J = self.manifold.fisher_information_matrix(s)
            try:
                # Use pseudo-inverse for numerical stability
                J_inv = np.linalg.pinv(J, rcond=1e-10)
                return J_inv @ grad
            except np.linalg.LinAlgError:
                # Fall back to standard gradient if FIM is singular
                return grad
        return grad

    def apply(self, s: np.ndarray) -> np.ndarray:
        """
        Apply reflective operator to update cognitive state.
        
        Args:
            s: Current cognitive state
            
        Returns:
            Updated cognitive state
        """
        # Compute gradient
        grad = numeric_gradient(self.log_posterior, s)
        
        # Convert to natural gradient
        nat_grad = self.natural_gradient(s, grad)
        
        # Apply momentum if enabled
        if self.momentum > 0:
            if self.velocity is None:
                self.velocity = np.zeros_like(s)
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * nat_grad
            update = self.step_size * self.velocity
        else:
            update = self.step_size * nat_grad
        
        # Update state
        s_new = s + update
        
        # Project back to manifold if needed
        s_new = self.manifold.project_to_manifold(s_new)
        
        return s_new
    
    def apply_with_line_search(self, s: np.ndarray, 
                              max_iter: int = 10,
                              c1: float = 1e-4) -> np.ndarray:
        """
        Apply reflective operator with backtracking line search.
        
        This ensures sufficient increase in log posterior.
        
        Args:
            s: Current cognitive state
            max_iter: Maximum line search iterations
            c1: Armijo condition parameter
            
        Returns:
            Updated cognitive state
        """
        # Compute gradient and natural gradient
        grad = numeric_gradient(self.log_posterior, s)
        nat_grad = self.natural_gradient(s, grad)
        
        # Initial function value
        f0 = self.log_posterior(s)
        
        # Backtracking line search
        alpha = self.step_size
        for _ in range(max_iter):
            s_new = s + alpha * nat_grad
            f_new = self.log_posterior(s_new)
            
            # Check Armijo condition
            if f_new >= f0 + c1 * alpha * np.dot(grad, nat_grad):
                break
            
            alpha *= 0.5
        
        return s + alpha * nat_grad
    
    def reset_momentum(self):
        """Reset momentum velocity to zero."""
        self.velocity = None