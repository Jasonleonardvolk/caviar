from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\self_modification.py

"""
Self-Modification Operator Implementation
========================================

Implements the self-modification endofunctor SM that optimizes cognitive
states by minimizing free energy while preserving consciousness (via IIT).
"""

import numpy as np
from typing import Callable, Optional, Dict, Any
from .utils import numeric_gradient
from .manifold import MetaCognitiveManifold


class SelfModificationOperator:
    """
    Endofunctor SM minimizing free energy + IIT term.
    
    This operator enables the cognitive system to modify its own state
    to reduce surprise (free energy) while maintaining integrated
    information (consciousness).
    
    Attributes:
        manifold: The cognitive manifold
        free_energy: Function computing free energy F(s)
        iit: Function computing integrated information Φ(s)
        iit_weight: Weight for consciousness preservation term
        step_size: Optimization step size
        max_iter: Maximum optimization iterations
    """
    
    def __init__(self, 
                 manifold: MetaCognitiveManifold,
                 free_energy_func: Callable[[np.ndarray], float],
                 iit_func: Callable[[np.ndarray], float],
                 iit_weight: float = 1.0,
                 step_size: float = 0.01,
                 max_iter: int = 100,
                 convergence_tol: float = 1e-6):
        """
        Initialize self-modification operator.
        
        Args:
            manifold: Cognitive manifold
            free_energy_func: Function computing free energy
            iit_func: Function computing integrated information
            iit_weight: Weight for IIT preservation (higher = more consciousness)
            step_size: Gradient descent step size
            max_iter: Maximum iterations
            convergence_tol: Convergence tolerance
        """
        self.manifold = manifold
        self.free_energy = free_energy_func
        self.iit = iit_func
        self.iit_weight = iit_weight
        self.step_size = step_size
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.optimization_history = []

    def optimize(self, s: np.ndarray, lambda_d: float = 1.0) -> np.ndarray:
        """
        Optimize cognitive state via gradient descent.
        
        Minimizes: F(s') + λ_d * d(s, s') + α * Φ(s')
        
        Args:
            s: Initial cognitive state
            lambda_d: Regularization weight for distance from initial state
            
        Returns:
            Optimized cognitive state
        """
        x = s.copy()
        self.optimization_history = []
        
        for iteration in range(self.max_iter):
            # Define objective function
            def objective(z):
                return (
                    self.free_energy(z)
                    + lambda_d * self.manifold.distance(s, z)
                    - self.iit_weight * self.iit(z)  # Negative because we want to preserve IIT
                )
            
            # Compute gradient
            grad = numeric_gradient(objective, x)
            
            # Natural gradient if using Fisher-Rao metric
            if self.manifold.metric == "fisher_rao":
                J = self.manifold.fisher_information_matrix(x)
                try:
                    grad = np.linalg.solve(J, grad)
                except np.linalg.LinAlgError:
                    pass  # Use standard gradient if FIM is singular
            
            # Update with gradient descent
            x_next = x - self.step_size * grad
            
            # Record history
            self.optimization_history.append({
                'iteration': iteration,
                'state': x.copy(),
                'objective': objective(x),
                'free_energy': self.free_energy(x),
                'iit': self.iit(x),
                'gradient_norm': np.linalg.norm(grad)
            })
            
            # Check convergence
            if np.linalg.norm(x_next - x) < self.convergence_tol:
                break
            
            x = x_next
        
        return x
    
    def apply(self, s: np.ndarray) -> np.ndarray:
        """
        Apply self-modification operator (alias for optimize).
        
        Args:
            s: Current cognitive state
            
        Returns:
            Modified cognitive state
        """
        return self.optimize(s)
    
    def adaptive_optimize(self, s: np.ndarray, 
                         lambda_d_init: float = 1.0,
                         lambda_d_decay: float = 0.9) -> np.ndarray:
        """
        Optimize with adaptive regularization.
        
        Gradually reduces the constraint to stay near initial state,
        allowing more exploration over time.
        
        Args:
            s: Initial cognitive state
            lambda_d_init: Initial regularization weight
            lambda_d_decay: Decay factor per iteration
            
        Returns:
            Optimized cognitive state
        """
        x = s.copy()
        lambda_d = lambda_d_init
        
        for outer_iter in range(5):  # Multiple rounds with decreasing lambda
            x = self.optimize(x, lambda_d)
            lambda_d *= lambda_d_decay
            
            # Early stopping if converged
            if len(self.optimization_history) > 1:
                if self.optimization_history[-1]['gradient_norm'] < self.convergence_tol:
                    break
        
        return x
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from last optimization.
        
        Returns:
            Dictionary with optimization metrics
        """
        if not self.optimization_history:
            return {}
        
        history = self.optimization_history
        return {
            'iterations': len(history),
            'initial_objective': history[0]['objective'],
            'final_objective': history[-1]['objective'],
            'initial_free_energy': history[0]['free_energy'],
            'final_free_energy': history[-1]['free_energy'],
            'initial_iit': history[0]['iit'],
            'final_iit': history[-1]['iit'],
            'free_energy_reduction': history[0]['free_energy'] - history[-1]['free_energy'],
            'iit_change': history[-1]['iit'] - history[0]['iit'],
            'converged': history[-1]['gradient_norm'] < self.convergence_tol
        }


# Placeholder for compute_free_energy function
def compute_free_energy(*args, **kwargs):
    """
    Compute free energy for the system.
    Placeholder implementation - returns 0.0.
    """
    return 0.0