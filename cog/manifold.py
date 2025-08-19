from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\manifold.py

"""
MetaCognitive Manifold Implementation
====================================

Implements the cognitive state manifold M with various metric structures:
- Euclidean metric for simple distance calculations
- Fisher-Rao metric for information geometric distances
- Support for custom metrics via extension
"""

import numpy as np
from typing import Optional, Callable
import warnings


class MetaCognitiveManifold:
    """
    Represents the cognitive state manifold M with chosen metric.
    
    The manifold provides a geometric structure for cognitive states,
    enabling distance calculations and differential geometric operations.
    
    Attributes:
        dimension (int): Dimensionality of the manifold
        metric (str): Type of metric ('euclidean' or 'fisher_rao')
        log_prob_model (Optional[Callable]): Log probability model for Fisher-Rao metric
    """
    
    def __init__(self, dimension: int, metric: str = "euclidean", 
                 log_prob_model: Optional[Callable] = None):
        """
        Initialize the metacognitive manifold.
        
        Args:
            dimension: Dimension of the cognitive state space
            metric: Type of metric to use ('euclidean' or 'fisher_rao')
            log_prob_model: Optional log probability function for Fisher-Rao metric
        """
        self.dimension = dimension
        self.metric = metric
        self.log_prob_model = log_prob_model
        
        if metric == "fisher_rao" and log_prob_model is None:
            if not hasattr(self, "log_prob_model"):
                warnings.warn("Fisher-Rao metric requires log_prob_model. "
                             "Using default implementation.")

    def distance(self, s: np.ndarray, s_prime: np.ndarray) -> float:
        """
        Compute distance between two cognitive states.
        
        Args:
            s: First cognitive state
            s_prime: Second cognitive state
            
        Returns:
            Distance between states according to chosen metric
        """
        if self.metric == "euclidean":
            return np.linalg.norm(s - s_prime)
        elif self.metric == "fisher_rao":
            return self.fisher_rao_distance(s, s_prime)
        else:
            raise NotImplementedError(f"Metric '{self.metric}' not implemented.")

    def fisher_information_matrix(self, s: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        """
        Compute Fisher Information Matrix at a given state.
        
        The FIM provides the local metric tensor for the Fisher-Rao metric.
        
        Args:
            s: Cognitive state
            epsilon: Finite difference step size
            
        Returns:
            Fisher Information Matrix (symmetric positive definite)
        """
        dim = len(s)
        fim = np.zeros((dim, dim))
        
        if self.log_prob_model is not None:
            # Compute FIM using finite differences
            for i in range(dim):
                for j in range(dim):
                    s_ij = s.copy()
                    s_ij[i] += epsilon
                    s_ij[j] += epsilon
                    
                    s_i = s.copy()
                    s_i[i] += epsilon
                    
                    s_j = s.copy()
                    s_j[j] += epsilon
                    
                    fim[i, j] = -(
                        self.log_prob_model(s_ij)
                        - self.log_prob_model(s_i)
                        - self.log_prob_model(s_j)
                        + self.log_prob_model(s)
                    ) / epsilon**2
        else:
            # Default implementation: identity scaled by state norm
            fim = np.eye(dim) * (1.0 + np.linalg.norm(s))
        
        # Ensure symmetry
        fim = 0.5 * (fim + fim.T)
        
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(fim)
        if np.min(eigvals) < 1e-6:
            fim += np.eye(dim) * (1e-6 - np.min(eigvals))
        
        return fim

    def fisher_rao_distance(self, s: np.ndarray, s_prime: np.ndarray) -> float:
        """
        Compute Fisher-Rao distance between states.
        
        This is the geodesic distance on the statistical manifold.
        
        Args:
            s: First cognitive state
            s_prime: Second cognitive state
            
        Returns:
            Fisher-Rao distance
        """
        # Compute FIM at midpoint for symmetry
        midpoint = 0.5 * (s + s_prime)
        J = self.fisher_information_matrix(midpoint)
        
        # Compute distance
        diff = s - s_prime
        return np.sqrt(np.abs(diff.T @ J @ diff))
    
    def tangent_space_dim(self, s: np.ndarray) -> int:
        """
        Get dimension of tangent space at a point.
        
        Args:
            s: Point on manifold
            
        Returns:
            Dimension of tangent space (equals manifold dimension)
        """
        return self.dimension
    
    def project_to_manifold(self, s: np.ndarray) -> np.ndarray:
        """
        Project a point onto the manifold.
        
        Args:
            s: Point to project
            
        Returns:
            Projected point on manifold
        """
        # For now, just ensure correct dimension
        if len(s) != self.dimension:
            raise ValueError(f"State dimension {len(s)} doesn't match "
                           f"manifold dimension {self.dimension}")
        return s.copy()