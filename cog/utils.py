from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\utils.py

"""
Utility Functions
=================

Common utilities for the TORI Cognitive Framework including:
- Random seed management
- Numerical differentiation
- IIT computation
- Mathematical helpers
"""

import numpy as np
from typing import Callable, Optional, Union, Tuple, List
import warnings
from scipy.special import softmax
from scipy.stats import entropy


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    # Also set for any other random libraries if needed
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def numeric_gradient(func: Callable[[np.ndarray], float], 
                    x: np.ndarray, 
                    eps: float = 1e-5,
                    method: str = 'central') -> np.ndarray:
    """
    Compute numerical gradient of a scalar function.
    
    Args:
        func: Scalar function f: R^n -> R
        x: Point at which to compute gradient
        eps: Finite difference step size
        method: 'forward', 'backward', or 'central' differences
        
    Returns:
        Gradient vector ∇f(x)
    """
    grad = np.zeros_like(x)
    
    if method == 'central':
        # Central differences: (f(x+h) - f(x-h)) / 2h
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
    
    elif method == 'forward':
        # Forward differences: (f(x+h) - f(x)) / h
        fx = func(x)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            grad[i] = (func(x_eps) - fx) / eps
    
    elif method == 'backward':
        # Backward differences: (f(x) - f(x-h)) / h
        fx = func(x)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] -= eps
            grad[i] = (fx - func(x_eps)) / eps
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return grad


def numeric_hessian(func: Callable[[np.ndarray], float],
                   x: np.ndarray,
                   eps: float = 1e-5) -> np.ndarray:
    """
    Compute numerical Hessian matrix.
    
    Args:
        func: Scalar function f: R^n -> R
        x: Point at which to compute Hessian
        eps: Finite difference step size
        
    Returns:
        Hessian matrix H[i,j] = ∂²f/∂x_i∂x_j
    """
    n = len(x)
    H = np.zeros((n, n))
    
    # Use central differences for second derivatives
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal: (f(x+h) - 2f(x) + f(x-h)) / h²
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                H[i, i] = (func(x_plus) - 2*func(x) + func(x_minus)) / eps**2
            else:
                # Off-diagonal: (f(x+hi+hj) - f(x+hi) - f(x+hj) + f(x)) / (hi*hj)
                x_ij = x.copy()
                x_i = x.copy()
                x_j = x.copy()
                x_ij[i] += eps
                x_ij[j] += eps
                x_i[i] += eps
                x_j[j] += eps
                
                H[i, j] = (func(x_ij) - func(x_i) - func(x_j) + func(x)) / eps**2
                H[j, i] = H[i, j]  # Symmetry
    
    return H


def compute_iit_phi(s: np.ndarray, 
                   connectivity_matrix: Optional[np.ndarray] = None,
                   integration_method: str = 'entropy') -> float:
    """
    Compute Integrated Information Theory (IIT) measure Φ.
    
    This is a simplified version of IIT 3.0 focusing on
    the key idea of integration vs segregation.
    
    Args:
        s: System state vector
        connectivity_matrix: Optional connectivity structure
        integration_method: Method for computing integration
        
    Returns:
        Φ (phi) value measuring integrated information
    """
    dim = len(s)
    
    # Default to full connectivity if not provided
    if connectivity_matrix is None:
        connectivity_matrix = np.ones((dim, dim)) - np.eye(dim)
    
    # Normalize state to probability distribution
    p = softmax(np.abs(s))
    
    if integration_method == 'entropy':
        # Entropy-based Φ
        # Whole system entropy
        H_whole = entropy(p)
        
        # Find minimum information partition (MIP)
        min_partition_info = float('inf')
        
        # Try all bipartitions
        for partition_size in range(1, dim // 2 + 1):
            # Partition indices
            part1_indices = list(range(partition_size))
            part2_indices = list(range(partition_size, dim))
            
            # Partition probabilities
            p1 = p[part1_indices]
            p2 = p[part2_indices]
            
            # Normalize partitions
            p1 = p1 / (np.sum(p1) + 1e-10)
            p2 = p2 / (np.sum(p2) + 1e-10)
            
            # Partition entropies
            H1 = entropy(p1) if np.sum(p1) > 0 else 0
            H2 = entropy(p2) if np.sum(p2) > 0 else 0
            
            # Total partition information
            partition_info = H1 + H2
            
            # Update minimum
            if partition_info < min_partition_info:
                min_partition_info = partition_info
        
        # Φ is the difference between whole and minimum partition
        phi = max(0, H_whole - min_partition_info)
        
    elif integration_method == 'correlation':
        # Correlation-based Φ
        # Compute correlation matrix
        state_matrix = s.reshape(-1, 1)
        corr = np.abs(np.corrcoef(state_matrix.T, state_matrix.T)[0, 1])
        
        # Weight by connectivity
        weighted_corr = corr * np.mean(connectivity_matrix)
        phi = weighted_corr
        
    elif integration_method == 'kld':
        # KL divergence based Φ
        # Compare joint distribution to product of marginals
        # Simplified version using Gaussian assumption
        
        # Covariance of full system
        cov_full = np.outer(s, s)
        
        # Product of marginal covariances
        cov_marginal = np.diag(np.diag(cov_full))
        
        # KL divergence (simplified)
        try:
            phi = 0.5 * (np.trace(np.linalg.inv(cov_marginal) @ cov_full) - dim + 
                        np.log(np.linalg.det(cov_marginal) / (np.linalg.det(cov_full) + 1e-10)))
            phi = max(0, phi)
        except:
            phi = 0.0
    
    else:
        raise ValueError(f"Unknown integration method: {integration_method}")
    
    # Scale by connectivity
    mean_connectivity = np.mean(connectivity_matrix)
    phi *= mean_connectivity
    
    return phi


def compute_free_energy(s: np.ndarray,
                       generative_model: Optional[Callable] = None,
                       prior_mean: Optional[np.ndarray] = None,
                       prior_cov: Optional[np.ndarray] = None,
                       beta: float = 1.0) -> float:
    """
    Compute variational free energy.
    
    F = -log P(o|s) + KL[Q(s)||P(s)]
    
    Args:
        s: Cognitive state
        generative_model: Optional generative model P(o|s)
        prior_mean: Prior mean for P(s)
        prior_cov: Prior covariance for P(s)
        beta: Inverse temperature parameter
        
    Returns:
        Free energy value
    """
    # Default prior
    if prior_mean is None:
        prior_mean = np.zeros_like(s)
    if prior_cov is None:
        prior_cov = np.eye(len(s))
    
    # KL divergence term (assuming Gaussian)
    diff = s - prior_mean
    try:
        kl_term = 0.5 * (diff.T @ np.linalg.solve(prior_cov, diff) + 
                        np.log(np.linalg.det(prior_cov)) - len(s))
    except:
        # Fallback for singular covariance
        kl_term = 0.5 * np.sum(diff**2)
    
    # Likelihood term
    if generative_model is not None:
        log_likelihood = generative_model(s)
    else:
        # Default: quadratic cost
        log_likelihood = -0.5 * np.sum(s**2)
    
    # Free energy
    free_energy = -beta * log_likelihood + kl_term
    
    return free_energy


def create_connectivity_matrix(n_nodes: int,
                             connection_type: str = 'full',
                             sparsity: float = 0.5,
                             seed: Optional[int] = None) -> np.ndarray:
    """
    Create connectivity matrix for IIT calculations.
    
    Args:
        n_nodes: Number of nodes
        connection_type: 'full', 'sparse', 'small_world', 'modular'
        sparsity: Fraction of connections (for sparse types)
        seed: Random seed
        
    Returns:
        Connectivity matrix (symmetric, no self-connections)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if connection_type == 'full':
        # Fully connected
        C = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        
    elif connection_type == 'sparse':
        # Random sparse connections
        C = np.random.rand(n_nodes, n_nodes) < sparsity
        C = C.astype(float)
        # Make symmetric
        C = (C + C.T) / 2
        # Remove self-connections
        np.fill_diagonal(C, 0)
        
    elif connection_type == 'small_world':
        # Watts-Strogatz small world
        k = max(2, int(n_nodes * sparsity))  # Average degree
        p_rewire = 0.1
        
        # Start with ring lattice
        C = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(1, k//2 + 1):
                C[i, (i+j) % n_nodes] = 1
                C[i, (i-j) % n_nodes] = 1
        
        # Rewire edges
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if C[i, j] and np.random.rand() < p_rewire:
                    C[i, j] = 0
                    C[j, i] = 0
                    # Add new random edge
                    new_target = np.random.randint(n_nodes)
                    if new_target != i:
                        C[i, new_target] = 1
                        C[new_target, i] = 1
        
    elif connection_type == 'modular':
        # Modular structure
        n_modules = max(2, n_nodes // 5)
        module_size = n_nodes // n_modules
        
        C = np.zeros((n_nodes, n_nodes))
        
        # Within-module connections
        for m in range(n_modules):
            start = m * module_size
            end = min((m+1) * module_size, n_nodes)
            for i in range(start, end):
                for j in range(start, end):
                    if i != j and np.random.rand() < 0.8:
                        C[i, j] = 1
        
        # Between-module connections
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if C[i, j] == 0 and np.random.rand() < 0.1:
                    C[i, j] = 1
                    C[j, i] = 1
    
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")
    
    return C


def normalize_state(s: np.ndarray, 
                   method: str = 'l2',
                   epsilon: float = 1e-10) -> np.ndarray:
    """
    Normalize cognitive state vector.
    
    Args:
        s: State vector
        method: Normalization method ('l2', 'l1', 'max', 'softmax')
        epsilon: Small value to prevent division by zero
        
    Returns:
        Normalized state
    """
    if method == 'l2':
        norm = np.linalg.norm(s) + epsilon
        return s / norm
        
    elif method == 'l1':
        norm = np.sum(np.abs(s)) + epsilon
        return s / norm
        
    elif method == 'max':
        max_val = np.max(np.abs(s)) + epsilon
        return s / max_val
        
    elif method == 'softmax':
        return softmax(s)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def interpolate_states(s1: np.ndarray,
                      s2: np.ndarray,
                      alpha: float,
                      method: str = 'linear') -> np.ndarray:
    """
    Interpolate between two cognitive states.
    
    Args:
        s1: First state
        s2: Second state
        alpha: Interpolation parameter (0 = s1, 1 = s2)
        method: Interpolation method
        
    Returns:
        Interpolated state
    """
    if method == 'linear':
        return (1 - alpha) * s1 + alpha * s2
        
    elif method == 'slerp':
        # Spherical linear interpolation
        s1_norm = normalize_state(s1)
        s2_norm = normalize_state(s2)
        
        dot = np.clip(np.dot(s1_norm, s2_norm), -1, 1)
        theta = np.arccos(dot)
        
        if abs(theta) < 1e-6:
            # States are parallel
            return (1 - alpha) * s1 + alpha * s2
        
        sin_theta = np.sin(theta)
        w1 = np.sin((1 - alpha) * theta) / sin_theta
        w2 = np.sin(alpha * theta) / sin_theta
        
        interpolated = w1 * s1 + w2 * s2
        
        # Preserve magnitude
        target_norm = (1 - alpha) * np.linalg.norm(s1) + alpha * np.linalg.norm(s2)
        return interpolated * target_norm / (np.linalg.norm(interpolated) + 1e-10)
        
    else:
        raise ValueError(f"Unknown interpolation method: {method}")