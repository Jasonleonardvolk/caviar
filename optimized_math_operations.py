"""
Optimized Mathematical Operations for Soliton Architecture
=========================================================

This module provides optimized implementations of expensive mathematical operations
used in the Soliton architecture, including:
- Fisher Information Matrix (FIM) computation
- Numeric gradient calculations
- Vectorized operations
- Block-diagonal FIM approximations

These optimizations ensure the system can scale to high-dimensional data and
operate in real-time environments.
"""

import numpy as np
import numba
from numba import njit, prange
from typing import Callable, Optional, Union, Tuple, List
import warnings
from scipy.special import softmax

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ======================================================
# Numeric Gradient Implementations
# ======================================================

def numeric_gradient_naive(func: Callable[[np.ndarray], float], 
                         x: np.ndarray, 
                         eps: float = 1e-5) -> np.ndarray:
    """
    Original naive implementation of numeric gradient.
    This is the baseline for comparison.
    
    Args:
        func: Scalar function f: R^n -> R
        x: Point at which to compute gradient
        eps: Finite difference step size
        
    Returns:
        Gradient vector ∇f(x)
    """
    grad = np.zeros_like(x)
    
    # Forward differences (non-vectorized)
    fx = func(x)
    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (func(x_eps) - fx) / eps
    
    return grad


@njit
def numeric_gradient_jit(func: Callable[[np.ndarray], float], 
                        x: np.ndarray, 
                        eps: float = 1e-5) -> np.ndarray:
    """
    JIT-compiled version of numeric gradient using Numba.
    
    Args:
        func: Scalar function f: R^n -> R (must be JIT-compatible)
        x: Point at which to compute gradient
        eps: Finite difference step size
        
    Returns:
        Gradient vector ∇f(x)
    """
    grad = np.zeros_like(x)
    
    # Forward differences with JIT compilation
    fx = func(x)
    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (func(x_eps) - fx) / eps
    
    return grad


def numeric_gradient_vectorized(func: Callable[[np.ndarray], np.ndarray], 
                               x: np.ndarray, 
                               eps: float = 1e-5) -> np.ndarray:
    """
    Vectorized implementation of numeric gradient.
    Evaluates all perturbed states in a single function call.
    
    Args:
        func: Vectorized function that accepts batch of inputs
        x: Point at which to compute gradient
        eps: Finite difference step size
        
    Returns:
        Gradient vector ∇f(x)
    """
    dim = len(x)
    
    # Create batch of perturbed states
    perturbations = np.eye(dim) * eps
    X_batch = np.tile(x, (dim, 1)) + perturbations
    
    # Prepend original point for base value
    X_full = np.vstack([x.reshape(1, -1), X_batch])
    
    # Evaluate function on all points at once
    y_full = func(X_full)
    fx = y_full[0]
    fx_perturbed = y_full[1:]
    
    # Compute gradient
    grad = (fx_perturbed - fx) / eps
    
    return grad


def numeric_gradient_torch(func: Callable[[torch.Tensor], torch.Tensor],
                          x: np.ndarray) -> np.ndarray:
    """
    Compute gradient using PyTorch's autograd.
    
    Args:
        func: PyTorch function that accepts and returns tensors
        x: Point at which to compute gradient
        
    Returns:
        Gradient vector ∇f(x)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available")
    
    # Convert to PyTorch tensor and enable gradient tracking
    x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    
    # Compute function and its gradient
    y = func(x_torch)
    y.backward()
    
    # Return gradient as numpy array
    return x_torch.grad.detach().numpy()


def numeric_gradient_jax(func: Callable[[jnp.ndarray], float],
                        x: np.ndarray) -> np.ndarray:
    """
    Compute gradient using JAX's automatic differentiation.
    
    Args:
        func: JAX-compatible function
        x: Point at which to compute gradient
        
    Returns:
        Gradient vector ∇f(x)
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available")
    
    # Define gradient function
    grad_func = jax.grad(func)
    
    # JIT-compile for speed
    jit_grad_func = jax.jit(grad_func)
    
    # Compute gradient
    x_jax = jnp.array(x)
    gradient = jit_grad_func(x_jax)
    
    return np.array(gradient)


# ======================================================
# Fisher Information Matrix Implementations
# ======================================================

def fisher_information_matrix_naive(log_prob_model: Callable[[np.ndarray], float],
                                   s: np.ndarray,
                                   epsilon: float = 1e-4) -> np.ndarray:
    """
    Original naive implementation of Fisher Information Matrix.
    
    Args:
        log_prob_model: Log probability function
        s: State vector
        epsilon: Finite difference step size
        
    Returns:
        Fisher Information Matrix
    """
    dim = len(s)
    fim = np.zeros((dim, dim))
    
    # Double loop implementation (O(n²) complexity)
    for i in range(dim):
        for j in range(dim):
            s_ij = s.copy()
            s_ij[i] += epsilon
            s_ij[j] += epsilon
            
            s_i = s.copy()
            s_i[i] += epsilon
            
            s_j = s.copy()
            s_j[j] += epsilon
            
            # Second derivative approximation
            fim[i, j] = -(
                log_prob_model(s_ij)
                - log_prob_model(s_i)
                - log_prob_model(s_j)
                + log_prob_model(s)
            ) / epsilon**2
    
    # Ensure symmetry
    fim = 0.5 * (fim + fim.T)
    
    return fim


@njit(parallel=True)
def fisher_information_matrix_jit(log_prob_model,
                                 s: np.ndarray,
                                 epsilon: float = 1e-4) -> np.ndarray:
    """
    JIT-compiled implementation of Fisher Information Matrix
    with parallel execution.
    
    Args:
        log_prob_model: Log probability function (must be JIT-compatible)
        s: State vector
        epsilon: Finite difference step size
        
    Returns:
        Fisher Information Matrix
    """
    dim = len(s)
    fim = np.zeros((dim, dim))
    
    # Compute base value once
    log_prob_s = log_prob_model(s)
    
    # Pre-compute all single perturbations
    single_perturbations = np.zeros(dim)
    for i in range(dim):
        s_i = s.copy()
        s_i[i] += epsilon
        single_perturbations[i] = log_prob_model(s_i)
    
    # Parallel loop for upper triangular part
    for i in prange(dim):
        for j in range(i, dim):
            if i == j:
                # Diagonal element (reuse pre-computed value)
                fim[i, i] = -(
                    single_perturbations[i]
                    - 2 * log_prob_s
                    + log_prob_s
                ) / epsilon**2
            else:
                # Off-diagonal element
                s_ij = s.copy()
                s_ij[i] += epsilon
                s_ij[j] += epsilon
                
                fim[i, j] = -(
                    log_prob_model(s_ij)
                    - single_perturbations[i]
                    - single_perturbations[j]
                    + log_prob_s
                ) / epsilon**2
                
                # Symmetry
                fim[j, i] = fim[i, j]
    
    return fim


def fisher_information_matrix_vectorized(log_prob_model: Callable[[np.ndarray], np.ndarray],
                                        s: np.ndarray,
                                        epsilon: float = 1e-4) -> np.ndarray:
    """
    Vectorized implementation of Fisher Information Matrix.
    
    Args:
        log_prob_model: Vectorized log probability function
        s: State vector
        epsilon: Finite difference step size
        
    Returns:
        Fisher Information Matrix
    """
    dim = len(s)
    
    # Create all necessary perturbed states at once
    # 1. Original state
    # 2. Single perturbations (dim states)
    # 3. Double perturbations (dim*(dim+1)/2 states)
    
    # Original state
    states = [s.copy()]
    
    # Single perturbations
    single_indices = []
    for i in range(dim):
        s_i = s.copy()
        s_i[i] += epsilon
        states.append(s_i)
        single_indices.append(len(states) - 1)
    
    # Double perturbations
    double_indices = {}
    for i in range(dim):
        for j in range(i, dim):
            s_ij = s.copy()
            s_ij[i] += epsilon
            if i != j:
                s_ij[j] += epsilon
            states.append(s_ij)
            double_indices[(i, j)] = len(states) - 1
    
    # Stack all states and evaluate in one batch
    all_states = np.vstack(states)
    all_log_probs = log_prob_model(all_states)
    
    # Extract base value
    log_prob_s = all_log_probs[0]
    
    # Construct FIM
    fim = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            if i == j:
                # Diagonal element
                fim[i, i] = -(
                    all_log_probs[single_indices[i]]
                    - 2 * log_prob_s
                    + log_prob_s
                ) / epsilon**2
            else:
                # Off-diagonal element
                fim[i, j] = -(
                    all_log_probs[double_indices[(i, j)]]
                    - all_log_probs[single_indices[i]]
                    - all_log_probs[single_indices[j]]
                    + log_prob_s
                ) / epsilon**2
                fim[j, i] = fim[i, j]
    
    return fim


def fisher_information_matrix_torch(log_prob_model, 
                                   s: np.ndarray) -> np.ndarray:
    """
    Compute Fisher Information Matrix using PyTorch autograd.
    
    Args:
        log_prob_model: PyTorch log probability function
        s: State vector
        
    Returns:
        Fisher Information Matrix
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available")
    
    # Convert to PyTorch tensor
    s_torch = torch.tensor(s, dtype=torch.float32, requires_grad=True)
    
    # Compute log probability
    log_prob = log_prob_model(s_torch)
    
    # Compute gradient
    log_prob.backward(retain_graph=True)
    grad_s = s_torch.grad.clone()
    
    # Outer product of gradient is Fisher Information
    fim = torch.outer(grad_s, grad_s)
    
    return fim.detach().numpy()


def fisher_information_matrix_jax(log_prob_model, 
                                 s: np.ndarray) -> np.ndarray:
    """
    Compute Fisher Information Matrix using JAX.
    
    Args:
        log_prob_model: JAX-compatible log probability function
        s: State vector
        
    Returns:
        Fisher Information Matrix
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available")
    
    # Convert to JAX array
    s_jax = jnp.array(s)
    
    # Define gradient and Hessian functions
    grad_func = jax.grad(log_prob_model)
    hessian_func = jax.hessian(log_prob_model)
    
    # JIT-compile
    jit_hessian = jax.jit(hessian_func)
    
    # Compute Hessian (negative of Fisher Information)
    fim = -jit_hessian(s_jax)
    
    # Ensure symmetry and positive definiteness
    fim = 0.5 * (fim + fim.T)
    
    return np.array(fim)


# ======================================================
# Block-Diagonal Fisher Information Matrix
# ======================================================

def fisher_information_matrix_block_diagonal(log_prob_model: Callable[[np.ndarray], float],
                                           s: np.ndarray,
                                           block_sizes: List[int],
                                           epsilon: float = 1e-4) -> np.ndarray:
    """
    Compute Fisher Information Matrix using block-diagonal approximation.
    
    Args:
        log_prob_model: Log probability function
        s: State vector
        block_sizes: List of sizes for each block
        epsilon: Finite difference step size
        
    Returns:
        Block-diagonal Fisher Information Matrix
    """
    dim = len(s)
    fim = np.zeros((dim, dim))
    
    # Validate block sizes
    if sum(block_sizes) != dim:
        raise ValueError(f"Sum of block sizes {sum(block_sizes)} doesn't match dimension {dim}")
    
    # Compute each block separately
    start_idx = 0
    for block_size in block_sizes:
        end_idx = start_idx + block_size
        
        # Extract block indices and values
        block_indices = slice(start_idx, end_idx)
        s_block = s[block_indices]
        
        # Define block log probability function
        def block_log_prob(x_block):
            s_full = s.copy()
            s_full[block_indices] = x_block
            return log_prob_model(s_full)
        
        # Compute block FIM
        block_fim = np.zeros((block_size, block_size))
        
        # Base value
        log_prob_s = block_log_prob(s_block)
        
        # Compute block FIM
        for i in range(block_size):
            for j in range(i, block_size):
                s_i = s_block.copy()
                s_i[i] += epsilon
                
                s_j = s_block.copy()
                s_j[j] += epsilon
                
                s_ij = s_block.copy()
                s_ij[i] += epsilon
                s_ij[j] += epsilon
                
                block_fim[i, j] = -(
                    block_log_prob(s_ij)
                    - block_log_prob(s_i)
                    - block_log_prob(s_j)
                    + log_prob_s
                ) / epsilon**2
                
                block_fim[j, i] = block_fim[i, j]
        
        # Place block in the full matrix
        fim[start_idx:end_idx, start_idx:end_idx] = block_fim
        
        # Move to next block
        start_idx = end_idx
    
    return fim


def fisher_information_matrix_parallel_blocks(log_prob_model: Callable[[np.ndarray], float],
                                            s: np.ndarray,
                                            block_sizes: List[int],
                                            epsilon: float = 1e-4,
                                            n_jobs: int = -1) -> np.ndarray:
    """
    Compute Fisher Information Matrix with parallel block computation.
    
    Args:
        log_prob_model: Log probability function
        s: State vector
        block_sizes: List of sizes for each block
        epsilon: Finite difference step size
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        Block-diagonal Fisher Information Matrix
    """
    from joblib import Parallel, delayed
    
    dim = len(s)
    fim = np.zeros((dim, dim))
    
    # Validate block sizes
    if sum(block_sizes) != dim:
        raise ValueError(f"Sum of block sizes {sum(block_sizes)} doesn't match dimension {dim}")
    
    def compute_block(start_idx, block_size):
        end_idx = start_idx + block_size
        
        # Extract block indices and values
        block_indices = slice(start_idx, end_idx)
        s_block = s[block_indices]
        
        # Define block log probability function
        def block_log_prob(x_block):
            s_full = s.copy()
            s_full[block_indices] = x_block
            return log_prob_model(s_full)
        
        # Compute block FIM
        block_fim = np.zeros((block_size, block_size))
        
        # Base value
        log_prob_s = block_log_prob(s_block)
        
        # Compute block FIM
        for i in range(block_size):
            for j in range(i, block_size):
                s_i = s_block.copy()
                s_i[i] += epsilon
                
                s_j = s_block.copy()
                s_j[j] += epsilon
                
                s_ij = s_block.copy()
                s_ij[i] += epsilon
                s_ij[j] += epsilon
                
                block_fim[i, j] = -(
                    block_log_prob(s_ij)
                    - block_log_prob(s_i)
                    - block_log_prob(s_j)
                    + log_prob_s
                ) / epsilon**2
                
                block_fim[j, i] = block_fim[i, j]
        
        return (start_idx, end_idx, block_fim)
    
    # Prepare block computations
    block_starts = [sum(block_sizes[:i]) for i in range(len(block_sizes))]
    
    # Compute blocks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_block)(start, size) 
        for start, size in zip(block_starts, block_sizes)
    )
    
    # Assemble results
    for start, end, block_fim in results:
        fim[start:end, start:end] = block_fim
    
    return fim


# ======================================================
# Usage Examples
# ======================================================

def example_function(x):
    """Example function for testing gradients and FIM."""
    return -0.5 * np.sum(x**2)  # Negative quadratic (log of Gaussian)


def example_with_torch():
    """Example using PyTorch for automatic differentiation."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        return
    
    import torch
    
    def torch_func(x):
        return -0.5 * torch.sum(x**2)
    
    x = np.array([1.0, 2.0, 3.0])
    grad = numeric_gradient_torch(torch_func, x)
    print(f"PyTorch gradient: {grad}")


def example_with_jax():
    """Example using JAX for automatic differentiation."""
    if not JAX_AVAILABLE:
        print("JAX not available")
        return
    
    def jax_func(x):
        return -0.5 * jnp.sum(x**2)
    
    x = np.array([1.0, 2.0, 3.0])
    grad = numeric_gradient_jax(jax_func, x)
    print(f"JAX gradient: {grad}")


def performance_comparison(dim=100, n_trials=5):
    """Compare performance of different FIM implementations."""
    import time
    
    def test_func(x):
        return -0.5 * np.sum(x**2)
    
    x = np.random.randn(dim)
    
    implementations = [
        ("Naive", fisher_information_matrix_naive),
        ("JIT", fisher_information_matrix_jit),
        ("Vectorized", fisher_information_matrix_vectorized),
        ("Block-Diagonal", lambda f, x: fisher_information_matrix_block_diagonal(f, x, [dim//4, dim//4, dim//4, dim//4]))
    ]
    
    if TORCH_AVAILABLE:
        def torch_test_func(x):
            return -0.5 * torch.sum(x**2)
        implementations.append(("PyTorch", lambda f, x: fisher_information_matrix_torch(torch_test_func, x)))
    
    if JAX_AVAILABLE:
        def jax_test_func(x):
            return -0.5 * jnp.sum(x**2)
        implementations.append(("JAX", lambda f, x: fisher_information_matrix_jax(jax_test_func, x)))
    
    # Run performance tests
    results = {}
    for name, impl in implementations:
        times = []
        for _ in range(n_trials):
            start = time.time()
            try:
                if name in ["PyTorch", "JAX"]:
                    _ = impl(None, x)
                else:
                    _ = impl(test_func, x)
                times.append(time.time() - start)
            except Exception as e:
                print(f"Error with {name}: {e}")
                times.append(float('inf'))
        
        avg_time = sum(times) / len(times)
        results[name] = avg_time
    
    # Print results
    print(f"Performance comparison for dimension {dim}:")
    for name, avg_time in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name}: {avg_time:.6f} seconds")


if __name__ == "__main__":
    # Run performance comparison
    performance_comparison(dim=50, n_trials=3)
    
    # Examples
    example_with_torch()
    example_with_jax()
