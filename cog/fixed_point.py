from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\fixed_point.py

"""
Fixed Point Computation
======================

Implements algorithms for finding fixed points of cognitive operators,
representing stable cognitive states.
"""

import numpy as np
from typing import Callable, Optional, Dict, List, Tuple
import warnings


def find_fixed_point(endofunctor,
                    initial_guess: np.ndarray,
                    tol: float = 1e-6,
                    max_iter: int = 1000,
                    method: str = 'iteration',
                    verbose: bool = False) -> np.ndarray:
    """
    Find fixed point of an endofunctor.
    
    Computes x* such that F(x*) = x*.
    
    Args:
        endofunctor: Object with apply(x) method
        initial_guess: Starting point
        tol: Convergence tolerance
        max_iter: Maximum iterations
        method: Algorithm ('iteration', 'newton', 'anderson')
        verbose: Print convergence info
        
    Returns:
        Fixed point x*
        
    Raises:
        RuntimeError: If convergence fails
    """
    if method == 'iteration':
        return _fixed_point_iteration(endofunctor, initial_guess, tol, max_iter, verbose)
    elif method == 'newton':
        return _newton_method(endofunctor, initial_guess, tol, max_iter, verbose)
    elif method == 'anderson':
        return _anderson_acceleration(endofunctor, initial_guess, tol, max_iter, verbose)
    else:
        raise ValueError(f"Unknown method: {method}")


def _fixed_point_iteration(endofunctor,
                          initial_guess: np.ndarray,
                          tol: float,
                          max_iter: int,
                          verbose: bool) -> np.ndarray:
    """
    Simple fixed point iteration: x_{n+1} = F(x_n).
    
    Args:
        endofunctor: Operator to iterate
        initial_guess: Starting point
        tol: Convergence tolerance
        max_iter: Maximum iterations
        verbose: Print progress
        
    Returns:
        Fixed point
    """
    x = initial_guess.copy()
    
    for iteration in range(max_iter):
        x_next = endofunctor.apply(x)
        error = np.linalg.norm(x_next - x)
        
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: error = {error:.6e}")
        
        if error < tol:
            if verbose:
                print(f"Converged in {iteration + 1} iterations")
            return x_next
        
        x = x_next
    
    raise RuntimeError(f"Fixed point not found within tolerance after {max_iter} iterations")


def _newton_method(endofunctor,
                  initial_guess: np.ndarray,
                  tol: float,
                  max_iter: int,
                  verbose: bool) -> np.ndarray:
    """
    Newton's method for fixed points: solve F(x) - x = 0.
    
    Args:
        endofunctor: Operator
        initial_guess: Starting point
        tol: Convergence tolerance
        max_iter: Maximum iterations
        verbose: Print progress
        
    Returns:
        Fixed point
    """
    x = initial_guess.copy()
    
    for iteration in range(max_iter):
        # Compute F(x) - x
        fx = endofunctor.apply(x)
        residual = fx - x
        error = np.linalg.norm(residual)
        
        if verbose and iteration % 50 == 0:
            print(f"Newton iteration {iteration}: error = {error:.6e}")
        
        if error < tol:
            if verbose:
                print(f"Newton converged in {iteration + 1} iterations")
            return x
        
        # Approximate Jacobian of F(x) - x
        n = len(x)
        J = np.zeros((n, n))
        eps = 1e-7
        
        for i in range(n):
            x_pert = x.copy()
            x_pert[i] += eps
            fx_pert = endofunctor.apply(x_pert)
            J[:, i] = (fx_pert - fx) / eps
        
        # Jacobian of F(x) - x
        J -= np.eye(n)
        
        # Newton update
        try:
            delta = np.linalg.solve(J, -residual)
        except np.linalg.LinAlgError:
            # Fall back to gradient descent if singular
            delta = -residual * 0.1
        
        # Line search for stability
        alpha = 1.0
        for _ in range(10):
            x_new = x + alpha * delta
            fx_new = endofunctor.apply(x_new)
            error_new = np.linalg.norm(fx_new - x_new)
            
            if error_new < error:
                break
            alpha *= 0.5
        
        x = x + alpha * delta
    
    raise RuntimeError(f"Newton method failed to converge after {max_iter} iterations")


def _anderson_acceleration(endofunctor,
                         initial_guess: np.ndarray,
                         tol: float,
                         max_iter: int,
                         verbose: bool,
                         m: int = 5) -> np.ndarray:
    """
    Anderson acceleration for fixed point problems.
    
    Uses a moving window of previous iterates to accelerate convergence.
    
    Args:
        endofunctor: Operator
        initial_guess: Starting point
        tol: Convergence tolerance
        max_iter: Maximum iterations
        verbose: Print progress
        m: Number of previous iterates to use
        
    Returns:
        Fixed point
    """
    x = initial_guess.copy()
    
    # History buffers
    X_history = []  # Previous iterates
    F_history = []  # F(x) values
    R_history = []  # Residuals F(x) - x
    
    for iteration in range(max_iter):
        # Compute F(x)
        fx = endofunctor.apply(x)
        residual = fx - x
        error = np.linalg.norm(residual)
        
        if verbose and iteration % 50 == 0:
            print(f"Anderson iteration {iteration}: error = {error:.6e}")
        
        if error < tol:
            if verbose:
                print(f"Anderson converged in {iteration + 1} iterations")
            return x
        
        # Update history
        X_history.append(x.copy())
        F_history.append(fx.copy())
        R_history.append(residual.copy())
        
        # Keep only last m iterates
        if len(X_history) > m:
            X_history.pop(0)
            F_history.pop(0)
            R_history.pop(0)
        
        # Anderson acceleration step
        if len(R_history) >= 2:
            # Build residual matrix
            R_matrix = np.column_stack(R_history)
            
            # Solve least squares for mixing coefficients
            try:
                # min ||R_matrix @ alpha||^2 s.t. sum(alpha) = 1
                n_hist = len(R_history)
                A = np.vstack([R_matrix, np.ones((1, n_hist))])
                b = np.zeros(len(residual) + 1)
                b[-1] = 1.0
                
                alpha = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Compute accelerated iterate
                x_accel = np.zeros_like(x)
                fx_accel = np.zeros_like(x)
                
                for i, a in enumerate(alpha):
                    x_accel += a * X_history[i]
                    fx_accel += a * F_history[i]
                
                # Mixing parameter for stability
                beta = 0.5
                x = beta * fx_accel + (1 - beta) * x_accel
                
            except np.linalg.LinAlgError:
                # Fall back to simple iteration
                x = fx
        else:
            # Not enough history yet
            x = fx
    
    raise RuntimeError(f"Anderson acceleration failed to converge after {max_iter} iterations")


def find_multiple_fixed_points(endofunctor,
                             n_attempts: int = 10,
                             bounds: Tuple[float, float] = (-1, 1),
                             tol: float = 1e-6,
                             max_iter: int = 1000,
                             method: str = 'iteration') -> List[np.ndarray]:
    """
    Find multiple fixed points using random initialization.
    
    Args:
        endofunctor: Operator
        n_attempts: Number of random starting points
        bounds: Range for random initialization
        tol: Convergence tolerance
        max_iter: Maximum iterations per attempt
        method: Fixed point algorithm
        
    Returns:
        List of unique fixed points found
    """
    fixed_points = []
    dim = None
    
    for attempt in range(n_attempts):
        # Random initialization
        if dim is None:
            # Try to infer dimension from a test call
            test_point = np.random.uniform(bounds[0], bounds[1], 10)
            try:
                result = endofunctor.apply(test_point)
                dim = len(result)
            except:
                raise ValueError("Cannot infer dimension from endofunctor")
        
        initial = np.random.uniform(bounds[0], bounds[1], dim)
        
        try:
            fp = find_fixed_point(endofunctor, initial, tol, max_iter, method, verbose=False)
            
            # Check if this fixed point is new
            is_new = True
            for existing_fp in fixed_points:
                if np.linalg.norm(fp - existing_fp) < tol * 10:
                    is_new = False
                    break
            
            if is_new:
                fixed_points.append(fp)
                
        except RuntimeError:
            # This attempt didn't converge
            continue
    
    return fixed_points


def analyze_fixed_point_stability(endofunctor,
                                fixed_point: np.ndarray,
                                epsilon: float = 1e-6) -> Dict[str, any]:
    """
    Analyze stability of a fixed point.
    
    Args:
        endofunctor: Operator
        fixed_point: Fixed point to analyze
        epsilon: Perturbation size for linearization
        
    Returns:
        Dictionary with stability analysis
    """
    n = len(fixed_point)
    
    # Compute Jacobian at fixed point
    J = np.zeros((n, n))
    fx = endofunctor.apply(fixed_point)
    
    for i in range(n):
        x_pert = fixed_point.copy()
        x_pert[i] += epsilon
        fx_pert = endofunctor.apply(x_pert)
        J[:, i] = (fx_pert - fx) / epsilon
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(J)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    
    # Determine stability
    is_stable = max_eigenvalue < 1.0
    is_attracting = max_eigenvalue < 0.99
    
    # Compute basin of attraction estimate
    basin_radius = 0.0
    if is_stable:
        # Estimate basin radius by testing perturbations
        test_radii = np.logspace(-3, 0, 10)
        for radius in test_radii:
            n_tests = 20
            all_converged = True
            
            for _ in range(n_tests):
                # Random perturbation
                direction = np.random.randn(n)
                direction /= np.linalg.norm(direction)
                perturbed = fixed_point + radius * direction
                
                # Test convergence
                try:
                    x = perturbed
                    for _ in range(100):
                        x = endofunctor.apply(x)
                        if np.linalg.norm(x - fixed_point) < 0.01:
                            break
                    else:
                        all_converged = False
                        break
                except:
                    all_converged = False
                    break
            
            if all_converged:
                basin_radius = radius
            else:
                break
    
    return {
        'eigenvalues': eigenvalues,
        'max_eigenvalue': max_eigenvalue,
        'is_stable': is_stable,
        'is_attracting': is_attracting,
        'spectral_radius': max_eigenvalue,
        'basin_radius_estimate': basin_radius,
        'fixed_point_error': np.linalg.norm(fx - fixed_point)
    }