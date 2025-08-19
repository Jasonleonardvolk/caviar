"""
Barrier Certificate learning module.

This module provides functionality for learning barrier certificates
from data. The barrier certificates ensure safety properties of dynamical
systems by providing a function B(x) that is positive in the unsafe region
and has negative derivative on the boundary of the safe set.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger("elfin.barrier.learner")

try:
    # Try to import optimization libraries
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("CVXPY not available, some optimization methods will be disabled")


class BarrierFunction:
    """
    Barrier certificate function B(x).
    
    A barrier certificate is a function B(x) that satisfies:
    1. B(x) > 0 for all x in the unsafe set
    2. B(x) ≤ 0 for all x in the safe set
    3. ∇B(x) · f(x) < 0 for all x on the boundary of the safe set
    
    Attributes:
        dictionary: Dictionary object that provides basis functions
        weights: Coefficient vector q for B(x) = q^T Φ(x)
        safe_region: Function that returns True if x is in the safe region
    """
    
    def __init__(
        self,
        dictionary: Any,
        weights: np.ndarray,
        safe_region: Callable[[np.ndarray], bool]
    ):
        """
        Initialize barrier function.
        
        Args:
            dictionary: Dictionary object that provides basis functions
            weights: Coefficient vector q for B(x) = q^T Φ(x)
            safe_region: Function that returns True if x is in the safe region
        """
        self.dictionary = dictionary
        self.weights = weights
        self.safe_region = safe_region
        
        # Cache for gradients
        self.gradient_cache = {}
    
    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate barrier function at state x.
        
        Args:
            x: State vector
            
        Returns:
            Barrier function value B(x)
        """
        # Apply dictionary to get Φ(x)
        if hasattr(self.dictionary, 'evaluate'):
            phi_x = self.dictionary.evaluate(x)
        else:
            # Assume dictionary is a callable
            phi_x = self.dictionary(x)
        
        # Compute B(x) = q^T Φ(x)
        return float(np.dot(self.weights, phi_x))
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of barrier function at state x.
        
        Args:
            x: State vector
            
        Returns:
            Gradient ∇B(x)
        """
        # Check cache for exact matches (for efficiency)
        x_tuple = tuple(x.flatten())
        if x_tuple in self.gradient_cache:
            return self.gradient_cache[x_tuple]
        
        # Use dictionary gradient if available
        if hasattr(self.dictionary, 'gradient'):
            # Get gradient of each basis function
            grad_phi = self.dictionary.gradient(x)
            # Compute ∇B(x) = q^T ∇Φ(x)
            grad_b = np.dot(self.weights, grad_phi)
        else:
            # Fallback to numerical differentiation
            h = 1e-6  # Small step size
            dim = len(x)
            grad_b = np.zeros(dim)
            
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                
                # Central difference
                grad_b[i] = (self(x_plus) - self(x_minus)) / (2 * h)
        
        # Cache the result
        self.gradient_cache[x_tuple] = grad_b
        return grad_b
    
    def decreasing_condition(self, x: np.ndarray, f_x: np.ndarray) -> float:
        """
        Compute the decreasing condition value at state x.
        
        For continuous-time systems, this is ∇B(x) · f(x) which should be < 0
        on the boundary of the safe set.
        
        Args:
            x: State vector
            f_x: Vector field value at x (dynamics)
            
        Returns:
            Decreasing condition value
        """
        grad_b = self.gradient(x)
        return float(np.dot(grad_b, f_x))
    
    def is_boundary_point(self, x: np.ndarray, epsilon: float = 1e-6) -> bool:
        """
        Check if a point is on the boundary of the safe set.
        
        Args:
            x: State vector
            epsilon: Tolerance for boundary check
            
        Returns:
            True if x is on the boundary of the safe set
        """
        return abs(self(x)) < epsilon
    
    def is_safe(self, x: np.ndarray) -> bool:
        """
        Check if a point is in the safe set according to the barrier function.
        
        Args:
            x: State vector
            
        Returns:
            True if B(x) ≤ 0 (indicating x is in the safe set)
        """
        return self(x) <= 0
    
    def distance_to_boundary(self, x: np.ndarray) -> float:
        """
        Compute approximate distance to the boundary of the safe set.
        
        Args:
            x: State vector
            
        Returns:
            Approximate distance to the boundary (B(x) = 0)
        """
        b_x = self(x)
        grad_b = self.gradient(x)
        grad_norm = np.linalg.norm(grad_b)
        
        if grad_norm < 1e-10:
            # Avoid division by zero
            return float('inf') if b_x <= 0 else float('inf')
        
        # Distance approximation based on first-order Taylor expansion
        return abs(b_x) / grad_norm
        
    def __str__(self):
        """String representation of barrier function."""
        dict_type = type(self.dictionary).__name__
        basis_count = len(self.weights) if hasattr(self.weights, '__len__') else 0
        return f"BarrierFunction(dict_type={dict_type}, basis={basis_count}, weights_shape={self.weights.shape})"
    
    def __repr__(self):
        """Representation of barrier function."""
        return self.__str__()


class BarrierLearner:
    """
    Learner for barrier certificates from data.
    
    This class implements methods for learning barrier certificates from data
    using convex optimization techniques. It fits a function B(x) = q^T Φ(x)
    such that B(x) > 0 for unsafe points and B(x) ≤ 0 for safe points.
    """
    
    def __init__(
        self,
        dictionary: Any,
        safe_region: Callable[[np.ndarray], bool],
        dynamics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize barrier learner.
        
        Args:
            dictionary: Dictionary object that provides basis functions
            safe_region: Function that returns True if x is in the safe region
            dynamics_fn: System dynamics function (optional, for decreasing condition)
            options: Additional options for learning
        """
        self.dictionary = dictionary
        self.safe_region = safe_region
        self.dynamics_fn = dynamics_fn
        self.options = options or {}
        
        # Check if CVXPY is available
        if not CVXPY_AVAILABLE:
            logger.warning("CVXPY not available, some learning methods will be disabled")
        
        # Initialize counters
        self.n_calls = 0
        self.refinements = 0
    
    def fit(
        self,
        safe_samples: np.ndarray,
        unsafe_samples: np.ndarray,
        boundary_samples: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        method: str = 'cvxopt'
    ) -> BarrierFunction:
        """
        Fit a barrier certificate to safe and unsafe samples.
        
        Args:
            safe_samples: Array of safe samples (n_safe x dim)
            unsafe_samples: Array of unsafe samples (n_unsafe x dim)
            boundary_samples: Array of boundary samples (n_boundary x dim)
            weights: Initial weights for barrier function (optional)
            method: Optimization method to use ('cvxopt', 'scs', etc.)
            
        Returns:
            Learned barrier function
        """
        self.n_calls += 1
        
        # Get dictionary dimension
        if hasattr(self.dictionary, 'evaluate'):
            # Get dimension from first sample
            if len(safe_samples) > 0:
                phi_dim = len(self.dictionary.evaluate(safe_samples[0]))
            elif len(unsafe_samples) > 0:
                phi_dim = len(self.dictionary.evaluate(unsafe_samples[0]))
            else:
                raise ValueError("No samples provided")
        elif hasattr(self.dictionary, 'dictionary_dimension'):
            phi_dim = self.dictionary.dictionary_dimension
        else:
            raise ValueError("Dictionary must have evaluate method or dictionary_dimension attribute")
        
        # Initialize weights if not provided
        if weights is None:
            weights = np.zeros(phi_dim)
        
        # Check if CVXPY is available
        if not CVXPY_AVAILABLE and method != 'direct':
            logger.warning("CVXPY not available, falling back to direct method")
            method = 'direct'
        
        # Choose optimization method
        if method == 'direct':
            weights = self._fit_direct(safe_samples, unsafe_samples, boundary_samples, weights)
        elif method == 'cvxopt' and CVXPY_AVAILABLE:
            weights = self._fit_cvxopt(safe_samples, unsafe_samples, boundary_samples, weights)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create and return barrier function
        return BarrierFunction(
            dictionary=self.dictionary,
            weights=weights,
            safe_region=self.safe_region
        )
    
    def _fit_direct(
        self,
        safe_samples: np.ndarray,
        unsafe_samples: np.ndarray,
        boundary_samples: Optional[np.ndarray],
        initial_weights: np.ndarray
    ) -> np.ndarray:
        """
        Fit a barrier certificate using direct optimization.
        
        This is a simple gradient-based optimization approach
        that should work without specialized optimization libraries.
        
        Args:
            safe_samples: Array of safe samples (n_safe x dim)
            unsafe_samples: Array of unsafe samples (n_unsafe x dim)
            boundary_samples: Array of boundary samples (n_boundary x dim)
            initial_weights: Initial weights for barrier function
            
        Returns:
            Optimized weights
        """
        # TODO: Implement direct optimization method
        # This would use gradient descent or another simple optimization method
        # to minimize the violations of the barrier constraints.
        
        # For now, just return the initial weights
        logger.warning("Direct optimization not implemented yet, returning initial weights")
        return initial_weights
    
    def _fit_cvxopt(
        self,
        safe_samples: np.ndarray,
        unsafe_samples: np.ndarray,
        boundary_samples: Optional[np.ndarray],
        initial_weights: np.ndarray
    ) -> np.ndarray:
        """
        Fit a barrier certificate using CVXPY with CVXOPT solver.
        
        This method uses convex optimization to find a barrier certificate
        that satisfies the safety constraints.
        
        Args:
            safe_samples: Array of safe samples (n_safe x dim)
            unsafe_samples: Array of unsafe samples (n_unsafe x dim)
            boundary_samples: Array of boundary samples (n_boundary x dim)
            initial_weights: Initial weights for barrier function
            
        Returns:
            Optimized weights
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is required for this method")
        
        # Get parameters
        n_safe = len(safe_samples)
        n_unsafe = len(unsafe_samples)
        n_boundary = 0 if boundary_samples is None else len(boundary_samples)
        phi_dim = len(initial_weights)
        
        # Margin parameters
        safe_margin = self.options.get('safe_margin', 0.1)
        unsafe_margin = self.options.get('unsafe_margin', 0.1)
        boundary_margin = self.options.get('boundary_margin', 0.1)
        
        # Regularization parameter
        reg_param = self.options.get('regularization', 1e-6)
        
        # Initialize variables
        q = cp.Variable(phi_dim)
        safe_slack = cp.Variable(n_safe)
        unsafe_slack = cp.Variable(n_unsafe)
        boundary_slack = cp.Variable(n_boundary)
        
        # Compute basis function values for all samples
        Phi_safe = np.zeros((n_safe, phi_dim))
        Phi_unsafe = np.zeros((n_unsafe, phi_dim))
        
        for i, sample in enumerate(safe_samples):
            Phi_safe[i] = self.dictionary.evaluate(sample)
        
        for i, sample in enumerate(unsafe_samples):
            Phi_unsafe[i] = self.dictionary.evaluate(sample)
        
        # Compute constraints for the boundary samples
        constraints = []
        objective_terms = []
        
        # Safe region constraints: B(x) ≤ -safe_margin + slack
        for i in range(n_safe):
            constraints.append(cp.matmul(Phi_safe[i], q) <= -safe_margin + safe_slack[i])
            constraints.append(safe_slack[i] >= 0)
            objective_terms.append(safe_slack[i])
        
        # Unsafe region constraints: B(x) ≥ unsafe_margin - slack
        for i in range(n_unsafe):
            constraints.append(cp.matmul(Phi_unsafe[i], q) >= unsafe_margin - unsafe_slack[i])
            constraints.append(unsafe_slack[i] >= 0)
            objective_terms.append(unsafe_slack[i])
        
        # Boundary decreasing constraints if dynamics function is provided
        if self.dynamics_fn is not None and boundary_samples is not None:
            # Compute basis function gradients and dynamics at boundary points
            for i, sample in enumerate(boundary_samples):
                # Get gradient of basis functions at this point
                if hasattr(self.dictionary, 'gradient'):
                    grad_phi = self.dictionary.gradient(sample)
                else:
                    # Fallback to numerical differentiation
                    h = 1e-6  # Small step size
                    dim = len(sample)
                    grad_phi = np.zeros((phi_dim, dim))
                    
                    for j in range(dim):
                        sample_plus = sample.copy()
                        sample_plus[j] += h
                        sample_minus = sample.copy()
                        sample_minus[j] -= h
                        
                        phi_plus = self.dictionary.evaluate(sample_plus)
                        phi_minus = self.dictionary.evaluate(sample_minus)
                        
                        # Central difference
                        grad_phi[:, j] = (phi_plus - phi_minus) / (2 * h)
                
                # Get dynamics at this point
                f_x = self.dynamics_fn(sample)
                
                # Compute inner product of gradient and dynamics
                dot_product = np.sum(grad_phi * f_x.reshape(1, -1), axis=1)
                
                # Add decreasing constraint: ∇B(x) · f(x) ≤ -boundary_margin + slack
                constraints.append(cp.matmul(dot_product, q) <= -boundary_margin + boundary_slack[i])
                constraints.append(boundary_slack[i] >= 0)
                objective_terms.append(boundary_slack[i])
        
        # Regularization term: small L2 norm of weights
        objective_terms.append(reg_param * cp.norm(q, 2))
        
        # Define objective: minimize sum of slacks and regularization
        objective = cp.Minimize(cp.sum(objective_terms))
        
        # Define and solve problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # Try with CVXOPT solver first
            prob.solve(solver=cp.CVXOPT)
        except Exception as e:
            logger.warning(f"CVXOPT solver failed: {e}, trying SCS")
            try:
                # Try with SCS solver as backup
                prob.solve(solver=cp.SCS)
            except Exception as e:
                logger.error(f"SCS solver failed: {e}")
                raise ValueError("Optimization failed") from e
        
        # Check if problem was successfully solved
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Problem status: {prob.status}, solution may be suboptimal")
        
        # Convert to numpy array
        weights = q.value
        
        # Ensure weights is a 1D array
        weights = np.array(weights).flatten()
        
        return weights
    
    def refine_with_counterexample(
        self,
        barrier_fn: BarrierFunction,
        counterexample: np.ndarray,
        is_unsafe: bool = True,
        is_boundary: bool = False
    ) -> BarrierFunction:
        """
        Refine barrier function with a counterexample.
        
        Args:
            barrier_fn: Current barrier function
            counterexample: Counterexample point where constraints are violated
            is_unsafe: Whether counterexample is from unsafe region
            is_boundary: Whether counterexample is a boundary point
            
        Returns:
            Refined barrier function
        """
        self.refinements += 1
        
        # Create new samples that include the counterexample
        if is_unsafe:
            unsafe_samples = np.array([counterexample])
            safe_samples = np.array([])
            boundary_samples = np.array([])
        elif is_boundary:
            unsafe_samples = np.array([])
            safe_samples = np.array([])
            boundary_samples = np.array([counterexample])
        else:
            unsafe_samples = np.array([])
            safe_samples = np.array([counterexample])
            boundary_samples = np.array([])
        
        # Get current weights
        current_weights = barrier_fn.weights
        
        # Fit with the counterexample
        method = 'cvxopt' if CVXPY_AVAILABLE else 'direct'
        
        # Use stronger margins for the counterexample
        options_copy = self.options.copy()
        options_copy['safe_margin'] = self.options.get('safe_margin', 0.1) * 2
        options_copy['unsafe_margin'] = self.options.get('unsafe_margin', 0.1) * 2
        options_copy['boundary_margin'] = self.options.get('boundary_margin', 0.1) * 2
        
        old_options = self.options
        self.options = options_copy
        
        # Fit new barrier function
        refined_fn = self.fit(
            safe_samples=safe_samples,
            unsafe_samples=unsafe_samples,
            boundary_samples=boundary_samples,
            weights=current_weights,
            method=method
        )
        
        # Restore original options
        self.options = old_options
        
        return refined_fn
