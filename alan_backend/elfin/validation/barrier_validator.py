"""
Barrier Function Validator

This module provides tools for validating barrier functions against system dynamics.
"""

import numpy as np
import sympy as sp
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import logging
from scipy.optimize import minimize, LinearConstraint

from .validation_result import ValidationResult, ValidationStatus, passed, failed, warning

logger = logging.getLogger(__name__)


class BarrierValidator:
    """
    Class for validating barrier functions against system dynamics.
    """
    
    def __init__(
        self,
        system_dynamics: Callable,
        barrier_function: Callable,
        state_dim: int,
        input_dim: int,
        barrier_derivative: Optional[Callable] = None,
        state_constraints: Optional[List[Callable]] = None,
        input_constraints: Optional[List[Callable]] = None,
        params: Optional[Dict[str, Any]] = None,
        symbolic: bool = False,
        sampling_method: str = "grid"
    ):
        """
        Initialize a barrier validator.
        
        Args:
            system_dynamics: Function representing the system dynamics dx/dt = f(x, u)
            barrier_function: Function representing the barrier function B(x)
            state_dim: Dimension of the state space
            input_dim: Dimension of the input space
            barrier_derivative: Optional function representing the barrier function derivative
            state_constraints: List of functions representing state constraints
            input_constraints: List of functions representing input constraints
            params: Dictionary of system and barrier function parameters
            symbolic: Whether to use symbolic computation (sympy)
            sampling_method: Method for sampling states ("grid", "random", "uniform", "sobol")
        """
        self.system_dynamics = system_dynamics
        self.barrier_function = barrier_function
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.barrier_derivative = barrier_derivative
        self.state_constraints = state_constraints or []
        self.input_constraints = input_constraints or []
        self.params = params or {}
        self.symbolic = symbolic
        self.sampling_method = sampling_method
        
        # Initialize symbolic variables if using symbolic computation
        if self.symbolic:
            self.init_symbolic_variables()
    
    def init_symbolic_variables(self):
        """Initialize symbolic variables for sympy computation."""
        # State variables
        self.state_vars = sp.symbols(f'x:{self.state_dim}')
        
        # Input variables
        if self.input_dim > 0:
            self.input_vars = sp.symbols(f'u:{self.input_dim}')
        else:
            self.input_vars = []
        
        # Parameter variables
        self.param_vars = {}
        for param_name, param_value in self.params.items():
            self.param_vars[param_name] = sp.Symbol(param_name)
    
    def validate_barrier_function(
        self,
        state_space_bounds: Optional[np.ndarray] = None,
        input_space_bounds: Optional[np.ndarray] = None,
        samples: int = 1000,
        tolerance: float = 1e-6
    ) -> ValidationResult:
        """
        Validate that the barrier function is positive in the safe set.
        
        Args:
            state_space_bounds: Bounds of the state space, shape (state_dim, 2)
            input_space_bounds: Bounds of the input space, shape (input_dim, 2)
            samples: Number of samples to check
            tolerance: Tolerance for numeric comparisons
            
        Returns:
            ValidationResult: Result of the validation
        """
        # Set default bounds if not provided
        if state_space_bounds is None:
            state_space_bounds = np.array([[-10.0, 10.0]] * self.state_dim)
        
        if input_space_bounds is None and self.input_dim > 0:
            input_space_bounds = np.array([[-10.0, 10.0]] * self.input_dim)
        
        if self.symbolic:
            return self._validate_barrier_function_symbolic(
                state_space_bounds, tolerance
            )
        else:
            return self._validate_barrier_function_numeric(
                state_space_bounds, samples, tolerance
            )
    
    def _validate_barrier_function_symbolic(
        self,
        state_space_bounds: np.ndarray,
        tolerance: float
    ) -> ValidationResult:
        """
        Validate the barrier function symbolically.
        
        Args:
            state_space_bounds: Bounds of the state space
            tolerance: Tolerance for numeric comparisons
            
        Returns:
            ValidationResult: Result of the validation
        """
        try:
            # Build symbolic expression for the barrier function
            barrier_expr = self.barrier_function(*self.state_vars, **self.param_vars)
            
            # Check if the barrier function is positive in the safe set
            # This is a simplified check - in practice, one would use formal verification
            # tools to prove positivity in the safe set
            
            # For now, we'll just return a warning
            return warning(
                "Symbolic validation is not fully implemented. "
                "Consider using numerical validation instead.",
                {"barrier_expr": str(barrier_expr)}
            )
        except Exception as e:
            logger.exception("Error during symbolic barrier function validation")
            return failed(
                f"Error during symbolic barrier function validation: {str(e)}",
                {"error": str(e)}
            )
    
    def _validate_barrier_function_numeric(
        self,
        state_space_bounds: np.ndarray,
        samples: int,
        tolerance: float
    ) -> ValidationResult:
        """
        Validate the barrier function numerically by sampling points.
        
        Args:
            state_space_bounds: Bounds of the state space
            samples: Number of samples to check
            tolerance: Tolerance for numeric comparisons
            
        Returns:
            ValidationResult: Result of the validation
        """
        # Sample points from the state space
        states = self._sample_states(state_space_bounds, samples)
        
        # Evaluate barrier function at each state
        barrier_values = np.array([
            self.barrier_function(state, **self.params) for state in states
        ])
        
        # Check state constraints to determine which states are in the safe set
        if self.state_constraints:
            in_safe_set = np.ones(samples, dtype=bool)
            for constraint in self.state_constraints:
                constraint_values = np.array([
                    constraint(state, **self.params) for state in states
                ])
                in_safe_set = in_safe_set & (constraint_values >= 0)
        else:
            # If no constraints, assume all states are in the safe set
            in_safe_set = np.ones(samples, dtype=bool)
        
        # Check if barrier function is positive in the safe set
        violations = []
        for i, (state, barrier_value) in enumerate(zip(states, barrier_values)):
            if in_safe_set[i] and barrier_value < -tolerance:
                violations.append({
                    "state": state.tolist(),
                    "barrier_value": float(barrier_value),
                    "message": f"Barrier function negative ({barrier_value:.6f}) at state {state}"
                })
        
        # Check if there are any violations
        if not violations:
            return passed(
                "Barrier function is positive in the sampled safe set",
                {
                    "samples": samples,
                    "min_barrier_value": float(np.min(barrier_values[in_safe_set])),
                    "max_barrier_value": float(np.max(barrier_values[in_safe_set])),
                    "mean_barrier_value": float(np.mean(barrier_values[in_safe_set]))
                }
            )
        else:
            return failed(
                f"Barrier function is negative at {len(violations)} sampled points in the safe set",
                {
                    "samples": samples,
                    "violation_count": len(violations),
                    "min_barrier_value": float(np.min(barrier_values[in_safe_set])),
                    "max_barrier_value": float(np.max(barrier_values[in_safe_set])),
                    "mean_barrier_value": float(np.mean(barrier_values[in_safe_set]))
                },
                violations=violations
            )
    
    def validate_barrier_derivative(
        self,
        state_space_bounds: Optional[np.ndarray] = None,
        input_space_bounds: Optional[np.ndarray] = None,
        samples: int = 1000,
        tolerance: float = 1e-6,
        alpha_function: Optional[Callable] = None
    ) -> ValidationResult:
        """
        Validate that the barrier function derivative is properly constrained.
        
        The barrier certificate condition is:
        ∇B(x) · f(x, u) ≥ -α(B(x))
        
        Args:
            state_space_bounds: Bounds of the state space, shape (state_dim, 2)
            input_space_bounds: Bounds of the input space, shape (input_dim, 2)
            samples: Number of samples to check
            tolerance: Tolerance for numeric comparisons
            alpha_function: Class-K function α(B(x)), defaults to α(B(x)) = B(x)
            
        Returns:
            ValidationResult: Result of the validation
        """
        # Set default bounds if not provided
        if state_space_bounds is None:
            state_space_bounds = np.array([[-10.0, 10.0]] * self.state_dim)
        
        if input_space_bounds is None and self.input_dim > 0:
            input_space_bounds = np.array([[-10.0, 10.0]] * self.input_dim)
        
        # Set default alpha function if not provided
        if alpha_function is None:
            alpha_function = lambda b: b  # Identity function
        
        if self.symbolic:
            return self._validate_barrier_derivative_symbolic(
                state_space_bounds, input_space_bounds, tolerance, alpha_function
            )
        else:
            return self._validate_barrier_derivative_numeric(
                state_space_bounds, input_space_bounds, samples, tolerance, alpha_function
            )
    
    def _validate_barrier_derivative_symbolic(
        self,
        state_space_bounds: np.ndarray,
        input_space_bounds: Optional[np.ndarray],
        tolerance: float,
        alpha_function: Callable
    ) -> ValidationResult:
        """
        Validate the barrier derivative symbolically.
        
        Args:
            state_space_bounds: Bounds of the state space
            input_space_bounds: Bounds of the input space
            tolerance: Tolerance for numeric comparisons
            alpha_function: Class-K function α(B(x))
            
        Returns:
            ValidationResult: Result of the validation
        """
        try:
            # Build symbolic expression for the barrier function
            barrier_expr = self.barrier_function(*self.state_vars, **self.param_vars)
            
            # Calculate gradient of the barrier function
            grad_barrier = [sp.diff(barrier_expr, x) for x in self.state_vars]
            
            # Build symbolic expression for the system dynamics
            if self.input_dim > 0:
                dynamics_expr = self.system_dynamics(
                    self.state_vars, self.input_vars, **self.param_vars
                )
            else:
                dynamics_expr = self.system_dynamics(
                    self.state_vars, **self.param_vars
                )
            
            # Calculate dot product of gradient and dynamics
            lie_derivative = sum(
                g * f for g, f in zip(grad_barrier, dynamics_expr)
            )
            
            # Build symbolic expression for the alpha function
            alpha_expr = alpha_function(barrier_expr)
            
            # Check if Lie derivative satisfies the barrier condition
            # Lie derivative ≥ -α(B(x))
            condition_expr = lie_derivative + alpha_expr
            
            # For now, we'll just return a warning
            return warning(
                "Symbolic validation of barrier derivative is not fully implemented. "
                "Consider using numerical validation instead.",
                {
                    "barrier_expr": str(barrier_expr),
                    "lie_derivative": str(lie_derivative),
                    "alpha_expr": str(alpha_expr),
                    "condition_expr": str(condition_expr)
                }
            )
        except Exception as e:
            logger.exception("Error during symbolic barrier derivative validation")
            return failed(
                f"Error during symbolic barrier derivative validation: {str(e)}",
                {"error": str(e)}
            )
    
    def _validate_barrier_derivative_numeric(
        self,
        state_space_bounds: np.ndarray,
        input_space_bounds: Optional[np.ndarray],
        samples: int,
        tolerance: float,
        alpha_function: Callable
    ) -> ValidationResult:
        """
        Validate the barrier derivative numerically by sampling points.
        
        Args:
            state_space_bounds: Bounds of the state space
            input_space_bounds: Bounds of the input space
            samples: Number of samples to check
            tolerance: Tolerance for numeric comparisons
            alpha_function: Class-K function α(B(x))
            
        Returns:
            ValidationResult: Result of the validation
        """
        # Sample points from the state space
        states = self._sample_states(state_space_bounds, samples)
        
        # Evaluate barrier function at each state
        barrier_values = np.array([
            self.barrier_function(state, **self.params) for state in states
        ])
        
        # For each state, find the worst-case input
        violations = []
        min_margin = float('inf')
        
        for i, state in enumerate(states):
            if self.input_dim > 0:
                # If there are control inputs, find the worst-case input
                result = self._find_worst_case_input(
                    state, input_space_bounds, alpha_function
                )
                worst_input = result["input"]
                margin = result["margin"]
            else:
                # If there are no control inputs, just evaluate the derivative
                worst_input = None
                gradient = self._compute_barrier_gradient(state)
                
                dynamics = self.system_dynamics(state, **self.params)
                lie_derivative = np.dot(gradient, dynamics)
                
                barrier_value = barrier_values[i]
                alpha_value = alpha_function(barrier_value)
                
                margin = lie_derivative + alpha_value
            
            # Update minimum margin
            min_margin = min(min_margin, margin)
            
            # Check if the barrier condition is violated
            if margin < -tolerance:
                if worst_input is not None:
                    violations.append({
                        "state": state.tolist(),
                        "input": worst_input.tolist(),
                        "barrier_value": float(barrier_values[i]),
                        "margin": float(margin),
                        "message": (
                            f"Barrier condition violated at state {state} "
                            f"with input {worst_input}, margin: {margin:.6f}"
                        )
                    })
                else:
                    violations.append({
                        "state": state.tolist(),
                        "barrier_value": float(barrier_values[i]),
                        "margin": float(margin),
                        "message": (
                            f"Barrier condition violated at state {state}, "
                            f"margin: {margin:.6f}"
                        )
                    })
        
        # Check if there are any violations
        if not violations:
            return passed(
                "Barrier derivative condition satisfied at all sampled points",
                {
                    "samples": samples,
                    "min_margin": float(min_margin)
                }
            )
        else:
            return failed(
                f"Barrier derivative condition violated at {len(violations)} sampled points",
                {
                    "samples": samples,
                    "violation_count": len(violations),
                    "min_margin": float(min_margin)
                },
                violations=violations
            )
    
    def _find_worst_case_input(
        self,
        state: np.ndarray,
        input_space_bounds: np.ndarray,
        alpha_function: Callable
    ) -> Dict[str, Any]:
        """
        Find the worst-case input that minimizes the barrier condition margin.
        
        Args:
            state: Current state
            input_space_bounds: Bounds of the input space
            alpha_function: Class-K function α(B(x))
            
        Returns:
            Dictionary with worst-case input and margin
        """
        # Define the objective function: -margin = -(∇B(x) · f(x, u) + α(B(x)))
        def objective(input_vec):
            # Compute gradient of the barrier function
            gradient = self._compute_barrier_gradient(state)
            
            # Compute system dynamics
            dynamics = self.system_dynamics(state, input_vec, **self.params)
            
            # Compute Lie derivative
            lie_derivative = np.dot(gradient, dynamics)
            
            # Compute barrier value and alpha function
            barrier_value = self.barrier_function(state, **self.params)
            alpha_value = alpha_function(barrier_value)
            
            # Compute margin
            margin = lie_derivative + alpha_value
            
            # Return negative margin (to minimize)
            return -margin
        
        # Prepare input bounds for the optimizer
        bounds = [(low, high) for low, high in input_space_bounds]
        
        # Initialize with zero input
        x0 = np.zeros(self.input_dim)
        
        # Check if there are input constraints
        if self.input_constraints:
            # Define constraint functions for scipy.optimize
            def constraint_func(input_vec):
                return np.array([
                    constraint(state, input_vec, **self.params)
                    for constraint in self.input_constraints
                ])
            
            # Define linear constraint for scipy.optimize
            linear_constraint = LinearConstraint(
                np.eye(self.input_dim), 
                input_space_bounds[:, 0], 
                input_space_bounds[:, 1]
            )
            
            # Run the optimization with constraints
            result = minimize(
                objective, x0, 
                method="SLSQP", 
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': constraint_func},
                    linear_constraint
                ]
            )
        else:
            # Run the optimization without constraints
            result = minimize(
                objective, x0, 
                method="L-BFGS-B", 
                bounds=bounds
            )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
        
        # Compute the margin at the optimal input
        worst_input = result.x
        
        # Recompute gradient and dynamics for the worst input
        gradient = self._compute_barrier_gradient(state)
        dynamics = self.system_dynamics(state, worst_input, **self.params)
        
        # Compute Lie derivative
        lie_derivative = np.dot(gradient, dynamics)
        
        # Compute barrier value and alpha function
        barrier_value = self.barrier_function(state, **self.params)
        alpha_value = alpha_function(barrier_value)
        
        # Compute margin
        margin = lie_derivative + alpha_value
        
        return {
            "input": worst_input,
            "margin": margin,
            "success": result.success
        }
    
    def _compute_barrier_gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the barrier function at a given state.
        
        Args:
            state: Current state
            
        Returns:
            Gradient of the barrier function
        """
        if self.barrier_derivative is not None:
            # Use provided derivative function
            return self.barrier_derivative(state, **self.params)
        else:
            # Compute gradient numerically
            h = 1e-6  # Step size for finite difference
            gradient = np.zeros(self.state_dim)
            
            for i in range(self.state_dim):
                # Create perturbation vectors
                state_plus = state.copy()
                state_plus[i] += h
                
                state_minus = state.copy()
                state_minus[i] -= h
                
                # Compute central difference
                b_plus = self.barrier_function(state_plus, **self.params)
                b_minus = self.barrier_function(state_minus, **self.params)
                
                gradient[i] = (b_plus - b_minus) / (2 * h)
            
            return gradient
    
    def _sample_states(
        self,
        state_space_bounds: np.ndarray,
        samples: int
    ) -> np.ndarray:
        """
        Sample states from the state space according to the specified method.
        
        Args:
            state_space_bounds: Bounds of the state space, shape (state_dim, 2)
            samples: Number of samples to generate
            
        Returns:
            Array of sampled states, shape (samples, state_dim)
        """
        if self.sampling_method == "grid":
            return self._sample_grid(state_space_bounds, samples)
        elif self.sampling_method == "random":
            return self._sample_random(state_space_bounds, samples)
        elif self.sampling_method == "uniform":
            return self._sample_uniform(state_space_bounds, samples)
        elif self.sampling_method == "sobol":
            return self._sample_sobol(state_space_bounds, samples)
        else:
            logger.warning(f"Unknown sampling method: {self.sampling_method}, using random")
            return self._sample_random(state_space_bounds, samples)
    
    def _sample_grid(
        self,
        state_space_bounds: np.ndarray,
        samples: int
    ) -> np.ndarray:
        """
        Sample states on a grid.
        
        Args:
            state_space_bounds: Bounds of the state space
            samples: Number of samples to generate
            
        Returns:
            Array of sampled states
        """
        # Compute number of points per dimension
        points_per_dim = max(2, int(np.power(samples, 1 / self.state_dim)))
        
        # Create grid for each dimension
        grids = []
        for i in range(self.state_dim):
            grid = np.linspace(
                state_space_bounds[i, 0],
                state_space_bounds[i, 1],
                points_per_dim
            )
            grids.append(grid)
        
        # Create meshgrid
        mesh = np.meshgrid(*grids)
        
        # Reshape to get all combinations
        states = np.column_stack([m.flatten() for m in mesh])
        
        # If too many points, subsample
        if len(states) > samples:
            indices = np.random.choice(len(states), samples, replace=False)
            states = states[indices]
        
        return states
    
    def _sample_random(
        self,
        state_space_bounds: np.ndarray,
        samples: int
    ) -> np.ndarray:
        """
        Sample states randomly.
        
        Args:
            state_space_bounds: Bounds of the state space
            samples: Number of samples to generate
            
        Returns:
            Array of sampled states
        """
        states = np.zeros((samples, self.state_dim))
        
        for i in range(self.state_dim):
            states[:, i] = np.random.uniform(
                state_space_bounds[i, 0],
                state_space_bounds[i, 1],
                samples
            )
        
        return states
    
    def _sample_uniform(
        self,
        state_space_bounds: np.ndarray,
        samples: int
    ) -> np.ndarray:
        """
        Sample states uniformly (Latin hypercube sampling).
        
        Args:
            state_space_bounds: Bounds of the state space
            samples: Number of samples to generate
            
        Returns:
            Array of sampled states
        """
        try:
            from scipy.stats.qmc import LatinHypercube
            
            # Create sampler
            sampler = LatinHypercube(d=self.state_dim)
            
            # Generate samples in [0, 1]^d
            unit_samples = sampler.random(n=samples)
            
            # Scale to state space bounds
            states = np.zeros((samples, self.state_dim))
            for i in range(self.state_dim):
                low, high = state_space_bounds[i]
                states[:, i] = low + (high - low) * unit_samples[:, i]
            
            return states
        except ImportError:
            logger.warning("scipy.stats.qmc not available, using random sampling")
            return self._sample_random(state_space_bounds, samples)
    
    def _sample_sobol(
        self,
        state_space_bounds: np.ndarray,
        samples: int
    ) -> np.ndarray:
        """
        Sample states using Sobol sequences.
        
        Args:
            state_space_bounds: Bounds of the state space
            samples: Number of samples to generate
            
        Returns:
            Array of sampled states
        """
        try:
            from scipy.stats.qmc import Sobol
            
            # Create sampler
            sampler = Sobol(d=self.state_dim, scramble=True)
            
            # Generate samples in [0, 1]^d
            unit_samples = sampler.random(n=samples)
            
            # Scale to state space bounds
            states = np.zeros((samples, self.state_dim))
            for i in range(self.state_dim):
                low, high = state_space_bounds[i]
                states[:, i] = low + (high - low) * unit_samples[:, i]
            
            return states
        except ImportError:
            logger.warning("scipy.stats.qmc not available, using random sampling")
            return self._sample_random(state_space_bounds, samples)


# Convenience functions

def validate_barrier_function(
    barrier_function: Callable,
    state_dim: int,
    state_space_bounds: Optional[np.ndarray] = None,
    state_constraints: Optional[List[Callable]] = None,
    params: Optional[Dict[str, Any]] = None,
    samples: int = 1000,
    tolerance: float = 1e-6,
    sampling_method: str = "grid"
) -> ValidationResult:
    """
    Validate that a barrier function is positive in the safe set.
    
    Args:
        barrier_function: Function representing the barrier function B(x)
        state_dim: Dimension of the state space
        state_space_bounds: Bounds of the state space, shape (state_dim, 2)
        state_constraints: List of functions representing state constraints
        params: Dictionary of barrier function parameters
        samples: Number of samples to check
        tolerance: Tolerance for numeric comparisons
        sampling_method: Method for sampling states ("grid", "random", "uniform", "sobol")
        
    Returns:
        ValidationResult: Result of the validation
    """
    # Create a dummy dynamics function (not used for barrier function validation)
    dummy_dynamics = lambda x, **p: np.zeros_like(x)
    
    # Create validator
    validator = BarrierValidator(
        system_dynamics=dummy_dynamics,
        barrier_function=barrier_function,
        state_dim=state_dim,
        input_dim=0,
        state_constraints=state_constraints,
        params=params,
        sampling_method=sampling_method
    )
    
    # Validate barrier function
    return validator.validate_barrier_function(
        state_space_bounds=state_space_bounds,
        samples=samples,
        tolerance=tolerance
    )


def validate_barrier_derivative(
    system_dynamics: Callable,
    barrier_function: Callable,
    state_dim: int,
    input_dim: int = 0,
    barrier_derivative: Optional[Callable] = None,
    state_space_bounds: Optional[np.ndarray] = None,
    input_space_bounds: Optional[np.ndarray] = None,
    state_constraints: Optional[List[Callable]] = None,
    input_constraints: Optional[List[Callable]] = None,
    params: Optional[Dict[str, Any]] = None,
    alpha_function: Optional[Callable] = None,
    samples: int = 1000,
    tolerance: float = 1e-6,
    sampling_method: str = "grid"
) -> ValidationResult:
    """
    Validate that the barrier function derivative satisfies the barrier condition.
    
    Args:
        system_dynamics: Function representing the system dynamics dx/dt = f(x, u)
        barrier_function: Function representing the barrier function B(x)
        state_dim: Dimension of the state space
        input_dim: Dimension of the input space
        barrier_derivative: Optional function representing the barrier function derivative
        state_space_bounds: Bounds of the state space, shape (state_dim, 2)
        input_space_bounds: Bounds of the input space, shape (input_dim, 2)
        state_constraints: List of functions representing state constraints
        input_constraints: List of functions representing input constraints
        params: Dictionary of system and barrier function parameters
        alpha_function: Class-K function α(B(x)), defaults to identity
        samples: Number of samples to check
        tolerance: Tolerance for numeric comparisons
        sampling_method: Method for sampling states
        
    Returns:
        ValidationResult: Result of the validation
    """
    # Create validator
    validator = BarrierValidator(
        system_dynamics=system_dynamics,
        barrier_function=barrier_function,
        state_dim=state_dim,
        input_dim=input_dim,
        barrier_derivative=barrier_derivative,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        params=params,
        sampling_method=sampling_method
    )
    
    # Validate barrier derivative
    return validator.validate_barrier_derivative(
        state_space_bounds=state_space_bounds,
        input_space_bounds=input_space_bounds,
        samples=samples,
        tolerance=tolerance,
        alpha_function=alpha_function
    )
