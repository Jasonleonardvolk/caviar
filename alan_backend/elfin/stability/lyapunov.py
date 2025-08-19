"""
Lyapunov Function Definitions for ELFIN DSL.

This module provides the base classes and implementations of various
Lyapunov function types for stability verification.
"""

from enum import Enum, auto
from typing import List, Dict, Optional, Any, Callable, Union, Tuple
import numpy as np


class LyapunovFunction:
    """
    Base class for Lyapunov functions.

    A Lyapunov function V(x) satisfies:
    1. V(0) = 0
    2. V(x) > 0 for all x ≠ 0
    3. V(f(x)) - V(x) < 0 for all x ≠ 0 (discrete-time)
       or dV/dt < 0 for all x ≠ 0 (continuous-time)
    """
    
    def __init__(self, name: str, domain_ids: Optional[List[str]] = None):
        """
        Initialize Lyapunov function.
        
        Args:
            name: Name of the function
            domain_ids: IDs of concepts in the domain
        """
        self.name = name
        self.domain_ids = domain_ids or []
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Lyapunov function at a state x.
        
        Args:
            x: State vector
            
        Returns:
            Lyapunov function value
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Lyapunov function for a batch of states.
        
        Args:
            X: Batch of state vectors, shape (n, N)
            
        Returns:
            Array of function values, shape (N,)
        """
        # Default implementation (can be overridden for efficiency)
        return np.array([self.evaluate(x) for x in X.T])
    
    def get_quadratic_form(self) -> Optional[np.ndarray]:
        """
        Get the quadratic form matrix Q for V(x) = x^T Q x.
        
        Returns:
            Quadratic form matrix or None if not applicable
        """
        return None


class PolynomialLyapunov(LyapunovFunction):
    """
    Polynomial Lyapunov function: V(x) = b_x^T Q b_x.
    
    This represents Lyapunov functions of the form:
    V(x) = b_x^T Q b_x
    
    where b_x is a vector of basis functions and Q is positive definite.
    For quadratic Lyapunov functions, b_x = x.
    """
    
    def __init__(
        self, 
        name: str, 
        Q: np.ndarray,
        basis_functions: Optional[List[str]] = None,
        domain_ids: Optional[List[str]] = None
    ):
        """
        Initialize polynomial Lyapunov function.
        
        Args:
            name: Name of the function
            Q: Positive definite matrix
            basis_functions: List of basis function expressions
            domain_ids: IDs of concepts in the domain
        """
        super().__init__(name, domain_ids)
        self.Q = Q
        self.basis_functions = basis_functions
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Lyapunov function at a state x.
        
        For simple quadratic case: V(x) = x^T Q x
        
        Args:
            x: State vector
            
        Returns:
            Lyapunov function value
        """
        x = np.asarray(x).flatten()
        
        if self.basis_functions:
            # Evaluate basis functions (placeholder)
            b_x = x  # In a real implementation, this would compute the basis vector
        else:
            b_x = x
            
        return float(b_x.T @ self.Q @ b_x)
    
    def get_quadratic_form(self) -> np.ndarray:
        """
        Get the quadratic form matrix Q.
        
        Returns:
            Quadratic form matrix
        """
        return self.Q


class NeuralLyapunov(LyapunovFunction):
    """
    Neural network Lyapunov function.
    
    This represents Lyapunov functions parameterized by neural networks,
    typically with properties enforced during training.
    """
    
    def __init__(
        self,
        name: str,
        network,
        domain_ids: Optional[List[str]] = None
    ):
        """
        Initialize neural Lyapunov function.
        
        Args:
            name: Name of the function
            network: Neural network model (e.g., PyTorch or TensorFlow)
            domain_ids: IDs of concepts in the domain
        """
        super().__init__(name, domain_ids)
        self.network = network
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Lyapunov function at a state x.
        
        Args:
            x: State vector
            
        Returns:
            Lyapunov function value
        """
        # Placeholder implementation (would depend on the network framework)
        # In real implementation, this would use self.network(x)
        x = np.asarray(x).flatten()
        return float(np.sum(x**2))  # Placeholder


class CLFFunction(LyapunovFunction):
    """
    Control Lyapunov Function (CLF).
    
    This represents Lyapunov functions that can be used for control design,
    typically with QP-based enforcement.
    """
    
    def __init__(
        self,
        name: str,
        value_function: Union[Callable, str],
        control_variables: List[str],
        gamma: float = 0.1,
        domain_ids: Optional[List[str]] = None
    ):
        """
        Initialize control Lyapunov function.
        
        Args:
            name: Name of the function
            value_function: Value function expression or callable
            control_variables: Names of control variables
            gamma: Convergence rate parameter
            domain_ids: IDs of concepts in the domain
        """
        super().__init__(name, domain_ids)
        self.value_function = value_function
        self.control_variables = control_variables
        self.gamma = gamma
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Lyapunov function at a state x.
        
        Args:
            x: State vector
            
        Returns:
            Lyapunov function value
        """
        # Placeholder implementation (would depend on the value_function type)
        if callable(self.value_function):
            return float(self.value_function(x))
        else:
            # For string expressions, would need a parser
            x = np.asarray(x).flatten()
            return float(np.sum(x**2))  # Placeholder


class CompositeLyapunov(LyapunovFunction):
    """
    Composite Lyapunov function.
    
    This represents Lyapunov functions composed of multiple component
    functions, useful for multi-agent systems and compositional verification.
    """
    
    def __init__(
        self,
        name: str,
        components: List[LyapunovFunction],
        weights: Optional[List[float]] = None,
        composition_type: str = "weighted_sum",
        domain_ids: Optional[List[str]] = None
    ):
        """
        Initialize composite Lyapunov function.
        
        Args:
            name: Name of the function
            components: List of component Lyapunov functions
            weights: Weights for each component (default: equal weights)
            composition_type: Type of composition (weighted_sum, max, min)
            domain_ids: IDs of concepts in the domain
        """
        super().__init__(name, domain_ids)
        self.components = components
        
        if weights is None:
            self.weights = [1.0] * len(components)
        else:
            assert len(weights) == len(components), "Weights and components must have same length"
            self.weights = weights
            
        assert composition_type in ["weighted_sum", "max", "min"], "Invalid composition type"
        self.composition_type = composition_type
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Lyapunov function at a state x.
        
        Args:
            x: State vector
            
        Returns:
            Lyapunov function value
        """
        component_values = [comp.evaluate(x) for comp in self.components]
        
        if self.composition_type == "weighted_sum":
            return float(sum(w * v for w, v in zip(self.weights, component_values)))
        elif self.composition_type == "max":
            return float(max(component_values))
        elif self.composition_type == "min":
            return float(min(component_values))
