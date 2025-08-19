"""
Van der Pol Oscillator System

This module provides a benchmark implementation of the Van der Pol oscillator.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..benchmark import BenchmarkSystem


class VanDerPolOscillator(BenchmarkSystem):
    """
    Van der Pol oscillator system with position and velocity as states.
    
    State variables:
        x[0]: position
        x[1]: velocity
    
    Input variables:
        u[0]: control input
    
    Parameters:
        mu: Oscillation parameter (controls nonlinearity)
        unsafe_radius: Radius of unsafe set in phase space
    """
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Van der Pol oscillator system.
        
        Args:
            params: System parameters
        """
        # Default parameters
        default_params = {
            "mu": 1.0,           # Oscillation parameter
            "unsafe_radius": 2.5  # Radius of unsafe set in phase space
        }
        
        # Override defaults with provided parameters
        params = {**default_params, **(params or {})}
        
        super().__init__(
            name="VanDerPolOscillator",
            state_dim=2,
            input_dim=1,
            params=params
        )
    
    def dynamics(self, state: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        """
        Compute the Van der Pol oscillator dynamics: dx/dt = f(x, u).
        
        Args:
            state: System state [position, velocity]
            input_vec: Control input
        
        Returns:
            State derivative
        """
        x, y = state
        u = input_vec[0] if input_vec.size > 0 else 0.0
        
        # Extract parameters
        mu = self.params["mu"]
        
        # Compute derivatives
        x_dot = y
        y_dot = mu * (1 - x**2) * y - x + u
        
        return np.array([x_dot, y_dot])
    
    def barrier_function(self, state: np.ndarray) -> float:
        """
        Compute the barrier function B(x).
        
        This barrier function is based on the distance from the origin.
        It is positive inside the safe set and negative outside.
        
        Args:
            state: System state [position, velocity]
        
        Returns:
            Barrier function value
        """
        x, y = state
        
        # Extract parameters
        unsafe_radius = self.params["unsafe_radius"]
        
        # Compute distance from origin
        distance_squared = x**2 + y**2
        
        # Barrier function: positive inside safe set, negative outside
        return unsafe_radius**2 - distance_squared
    
    def barrier_derivative(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the barrier function.
        
        Args:
            state: System state [position, velocity]
        
        Returns:
            Gradient of the barrier function
        """
        x, y = state
        
        # Gradient of B(x, y) = unsafe_radius^2 - (x^2 + y^2)
        grad_x = -2 * x
        grad_y = -2 * y
        
        return np.array([grad_x, grad_y])
    
    def is_safe(self, state: np.ndarray) -> bool:
        """
        Check if a state is safe.
        
        Args:
            state: System state [position, velocity]
        
        Returns:
            True if the state is safe, False otherwise
        """
        x, y = state
        
        # Extract parameters
        unsafe_radius = self.params["unsafe_radius"]
        
        # Compute distance from origin
        distance_squared = x**2 + y**2
        
        # Safe if distance is less than unsafe radius
        return distance_squared < unsafe_radius**2
    
    def get_state_bounds(self) -> np.ndarray:
        """
        Get state space bounds for the Van der Pol oscillator.
        
        Returns:
            Array of state bounds, shape (2, 2)
        """
        # Use a range that encompasses the limit cycle
        bound = 4.0
        return np.array([
            [-bound, bound],  # x range
            [-bound, bound]   # y range
        ])
    
    def get_input_bounds(self) -> np.ndarray:
        """
        Get input space bounds for the Van der Pol oscillator.
        
        Returns:
            Array of input bounds, shape (1, 2)
        """
        return np.array([
            [-5.0, 5.0]  # control input range
        ])
    
    def get_initial_states(self, num_states: int = 10) -> np.ndarray:
        """
        Get representative initial states for simulation.
        
        For Van der Pol, we sample states inside the safe region.
        
        Args:
            num_states: Number of initial states to generate
        
        Returns:
            Array of initial states, shape (num_states, state_dim)
        """
        # Sample uniformly inside a circle
        unsafe_radius = self.params["unsafe_radius"]
        safe_radius = unsafe_radius * 0.8  # Stay away from boundary
        
        # Generate random angles and radii
        angles = np.random.uniform(0, 2 * np.pi, num_states)
        radii = np.random.uniform(0, safe_radius, num_states)
        
        # Convert to Cartesian coordinates
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        
        return np.column_stack([x, y])
    
    def get_elfin_spec(self) -> str:
        """
        Get the ELFIN specification for the Van der Pol oscillator.
        
        Returns:
            ELFIN specification as a string
        """
        # Extract parameters
        mu = self.params["mu"]
        unsafe_radius = self.params["unsafe_radius"]
        
        spec = f"""
// Van der Pol oscillator ELFIN specification
param mu = {mu};           // Oscillation parameter
param unsafe_radius = {unsafe_radius};  // Radius of unsafe set in phase space

// State variables
state x;  // Position
state y;  // Velocity

// Input variables
input u;  // Control input

// System dynamics
flow_map {{
    d/dt[x] = y;
    d/dt[y] = mu * (1 - x^2) * y - x + u;
}}

// Barrier function that keeps the system within a safe radius
barrier_certificate B {{
    B = unsafe_radius^2 - (x^2 + y^2);
    alpha = 0.1 * B;  // Class-K function
}}

// Controller that stabilizes the origin
controller stabilize {{
    u = -y - 0.5 * x;
}}
"""
        return spec
