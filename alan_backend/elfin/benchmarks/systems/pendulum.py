"""
Pendulum System

This module provides a benchmark implementation of a pendulum system.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..benchmark import BenchmarkSystem


class Pendulum(BenchmarkSystem):
    """
    Pendulum system with angle and angular velocity as states.
    
    State variables:
        x[0]: theta - pendulum angle (radians)
        x[1]: theta_dot - pendulum angular velocity (radians/s)
    
    Input variables:
        u[0]: torque - control torque applied to pendulum (Nm)
    
    Parameters:
        m: Mass of pendulum bob (kg)
        l: Length of pendulum rod (m)
        g: Gravitational acceleration (m/s^2)
        b: Damping coefficient (kg*m^2/s)
        unsafe_angle: Angle beyond which pendulum is considered unsafe (radians)
        unsafe_velocity: Angular velocity beyond which pendulum is considered unsafe (radians/s)
    """
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a pendulum system.
        
        Args:
            params: Pendulum parameters
        """
        # Default parameters
        default_params = {
            "m": 1.0,       # Mass of pendulum bob (kg)
            "l": 1.0,       # Length of pendulum rod (m)
            "g": 9.81,      # Gravitational acceleration (m/s^2)
            "b": 0.1,       # Damping coefficient (kg*m^2/s)
            "unsafe_angle": np.pi/3,  # Angle beyond which pendulum is unsafe (radians)
            "unsafe_velocity": 2.0    # Angular velocity beyond which pendulum is unsafe (radians/s)
        }
        
        # Override defaults with provided parameters
        params = {**default_params, **(params or {})}
        
        super().__init__(
            name="Pendulum",
            state_dim=2,
            input_dim=1,
            params=params
        )
    
    def dynamics(self, state: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        """
        Compute the pendulum dynamics: dx/dt = f(x, u).
        
        Args:
            state: Pendulum state [theta, theta_dot]
            input_vec: Control input [torque]
        
        Returns:
            State derivative [theta_dot, theta_ddot]
        """
        theta, theta_dot = state
        torque = input_vec[0] if input_vec.size > 0 else 0.0
        
        # Extract parameters
        m = self.params["m"]
        l = self.params["l"]
        g = self.params["g"]
        b = self.params["b"]
        
        # Compute angular acceleration
        ml2 = m * l * l
        theta_ddot = (-b * theta_dot - m * g * l * np.sin(theta) + torque) / ml2
        
        return np.array([theta_dot, theta_ddot])
    
    def barrier_function(self, state: np.ndarray) -> float:
        """
        Compute the barrier function B(x).
        
        This barrier function is positive in the safe set and negative outside.
        It uses a quadratic form: B(x) = c - (x/a)^2 - (y/b)^2
        
        Args:
            state: Pendulum state [theta, theta_dot]
        
        Returns:
            Barrier function value
        """
        theta, theta_dot = state
        
        # Normalize angle to [-pi, pi]
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        
        # Extract safety limits
        unsafe_angle = self.params["unsafe_angle"]
        unsafe_velocity = self.params["unsafe_velocity"]
        
        # Compute barrier function value
        barrier_angle = 1.0 - (theta / unsafe_angle) ** 2
        barrier_velocity = 1.0 - (theta_dot / unsafe_velocity) ** 2
        
        # Use minimum of the two components
        return min(barrier_angle, barrier_velocity)
    
    def barrier_derivative(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the barrier function.
        
        Args:
            state: Pendulum state [theta, theta_dot]
        
        Returns:
            Gradient of the barrier function
        """
        theta, theta_dot = state
        
        # Normalize angle to [-pi, pi]
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        
        # Extract safety limits
        unsafe_angle = self.params["unsafe_angle"]
        unsafe_velocity = self.params["unsafe_velocity"]
        
        # Compute barrier components
        barrier_angle = 1.0 - (theta / unsafe_angle) ** 2
        barrier_velocity = 1.0 - (theta_dot / unsafe_velocity) ** 2
        
        # Compute gradients
        grad_theta = -2.0 * theta / (unsafe_angle ** 2)
        grad_theta_dot = -2.0 * theta_dot / (unsafe_velocity ** 2)
        
        # Determine which barrier component is active
        if barrier_angle < barrier_velocity:
            # Angle constraint is active
            return np.array([grad_theta, 0.0])
        else:
            # Velocity constraint is active
            return np.array([0.0, grad_theta_dot])
    
    def is_safe(self, state: np.ndarray) -> bool:
        """
        Check if a state is safe.
        
        Args:
            state: Pendulum state [theta, theta_dot]
        
        Returns:
            True if the state is safe, False otherwise
        """
        theta, theta_dot = state
        
        # Normalize angle to [-pi, pi]
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        
        # Extract safety limits
        unsafe_angle = self.params["unsafe_angle"]
        unsafe_velocity = self.params["unsafe_velocity"]
        
        # Check safety conditions
        angle_safe = abs(theta) < unsafe_angle
        velocity_safe = abs(theta_dot) < unsafe_velocity
        
        return angle_safe and velocity_safe
    
    def get_state_bounds(self) -> np.ndarray:
        """
        Get state space bounds for the pendulum.
        
        Returns:
            Array of state bounds, shape (2, 2)
        """
        return np.array([
            [-np.pi, np.pi],         # theta range
            [-5.0, 5.0]              # theta_dot range
        ])
    
    def get_input_bounds(self) -> np.ndarray:
        """
        Get input space bounds for the pendulum.
        
        Returns:
            Array of input bounds, shape (1, 2)
        """
        return np.array([
            [-5.0, 5.0]              # torque range
        ])
    
    def get_elfin_spec(self) -> str:
        """
        Get the ELFIN specification for the pendulum.
        
        Returns:
            ELFIN specification as a string
        """
        # Extract parameters
        m = self.params["m"]
        l = self.params["l"]
        g = self.params["g"]
        b = self.params["b"]
        unsafe_angle = self.params["unsafe_angle"]
        unsafe_velocity = self.params["unsafe_velocity"]
        
        spec = f"""
// Pendulum system ELFIN specification
// Non-Dimensionalized Parameters
param m = {m};        // Mass of pendulum bob (kg)
param l = {l};        // Length of pendulum rod (m)
param g = {g};    // Gravitational acceleration (m/s^2)
param b = {b};      // Damping coefficient (kg*m^2/s)

// Safety parameters
param unsafe_angle = {unsafe_angle};  // Unsafe pendulum angle (radians)
param unsafe_velocity = {unsafe_velocity};   // Unsafe angular velocity (radians/s)

// State variables
state theta;        // Pendulum angle (radians)
state theta_dot;    // Angular velocity (radians/s)

// Input variables
input torque;       // Control torque (Nm)

// System dynamics
flow_map {{
    d/dt[theta] = theta_dot;
    d/dt[theta_dot] = (-b * theta_dot - m * g * l * sin(theta) + torque) / (m * l * l);
}}

// Barrier function
barrier_certificate B {{
    B = min(1.0 - (theta / unsafe_angle)^2, 1.0 - (theta_dot / unsafe_velocity)^2);
    alpha = 0.1 * B;  // Class-K function
}}

// Basic controller that stabilizes the pendulum at theta = 0
controller stabilize {{
    torque = -2.0 * theta - 1.0 * theta_dot;
}}
"""
        return spec
