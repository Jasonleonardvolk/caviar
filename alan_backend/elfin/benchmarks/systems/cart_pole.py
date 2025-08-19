"""
Cart-Pole System

This module provides a benchmark implementation of the cart-pole (inverted pendulum) system.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..benchmark import BenchmarkSystem


class CartPole(BenchmarkSystem):
    """
    Cart-pole system (inverted pendulum on a cart).
    
    State variables:
        x[0]: x - cart position (m)
        x[1]: x_dot - cart velocity (m/s)
        x[2]: theta - pole angle (radians, 0 = upright)
        x[3]: theta_dot - pole angular velocity (radians/s)
    
    Input variables:
        u[0]: force - horizontal force applied to cart (N)
    
    Parameters:
        m_c: Mass of cart (kg)
        m_p: Mass of pole (kg)
        l: Half-length of pole (m)
        g: Gravitational acceleration (m/s^2)
        track_limit: Limit of the track in both directions (m)
        unsafe_angle: Angle beyond which pole is considered unsafe (radians)
    """
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a cart-pole system.
        
        Args:
            params: System parameters
        """
        # Default parameters
        default_params = {
            "m_c": 1.0,         # Mass of cart (kg)
            "m_p": 0.1,         # Mass of pole (kg)
            "l": 0.5,           # Half-length of pole (m)
            "g": 9.81,          # Gravitational acceleration (m/s^2)
            "track_limit": 2.4, # Limit of track in both directions (m)
            "unsafe_angle": 0.2  # Angle beyond which pole is unsafe (radians)
        }
        
        # Override defaults with provided parameters
        params = {**default_params, **(params or {})}
        
        super().__init__(
            name="CartPole",
            state_dim=4,
            input_dim=1,
            params=params
        )
    
    def dynamics(self, state: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        """
        Compute the cart-pole dynamics: dx/dt = f(x, u).
        
        Args:
            state: System state [x, x_dot, theta, theta_dot]
            input_vec: Control input [force]
        
        Returns:
            State derivative
        """
        x, x_dot, theta, theta_dot = state
        force = input_vec[0] if input_vec.size > 0 else 0.0
        
        # Extract parameters
        m_c = self.params["m_c"]
        m_p = self.params["m_p"]
        l = self.params["l"]
        g = self.params["g"]
        
        # Compute intermediate terms
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Compute denominator term
        denom = m_c + m_p * sin_theta**2
        
        # Compute accelerations
        x_ddot = (force + m_p * sin_theta * (l * theta_dot**2 + g * cos_theta)) / denom
        
        theta_ddot = (-force * cos_theta - m_p * l * theta_dot**2 * sin_theta * cos_theta - 
                      (m_c + m_p) * g * sin_theta) / (l * denom)
        
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])
    
    def barrier_function(self, state: np.ndarray) -> float:
        """
        Compute the barrier function B(x).
        
        This barrier function combines position and angle constraints.
        It is positive in the safe set and negative outside.
        
        Args:
            state: System state [x, x_dot, theta, theta_dot]
        
        Returns:
            Barrier function value
        """
        x, _, theta, _ = state
        
        # Extract parameters
        track_limit = self.params["track_limit"]
        unsafe_angle = self.params["unsafe_angle"]
        
        # Position barrier (keeps cart on track)
        position_barrier = (track_limit**2 - x**2) / track_limit**2
        
        # Angle barrier (keeps pole upright)
        angle_barrier = (unsafe_angle**2 - theta**2) / unsafe_angle**2
        
        # Combined barrier (minimum of the two)
        return min(position_barrier, angle_barrier)
    
    def barrier_derivative(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the barrier function.
        
        Args:
            state: System state [x, x_dot, theta, theta_dot]
        
        Returns:
            Gradient of the barrier function
        """
        x, _, theta, _ = state
        
        # Extract parameters
        track_limit = self.params["track_limit"]
        unsafe_angle = self.params["unsafe_angle"]
        
        # Position barrier value
        position_barrier = (track_limit**2 - x**2) / track_limit**2
        
        # Angle barrier value
        angle_barrier = (unsafe_angle**2 - theta**2) / unsafe_angle**2
        
        # Initialize gradient
        gradient = np.zeros(4)
        
        # Determine which barrier component is active
        if position_barrier < angle_barrier:
            # Position constraint is active
            gradient[0] = -2 * x / track_limit**2
        else:
            # Angle constraint is active
            gradient[2] = -2 * theta / unsafe_angle**2
        
        return gradient
    
    def is_safe(self, state: np.ndarray) -> bool:
        """
        Check if a state is safe.
        
        Args:
            state: System state [x, x_dot, theta, theta_dot]
        
        Returns:
            True if the state is safe, False otherwise
        """
        x, _, theta, _ = state
        
        # Extract parameters
        track_limit = self.params["track_limit"]
        unsafe_angle = self.params["unsafe_angle"]
        
        # Check safety conditions
        position_safe = abs(x) < track_limit
        angle_safe = abs(theta) < unsafe_angle
        
        return position_safe and angle_safe
    
    def get_state_bounds(self) -> np.ndarray:
        """
        Get state space bounds for the cart-pole.
        
        Returns:
            Array of state bounds, shape (4, 2)
        """
        track_limit = self.params["track_limit"]
        
        return np.array([
            [-track_limit, track_limit],  # x range
            [-5.0, 5.0],                 # x_dot range
            [-np.pi/4, np.pi/4],         # theta range
            [-5.0, 5.0]                  # theta_dot range
        ])
    
    def get_input_bounds(self) -> np.ndarray:
        """
        Get input space bounds for the cart-pole.
        
        Returns:
            Array of input bounds, shape (1, 2)
        """
        return np.array([
            [-10.0, 10.0]  # force range
        ])
    
    def get_initial_states(self, num_states: int = 10) -> np.ndarray:
        """
        Get representative initial states for simulation.
        
        For cart-pole, we sample states close to the upright equilibrium.
        
        Args:
            num_states: Number of initial states to generate
        
        Returns:
            Array of initial states, shape (num_states, state_dim)
        """
        track_limit = self.params["track_limit"]
        unsafe_angle = self.params["unsafe_angle"]
        
        # Sample near the upright position with small perturbations
        states = np.zeros((num_states, 4))
        
        # Position: uniform in track
        states[:, 0] = np.random.uniform(-0.5 * track_limit, 0.5 * track_limit, num_states)
        
        # Velocity: small random values
        states[:, 1] = np.random.uniform(-0.5, 0.5, num_states)
        
        # Angle: small deviations from upright
        states[:, 2] = np.random.uniform(-0.5 * unsafe_angle, 0.5 * unsafe_angle, num_states)
        
        # Angular velocity: small random values
        states[:, 3] = np.random.uniform(-0.5, 0.5, num_states)
        
        return states
    
    def get_elfin_spec(self) -> str:
        """
        Get the ELFIN specification for the cart-pole.
        
        Returns:
            ELFIN specification as a string
        """
        # Extract parameters
        m_c = self.params["m_c"]
        m_p = self.params["m_p"]
        l = self.params["l"]
        g = self.params["g"]
        track_limit = self.params["track_limit"]
        unsafe_angle = self.params["unsafe_angle"]
        
        spec = f"""
// Cart-Pole (Inverted Pendulum) system ELFIN specification
param m_c = {m_c};           // Mass of cart (kg)
param m_p = {m_p};          // Mass of pole (kg)
param l = {l};            // Half-length of pole (m)
param g = {g};           // Gravitational acceleration (m/s^2)
param track_limit = {track_limit};   // Track limit (m)
param unsafe_angle = {unsafe_angle};    // Unsafe angle (radians)

// State variables
state x;           // Cart position (m)
state x_dot;       // Cart velocity (m/s)
state theta;       // Pole angle (radians, 0 = upright)
state theta_dot;   // Pole angular velocity (radians/s)

// Input variables
input force;       // Horizontal force applied to cart (N)

// System dynamics
flow_map {{
    // Define intermediate terms
    let sin_theta = sin(theta);
    let cos_theta = cos(theta);
    let denom = m_c + m_p * sin_theta^2;
    
    d/dt[x] = x_dot;
    d/dt[x_dot] = (force + m_p * sin_theta * (l * theta_dot^2 + g * cos_theta)) / denom;
    d/dt[theta] = theta_dot;
    d/dt[theta_dot] = (-force * cos_theta - m_p * l * theta_dot^2 * sin_theta * cos_theta - 
                      (m_c + m_p) * g * sin_theta) / (l * denom);
}}

// Barrier function that combines position and angle constraints
barrier_certificate B {{
    // Position barrier (keeps cart on track)
    let position_barrier = (track_limit^2 - x^2) / track_limit^2;
    
    // Angle barrier (keeps pole upright)
    let angle_barrier = (unsafe_angle^2 - theta^2) / unsafe_angle^2;
    
    // Combined barrier
    B = min(position_barrier, angle_barrier);
    alpha = 0.1 * B;  // Class-K function
}}

// LQR controller for stabilizing the inverted pendulum
controller stabilize {{
    // Linearized gains
    let K1 = 1.0;     // Position gain
    let K2 = 1.5;     // Velocity gain
    let K3 = 20.0;    // Angle gain
    let K4 = 3.0;     // Angular velocity gain
    
    force = -K1 * x - K2 * x_dot - K3 * theta - K4 * theta_dot;
}}
"""
        return spec
