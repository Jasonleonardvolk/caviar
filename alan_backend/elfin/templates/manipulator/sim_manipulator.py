#!/usr/bin/env python3
"""
Manipulator Simulation Scaffold

This script provides a JAX-based simulation environment for the 6-DOF manipulator
defined in the ELFIN template. It allows for testing different control modes,
visualizing barrier functions, and validating safety guarantees.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental.ode import odeint
from typing import Dict, List, Tuple, Callable, Union, Optional

# Set up JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

# Define path to ELFIN file
ELFIN_FILE = Path(__file__).parent / "src" / "manipulator_controller.elfin"

class ManipulatorSimulator:
    """
    Simulator for the 6-DOF manipulator system defined in the ELFIN template.
    """
    def __init__(self, config_file: Path = ELFIN_FILE):
        """
        Initialize the simulator.
        
        Args:
            config_file: Path to the ELFIN controller specification file
        """
        self.config_file = config_file
        self.params = self._load_default_params()
        self.state_dim = 12  # 6 joint positions + 6 joint velocities
        self.input_dim = 6   # 6 joint torques
        
        # Initialize the system state
        self.reset_state()
        
        # Compile dynamics and controllers
        self._init_dynamics()
        self._init_controllers()
        self._init_barriers()
    
    def _load_default_params(self) -> Dict:
        """
        Load default parameters from the ELFIN file.
        In a real implementation, this would parse the ELFIN file.
        
        Returns:
            Dictionary of parameters
        """
        # For now, hardcode the parameters from the ELFIN file
        return {
            # Link masses (kg)
            "m1": 5.0, "m2": 4.0, "m3": 3.0, "m4": 2.0, "m5": 1.0, "m6": 0.5,
            
            # Link lengths (m)
            "l1": 0.3, "l2": 0.4, "l3": 0.4, "l4": 0.2, "l5": 0.1, "l6": 0.1,
            
            # Moments of inertia
            "I1": 0.1, "I2": 0.15, "I3": 0.12, "I4": 0.08, "I5": 0.05, "I6": 0.02,
            
            # Joint damping coefficients
            "d1": 0.5, "d2": 0.5, "d3": 0.5, "d4": 0.3, "d5": 0.3, "d6": 0.2,
            
            # Gravity
            "g": 9.81,
            
            # Controller gains
            "k_p1": 10.0, "k_p2": 10.0, "k_p3": 10.0, "k_p4": 8.0, "k_p5": 5.0, "k_p6": 3.0,
            "k_d1": 2.0, "k_d2": 2.0, "k_d3": 2.0, "k_d4": 1.5, "k_d5": 1.0, "k_d6": 0.5,
            "k_f": 0.1,
            
            # Desired joint positions
            "q1_d": 0.0, "q2_d": 0.0, "q3_d": 0.0, "q4_d": 0.0, "q5_d": 0.0, "q6_d": 0.0,
            
            # Joint limits
            "q1_min": -2.0, "q1_max": 2.0,
            "q2_min": -1.5, "q2_max": 1.5,
            "q3_min": -2.5, "q3_max": 2.5,
            "q4_min": -1.8, "q4_max": 1.8,
            "q5_min": -1.5, "q5_max": 1.5,
            "q6_min": -3.0, "q6_max": 3.0,
            
            # Barrier parameters
            "r_max": 1.0,
            "d_min": 0.1,
            "d_safe": 0.3,
            "x_h": 0.5,
            "y_h": 0.5,
            "alpha": 1.0,
            
            # Human collaboration parameters
            "human_proximity_factor": 0.5,
            
            # Force control parameters
            "F_d": 5.0,
            "F_measured": 0.0,
        }
    
    def reset_state(self, initial_state: Optional[np.ndarray] = None) -> None:
        """
        Reset the system state.
        
        Args:
            initial_state: Optional initial state, defaults to zeros
        """
        if initial_state is None:
            # Default to zeros
            self.state = np.zeros(self.state_dim)
        else:
            assert len(initial_state) == self.state_dim, f"State dimension mismatch: {len(initial_state)} != {self.state_dim}"
            self.state = initial_state.copy()
    
    def _init_dynamics(self) -> None:
        """Initialize the system dynamics function."""
        
        # Define the dynamics function
        def dynamics(state, u, params):
            """
            System dynamics for the 6-DOF manipulator.
            
            Args:
                state: System state [q1, q2, q3, q4, q5, q6, dq1, dq2, dq3, dq4, dq5, dq6]
                u: Control input [tau1, tau2, tau3, tau4, tau5, tau6]
                params: System parameters
            
            Returns:
                State derivatives [dq1, dq2, dq3, dq4, dq5, dq6, ddq1, ddq2, ddq3, ddq4, ddq5, ddq6]
            """
            # Extract state variables
            q1, q2, q3, q4, q5, q6 = state[0:6]
            dq1, dq2, dq3, dq4, dq5, dq6 = state[6:12]
            
            # Extract control inputs
            tau1, tau2, tau3, tau4, tau5, tau6 = u
            
            # Extract parameters
            m1, m2, m3, m4, m5, m6 = params['m1'], params['m2'], params['m3'], params['m4'], params['m5'], params['m6']
            l1, l2, l3, l4, l5, l6 = params['l1'], params['l2'], params['l3'], params['l4'], params['l5'], params['l6']
            I1, I2, I3, I4, I5, I6 = params['I1'], params['I2'], params['I3'], params['I4'], params['I5'], params['I6']
            d1, d2, d3, d4, d5, d6 = params['d1'], params['d2'], params['d3'], params['d4'], params['d5'], params['d6']
            g = params['g']
            
            # Compute state derivatives
            # Position derivatives = velocities
            dx1 = dq1
            dx2 = dq2
            dx3 = dq3
            dx4 = dq4
            dx5 = dq5
            dx6 = dq6
            
            # Velocity derivatives from torque inputs and system dynamics
            # Simplified dynamics from the ELFIN file
            ddq1 = (tau1 - d1*dq1 - g*m1*l1*jnp.sin(q1))/I1
            ddq2 = (tau2 - d2*dq2 - g*m2*l2*jnp.sin(q2))/I2
            ddq3 = (tau3 - d3*dq3 - g*m3*l3*jnp.sin(q3))/I3
            ddq4 = (tau4 - d4*dq4)/I4
            ddq5 = (tau5 - d5*dq5)/I5
            ddq6 = (tau6 - d6*dq6)/I6
            
            return jnp.array([dx1, dx2, dx3, dx4, dx5, dx6, ddq1, ddq2, ddq3, ddq4, ddq5, ddq6])
        
        # JIT-compile the dynamics function for speed
        self.dynamics_fn = jit(dynamics)
    
    def _init_controllers(self) -> None:
        """Initialize the controller functions for each mode."""
        
        # Joint position control mode
        def joint_position_control(state, params):
            """
            Joint position control with PD + gravity compensation.
            
            Args:
                state: System state
                params: Controller parameters
            
            Returns:
                Control inputs [tau1, tau2, tau3, tau4, tau5, tau6]
            """
            # Extract state variables
            q1, q2, q3, q4, q5, q6 = state[0:6]
            dq1, dq2, dq3, dq4, dq5, dq6 = state[6:12]
            
            # Extract parameters
            q1_d, q2_d, q3_d, q4_d, q5_d, q6_d = params['q1_d'], params['q2_d'], params['q3_d'], params['q4_d'], params['q5_d'], params['q6_d']
            k_p1, k_p2, k_p3, k_p4, k_p5, k_p6 = params['k_p1'], params['k_p2'], params['k_p3'], params['k_p4'], params['k_p5'], params['k_p6']
            k_d1, k_d2, k_d3, k_d4, k_d5, k_d6 = params['k_d1'], params['k_d2'], params['k_d3'], params['k_d4'], params['k_d5'], params['k_d6']
            g, m1, m2, m3 = params['g'], params['m1'], params['m2'], params['m3']
            l1, l2, l3 = params['l1'], params['l2'], params['l3']
            
            # Compute control inputs
            tau1 = k_p1*(q1_d - q1) + k_d1*(0 - dq1) + g*m1*l1*jnp.sin(q1)
            tau2 = k_p2*(q2_d - q2) + k_d2*(0 - dq2) + g*m2*l2*jnp.sin(q2)
            tau3 = k_p3*(q3_d - q3) + k_d3*(0 - dq3) + g*m3*l3*jnp.sin(q3)
            tau4 = k_p4*(q4_d - q4) + k_d4*(0 - dq4)
            tau5 = k_p5*(q5_d - q5) + k_d5*(0 - dq5)
            tau6 = k_p6*(q6_d - q6) + k_d6*(0 - dq6)
            
            return jnp.array([tau1, tau2, tau3, tau4, tau5, tau6])
        
        # Human collaboration control mode
        def human_collaboration_control(state, params):
            """
            Human collaboration control with variable impedance.
            
            Args:
                state: System state
                params: Controller parameters
            
            Returns:
                Control inputs [tau1, tau2, tau3, tau4, tau5, tau6]
            """
            # Extract state variables
            q1, q2, q3, q4, q5, q6 = state[0:6]
            dq1, dq2, dq3, dq4, dq5, dq6 = state[6:12]
            
            # Extract parameters
            q1_d, q2_d, q3_d, q4_d, q5_d, q6_d = params['q1_d'], params['q2_d'], params['q3_d'], params['q4_d'], params['q5_d'], params['q6_d']
            k_p1, k_p2, k_p3, k_p4, k_p5, k_p6 = params['k_p1'], params['k_p2'], params['k_p3'], params['k_p4'], params['k_p5'], params['k_p6']
            k_d1, k_d2, k_d3, k_d4, k_d5, k_d6 = params['k_d1'], params['k_d2'], params['k_d3'], params['k_d4'], params['k_d5'], params['k_d6']
            g, m1, m2, m3 = params['g'], params['m1'], params['m2'], params['m3']
            l1, l2, l3 = params['l1'], params['l2'], params['l3']
            human_proximity_factor = params['human_proximity_factor']
            
            # Adjust gains based on human proximity
            k_p1_h = k_p1 * human_proximity_factor
            k_p2_h = k_p2 * human_proximity_factor
            k_p3_h = k_p3 * human_proximity_factor
            k_p4_h = k_p4 * human_proximity_factor
            k_p5_h = k_p5 * human_proximity_factor
            k_p6_h = k_p6 * human_proximity_factor
            
            k_d1_h = k_d1 * (2.0 - human_proximity_factor)
            k_d2_h = k_d2 * (2.0 - human_proximity_factor)
            k_d3_h = k_d3 * (2.0 - human_proximity_factor)
            k_d4_h = k_d4 * (2.0 - human_proximity_factor)
            k_d5_h = k_d5 * (2.0 - human_proximity_factor)
            k_d6_h = k_d6 * (2.0 - human_proximity_factor)
            
            # Compute control inputs
            tau1 = k_p1_h*(q1_d - q1) + k_d1_h*(0 - dq1) + g*m1*l1*jnp.sin(q1)
            tau2 = k_p2_h*(q2_d - q2) + k_d2_h*(0 - dq2) + g*m2*l2*jnp.sin(q2)
            tau3 = k_p3_h*(q3_d - q3) + k_d3_h*(0 - dq3) + g*m3*l3*jnp.sin(q3)
            tau4 = k_p4_h*(q4_d - q4) + k_d4_h*(0 - dq4)
            tau5 = k_p5_h*(q5_d - q5) + k_d5_h*(0 - dq5)
            tau6 = k_p6_h*(q6_d - q6) + k_d6_h*(0 - dq6)
            
            return jnp.array([tau1, tau2, tau3, tau4, tau5, tau6])
        
        # Force control mode
        def force_control(state, params):
            """
            Hybrid position/force control.
            
            Args:
                state: System state
                params: Controller parameters
            
            Returns:
                Control inputs [tau1, tau2, tau3, tau4, tau5, tau6]
            """
            # Extract state variables
            q1, q2, q3, q4, q5, q6 = state[0:6]
            dq1, dq2, dq3, dq4, dq5, dq6 = state[6:12]
            
            # Extract parameters
            q1_d, q2_d, q3_d, q4_d, q5_d, q6_d = params['q1_d'], params['q2_d'], params['q3_d'], params['q4_d'], params['q5_d'], params['q6_d']
            k_p1, k_p2, k_p4, k_p5, k_p6 = params['k_p1'], params['k_p2'], params['k_p4'], params['k_p5'], params['k_p6']
            k_d1, k_d2, k_d4, k_d5, k_d6 = params['k_d1'], params['k_d2'], params['k_d4'], params['k_d5'], params['k_d6']
            k_f = params['k_f']
            F_d, F_measured = params['F_d'], params['F_measured']
            g, m1, m2, m3 = params['g'], params['m1'], params['m2'], params['m3']
            l1, l2, l3 = params['l1'], params['l2'], params['l3']
            
            # Position control components
            tau_pos1 = k_p1*(q1_d - q1) + k_d1*(0 - dq1)
            tau_pos2 = k_p2*(q2_d - q2) + k_d2*(0 - dq2)
            
            # Force control component
            tau_force3 = k_f*(F_d - F_measured)
            
            # Orientation control
            tau_ori4 = k_p4*(q4_d - q4) + k_d4*(0 - dq4)
            tau_ori5 = k_p5*(q5_d - q5) + k_d5*(0 - dq5)
            tau_ori6 = k_p6*(q6_d - q6) + k_d6*(0 - dq6)
            
            # Gravity compensation
            tau_g1 = g*m1*l1*jnp.sin(q1)
            tau_g2 = g*m2*l2*jnp.sin(q2)
            tau_g3 = g*m3*l3*jnp.sin(q3)
            
            # Combined control
            tau1 = tau_pos1 + tau_g1
            tau2 = tau_pos2 + tau_g2
            tau3 = tau_force3 + tau_g3
            tau4 = tau_ori4
            tau5 = tau_ori5
            tau6 = tau_ori6
            
            return jnp.array([tau1, tau2, tau3, tau4, tau5, tau6])
        
        # JIT-compile the controller functions
        self.joint_position_control = jit(joint_position_control)
        self.human_collaboration_control = jit(human_collaboration_control)
        self.force_control = jit(force_control)
        
        # Dictionary mapping controller names to functions
        self.controllers = {
            'joint_position': self.joint_position_control,
            'human_collaboration': self.human_collaboration_control,
            'force': self.force_control
        }
    
    def _init_barriers(self) -> None:
        """Initialize barrier functions for safety verification."""
        
        # Joint limits barrier
        def joint_limits_barrier(state, params):
            """
            Barrier function for joint limits.
            B > 0 means safe, B = 0 at boundary, B < 0 means unsafe.
            
            Args:
                state: System state
                params: Barrier parameters
            
            Returns:
                Barrier value
            """
            # Extract state variables
            q1, q2, q3, q4, q5, q6 = state[0:6]
            
            # Extract parameters
            q1_min, q1_max = params['q1_min'], params['q1_max']
            q2_min, q2_max = params['q2_min'], params['q2_max']
            q3_min, q3_max = params['q3_min'], params['q3_max']
            q4_min, q4_max = params['q4_min'], params['q4_max']
            q5_min, q5_max = params['q5_min'], params['q5_max']
            q6_min, q6_max = params['q6_min'], params['q6_max']
            
            # Compute barrier value
            B = (q1_max - q1)*(q1 - q1_min) * \
                (q2_max - q2)*(q2 - q2_min) * \
                (q3_max - q3)*(q3 - q3_min) * \
                (q4_max - q4)*(q4 - q4_min) * \
                (q5_max - q5)*(q5 - q5_min) * \
                (q6_max - q6)*(q6 - q6_min)
            
            return B
        
        # Self-collision barrier
        def self_collision_barrier(state, params):
            """
            Barrier function for self-collision avoidance.
            B > 0 means safe, B = 0 at boundary, B < 0 means unsafe.
            
            Args:
                state: System state
                params: Barrier parameters
            
            Returns:
                Barrier value
            """
            # Extract state variables
            q1, q2, q3, q4, q5, q6 = state[0:6]
            
            # Extract parameters
            l2, l4 = params['l2'], params['l4']
            d_min = params['d_min']
            
            # Compute barrier value
            # Simplified example: distance between link 2 and link 4
            B = d_min**2 - (l2*jnp.sin(q2) - l4*jnp.sin(q4))**2 - (l2*jnp.cos(q2) - l4*jnp.cos(q4))**2
            
            return B
        
        # Workspace barrier
        def workspace_barrier(state, params):
            """
            Barrier function for workspace constraints.
            B > 0 means safe, B = 0 at boundary, B < 0 means unsafe.
            
            Args:
                state: System state
                params: Barrier parameters
            
            Returns:
                Barrier value
            """
            # Extract state variables
            q1, q2, q3, q4, q5, q6 = state[0:6]
            
            # Extract parameters
            l1, l2, l3 = params['l1'], params['l2'], params['l3']
            r_max = params['r_max']
            
            # Compute end-effector position (simplified kinematics)
            x_ee = l1*jnp.cos(q1) + l2*jnp.cos(q1+q2) + l3*jnp.cos(q1+q2+q3)
            y_ee = l1*jnp.sin(q1) + l2*jnp.sin(q1+q2) + l3*jnp.sin(q1+q2+q3)
            
            # Compute barrier value
            B = r_max**2 - x_ee**2 - y_ee**2
            
            return B
        
        # Human safety barrier
        def human_safety_barrier(state, params):
            """
            Barrier function for human safety.
            B > 0 means safe, B = 0 at boundary, B < 0 means unsafe.
            
            Args:
                state: System state
                params: Barrier parameters
            
            Returns:
                Barrier value
            """
            # Extract state variables
            q1, q2, q3, q4, q5, q6 = state[0:6]
            
            # Extract parameters
            l1, l2 = params['l1'], params['l2']
            x_h, y_h = params['x_h'], params['y_h']
            d_safe = params['d_safe']
            
            # Compute link position (simplified kinematics)
            x_link = l1*jnp.cos(q1) + l2*jnp.cos(q1+q2)
            y_link = l1*jnp.sin(q1) + l2*jnp.sin(q1+q2)
            
            # Compute squared distance to human
            dist_squared = (x_h - x_link)**2 + (y_h - y_link)**2
            
            # Compute barrier value
            B = dist_squared - d_safe**2
            
            return B
        
        # JIT-compile the barrier functions
        self.joint_limits_barrier = jit(joint_limits_barrier)
        self.self_collision_barrier = jit(self_collision_barrier)
        self.workspace_barrier = jit(workspace_barrier)
        self.human_safety_barrier = jit(human_safety_barrier)
        
        # Dictionary mapping barrier names to functions
        self.barriers = {
            'joint_limits': self.joint_limits_barrier,
            'self_collision': self.self_collision_barrier,
            'workspace': self.workspace_barrier,
            'human_safety': self.human_safety_barrier
        }
    
    def evaluate_barriers(self, state: np.ndarray) -> Dict[str, float]:
        """
        Evaluate all barrier functions at the given state.
        
        Args:
            state: System state
        
        Returns:
            Dictionary mapping barrier names to values
        """
        return {
            name: float(barrier_fn(state, self.params))
            for name, barrier_fn in self.barriers.items()
        }
    
    def compute_forward_kinematics(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute forward kinematics for visualization.
        
        Args:
            state: System state
        
        Returns:
            Dictionary with link positions
        """
        # Extract joint angles
        q1, q2, q3, q4, q5, q6 = state[0:6]
        
        # Extract link lengths
        l1, l2, l3, l4, l5, l6 = self.params['l1'], self.params['l2'], self.params['l3'], \
                                 self.params['l4'], self.params['l5'], self.params['l6']
        
        # Base position
        p0 = np.array([0., 0., 0.])
        
        # Link positions (simplified kinematics)
        p1 = p0 + np.array([l1*np.cos(q1), l1*np.sin(q1), 0.])
        p2 = p1 + np.array([l2*np.cos(q1+q2), l2*np.sin(q1+q2), 0.])
        p3 = p2 + np.array([l3*np.cos(q1+q2+q3), l3*np.sin(q1+q2+q3), 0.])
        p4 = p3 + np.array([l4*np.cos(q1+q2+q3+q4), l4*np.sin(q1+q2+q3+q4), 0.])
        p5 = p4 + np.array([l5*np.cos(q1+q2+q3+q4+q5), l5*np.sin(q1+q2+q3+q4+q5), 0.])
        p6 = p5 + np.array([l6*np.cos(q1+q2+q3+q4+q5+q6), l6*np.sin(q1+q2+q3+q4+q5+q6), 0.])
        
        return {
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6
        }
    
    def simulate(self, 
                 controller_name: str, 
                 T: float = 10.0, 
                 dt: float = 0.01, 
                 initial_state: Optional[np.ndarray] = None,
                 reference_trajectory: Optional[Callable] = None) -> Dict:
        """
        Simulate the system with the specified controller.
        
        Args:
            controller_name: Name of the controller to use
            T: Simulation time (seconds)
            dt: Time step (seconds)
            initial_state: Optional initial state
            reference_trajectory: Optional reference trajectory function
        
        Returns:
            Dictionary with simulation results
        """
        # Reset state if initial_state is provided
        if initial_state is not None:
            self.reset_state(initial_state)
        
        # Get controller function
        if controller_name not in self.controllers:
            raise ValueError(f"Unknown controller: {controller_name}")
        controller_fn = self.controllers[controller_name]
        
        # Prepare time vector and storage for results
        t_values = np.arange(0, T, dt)
        n_steps = len(t_values)
        state_history = np.zeros((n_steps, self.state_dim))
        control_history = np.zeros((n_steps, self.input_dim))
        barrier_history = {name: np.zeros(n_steps) for name in self.barriers}
        
        # Current state
        state = self.state.copy()
        
        # Simulation loop
        for i, t in enumerate(t_values):
            # Store current state
            state_history[i] = state
            
            # Update reference if provided
            if reference_trajectory is not None:
                self.params.update(reference_trajectory(t))
            
            # Compute control input
            u = controller_fn(state, self.params)
            control_history[i] = u
            
            # Evaluate barrier functions
            for name, barrier_fn in self.barriers.items():
                barrier_history[name][i] = barrier_fn(state, self.params)
            
            # Integrate dynamics for one step
            # Convert to JAX arrays for JIT compilation
            state_jax = jnp.array(state)
            u_jax = jnp.array(u)
            
            # Compute state derivative
            state_dot = self.dynamics_fn(state_jax, u_jax, self.params)
            
            # Euler integration (simple but not very accurate)
            state = state + dt * np.array(state_dot)
        
        # Return results
        return {
            't': t_values,
            'state': state_history,
            'control': control_history,
            'barrier': barrier_history
        }
        
    def plot_barriers(self, sim_results: Dict) -> plt.Figure:
        """
        Plot barrier values from simulation results.
        
        Args:
            sim_results: Simulation results from the simulate method
        
        Returns:
            Matplotlib figure with barrier plots
        """
        t = sim_results['t']
        barriers = sim_results['barrier']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Barrier Functions (B > 0 means safe)')
        
        # Plot each barrier function
        for name, values in barriers.items():
            ax.plot(t, values, label=name)
            
        # Add zero line to indicate boundary
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Safety Boundary')
        
        # Add labels and legend
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Barrier Value')
        ax.legend()
        ax.grid(True)
        
        # Log scale for y-axis if barrier values span multiple orders of magnitude
        if any([np.any(b > 0) for b in barriers.values()]):
            y_min = min([np.min(b) for b in barriers.values() if np.any(b > 0)])
            y_max = max([np.max(b) for b in barriers.values()])
            if y_max / y_min > 100:
                ax.set_yscale('symlog')  # symlog allows negative values
        
        plt.tight_layout()
        return fig
    
    def plot_joint_states(self, sim_results: Dict) -> plt.Figure:
        """
        Plot joint positions and velocities from simulation results.
        
        Args:
            sim_results: Simulation results from the simulate method
        
        Returns:
            Matplotlib figure with joint plots
        """
        t = sim_results['t']
        state = sim_results['state']
        
        # Create figure with 2 rows for positions and velocities
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('Joint States')
        
        # Plot joint positions
        ax_pos = axes[0]
        for i in range(6):
            ax_pos.plot(t, state[:, i], label=f'q{i+1}')
        ax_pos.set_ylabel('Position (rad)')
        ax_pos.legend()
        ax_pos.grid(True)
        
        # Plot joint velocities
        ax_vel = axes[1]
        for i in range(6):
            ax_vel.plot(t, state[:, i+6], label=f'dq{i+1}')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Velocity (rad/s)')
        ax_vel.legend()
        ax_vel.grid(True)
        
        plt.tight_layout()
        return fig
    
    def visualize_robot(self, state: Union[np.ndarray, Dict], ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualize the robot configuration in 3D.
        
        Args:
            state: Either a state vector or simulation results dictionary
            ax: Optional existing matplotlib 3D axis to plot on
            
        Returns:
            Matplotlib 3D axis with the plotted robot
        """
        # If state is a simulation results dictionary, use the last state
        if isinstance(state, dict):
            state = state['state'][-1]
        
        # Compute forward kinematics
        kinematics = self.compute_forward_kinematics(state)
        
        # Create figure if not provided
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions
        p0, p1, p2, p3, p4, p5, p6 = kinematics['p0'], kinematics['p1'], kinematics['p2'], \
                                     kinematics['p3'], kinematics['p4'], kinematics['p5'], \
                                     kinematics['p6']
        
        # Convert to arrays for plotting
        x = [p0[0], p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]]
        y = [p0[1], p1[1], p2[1], p3[1], p4[1], p5[1], p6[1]]
        z = [p0[2], p1[2], p2[2], p3[2], p4[2], p5[2], p6[2]]
        
        # Plot links
        ax.plot(x, y, z, 'k-', linewidth=3)
        
        # Plot joints
        ax.scatter(x, y, z, c='red', s=100)
        
        # Set labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Manipulator Configuration')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set limits based on link lengths
        max_reach = sum([self.params[f'l{i}'] for i in range(1, 7)])
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([-max_reach/2, max_reach])
        
        return ax

# Example usage
if __name__ == "__main__":
    # Create simulator instance
    simulator = ManipulatorSimulator()
    
    # Set initial joint configuration
    initial_state = np.zeros(12)
    initial_state[0:6] = [0.3, 0.5, -0.2, 0.1, 0.0, 0.0]  # Joint positions
    
    # Define reference trajectory
    def reference_trajectory(t):
        """Simple sinusoidal reference for demonstration."""
        amplitude = 0.5
        frequency = 0.2  # Hz
        phase_shift = np.pi/3
        
        return {
            "q1_d": amplitude * np.sin(2*np.pi*frequency*t),
            "q2_d": amplitude * np.sin(2*np.pi*frequency*t + phase_shift),
            "q3_d": amplitude * np.sin(2*np.pi*frequency*t + 2*phase_shift)
        }
    
    # Run simulations with different controllers
    print("Running simulations...")
    
    # Joint position control
    sim_joint = simulator.simulate(
        controller_name='joint_position',
        T=10.0,
        dt=0.01,
        initial_state=initial_state,
        reference_trajectory=reference_trajectory
    )
    
    # Human collaboration control
    # Update proximity factor for demonstration
    simulator.params['human_proximity_factor'] = 0.3
    
    sim_human = simulator.simulate(
        controller_name='human_collaboration',
        T=10.0,
        dt=0.01,
        initial_state=initial_state,
        reference_trajectory=reference_trajectory
    )
    
    # Force control
    # Update force measurement for demonstration
    simulator.params['F_measured'] = 2.0
    
    sim_force = simulator.simulate(
        controller_name='force',
        T=10.0,
        dt=0.01,
        initial_state=initial_state,
        reference_trajectory=reference_trajectory
    )
    
    # Plot results
    print("Plotting results...")
    
    # Plot joint states for each controller
    fig_joint = simulator.plot_joint_states(sim_joint)
    fig_joint.suptitle('Joint Position Control')
    
    fig_human = simulator.plot_joint_states(sim_human)
    fig_human.suptitle('Human Collaboration Control')
    
    fig_force = simulator.plot_joint_states(sim_force)
    fig_force.suptitle('Force Control')
    
    # Plot barrier values for joint position control
    fig_barrier = simulator.plot_barriers(sim_joint)
    
    # Visualize final robot configuration
    fig_robot = plt.figure(figsize=(10, 8))
    ax = fig_robot.add_subplot(111, projection='3d')
    simulator.visualize_robot(sim_joint, ax)
    
    # Show all plots
    plt.show()
    
    print("Simulation complete.")
