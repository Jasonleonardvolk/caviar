"""
Data Generator for Neural Barrier and Lyapunov Training

This module provides utilities for generating synthetic data for training
neural barrier and Lyapunov networks.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Callable, Sequence
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class DataGenerator:
    """
    Generate synthetic data for training barrier and Lyapunov functions.
    
    This class provides methods for generating:
    1. Safe and unsafe states for barrier functions
    2. Stable and unstable states for Lyapunov functions
    3. Trajectories for dynamic verification
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        dynamics_fn: Optional[Callable] = None,
        controller_fn: Optional[Callable] = None,
        state_bounds: Optional[np.ndarray] = None,
        input_bounds: Optional[np.ndarray] = None,
        safe_fn: Optional[Callable] = None,
        seed: int = 42
    ):
        """
        Initialize a data generator.
        
        Args:
            state_dim: Dimension of the state space
            input_dim: Dimension of the input space (0 for autonomous systems)
            dynamics_fn: Function mapping (state, input) to state derivative
            controller_fn: Function mapping state to control input
            state_bounds: Bounds of the state space, shape (state_dim, 2)
            input_bounds: Bounds of the input space, shape (input_dim, 2)
            safe_fn: Function that determines if a state is safe
            seed: Random seed for reproducibility
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dynamics_fn = dynamics_fn
        self.controller_fn = controller_fn
        
        # Set default bounds if not provided
        if state_bounds is None:
            self.state_bounds = np.array([[-10.0, 10.0]] * state_dim)
        else:
            self.state_bounds = state_bounds
            
        if input_bounds is None and input_dim > 0:
            self.input_bounds = np.array([[-10.0, 10.0]] * input_dim)
        else:
            self.input_bounds = input_bounds
        
        self.safe_fn = safe_fn
        np.random.seed(seed)
    
    def generate_uniform_states(
        self,
        num_samples: int,
        bounds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate uniformly distributed states in the specified bounds.
        
        Args:
            num_samples: Number of samples to generate
            bounds: Bounds of the state space, shape (state_dim, 2)
            
        Returns:
            Array of states, shape (num_samples, state_dim)
        """
        if bounds is None:
            bounds = self.state_bounds
        
        states = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(num_samples, bounds.shape[0])
        )
        
        return states
    
    def generate_safe_unsafe_data(
        self,
        num_samples: int,
        safe_ratio: float = 0.5,
        balance: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate safe and unsafe states for barrier function training.
        
        Args:
            num_samples: Total number of samples to generate
            safe_ratio: Ratio of safe samples (if balance=False)
            balance: Whether to balance the number of safe/unsafe samples
            
        Returns:
            Tuple of (states, labels) where labels are 1 for safe, 0 for unsafe
        """
        if self.safe_fn is None:
            raise ValueError("Safe function (safe_fn) must be provided to generate safe/unsafe data")
        
        if balance:
            # Generate samples until we have enough safe and unsafe states
            num_safe = num_samples // 2
            num_unsafe = num_samples - num_safe
            
            all_states = []
            all_labels = []
            
            safe_count = 0
            unsafe_count = 0
            
            # Generate batches for efficiency
            batch_size = 1000
            max_iterations = 1000
            iterations = 0
            
            while (safe_count < num_safe or unsafe_count < num_unsafe) and iterations < max_iterations:
                # Generate a batch of random states
                states_batch = self.generate_uniform_states(batch_size)
                
                # Evaluate safety function
                safety = np.array([self.safe_fn(state) for state in states_batch])
                
                # Select safe states if needed
                if safe_count < num_safe:
                    safe_indices = np.where(safety)[0]
                    safe_to_take = min(len(safe_indices), num_safe - safe_count)
                    
                    if safe_to_take > 0:
                        all_states.append(states_batch[safe_indices[:safe_to_take]])
                        all_labels.append(np.ones(safe_to_take))
                        safe_count += safe_to_take
                
                # Select unsafe states if needed
                if unsafe_count < num_unsafe:
                    unsafe_indices = np.where(~safety)[0]
                    unsafe_to_take = min(len(unsafe_indices), num_unsafe - unsafe_count)
                    
                    if unsafe_to_take > 0:
                        all_states.append(states_batch[unsafe_indices[:unsafe_to_take]])
                        all_labels.append(np.zeros(unsafe_to_take))
                        unsafe_count += unsafe_to_take
                
                iterations += 1
            
            if iterations == max_iterations:
                print(f"Warning: Reached maximum iterations with {safe_count} safe and {unsafe_count} unsafe samples")
            
            # Combine and shuffle data
            states = np.vstack(all_states)
            labels = np.concatenate(all_labels)
            
            # Shuffle data
            shuffle_indices = np.random.permutation(len(states))
            states = states[shuffle_indices]
            labels = labels[shuffle_indices]
            
        else:
            # Generate without balancing
            states = self.generate_uniform_states(num_samples)
            
            # Evaluate safety function
            labels = np.array([1.0 if self.safe_fn(state) else 0.0 for state in states])
        
        return states, labels.reshape(-1, 1)
    
    def generate_trajectories(
        self,
        num_trajectories: int,
        T: float,
        dt: float,
        initial_states: Optional[np.ndarray] = None,
        controller: Optional[Callable] = None,
        noise_level: float = 0.0
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate system trajectories by simulating dynamics.
        
        Args:
            num_trajectories: Number of trajectories to generate
            T: Simulation time horizon
            dt: Time step for simulation
            initial_states: Initial states for trajectories
            controller: Controller function (overrides self.controller_fn)
            noise_level: Standard deviation of Gaussian noise added to dynamics
            
        Returns:
            List of trajectory dictionaries with keys:
            - 'states': Array of states, shape (num_steps, state_dim)
            - 'inputs': Array of inputs, shape (num_steps, input_dim)
            - 'times': Array of times, shape (num_steps,)
        """
        if self.dynamics_fn is None:
            raise ValueError("Dynamics function (dynamics_fn) must be provided to generate trajectories")
        
        # Use provided controller or default
        ctrl_fn = controller if controller is not None else self.controller_fn
        
        # Generate initial states if not provided
        if initial_states is None:
            initial_states = self.generate_uniform_states(num_trajectories)
        elif len(initial_states) < num_trajectories:
            # Generate additional initial states if needed
            additional_states = self.generate_uniform_states(
                num_trajectories - len(initial_states)
            )
            initial_states = np.vstack([initial_states, additional_states])
        
        # Initialize list of trajectories
        trajectories = []
        
        # Number of steps in each trajectory
        num_steps = int(T / dt) + 1
        
        # Generate each trajectory
        for i in range(num_trajectories):
            # Initialize trajectory data
            states = np.zeros((num_steps, self.state_dim))
            times = np.linspace(0, T, num_steps)
            
            # Set initial state
            states[0] = initial_states[i]
            
            # Initialize inputs if using a controller
            if self.input_dim > 0 and ctrl_fn is not None:
                inputs = np.zeros((num_steps, self.input_dim))
                inputs[0] = ctrl_fn(states[0])
            else:
                inputs = None
            
            # Simulate trajectory
            for j in range(1, num_steps):
                # Get current state
                current_state = states[j-1]
                
                # Compute control input if using a controller
                if self.input_dim > 0 and ctrl_fn is not None:
                    current_input = ctrl_fn(current_state)
                    inputs[j-1] = current_input
                else:
                    current_input = None
                
                # Compute dynamics
                if current_input is not None:
                    derivative = self.dynamics_fn(current_state, current_input)
                else:
                    derivative = self.dynamics_fn(current_state, None)
                
                # Add noise if specified
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, size=self.state_dim)
                    derivative += noise
                
                # Euler integration
                states[j] = current_state + dt * derivative
            
            # Compute final input if using a controller
            if self.input_dim > 0 and ctrl_fn is not None:
                inputs[-1] = ctrl_fn(states[-1])
            
            # Store trajectory
            trajectory = {
                'states': states,
                'times': times
            }
            
            if inputs is not None:
                trajectory['inputs'] = inputs
                
            trajectories.append(trajectory)
        
        return trajectories
    
    def visualize_safe_unsafe(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        dims: Tuple[int, int] = (0, 1),
        figsize: Tuple[int, int] = (10, 8),
        alpha: float = 0.6,
        s: int = 20
    ) -> plt.Figure:
        """
        Visualize safe and unsafe states in a 2D plot.
        
        Args:
            states: Array of states, shape (num_samples, state_dim)
            labels: Array of labels (1 for safe, 0 for unsafe), shape (num_samples, 1)
            dims: Dimensions to plot
            figsize: Figure size
            alpha: Alpha value for scatter plots
            s: Size of scatter points
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract dimensions to plot
        x_dim, y_dim = dims
        
        # Split states into safe and unsafe
        safe_indices = np.where(labels > 0.5)[0]
        unsafe_indices = np.where(labels <= 0.5)[0]
        
        safe_states = states[safe_indices]
        unsafe_states = states[unsafe_indices]
        
        # Plot safe states in green
        if len(safe_states) > 0:
            ax.scatter(
                safe_states[:, x_dim],
                safe_states[:, y_dim],
                c='green',
                label='Safe',
                alpha=alpha,
                s=s
            )
        
        # Plot unsafe states in red
        if len(unsafe_states) > 0:
            ax.scatter(
                unsafe_states[:, x_dim],
                unsafe_states[:, y_dim],
                c='red',
                label='Unsafe',
                alpha=alpha,
                s=s
            )
        
        # Set plot properties
        ax.set_xlabel(f'Dimension {x_dim}')
        ax.set_ylabel(f'Dimension {y_dim}')
        ax.set_title('Safe and Unsafe States')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_trajectory(
        self,
        trajectory: Dict[str, np.ndarray],
        dims: Tuple[int, int] = (0, 1),
        figsize: Tuple[int, int] = (10, 8),
        plot_derivatives: bool = False,
        safe_fn: Optional[Callable] = None
    ) -> plt.Figure:
        """
        Visualize a trajectory in a 2D plot.
        
        Args:
            trajectory: Trajectory dictionary with 'states', 'times', and optionally 'inputs'
            dims: Dimensions to plot
            figsize: Figure size
            plot_derivatives: Whether to plot state derivatives as a quiver plot
            safe_fn: Function to determine if states are safe (for coloring)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract dimensions to plot
        x_dim, y_dim = dims
        
        # Extract trajectory data
        states = trajectory['states']
        times = trajectory['times']
        
        # Color trajectory points by safety if safe_fn is provided
        if safe_fn is not None:
            safety = np.array([safe_fn(state) for state in states])
            scatter = ax.scatter(
                states[:, x_dim],
                states[:, y_dim],
                c=safety,
                cmap='RdYlGn',
                s=30,
                alpha=0.8
            )
            plt.colorbar(scatter, label='Safety')
        else:
            # Color by time
            scatter = ax.scatter(
                states[:, x_dim],
                states[:, y_dim],
                c=times,
                cmap='viridis',
                s=30,
                alpha=0.8
            )
            plt.colorbar(scatter, label='Time')
        
        # Plot trajectory line
        ax.plot(states[:, x_dim], states[:, y_dim], 'k-', alpha=0.5)
        
        # Mark start and end points
        ax.plot(states[0, x_dim], states[0, y_dim], 'bo', markersize=10, label='Start')
        ax.plot(states[-1, x_dim], states[-1, y_dim], 'ro', markersize=10, label='End')
        
        # Plot derivatives if requested
        if plot_derivatives and self.dynamics_fn is not None:
            # Subsample for clarity
            stride = max(1, len(states) // 20)
            derivative_indices = range(0, len(states), stride)
            derivative_states = states[derivative_indices]
            
            # Compute derivatives
            derivatives = np.zeros_like(derivative_states)
            for i, state in enumerate(derivative_states):
                if 'inputs' in trajectory:
                    derivative = self.dynamics_fn(state, trajectory['inputs'][derivative_indices[i]])
                else:
                    derivative = self.dynamics_fn(state, None)
                derivatives[i] = derivative
            
            # Plot quiver
            ax.quiver(
                derivative_states[:, x_dim],
                derivative_states[:, y_dim],
                derivatives[:, x_dim],
                derivatives[:, y_dim],
                angles='xy',
                scale_units='xy',
                scale=0.5,
                width=0.005,
                color='blue',
                alpha=0.7
            )
        
        # Set plot properties
        ax.set_xlabel(f'Dimension {x_dim}')
        ax.set_ylabel(f'Dimension {y_dim}')
        ax.set_title('Trajectory Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class ObstacleDataGenerator(DataGenerator):
    """
    Data generator specialized for systems with obstacles.
    
    This generator creates barrier function data for obstacle avoidance problems,
    assuming a safety condition based on distance to obstacles.
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        dynamics_fn: Optional[Callable] = None,
        controller_fn: Optional[Callable] = None,
        state_bounds: Optional[np.ndarray] = None,
        input_bounds: Optional[np.ndarray] = None,
        obstacles: List[Dict[str, Any]] = None,
        robot_radius: float = 0.2,
        safety_margin: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize an obstacle data generator.
        
        Args:
            state_dim: Dimension of the state space
            input_dim: Dimension of the input space
            dynamics_fn: Function mapping (state, input) to state derivative
            controller_fn: Function mapping state to control input
            state_bounds: Bounds of the state space, shape (state_dim, 2)
            input_bounds: Bounds of the input space, shape (input_dim, 2)
            obstacles: List of obstacle dictionaries with keys:
                - 'type': Type of obstacle ('circle', 'rectangle', etc.)
                - 'position': Position of obstacle
                - 'radius': Radius for circular obstacles
                - 'width', 'height': Dimensions for rectangular obstacles
            robot_radius: Radius of the robot
            safety_margin: Additional safety margin beyond collision distance
            seed: Random seed for reproducibility
        """
        # Define safety function based on obstacles
        def obstacle_safe_fn(state):
            # Extract position (assuming first 2 dimensions are position)
            position = state[0:2]
            
            # Check distance to each obstacle
            for obstacle in obstacles:
                if obstacle['type'] == 'circle':
                    # Compute distance to circular obstacle
                    distance = np.linalg.norm(position - obstacle['position'])
                    safe_distance = obstacle['radius'] + robot_radius + safety_margin
                    
                    if distance < safe_distance:
                        return False
                
                elif obstacle['type'] == 'rectangle':
                    # Compute distance to rectangular obstacle
                    # (This is a simplified calculation using the center of the rectangle)
                    rect_center = obstacle['position']
                    rect_width = obstacle['width']
                    rect_height = obstacle['height']
                    
                    # Compute closest point on rectangle to position
                    closest_x = max(rect_center[0] - rect_width/2, 
                                   min(position[0], rect_center[0] + rect_width/2))
                    closest_y = max(rect_center[1] - rect_height/2,
                                   min(position[1], rect_center[1] + rect_height/2))
                    
                    closest_point = np.array([closest_x, closest_y])
                    distance = np.linalg.norm(position - closest_point)
                    
                    if distance < robot_radius + safety_margin:
                        return False
            
            # If we get here, the state is safe with respect to all obstacles
            return True
        
        # Initialize parent class
        super(ObstacleDataGenerator, self).__init__(
            state_dim=state_dim,
            input_dim=input_dim,
            dynamics_fn=dynamics_fn,
            controller_fn=controller_fn,
            state_bounds=state_bounds,
            input_bounds=input_bounds,
            safe_fn=obstacle_safe_fn,
            seed=seed
        )
        
        # Store obstacle data
        self.obstacles = obstacles or []
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
    
    def visualize_environment(
        self,
        figsize: Tuple[int, int] = (10, 8),
        states: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """
        Visualize the environment with obstacles and optionally states.
        
        Args:
            figsize: Figure size
            states: Optional array of states to plot
            labels: Optional array of labels for states
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot obstacles
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                circle = plt.Circle(
                    obstacle['position'],
                    obstacle['radius'],
                    color='red',
                    alpha=0.5
                )
                ax.add_patch(circle)
                
                # Plot safety region
                safety_circle = plt.Circle(
                    obstacle['position'],
                    obstacle['radius'] + self.robot_radius + self.safety_margin,
                    fill=False,
                    color='red',
                    linestyle='--',
                    alpha=0.3
                )
                ax.add_patch(safety_circle)
            
            elif obstacle['type'] == 'rectangle':
                rect_center = obstacle['position']
                rect_width = obstacle['width']
                rect_height = obstacle['height']
                
                rect = plt.Rectangle(
                    (rect_center[0] - rect_width/2, rect_center[1] - rect_height/2),
                    rect_width,
                    rect_height,
                    color='red',
                    alpha=0.5
                )
                ax.add_patch(rect)
                
                # Plot safety region
                safety_rect = plt.Rectangle(
                    (rect_center[0] - rect_width/2 - self.robot_radius - self.safety_margin,
                     rect_center[1] - rect_height/2 - self.robot_radius - self.safety_margin),
                    rect_width + 2 * (self.robot_radius + self.safety_margin),
                    rect_height + 2 * (self.robot_radius + self.safety_margin),
                    fill=False,
                    color='red',
                    linestyle='--',
                    alpha=0.3
                )
                ax.add_patch(safety_rect)
        
        # Plot states if provided
        if states is not None and labels is not None:
            # Split states into safe and unsafe
            safe_indices = np.where(labels > 0.5)[0]
            unsafe_indices = np.where(labels <= 0.5)[0]
            
            safe_states = states[safe_indices]
            unsafe_states = states[unsafe_indices]
            
            # Plot safe states in green
            if len(safe_states) > 0:
                ax.scatter(
                    safe_states[:, 0],
                    safe_states[:, 1],
                    c='green',
                    label='Safe',
                    alpha=0.6,
                    s=20
                )
            
            # Plot unsafe states in red
            if len(unsafe_states) > 0:
                ax.scatter(
                    unsafe_states[:, 0],
                    unsafe_states[:, 1],
                    c='red',
                    label='Unsafe',
                    alpha=0.6,
                    s=20
                )
        
        # Plot state bounds if available
        if self.state_bounds is not None:
            x_min, x_max = self.state_bounds[0]
            y_min, y_max = self.state_bounds[1]
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Environment with Obstacles')
        ax.grid(True, alpha=0.3)
        if states is not None and labels is not None:
            ax.legend()
        
        return fig
