"""
Demonstration of barrier certificates for safety verification.

This script demonstrates the use of barrier certificates for verifying safety
properties of dynamical systems. It creates a double integrator system with
a circular obstacle and learns a barrier certificate that ensures the system
will avoid the obstacle.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Add parent directory to path if script is run directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from alan_backend.elfin.barrier.barrier_bridge_agent import (
    create_double_integrator_agent, BarrierBridgeAgent, BarrierFunction
)

# Configure for visualization
try:
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (12, 8)
    matplotlib.rcParams['font.size'] = 12
except ImportError:
    print("Matplotlib not available, visualization will be limited")


def visualize_2d_barrier(
    barrier_fn: BarrierFunction,
    domain: Tuple[np.ndarray, np.ndarray],
    obstacle_center: np.ndarray = np.array([0.0, 0.0]),
    obstacle_radius: float = 1.0,
    resolution: int = 100,
    title: str = "Barrier Function",
    show_boundary: bool = True,
    fixed_velocities: Tuple[float, float] = (0.0, 0.0)
) -> None:
    """
    Visualize a 2D projection of the barrier function.
    
    Args:
        barrier_fn: Barrier function to visualize
        domain: Domain bounds (lower, upper)
        obstacle_center: Center of the obstacle (for display only)
        obstacle_radius: Radius of the obstacle (for display only)
        resolution: Resolution of the grid for visualization
        title: Title for the plot
        show_boundary: Whether to show the zero level set (boundary)
        fixed_velocities: Fixed velocity values for projection to position space
    """
    # Extract domain bounds for positions
    lower, upper = domain
    x_min, y_min = lower[0], lower[1]
    x_max, y_max = upper[0], upper[1]
    
    # Create grid of position points
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate barrier function at each point
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            # Create state with fixed velocities
            vx, vy = fixed_velocities
            state = np.array([X[i, j], Y[i, j], vx, vy])
            # Evaluate barrier function
            Z[i, j] = barrier_fn(state)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot barrier function as a filled contour
    cmap = plt.cm.RdBu_r  # Red for unsafe (> 0), blue for safe (< 0)
    contour = plt.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=0.8)
    plt.colorbar(contour, label="Barrier Function Value")
    
    # Plot zero level set (boundary between safe and unsafe regions)
    if show_boundary:
        boundary = plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=2)
        plt.clabel(boundary, fmt='B(x) = 0', fontsize=10)
    
    # Plot obstacle
    theta = np.linspace(0, 2 * np.pi, 100)
    obstacle_x = obstacle_center[0] + obstacle_radius * np.cos(theta)
    obstacle_y = obstacle_center[1] + obstacle_radius * np.sin(theta)
    plt.fill(obstacle_x, obstacle_y, 'r', alpha=0.3, label='Obstacle')
    
    # Set labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f"{title} (vx={fixed_velocities[0]}, vy={fixed_velocities[1]})")
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.tight_layout()


def simulate_system(
    barrier_fn: BarrierFunction,
    dynamics_fn: callable,
    initial_states: List[np.ndarray],
    t_max: float = 10.0,
    dt: float = 0.01,
    domain: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    obstacle_center: np.ndarray = np.array([0.0, 0.0]),
    obstacle_radius: float = 1.0
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """
    Simulate the system dynamics for multiple initial states.
    
    Args:
        barrier_fn: Barrier function for safety checking
        dynamics_fn: System dynamics function
        initial_states: List of initial states
        t_max: Maximum simulation time
        dt: Time step
        domain: Domain bounds (lower, upper)
        obstacle_center: Center of the obstacle (for display only)
        obstacle_radius: Radius of the obstacle (for display only)
        
    Returns:
        Time points and trajectories for each initial state
    """
    # Create time points
    t_points = np.arange(0, t_max, dt)
    n_steps = len(t_points)
    
    # Initialize trajectories
    trajectories = []
    
    # Simulate for each initial state
    for initial_state in initial_states:
        # Initialize trajectory
        trajectory = [initial_state.copy()]
        current_state = initial_state.copy()
        
        # Simulate
        for i in range(1, n_steps):
            # Compute derivative
            derivative = dynamics_fn(current_state)
            
            # Euler integration
            next_state = current_state + dt * derivative
            
            # Check if state is in domain
            if domain is not None:
                lower, upper = domain
                for j in range(len(next_state)):
                    next_state[j] = max(lower[j], min(upper[j], next_state[j]))
            
            # Store next state
            trajectory.append(next_state.copy())
            
            # Update current state
            current_state = next_state.copy()
        
        # Convert to numpy array
        trajectory = np.array(trajectory)
        trajectories.append(trajectory)
    
    return t_points, trajectories


def visualize_trajectories(
    t_points: np.ndarray,
    trajectories: List[np.ndarray],
    barrier_fn: BarrierFunction,
    domain: Tuple[np.ndarray, np.ndarray],
    obstacle_center: np.ndarray = np.array([0.0, 0.0]),
    obstacle_radius: float = 1.0,
    resolution: int = 100,
    title: str = "System Trajectories"
) -> None:
    """
    Visualize system trajectories along with the barrier function.
    
    Args:
        t_points: Time points
        trajectories: List of trajectories
        barrier_fn: Barrier function for coloring
        domain: Domain bounds (lower, upper)
        obstacle_center: Center of the obstacle (for display only)
        obstacle_radius: Radius of the obstacle (for display only)
        resolution: Resolution of the grid for visualization
        title: Title for the plot
    """
    # Extract domain bounds for positions
    lower, upper = domain
    x_min, y_min = lower[0], lower[1]
    x_max, y_max = upper[0], upper[1]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create grid of position points for barrier function visualization
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate barrier function at each point (with zero velocities)
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j], 0.0, 0.0])
            Z[i, j] = barrier_fn(state)
    
    # Plot barrier function as a filled contour in the first subplot
    cmap = plt.cm.RdBu_r  # Red for unsafe (> 0), blue for safe (< 0)
    contour = ax1.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=0.6)
    fig.colorbar(contour, ax=ax1, label="Barrier Function Value")
    
    # Plot zero level set (boundary between safe and unsafe regions)
    boundary = ax1.contour(X, Y, Z, levels=[0], colors='k', linewidths=2)
    ax1.clabel(boundary, fmt='B(x) = 0', fontsize=10)
    
    # Plot obstacle in both subplots
    theta = np.linspace(0, 2 * np.pi, 100)
    obstacle_x = obstacle_center[0] + obstacle_radius * np.cos(theta)
    obstacle_y = obstacle_center[1] + obstacle_radius * np.sin(theta)
    ax1.fill(obstacle_x, obstacle_y, 'r', alpha=0.3, label='Obstacle')
    ax2.fill(obstacle_x, obstacle_y, 'r', alpha=0.3, label='Obstacle')
    
    # Plot trajectories in both subplots
    for i, trajectory in enumerate(trajectories):
        # Position space
        ax1.plot(trajectory[:, 0], trajectory[:, 1], '-', linewidth=2, 
                 label=f'Trajectory {i+1}')
        
        # Phase space (position vs. velocity for x)
        ax2.plot(trajectory[:, 0], trajectory[:, 2], '-', linewidth=2,
                 label=f'Trajectory {i+1} (X)')
    
    # Set labels and title for first subplot (position space)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f"{title} - Position Space")
    ax1.legend()
    ax1.grid(True)
    
    # Set labels and title for second subplot (phase space)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('X Velocity')
    ax2.set_title(f"{title} - Phase Space")
    ax2.legend()
    ax2.grid(True)
    
    # Show plot
    plt.tight_layout()


def evaluate_barrier_along_trajectory(
    trajectory: np.ndarray,
    barrier_fn: BarrierFunction,
    t_points: np.ndarray,
    title: str = "Barrier Function Value Along Trajectory"
) -> None:
    """
    Evaluate and plot barrier function value along a trajectory.
    
    Args:
        trajectory: System trajectory
        barrier_fn: Barrier function
        t_points: Time points
        title: Title for the plot
    """
    # Evaluate barrier function at each point
    barrier_values = np.zeros(len(trajectory))
    for i, state in enumerate(trajectory):
        barrier_values[i] = barrier_fn(state)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot barrier values
    plt.plot(t_points, barrier_values, 'b-', linewidth=2)
    
    # Plot zero line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Highlight regions
    plt.fill_between(t_points, barrier_values, 0, 
                     where=(barrier_values >= 0),
                     interpolate=True, color='r', alpha=0.3, label='Unsafe Region')
    plt.fill_between(t_points, barrier_values, 0,
                     where=(barrier_values < 0),
                     interpolate=True, color='g', alpha=0.3, label='Safe Region')
    
    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Barrier Function Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.tight_layout()


def demo_barrier_certificate() -> None:
    """
    Demonstrate barrier certificates for safety verification.
    """
    print("\n=== Barrier Certificate Demonstration ===\n")
    
    # Create double integrator agent
    print("Creating double integrator agent...")
    agent, barrier_fn = create_double_integrator_agent(
        verify=True,
        dict_type="rbf",
        dict_size=100
    )
    
    # Get system information
    system_name = "double_integrator"
    result = agent.results[system_name]
    domain = result['domain']
    dynamics_fn = result['dynamics_fn']
    
    # Define obstacle properties
    obstacle_center = np.array([0.0, 0.0])
    obstacle_radius = 1.0
    
    # Print summary
    summary = agent.get_summary(system_name)
    print(f"\nSystem: {summary['system_name']}")
    print(f"Has barrier: {summary['has_barrier']}")
    print(f"Has verification: {summary['has_verification']}")
    
    if summary['has_verification']:
        verification = summary['verification']
        print(f"Verification success: {verification['success']}")
        print(f"Verification status: {verification['status']}")
        print(f"Verification method: {verification['method']}")
        print(f"Verification time: {verification['time']:.3f} seconds")
    
    # Verify the barrier certificate if not already verified
    if not summary.get('has_verification', False):
        print("\nVerifying barrier certificate...")
        verification_result = agent.verify(system_name)
        print(f"Verification success: {verification_result.success}")
        print(f"Verification status: {verification_result.status}")
    
    # If verification failed, try to refine the barrier certificate
    if summary.get('has_verification', False) and not summary['verification']['success']:
        print("\nRefinement needed. Automatically refining barrier certificate...")
        verification_result = agent.refine_auto(
            system_name=system_name,
            max_iterations=5
        )
        print(f"Verification after refinement: {verification_result.success}")
    
    # Visualize the barrier function
    print("\nVisualizing barrier function...")
    visualize_2d_barrier(
        barrier_fn=barrier_fn,
        domain=domain,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        title="Double Integrator Barrier Function",
        fixed_velocities=(0.0, 0.0)
    )
    plt.savefig("barrier_function.png")
    
    # Visualize the barrier function with different velocities
    visualize_2d_barrier(
        barrier_fn=barrier_fn,
        domain=domain,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        title="Double Integrator Barrier Function",
        fixed_velocities=(1.0, 0.0)
    )
    plt.savefig("barrier_function_with_velocity.png")
    
    # Generate initial states for simulation
    print("\nSimulating system dynamics...")
    initial_states = [
        np.array([-3.0, -3.0, 1.0, 1.0]),   # Bottom-left, moving toward obstacle
        np.array([3.0, -3.0, -1.0, 1.0]),   # Bottom-right, moving toward obstacle
        np.array([-3.0, 3.0, 1.0, -1.0]),   # Top-left, moving toward obstacle
        np.array([3.0, 3.0, -1.0, -1.0])    # Top-right, moving toward obstacle
    ]
    
    # Simulate system
    t_points, trajectories = simulate_system(
        barrier_fn=barrier_fn,
        dynamics_fn=dynamics_fn,
        initial_states=initial_states,
        domain=domain,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius
    )
    
    # Visualize trajectories
    visualize_trajectories(
        t_points=t_points,
        trajectories=trajectories,
        barrier_fn=barrier_fn,
        domain=domain,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        title="Double Integrator System"
    )
    plt.savefig("barrier_trajectories.png")
    
    # Evaluate barrier function along a trajectory
    evaluate_barrier_along_trajectory(
        trajectory=trajectories[0],
        barrier_fn=barrier_fn,
        t_points=t_points,
        title="Barrier Function Value Along Trajectory"
    )
    plt.savefig("barrier_trajectory_values.png")
    
    print("\nDemonstration complete.")
    print("Saved visualization images: barrier_function.png, barrier_trajectories.png, barrier_trajectory_values.png")
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    demo_barrier_certificate()
