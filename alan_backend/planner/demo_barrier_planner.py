"""
Demonstration of barrier-aware planning.

This script demonstrates how to use barrier certificates with the ALAN planner
to ensure safety constraints during planning. It creates a simple 2D navigation
problem with obstacles and shows how barrier certificates can be used to
ensure that the planned path avoids unsafe regions.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable

# Add parent directory to path if script is run directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from alan_backend.planner.goal_graph import GoalGraph, ConceptNode
from alan_backend.planner.barrier_planner import (
    BarrierGuard, SafetyAwareGoalGraph, create_goal_graph_with_safety, verify_plan_safety
)
from alan_backend.elfin.barrier.barrier_bridge_agent import BarrierBridgeAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alan.planner.demo_barrier_planner")


def create_2d_navigation_problem(
    obstacle_centers: List[np.ndarray],
    obstacle_radii: List[float],
    domain_bounds: Tuple[np.ndarray, np.ndarray]
) -> Tuple[BarrierBridgeAgent, str]:
    """
    Create a barrier-based safety system for 2D navigation with circular obstacles.
    
    Args:
        obstacle_centers: List of obstacle center points [(x1,y1), (x2,y2), ...]
        obstacle_radii: List of corresponding obstacle radii [r1, r2, ...]
        domain_bounds: Tuple of (lower_bounds, upper_bounds) for the domain
        
    Returns:
        Barrier bridge agent and system name
    """
    # Create agent
    agent = BarrierBridgeAgent(
        name="navigation_barrier",
        auto_verify=True
    )
    
    # System name
    system_name = "navigation_2d"
    
    # Generate safe and unsafe samples
    n_samples = 1000
    safe_samples = []
    unsafe_samples = []
    
    # Define unsafe region (inside any obstacle)
    def is_in_obstacle(x, y):
        for center, radius in zip(obstacle_centers, obstacle_radii):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < radius:
                return True
        return False
    
    # Sample points randomly from the domain
    lower, upper = domain_bounds
    x_min, y_min = lower[0], lower[1]
    x_max, y_max = upper[0], upper[1]
    
    for _ in range(n_samples * 2):  # Oversample to get enough points
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        
        point = np.array([x, y])
        if is_in_obstacle(x, y):
            unsafe_samples.append(point)
            if len(unsafe_samples) >= n_samples:
                break
        else:
            safe_samples.append(point)
            if len(safe_samples) >= n_samples:
                break
    
    # Convert to numpy arrays
    safe_samples = np.array(safe_samples[:n_samples])
    unsafe_samples = np.array(unsafe_samples[:n_samples])
    
    # Define barrier function (no dynamics for simple navigation)
    def safe_region(state):
        x, y = state
        return not is_in_obstacle(x, y)
    
    def unsafe_region(state):
        x, y = state
        return is_in_obstacle(x, y)
    
    # Learn barrier certificate
    barrier_fn = agent.learn_barrier(
        system_name=system_name,
        safe_samples=safe_samples,
        unsafe_samples=unsafe_samples,
        dictionary_type="rbf",
        dictionary_size=100,
        domain=domain_bounds,
        safe_region=safe_region,
        unsafe_region=unsafe_region,
        options={
            'safe_margin': 0.1,
            'unsafe_margin': 0.1
        }
    )
    
    return agent, system_name


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(a - b)


def get_2d_transition_model(
    step_size: float = 0.5,
    directions: int = 8
) -> Callable[[np.ndarray], List[Tuple[np.ndarray, str, float]]]:
    """
    Create a transition model for 2D grid navigation.
    
    Args:
        step_size: Size of each step
        directions: Number of possible directions (4 or 8)
        
    Returns:
        Transition model function
    """
    
    def transition_model(state: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """
        Generate possible transitions from a state.
        
        Args:
            state: Current state [x, y]
            
        Returns:
            List of (next_state, action, cost) tuples
        """
        x, y = state
        transitions = []
        
        # Define possible directions
        if directions == 4:
            # Four cardinal directions
            dirs = [
                (step_size, 0, "right"),
                (-step_size, 0, "left"),
                (0, step_size, "up"),
                (0, -step_size, "down")
            ]
        else:
            # Eight directions including diagonals
            dirs = [
                (step_size, 0, "right"),
                (-step_size, 0, "left"),
                (0, step_size, "up"),
                (0, -step_size, "down"),
                (step_size, step_size, "up-right"),
                (-step_size, step_size, "up-left"),
                (step_size, -step_size, "down-right"),
                (-step_size, -step_size, "down-left")
            ]
        
        # Create transitions
        for dx, dy, action in dirs:
            next_state = np.array([x + dx, y + dy])
            cost = np.sqrt(dx**2 + dy**2)  # Euclidean distance
            transitions.append((next_state, action, cost))
        
        return transitions
    
    return transition_model


def visualize_barriers_and_plan(
    barrier_agent: BarrierBridgeAgent,
    system_name: str,
    domain_bounds: Tuple[np.ndarray, np.ndarray],
    obstacle_centers: List[np.ndarray],
    obstacle_radii: List[float],
    plan: Optional[List[ConceptNode]] = None,
    start_state: Optional[np.ndarray] = None,
    goal_state: Optional[np.ndarray] = None,
    resolution: int = 100,
    title: str = "Barrier Function and Plan"
) -> None:
    """
    Visualize barrier function and plan.
    
    Args:
        barrier_agent: Barrier bridge agent
        system_name: System name
        domain_bounds: Domain bounds (lower, upper)
        obstacle_centers: List of obstacle centers
        obstacle_radii: List of obstacle radii
        plan: Optional plan from start to goal
        start_state: Optional start state
        goal_state: Optional goal state
        resolution: Resolution for visualization
        title: Title for the plot
    """
    # Extract domain bounds
    lower, upper = domain_bounds
    x_min, y_min = lower[0], lower[1]
    x_max, y_max = upper[0], upper[1]
    
    # Get barrier function
    barrier_fn = barrier_agent.results[system_name]['barrier']
    
    # Create grid for visualization
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate barrier function at each grid point
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j]])
            Z[i, j] = barrier_fn(state)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot barrier function as filled contour
    cmap = plt.cm.RdBu_r  # Red for unsafe (> 0), blue for safe (< 0)
    contour = plt.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=0.6)
    plt.colorbar(contour, label="Barrier Function Value")
    
    # Plot zero level set (boundary between safe and unsafe regions)
    boundary = plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=2)
    plt.clabel(boundary, fmt='B(x) = 0', fontsize=10)
    
    # Plot obstacles
    for center, radius in zip(obstacle_centers, obstacle_radii):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        plt.fill(x, y, 'r', alpha=0.3)
    
    # Plot start and goal points
    if start_state is not None:
        plt.plot(start_state[0], start_state[1], 'go', markersize=10, label='Start')
    
    if goal_state is not None:
        plt.plot(goal_state[0], goal_state[1], 'mo', markersize=10, label='Goal')
    
    # Plot plan
    if plan is not None:
        plan_x = [node.state[0] for node in plan]
        plan_y = [node.state[1] for node in plan]
        plt.plot(plan_x, plan_y, 'g-', linewidth=2, label='Plan')
        
        # Plot plan points
        plt.plot(plan_x, plan_y, 'g.', markersize=5)
    
    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.tight_layout()


def demo_2d_navigation_planning() -> None:
    """
    Demonstrate 2D navigation planning with barrier certificates.
    """
    print("\n=== 2D Navigation Planning with Barrier Certificates ===\n")
    
    # Define domain
    domain_bounds = (
        np.array([0.0, 0.0]),  # Lower bounds
        np.array([10.0, 10.0])  # Upper bounds
    )
    
    # Define obstacles
    obstacle_centers = [
        np.array([3.0, 3.0]),
        np.array([7.0, 7.0]),
        np.array([5.0, 5.0])
    ]
    obstacle_radii = [1.0, 1.0, 1.5]
    
    # Create barrier certificates
    print("Creating barrier certificates for navigation...")
    barrier_agent, system_name = create_2d_navigation_problem(
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        domain_bounds=domain_bounds
    )
    
    # Define start and goal
    start_state = np.array([1.0, 1.0])
    goal_state = np.array([9.0, 9.0])
    
    # Create transition model
    transition_model = get_2d_transition_model(step_size=0.5, directions=8)
    
    # Create safety-aware goal graph
    print("Creating safety-aware goal graph...")
    goal_graph = create_goal_graph_with_safety(
        distance_fn=euclidean_distance,
        transition_model=transition_model,
        barrier_agent=barrier_agent,
        system_name=system_name,
        margin=0.0,  # Barrier value must be <= 0 (safe region)
        strict=False  # Allow barrier value = 0 (boundary)
    )
    
    # Create start and goal nodes
    start_node = ConceptNode(id="start", state=start_state)
    goal_node = ConceptNode(id="goal", state=goal_state)
    
    # Set start and goal
    goal_graph.set_start_and_goal(start_node, goal_node)
    
    # Perform A* search
    print("Planning path from start to goal...")
    start_time = time.time()
    plan = goal_graph.a_star_search()
    planning_time = time.time() - start_time
    
    # Print planning results
    if plan:
        print(f"Plan found with {len(plan)} steps in {planning_time:.3f} seconds")
        
        # Verify plan safety
        is_safe = verify_plan_safety(
            plan=plan,
            barrier_agent=barrier_agent,
            system_name=system_name
        )
        print(f"Plan is safe: {is_safe}")
        
        # Print plan statistics
        stats = goal_graph.get_stats()
        print(f"Nodes explored: {stats['explored_nodes']}")
        print(f"Nodes expanded: {stats['expanded_nodes']}")
        print(f"Plan length: {stats['plan_length']}")
        print(f"Plan cost: {stats['plan_cost']:.2f}")
        
        # Print barrier guard statistics
        barrier_stats = stats.get('barrier_guard', {})
        print(f"Safety checks: {barrier_stats.get('total_checks', 0)}")
        print(f"Rejected transitions: {barrier_stats.get('rejected_transitions', 0)}")
        print(f"Rejection rate: {barrier_stats.get('rejection_rate', 0):.2%}")
    else:
        print("No plan found!")
    
    # Visualize barrier function and plan
    print("Visualizing barrier function and plan...")
    visualize_barriers_and_plan(
        barrier_agent=barrier_agent,
        system_name=system_name,
        domain_bounds=domain_bounds,
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        plan=plan,
        start_state=start_state,
        goal_state=goal_state,
        title="2D Navigation with Barrier Certificates"
    )
    plt.savefig("barrier_navigation_plan.png")
    
    print("\nDemonstration complete.")
    print("Saved visualization image: barrier_navigation_plan.png")
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    demo_2d_navigation_planning()
