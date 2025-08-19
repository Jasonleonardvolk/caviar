"""
Command-line interface for the ALAN planner.

This module provides command-line tools for running planning algorithms
with stability constraints provided by the ELFIN framework.
"""

import os
import sys
import argparse
import logging
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from alan_backend.elfin.koopman.koopman_bridge_agent import KoopmanBridgeAgent
from alan_backend.planner.goal_graph import GoalGraph, create_goal_graph_with_stability
from alan_backend.planner.guards import StabilityGuard
from alan_backend.elfin.cli import ELFINCli, get_cache_dir

# Configure logging
logger = logging.getLogger("alan.planner.cli")


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        a: First point
        b: Second point
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(a - b)


def create_grid_transition_model(grid_size: Tuple[int, int], obstacles: List[Tuple[int, int]] = None):
    """
    Create a transition model for a 2D grid world.
    
    Args:
        grid_size: Size of the grid (width, height)
        obstacles: List of obstacle positions
        
    Returns:
        Transition model function
    """
    obstacles = obstacles or []
    
    def transition_model(state):
        """
        Generate possible transitions from a state in grid world.
        
        Args:
            state: Current state
            
        Returns:
            List of (new_state, action, cost) tuples
        """
        # Possible moves: up, down, left, right
        moves = [
            (np.array([0, -1]), "up", 1.0),     # Up
            (np.array([0, 1]), "down", 1.0),    # Down
            (np.array([-1, 0]), "left", 1.0),   # Left
            (np.array([1, 0]), "right", 1.0)    # Right
        ]
        
        # Generate new states
        successors = []
        for delta, action, cost in moves:
            new_state = state + delta
            
            # Check if within bounds
            if (0 <= new_state[0] < grid_size[0]) and (0 <= new_state[1] < grid_size[1]):
                # Check if not an obstacle
                if tuple(new_state) not in obstacles:
                    successors.append((new_state, action, cost))
        
        return successors
    
    return transition_model


def add_planner_commands(cli: ELFINCli) -> None:
    """
    Add planner-related commands to the ELFIN CLI.
    
    Args:
        cli: ELFIN CLI instance
    """
    # Get the subparsers object
    parser = cli.parser
    subparsers = None
    
    # Find the subparsers object
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break
    
    if not subparsers:
        logger.error("Failed to find subparsers in ELFIN CLI")
        return
    
    # Add planner command
    planner_parser = subparsers.add_parser(
        "planner",
        help="Planning algorithms with stability constraints"
    )
    planner_subparsers = planner_parser.add_subparsers(
        dest="planner_command",
        help="Planner command to execute"
    )
    
    # Add planner run command
    run_parser = planner_subparsers.add_parser(
        "run",
        help="Run planner with stability constraints"
    )
    run_parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start state as comma-separated values"
    )
    run_parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="Goal state as comma-separated values"
    )
    run_parser.add_argument(
        "--grid-size",
        type=str,
        default="10,10",
        help="Grid size as width,height (default: 10,10)"
    )
    run_parser.add_argument(
        "--obstacles",
        type=str,
        default="",
        help="Comma-separated list of obstacle positions as x1,y1;x2,y2;..."
    )
    run_parser.add_argument(
        "--algorithm",
        type=str,
        choices=["astar", "bfs"],
        default="astar",
        help="Search algorithm to use (default: astar)"
    )
    run_parser.add_argument(
        "--stability-guard",
        action="store_true",
        help="Enable stability guard using Koopman Lyapunov function"
    )
    run_parser.add_argument(
        "--system",
        type=str,
        help="Koopman system name to use for stability guard"
    )
    run_parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Stability epsilon (tolerance) for Lyapunov value increase (default: 1e-3)"
    )
    run_parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict stability check (no Lyapunov value increase allowed)"
    )
    run_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the search progress and result"
    )
    run_parser.add_argument(
        "--output",
        type=str,
        help="Output file for search result (JSON)"
    )
    
    # Add planner demo command
    demo_parser = planner_subparsers.add_parser(
        "demo",
        help="Run a planner demo"
    )
    demo_parser.add_argument(
        "--scenario",
        type=str,
        choices=["grid", "pendulum", "custom"],
        default="grid",
        help="Demo scenario to run (default: grid)"
    )
    demo_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the demo"
    )
    
    # Update cli._handle_command to handle planner commands
    original_handle_command = cli._handle_command
    
    def new_handle_command(self, args: argparse.Namespace) -> int:
        """Handle commands."""
        if args.command == "planner":
            return _handle_planner_command(self, args)
        else:
            return original_handle_command(self, args)
    
    # Replace method
    cli._handle_command = new_handle_command.__get__(cli, type(cli))


def _handle_planner_command(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle planner commands.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    if not args.planner_command:
        logger.error("No planner command specified")
        return 1
        
    # Dispatch to planner command handler
    if args.planner_command == "run":
        return _handle_planner_run(cli, args)
    elif args.planner_command == "demo":
        return _handle_planner_demo(cli, args)
    else:
        logger.error(f"Unknown planner command: {args.planner_command}")
        return 1


def _handle_planner_run(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle planner run command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        # Parse start and goal states
        start_state = np.array([float(x) for x in args.start.split(",")])
        goal_state = np.array([float(x) for x in args.goal.split(",")])
        
        # Parse grid size
        grid_size = tuple(int(x) for x in args.grid_size.split(","))
        
        # Parse obstacles
        obstacles = []
        if args.obstacles:
            for obstacle in args.obstacles.split(";"):
                if obstacle:
                    obstacles.append(tuple(int(x) for x in obstacle.split(",")))
        
        # Create transition model
        transition_model = create_grid_transition_model(grid_size, obstacles)
        
        # Create goal graph
        if args.stability_guard:
            # Load Koopman bridge agent
            bridge_agent = KoopmanBridgeAgent(
                name=f"koopman_{args.system or 'default'}",
                cache_dir=cli.cache_dir
            )
            
            # Create goal graph with stability guard
            graph = create_goal_graph_with_stability(
                bridge_agent=bridge_agent,
                distance_fn=euclidean_distance,
                transition_model=transition_model,
                system_name=args.system,
                epsilon=args.epsilon,
                strict=args.strict
            )
            
            logger.info(f"Created goal graph with stability guard (system: {args.system}, epsilon: {args.epsilon})")
        else:
            # Create goal graph without stability guard
            graph = GoalGraph(
                distance_fn=euclidean_distance,
                transition_model=transition_model
            )
            
            logger.info("Created goal graph without stability guard")
        
        # Set start and goal
        graph.set_start(start_state)
        graph.set_goal(goal_state)
        
        # Run search algorithm
        start_time = time.time()
        if args.algorithm == "astar":
            path = graph.a_star_search()
        elif args.algorithm == "bfs":
            path = graph.breadth_first_search()
        else:
            logger.error(f"Unknown algorithm: {args.algorithm}")
            return 1
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Check if path was found
        if path is None:
            logger.error("No path found from start to goal")
            return 1
        
        # Get path states
        path_states = [node.state.tolist() for node in path]
        
        # Print statistics
        stats = graph.get_stats()
        logger.info(f"Search statistics: {stats}")
        logger.info(f"Path length: {len(path)}")
        logger.info(f"Search time: {elapsed_time:.3f} seconds")
        
        # Visualize result
        if args.visualize:
            try:
                _visualize_grid_search(
                    grid_size=grid_size,
                    obstacles=obstacles,
                    start=start_state,
                    goal=goal_state,
                    path=path_states
                )
            except Exception as e:
                logger.warning(f"Failed to visualize result: {e}")
        
        # Save result to file
        if args.output:
            try:
                result = {
                    "start": start_state.tolist(),
                    "goal": goal_state.tolist(),
                    "grid_size": grid_size,
                    "obstacles": obstacles,
                    "algorithm": args.algorithm,
                    "stability_guard": args.stability_guard,
                    "path": path_states,
                    "path_length": len(path),
                    "search_time": elapsed_time,
                    "stats": stats
                }
                
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Saved result to {args.output}")
            except Exception as e:
                logger.warning(f"Failed to save result: {e}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running planner: {e}", exc_info=True)
        return 1


def _handle_planner_demo(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle planner demo command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        if args.scenario == "grid":
            return _run_grid_demo(cli, args)
        elif args.scenario == "pendulum":
            return _run_pendulum_demo(cli, args)
        elif args.scenario == "custom":
            return _run_custom_demo(cli, args)
        else:
            logger.error(f"Unknown demo scenario: {args.scenario}")
            return 1
    
    except Exception as e:
        logger.error(f"Error running planner demo: {e}", exc_info=True)
        return 1


def _run_grid_demo(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Run grid world demo.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    # This demo creates a grid world with a "mountain" in the middle
    # represented by high Lyapunov values, and shows how the planner
    # avoids the mountain when stability guard is enabled
    
    try:
        from alan_backend.planner.tests.test_goal_graph import MockLyapunovFunction, MockKoopmanBridgeAgent
        
        # Create a grid world with a "mountain" in the middle
        grid_size = (5, 5)
        start_state = np.array([0.0, 0.0])
        goal_state = np.array([4.0, 4.0])
        
        # Create a mock Lyapunov function with a "mountain" in the middle
        lyap_values = {}
        
        # Base values (distance from goal)
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                dist = np.linalg.norm(np.array([x, y]) - goal_state)
                lyap_values[(float(x), float(y))] = dist
        
        # Add mountain in the middle (high Lyapunov values)
        mountain_center = (2, 2)
        mountain_radius = 1
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                dist_to_mountain = np.linalg.norm(np.array([x, y]) - np.array(mountain_center))
                if dist_to_mountain <= mountain_radius:
                    # Increase Lyapunov value for mountain area
                    lyap_values[(float(x), float(y))] += 10.0 * (mountain_radius - dist_to_mountain)
        
        # Create transition model
        transition_model = create_grid_transition_model(grid_size)
        
        # Create mock KoopmanBridgeAgent
        lyap_fn = MockLyapunovFunction(lyap_values)
        agent = MockKoopmanBridgeAgent({
            'demo': {
                'lyapunov': lyap_fn
            }
        })
        
        # Print demo information
        print("\n=== Grid World Demo with Stability Guard ===")
        print(f"Grid size: {grid_size}")
        print(f"Start: {start_state}")
        print(f"Goal: {goal_state}")
        print(f"Mountain center: {mountain_center} with radius {mountain_radius}")
        print("\nRunning search without stability guard...")
        
        # Run search without stability guard
        graph_without = GoalGraph(
            distance_fn=euclidean_distance,
            transition_model=transition_model
        )
        graph_without.set_start(start_state)
        graph_without.set_goal(goal_state)
        path_without = graph_without.a_star_search()
        
        if path_without:
            path_states_without = [node.state.tolist() for node in path_without]
            print(f"Path found with {len(path_without)} steps")
            print(f"Path: {path_states_without}")
            
            # Check if path goes through mountain
            goes_through_mountain = False
            for state in path_states_without:
                dist_to_mountain = np.linalg.norm(np.array(state) - np.array(mountain_center))
                if dist_to_mountain <= mountain_radius:
                    goes_through_mountain = True
                    break
            
            print(f"Path goes through mountain: {goes_through_mountain}")
        else:
            print("No path found")
        
        print("\nRunning search with stability guard...")
        
        # Run search with stability guard
        stability_guard = StabilityGuard(
            bridge_agent=agent,
            system_name='demo',
            epsilon=0.1
        )
        graph_with = GoalGraph(
            distance_fn=euclidean_distance,
            transition_model=transition_model,
            stability_guard=stability_guard
        )
        graph_with.set_start(start_state)
        graph_with.set_goal(goal_state)
        path_with = graph_with.a_star_search()
        
        if path_with:
            path_states_with = [node.state.tolist() for node in path_with]
            print(f"Path found with {len(path_with)} steps")
            print(f"Path: {path_states_with}")
            
            # Check if path goes through mountain
            goes_through_mountain = False
            for state in path_states_with:
                dist_to_mountain = np.linalg.norm(np.array(state) - np.array(mountain_center))
                if dist_to_mountain <= mountain_radius:
                    goes_through_mountain = True
                    break
            
            print(f"Path goes through mountain: {goes_through_mountain}")
            
            # Print statistics
            stats = graph_with.get_stats()
            print(f"Pruned transitions: {stats['pruned_transitions']}")
        else:
            print("No path found")
        
        # Visualize result
        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Plot grid without stability guard
                ax1.set_title("Without Stability Guard")
                _plot_grid(ax1, grid_size, [], start_state, goal_state, path_states_without if path_without else None)
                
                # Plot mountain
                mountain_x, mountain_y = mountain_center
                circle = plt.Circle((mountain_x, mountain_y), mountain_radius, color='orange', alpha=0.5)
                ax1.add_patch(circle)
                
                # Plot grid with stability guard
                ax2.set_title("With Stability Guard")
                _plot_grid(ax2, grid_size, [], start_state, goal_state, path_states_with if path_with else None)
                
                # Plot mountain
                circle = plt.Circle((mountain_x, mountain_y), mountain_radius, color='orange', alpha=0.5)
                ax2.add_patch(circle)
                
                # Plot Lyapunov values as a heatmap
                lyap_grid = np.zeros(grid_size)
                for x in range(grid_size[0]):
                    for y in range(grid_size[1]):
                        lyap_grid[y, x] = lyap_values.get((float(x), float(y)), 0.0)
                
                fig2, ax3 = plt.subplots(figsize=(8, 6))
                ax3.set_title("Lyapunov Values (Stability Landscape)")
                im = ax3.imshow(lyap_grid, origin='lower', cmap='viridis')
                fig2.colorbar(im, ax=ax3)
                
                # Show plots
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logger.warning(f"Failed to visualize result: {e}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running grid demo: {e}", exc_info=True)
        return 1


def _run_pendulum_demo(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Run pendulum demo using actual Koopman Lyapunov function.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        from alan_backend.elfin.koopman.koopman_bridge_agent import create_pendulum_agent
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Create pendulum agent
        agent, lyap_fn = create_pendulum_agent(verify=True)
        
        # Define grid bounds for pendulum state space
        x_min, x_max = -np.pi, np.pi
        y_min, y_max = -2.0, 2.0
        grid_size = (20, 20)
        
        # Create transition model for pendulum state space
        def pendulum_transition_model(state):
            # Simple discretization of pendulum dynamics
            # theta, omega = state
            x, y = state
            
            # Possible moves: small adjustments in each direction
            deltas = [
                np.array([0.2, 0.0]),    # Increase angle
                np.array([-0.2, 0.0]),   # Decrease angle
                np.array([0.0, 0.2]),    # Increase angular velocity
                np.array([0.0, -0.2])    # Decrease angular velocity
            ]
            
            actions = ["inc_angle", "dec_angle", "inc_vel", "dec_vel"]
            
            # Generate new states
            successors = []
            for i, delta in enumerate(deltas):
                new_state = state + delta
                
                # Check if within bounds
                if (x_min <= new_state[0] <= x_max) and (y_min <= new_state[1] <= y_max):
                    # Compute cost based on Euclidean distance
                    cost = np.linalg.norm(delta)
                    successors.append((new_state, actions[i], cost))
            
            return successors
        
        # Define start and goal states
        start_state = np.array([np.pi/2, 0.0])   # 90 degrees, no velocity
        goal_state = np.array([0.0, 0.0])        # Upright position
        
        # Print demo information
        print("\n=== Pendulum Demo with Stability Guard ===")
        print(f"Start: theta={start_state[0]:.2f} rad, omega={start_state[1]:.2f} rad/s")
        print(f"Goal: theta={goal_state[0]:.2f} rad, omega={goal_state[1]:.2f} rad/s")
        print("\nRunning search without stability guard...")
        
        # Run search without stability guard
        graph_without = GoalGraph(
            distance_fn=euclidean_distance,
            transition_model=pendulum_transition_model
        )
        graph_without.set_start(start_state)
        graph_without.set_goal(goal_state)
        path_without = graph_without.a_star_search()
        
        if path_without:
            path_states_without = [node.state.tolist() for node in path_without]
            print(f"Path found with {len(path_without)} steps")
        else:
            print("No path found")
        
        print("\nRunning search with stability guard...")
        
        # Run search with stability guard
        stability_guard = StabilityGuard(
            bridge_agent=agent,
            system_name='pendulum',
            epsilon=0.1
        )
        graph_with = GoalGraph(
            distance_fn=euclidean_distance,
            transition_model=pendulum_transition_model,
            stability_guard=stability_guard
        )
        graph_with.set_start(start_state)
        graph_with.set_goal(goal_state)
        path_with = graph_with.a_star_search()
        
        if path_with:
            path_states_with = [node.state.tolist() for node in path_with]
            print(f"Path found with {len(path_with)} steps")
            
            # Print statistics
            stats = graph_with.get_stats()
            print(f"Pruned transitions: {stats['pruned_transitions']}")
        else:
            print("No path found")
        
        # Visualize result
        if args.visualize:
            try:
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Plot pendulum phase space without stability guard
                ax1.set_title("Without Stability Guard")
                ax1.set_xlabel("Angle (rad)")
                ax1.set_ylabel("Angular Velocity (rad/s)")
                ax1.set_xlim(x_min, x_max)
                ax1.set_ylim(y_min, y_max)
                
                # Plot start and goal
                ax1.plot(start_state[0], start_state[1], 'go', markersize=10, label='Start')
                ax1.plot(goal_state[0], goal_state[1], 'ro', markersize=10, label='Goal')
                
                # Plot path without stability guard
                if path_without:
                    path_x = [state[0] for state in path_states_without]
                    path_y = [state[1] for state in path_states_without]
                    ax1.plot(path_x, path_y, 'b-', linewidth=2)
                    ax1.scatter(path_x, path_y, color='blue', s=30)
                
                ax1.legend()
                
                # Plot pendulum phase space with stability guard
                ax2.set_title("With Stability Guard")
                ax2.set_xlabel("Angle (rad)")
                ax2.set_ylabel("Angular Velocity (rad/s)")
                ax2.set_xlim(x_min, x_max)
                ax2.set_ylim(y_min, y_max)
                
                # Plot start and goal
                ax2.plot(start_state[0], start_state[1], 'go', markersize=10, label='Start')
                ax2.plot(goal_state[0], goal_state[1], 'ro', markersize=10, label='Goal')
                
                # Plot path with stability guard
                if path_with:
                    path_x = [state[0] for state in path_states_with]
                    path_y = [state[1] for state in path_states_with]
                    ax2.plot(path_x, path_y, 'b-', linewidth=2)
                    ax2.scatter(path_x, path_y, color='blue', s=30)
                
                ax2.legend()
                
                # Plot Lyapunov level sets
                resolution = 100
                x = np.linspace(x_min, x_max, resolution)
                y = np.linspace(y_min, y_max, resolution)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                
                for i in range(resolution):
                    for j in range(resolution):
                        state = np.array([X[i, j], Y[i, j]])
                        Z[i, j] = lyap_fn(state)
                
                fig2, ax3 = plt.subplots(figsize=(8, 6))
                ax3.set_title("Pendulum Lyapunov Function (Stability Landscape)")
                ax3.set_xlabel("Angle (rad)")
                ax3.set_ylabel("Angular Velocity (rad/s)")
                
                # Plot contour
                contour = ax3.contourf(X, Y, Z, cmap='viridis', levels=20)
                plt.colorbar(contour, ax=ax3)
                
                # Plot paths on Lyapunov landscape
                if path_without:
                    path_x = [state[0] for state in path_states_without]
                    path_y = [state[1] for state in path_states_without]
                    ax3.plot(path_x, path_y, 'r-', linewidth=2, label='Without Guard')
                
                if path_with:
                    path_x = [state[0] for state in path_states_with]
                    path_y = [state[1] for state in path_states_with]
                    ax3.plot(path_x, path_y, 'b-', linewidth=2, label='With Guard')
                
                ax3.legend()
                
                # Show plots
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logger.warning(f"Failed to visualize result: {e}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running pendulum demo: {e}", exc_info=True)
        return 1


def _run_custom_demo(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Run custom demo.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    logger.error("Custom demo not implemented yet")
    return 1


def _visualize_grid_search(grid_size, obstacles, start, goal, path=None):
    """
    Visualize grid search result.
    
    Args:
        grid_size: Size of the grid (width, height)
        obstacles: List of obstacle positions
        start: Start state
        goal: Goal state
        path: Path as list of states
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Grid Search Result")
        
        _plot_grid(ax, grid_size, obstacles, start, goal, path)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error visualizing grid search: {e}")


def _plot_grid(ax, grid_size, obstacles, start, goal, path=None):
    """
    Plot grid world.
    
    Args:
        ax: Matplotlib axis
        grid_size: Size of the grid (width, height)
        obstacles: List of obstacle positions
        start: Start state
        goal: Goal state
        path: Path as list of states
    """
    # Draw grid lines
    for i in range(grid_size[0] + 1):
        ax.axvline(i, color='gray', linestyle='-', alpha=0.5)
    for i in range(grid_size[1] + 1):
        ax.axhline(i, color='gray', linestyle='-', alpha=0.5)
    
    # Draw obstacles
    for obstacle in obstacles:
        rect = plt.Rectangle((obstacle[0], obstacle[1]), 1, 1, color='black')
        ax.add_patch(rect)
    
    # Draw start and goal
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Draw path
    if path:
        path_x = [state[0] for state in path]
        path_y = [state[1] for state in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2)
        ax.scatter(path_x, path_y, color='blue', s=30)
    
    # Set limits and labels
    ax.set_xlim(-0.5, grid_size[0] - 0.5)
    ax.set_ylim(-0.5, grid_size[1] - 0.5)
    ax.set_xticks(range(grid_size[0]))
    ax.set_yticks(range(grid_size[1]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()


def init_module():
    """Initialize the Planner CLI module."""
    # Register with ELFIN CLI
    try:
        from alan_backend.elfin.cli import ELFINCli
        
        # Create a dummy CLI instance for registration
        dummy_cli = ELFINCli()
        
        # Add Planner commands
        add_planner_commands(dummy_cli)
        
        logger.info("Planner commands registered with ELFIN CLI")
        
    except ImportError:
        logger.warning("ELFIN CLI not found, Planner commands not registered")


# Initialize when imported
init_module()
