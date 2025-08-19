"""
Unit tests for the planner goal graph with stability guards.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add the parent directory to the path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from alan_backend.planner.goal_graph import GoalGraph, ConceptNode, create_goal_graph_with_stability
from alan_backend.planner.guards import StabilityGuard
from alan_backend.planner.tests.test_guards import MockLyapunovFunction, MockKoopmanBridgeAgent


class TestGoalGraph(unittest.TestCase):
    """Test cases for goal graph with stability guards."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a simple 2D grid world for testing:
        # [0,0] [1,0] [2,0]
        # [0,1] [1,1] [2,1]
        # [0,2] [1,2] [2,2]
        
        # Euclidean distance function
        self.distance_fn = lambda a, b: np.linalg.norm(a - b)
        
        # Simple transition model with 4-connectivity (up, down, left, right)
        def transition_model(state):
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
                if (0 <= new_state[0] <= 2) and (0 <= new_state[1] <= 2):
                    successors.append((new_state, action, cost))
            
            return successors
        
        self.transition_model = transition_model
        
        # Create a mock Lyapunov function with a "hill" in the middle
        # that should be avoided when using stability guards
        lyap_values = {
            (0.0, 0.0): 1.0,      # Bottom left
            (1.0, 0.0): 2.0,
            (2.0, 0.0): 3.0,
            (0.0, 1.0): 2.0,
            (1.0, 1.0): 5.0,      # Hill in the middle
            (2.0, 1.0): 3.0,
            (0.0, 2.0): 2.0,
            (1.0, 2.0): 1.0,
            (2.0, 2.0): 1.0,      # Top right (goal)
        }
        self.lyap_fn = MockLyapunovFunction(lyap_values)
        
        # Create a mock KoopmanBridgeAgent
        self.agent = MockKoopmanBridgeAgent({
            'test_system': {
                'lyapunov': self.lyap_fn
            }
        })
    
    def test_goal_graph_without_stability(self):
        """Test goal graph search without stability constraints."""
        # Create goal graph without stability guard
        graph = GoalGraph(
            distance_fn=self.distance_fn,
            transition_model=self.transition_model
        )
        
        # Set start and goal
        graph.set_start(np.array([0.0, 0.0]))
        graph.set_goal(np.array([2.0, 2.0]))
        
        # Perform A* search
        path = graph.a_star_search()
        
        # Verify path exists
        self.assertIsNotNone(path)
        
        # Get path states
        path_states = [node.state.tolist() for node in path]
        
        # Check path length
        self.assertEqual(len(path), 5)
        
        # Check that the path goes through the "hill" in the middle
        # since we're not using stability constraints
        middle_states = [[1.0, 1.0], [1.0, 2.0]]
        
        # Check if at least one middle state is in the path
        has_middle = any(state in path_states for state in middle_states)
        self.assertTrue(has_middle)
    
    def test_goal_graph_with_stability(self):
        """Test goal graph search with stability constraints."""
        # Create stability guard
        stability_guard = StabilityGuard(
            bridge_agent=self.agent,
            system_name='test_system',
            epsilon=0.1
        )
        
        # Create goal graph with stability guard
        graph = GoalGraph(
            distance_fn=self.distance_fn,
            transition_model=self.transition_model,
            stability_guard=stability_guard
        )
        
        # Set start and goal
        graph.set_start(np.array([0.0, 0.0]))
        graph.set_goal(np.array([2.0, 2.0]))
        
        # Perform A* search
        path = graph.a_star_search()
        
        # Verify path exists
        self.assertIsNotNone(path)
        
        # Get path states
        path_states = [node.state.tolist() for node in path]
        
        # The path should avoid the "hill" with Lyapunov value 5.0
        # instead going around via [0, 1], [0, 2], [1, 2]
        self.assertNotIn([1.0, 1.0], path_states)
        
        # Check that the path includes safer states
        safe_states = [[0.0, 1.0], [0.0, 2.0], [1.0, 2.0]]
        for safe_state in safe_states:
            self.assertIn(safe_state, path_states)
        
        # Check path length (should be longer due to avoiding the hill)
        self.assertGreaterEqual(len(path), 5)
        
        # Check statistics
        stats = graph.get_stats()
        self.assertGreater(stats["pruned_transitions"], 0)
    
    def test_create_goal_graph_with_stability(self):
        """Test factory function for creating goal graph with stability guard."""
        # Create goal graph using factory function
        graph = create_goal_graph_with_stability(
            bridge_agent=self.agent,
            distance_fn=self.distance_fn,
            transition_model=self.transition_model,
            system_name='test_system',
            epsilon=0.1
        )
        
        # Verify that stability guard was created
        self.assertIsNotNone(graph.stability_guard)
        
        # Set start and goal
        graph.set_start(np.array([0.0, 0.0]))
        graph.set_goal(np.array([2.0, 2.0]))
        
        # Perform A* search
        path = graph.a_star_search()
        
        # Verify path exists
        self.assertIsNotNone(path)
        
        # Check statistics
        stats = graph.get_stats()
        self.assertTrue("stability_guard" in stats)
    
    def test_breadth_first_search_with_stability(self):
        """Test breadth-first search with stability constraints."""
        # Create stability guard
        stability_guard = StabilityGuard(
            bridge_agent=self.agent,
            system_name='test_system',
            epsilon=0.1
        )
        
        # Create goal graph with stability guard
        graph = GoalGraph(
            distance_fn=self.distance_fn,
            transition_model=self.transition_model,
            stability_guard=stability_guard
        )
        
        # Set start and goal
        graph.set_start(np.array([0.0, 0.0]))
        graph.set_goal(np.array([2.0, 2.0]))
        
        # Perform BFS search
        path = graph.breadth_first_search()
        
        # Verify path exists
        self.assertIsNotNone(path)
        
        # Get path states
        path_states = [node.state.tolist() for node in path]
        
        # The path should avoid the "hill" with Lyapunov value 5.0
        self.assertNotIn([1.0, 1.0], path_states)
        
        # Check statistics
        stats = graph.get_stats()
        self.assertGreater(stats["pruned_transitions"], 0)


if __name__ == '__main__':
    unittest.main()
