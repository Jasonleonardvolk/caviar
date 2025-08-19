"""
Goal Graph implementation for ALAN planner.

This module defines the GoalGraph class which is used for planning and
exploring paths to achieve goals while ensuring stability constraints.
"""

import os
import sys
import logging
import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Set
from collections import deque

from alan_backend.elfin.koopman.koopman_bridge_agent import KoopmanBridgeAgent
from alan_backend.planner.guards import verify_transition, StabilityGuard

# Configure logging
logger = logging.getLogger("alan.planner.goal_graph")


class ConceptNode:
    """
    A node in the goal graph representing a concept state.
    
    Attributes:
        id: Unique identifier for the node
        state: State vector representing the concept
        parent: Parent node
        g_score: Cost to reach this node from the start
        h_score: Heuristic estimate of cost to goal
        action: Action that led to this node
    """
    
    def __init__(
        self,
        id: str,
        state: np.ndarray,
        parent: Optional['ConceptNode'] = None,
        g_score: float = 0.0,
        h_score: float = 0.0,
        action: Optional[str] = None
    ):
        """
        Initialize a concept node.
        
        Args:
            id: Unique identifier for the node
            state: State vector representing the concept
            parent: Parent node
            g_score: Cost to reach this node from the start
            h_score: Heuristic estimate of cost to goal
            action: Action that led to this node
        """
        self.id = id
        self.state = state
        self.parent = parent
        self.g_score = g_score
        self.h_score = h_score
        self.action = action
        
        # For A* search
        self.closed = False
        self.open = False
        
        # Meta information
        self.metadata = {}
    
    @property
    def f_score(self) -> float:
        """
        Get the f-score (g + h) for this node.
        
        Returns:
            f-score
        """
        return self.g_score + self.h_score
    
    def __lt__(self, other: 'ConceptNode') -> bool:
        """
        Compare nodes based on f-score for priority queue.
        
        Args:
            other: Node to compare with
            
        Returns:
            True if this node has lower f-score
        """
        return self.f_score < other.f_score
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two nodes are equal based on ID.
        
        Args:
            other: Node to compare with
            
        Returns:
            True if nodes have the same ID
        """
        if not isinstance(other, ConceptNode):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """
        Hash function based on ID.
        
        Returns:
            Hash of the node's ID
        """
        return hash(self.id)


class GoalGraph:
    """
    Graph-based planner for navigating concept space.
    
    This class implements graph-based planning algorithms (A*, BFS, etc.)
    for finding paths between concept states in the semantic space.
    """
    
    def __init__(
        self,
        distance_fn: Callable[[np.ndarray, np.ndarray], float],
        transition_model: Callable[[np.ndarray], List[Tuple[np.ndarray, str, float]]],
        heuristic_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        stability_guard: Optional[StabilityGuard] = None
    ):
        """
        Initialize the goal graph.
        
        Args:
            distance_fn: Function to compute distance between states
            transition_model: Function that generates possible transitions from a state
            heuristic_fn: Function to estimate distance to goal (for A*)
            stability_guard: Optional StabilityGuard to enforce stability constraints
        """
        self.distance_fn = distance_fn
        self.transition_model = transition_model
        self.heuristic_fn = heuristic_fn or (lambda a, b: self.distance_fn(a, b))
        self.stability_guard = stability_guard
        
        # States
        self.nodes = {}  # id -> node
        self.start_node = None
        self.goal_node = None
        
        # Statistics
        self.expanded_nodes = 0
        self.generated_nodes = 0
        self.pruned_transitions = 0
    
    def add_node(self, node: ConceptNode) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: Node to add
        """
        self.nodes[node.id] = node
        self.generated_nodes += 1
    
    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: ID of the node to get
            
        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(node_id)
    
    def set_start(self, start_state: np.ndarray, start_id: str = "start") -> ConceptNode:
        """
        Set the start state.
        
        Args:
            start_state: Start state vector
            start_id: ID for the start node
            
        Returns:
            Start node
        """
        self.start_node = ConceptNode(id=start_id, state=start_state)
        self.add_node(self.start_node)
        return self.start_node
    
    def set_goal(self, goal_state: np.ndarray, goal_id: str = "goal") -> ConceptNode:
        """
        Set the goal state.
        
        Args:
            goal_state: Goal state vector
            goal_id: ID for the goal node
            
        Returns:
            Goal node
        """
        self.goal_node = ConceptNode(id=goal_id, state=goal_state)
        self.add_node(self.goal_node)
        return self.goal_node
    
    def a_star_search(self) -> Optional[List[ConceptNode]]:
        """
        Perform A* search from start to goal.
        
        Returns:
            List of nodes from start to goal if found, None otherwise
        """
        if not self.start_node or not self.goal_node:
            logger.error("Start or goal node not set")
            return None
        
        # Initialize open set with start node
        open_set = []
        heapq.heappush(open_set, (0, self.start_node))
        self.start_node.open = True
        
        # Track visited nodes
        visited = {self.start_node.id: self.start_node}
        
        while open_set:
            # Get node with lowest f-score
            _, current = heapq.heappop(open_set)
            current.open = False
            
            # Check if goal reached
            if current == self.goal_node:
                return self._reconstruct_path(current)
            
            # Mark as closed
            current.closed = True
            self.expanded_nodes += 1
            
            # Generate successors
            for next_state, action, cost in self.transition_model(current.state):
                # Create successor node
                next_id = f"node_{self.generated_nodes}"
                next_node = ConceptNode(
                    id=next_id,
                    state=next_state,
                    parent=current,
                    g_score=current.g_score + cost,
                    h_score=self.heuristic_fn(next_state, self.goal_node.state),
                    action=action
                )
                
                # Check stability constraint if guard is enabled
                if self.stability_guard and not self.stability_guard(current, next_node):
                    self.pruned_transitions += 1
                    logger.debug(f"Pruned transition from {current.id} to {next_id} due to stability constraint")
                    continue
                
                # Check if already visited with better cost
                if next_id in visited and visited[next_id].g_score <= next_node.g_score:
                    continue
                
                # Add to visited and open set
                visited[next_id] = next_node
                self.add_node(next_node)
                
                if not next_node.open and not next_node.closed:
                    heapq.heappush(open_set, (next_node.f_score, next_node))
                    next_node.open = True
        
        # No path found
        logger.warning("No path found from start to goal")
        return None
    
    def breadth_first_search(self) -> Optional[List[ConceptNode]]:
        """
        Perform breadth-first search from start to goal.
        
        Returns:
            List of nodes from start to goal if found, None otherwise
        """
        if not self.start_node or not self.goal_node:
            logger.error("Start or goal node not set")
            return None
        
        # Initialize queue with start node
        queue = deque([self.start_node])
        
        # Track visited nodes
        visited = {self.start_node.id: True}
        
        while queue:
            # Get next node
            current = queue.popleft()
            self.expanded_nodes += 1
            
            # Check if goal reached
            if current == self.goal_node:
                return self._reconstruct_path(current)
            
            # Generate successors
            for next_state, action, cost in self.transition_model(current.state):
                # Create successor node
                next_id = f"node_{self.generated_nodes}"
                next_node = ConceptNode(
                    id=next_id,
                    state=next_state,
                    parent=current,
                    g_score=current.g_score + cost,
                    action=action
                )
                
                # Check stability constraint if guard is enabled
                if self.stability_guard and not self.stability_guard(current, next_node):
                    self.pruned_transitions += 1
                    logger.debug(f"Pruned transition from {current.id} to {next_id} due to stability constraint")
                    continue
                
                # Check if already visited
                if next_id in visited:
                    continue
                
                # Add to visited and queue
                visited[next_id] = True
                self.add_node(next_node)
                queue.append(next_node)
        
        # No path found
        logger.warning("No path found from start to goal")
        return None
    
    def _reconstruct_path(self, node: ConceptNode) -> List[ConceptNode]:
        """
        Reconstruct path from start to node.
        
        Args:
            node: End node
            
        Returns:
            List of nodes from start to end
        """
        path = []
        current = node
        
        while current:
            path.append(current)
            current = current.parent
        
        # Reverse to get start to goal
        path.reverse()
        return path
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get search statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "expanded_nodes": self.expanded_nodes,
            "generated_nodes": self.generated_nodes,
            "pruned_transitions": self.pruned_transitions,
        }
        
        # Add stability guard stats if available
        if self.stability_guard:
            stats["stability_guard"] = self.stability_guard.get_stats()
        
        return stats


def create_goal_graph_with_stability(
    bridge_agent: KoopmanBridgeAgent,
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    transition_model: Callable[[np.ndarray], List[Tuple[np.ndarray, str, float]]],
    system_name: str = None,
    epsilon: float = 1e-3,
    strict: bool = False
) -> GoalGraph:
    """
    Create a goal graph with stability constraints.
    
    Args:
        bridge_agent: KoopmanBridgeAgent to use for stability checks
        distance_fn: Function to compute distance between states
        transition_model: Function that generates possible transitions from a state
        system_name: Name of the system (if None, use agent's first system)
        epsilon: Tolerance for determining stability
        strict: If True, reject all transitions that increase Lyapunov value
        
    Returns:
        GoalGraph with stability guard
    """
    # Create stability guard
    stability_guard = StabilityGuard(
        bridge_agent=bridge_agent,
        system_name=system_name,
        epsilon=epsilon,
        strict=strict
    )
    
    # Create goal graph with stability guard
    return GoalGraph(
        distance_fn=distance_fn,
        transition_model=transition_model,
        heuristic_fn=distance_fn,
        stability_guard=stability_guard
    )
