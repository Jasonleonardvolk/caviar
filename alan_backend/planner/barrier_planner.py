"""
Barrier-aware planner for ALAN.

This module provides an extension to the standard planner that incorporates
barrier certificates to ensure safety constraints during planning. It allows
the planner to avoid unsafe regions and ensure that trajectories will not
violate safety properties.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

from alan_backend.planner.goal_graph import GoalGraph, ConceptNode, create_goal_graph_with_stability
from alan_backend.planner.guards import StabilityGuard
from alan_backend.elfin.barrier.barrier_bridge_agent import BarrierBridgeAgent

# Configure logging
logger = logging.getLogger("alan.planner.barrier_planner")


class BarrierGuard:
    """
    Guard that ensures safety of transitions using barrier certificates.
    
    This class can be used as a callback in the planner to check if
    transitions between states maintain safety by evaluating barrier
    functions at the destination state.
    """
    
    def __init__(
        self,
        barrier_agent: BarrierBridgeAgent,
        system_name: str,
        margin: float = 0.0,
        strict: bool = False
    ):
        """
        Initialize the BarrierGuard.
        
        Args:
            barrier_agent: BarrierBridgeAgent to use for safety checking
            system_name: Name of the system in the barrier agent
            margin: Safety margin for barrier function (negative means stricter)
            strict: If True, reject states with barrier value exactly 0
        """
        self.barrier_agent = barrier_agent
        self.system_name = system_name
        self.margin = margin
        self.strict = strict
        
        # Check if system exists in barrier agent
        if system_name not in barrier_agent.results:
            raise ValueError(f"System '{system_name}' not found in barrier agent")
        
        # Initialize counters for statistics
        self.total_checks = 0
        self.rejected_transitions = 0
    
    def __call__(
        self,
        src_node: ConceptNode,
        dst_node: ConceptNode
    ) -> bool:
        """
        Check if a transition from src_node to dst_node is safe.
        
        A transition is considered safe if B(dst) ≤ margin, where B is the
        barrier function and margin is a safety tolerance (usually 0 or negative).
        
        Args:
            src_node: Source concept node
            dst_node: Destination concept node
            
        Returns:
            True if transition is safe, False otherwise
        """
        self.total_checks += 1
        
        # Extract state vectors from nodes
        if hasattr(dst_node, 'state'):
            dst_state = dst_node.state
        elif hasattr(dst_node, 'embedding'):
            dst_state = dst_node.embedding
        else:
            logger.error("Destination node does not have a state or embedding")
            return False
        
        # Get barrier function
        barrier_fn = self.barrier_agent.results[self.system_name]['barrier']
        
        # Evaluate barrier function at destination state
        barrier_value = barrier_fn(dst_state)
        
        # Check if destination is safe (B(x) ≤ margin)
        if self.strict:
            is_safe = barrier_value < self.margin
        else:
            is_safe = barrier_value <= self.margin
        
        # Update statistics
        if not is_safe:
            self.rejected_transitions += 1
            logger.debug(f"Rejected unsafe transition to {dst_node.id}, B(x) = {barrier_value:.6f}")
        
        # Emit event to dependency graph if available
        try:
            # Only emit event if agent has an interaction log
            if hasattr(self.barrier_agent, 'interaction_log') and self.barrier_agent.interaction_log is not None:
                self.barrier_agent.interaction_log.emit("planner_safety_checked", {
                    "system_id": self.system_name,
                    "dst_state": dst_state.tolist() if isinstance(dst_state, np.ndarray) else dst_state,
                    "barrier_value": float(barrier_value),
                    "is_safe": is_safe
                })
        except Exception as e:
            logger.warning(f"Failed to emit event: {e}")
        
        return is_safe
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the guard's operation.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_checks": self.total_checks,
            "rejected_transitions": self.rejected_transitions,
            "rejection_rate": self.rejected_transitions / max(1, self.total_checks),
            "margin": self.margin,
            "strict": self.strict,
            "system_name": self.system_name
        }
    
    def reset_stats(self) -> None:
        """Reset the guard's statistics."""
        self.total_checks = 0
        self.rejected_transitions = 0


class SafetyViolationError(Exception):
    """Exception raised when a safety constraint is violated."""
    pass


class SafetyAwareGoalGraph(GoalGraph):
    """
    Safety-aware goal graph for planning with safety constraints.
    
    This class extends the standard goal graph with safety checking using
    barrier certificates. It ensures that all transitions in the plan
    satisfy safety constraints defined by barrier functions.
    """
    
    def __init__(
        self,
        distance_fn: Callable[[np.ndarray, np.ndarray], float],
        transition_model: Callable[[np.ndarray], List[Tuple[np.ndarray, str, float]]],
        heuristic_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        stability_guard: Optional[StabilityGuard] = None,
        barrier_guard: Optional[BarrierGuard] = None
    ):
        """
        Initialize the safety-aware goal graph.
        
        Args:
            distance_fn: Function to compute distance between states
            transition_model: Function that generates possible transitions from a state
            heuristic_fn: Function to estimate distance to goal (for A*)
            stability_guard: Optional StabilityGuard to enforce stability constraints
            barrier_guard: Optional BarrierGuard to enforce safety constraints
        """
        super().__init__(
            distance_fn=distance_fn,
            transition_model=transition_model,
            heuristic_fn=heuristic_fn,
            stability_guard=stability_guard
        )
        self.barrier_guard = barrier_guard
        
        # Safety statistics
        self.safety_violations = 0
    
    def a_star_search(self) -> Optional[List[ConceptNode]]:
        """
        Perform A* search from start to goal with safety constraints.
        
        This extends the standard A* search with safety checking using
        the barrier guard.
        
        Returns:
            List of nodes from start to goal if found, None otherwise
        """
        # Reset safety statistics
        self.safety_violations = 0
        
        # Check if start node is safe
        if self.barrier_guard and not self.barrier_guard(self.start_node, self.start_node):
            logger.error("Start state violates safety constraints")
            self.safety_violations += 1
            return None
        
        return super().a_star_search()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get search statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = super().get_stats()
        
        # Add safety statistics
        stats["safety_violations"] = self.safety_violations
        
        # Add barrier guard stats if available
        if self.barrier_guard:
            stats["barrier_guard"] = self.barrier_guard.get_stats()
        
        return stats


def create_goal_graph_with_safety(
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    transition_model: Callable[[np.ndarray], List[Tuple[np.ndarray, str, float]]],
    barrier_agent: BarrierBridgeAgent,
    system_name: str,
    margin: float = 0.0,
    strict: bool = False,
    stability_bridge_agent: Optional[Any] = None,
    stability_system_name: Optional[str] = None,
    stability_epsilon: float = 1e-3,
    stability_strict: bool = False
) -> SafetyAwareGoalGraph:
    """
    Create a goal graph with both safety and stability constraints.
    
    Args:
        distance_fn: Function to compute distance between states
        transition_model: Function that generates possible transitions from a state
        barrier_agent: BarrierBridgeAgent to use for safety checking
        system_name: Name of the system in the barrier agent
        margin: Safety margin for barrier function (negative means stricter)
        strict: If True, reject states with barrier value exactly 0
        stability_bridge_agent: Optional KoopmanBridgeAgent for stability checking
        stability_system_name: Optional name of the system for stability checking
        stability_epsilon: Tolerance for determining stability
        stability_strict: If True, reject all transitions that increase Lyapunov value
        
    Returns:
        SafetyAwareGoalGraph with safety and stability guards
    """
    # Create barrier guard
    barrier_guard = BarrierGuard(
        barrier_agent=barrier_agent,
        system_name=system_name,
        margin=margin,
        strict=strict
    )
    
    # Create stability guard if bridge agent is provided
    stability_guard = None
    if stability_bridge_agent is not None:
        from alan_backend.planner.guards import StabilityGuard
        stability_guard = StabilityGuard(
            bridge_agent=stability_bridge_agent,
            system_name=stability_system_name,
            epsilon=stability_epsilon,
            strict=stability_strict
        )
    
    # Create safety-aware goal graph
    return SafetyAwareGoalGraph(
        distance_fn=distance_fn,
        transition_model=transition_model,
        heuristic_fn=distance_fn,
        stability_guard=stability_guard,
        barrier_guard=barrier_guard
    )


def verify_plan_safety(
    plan: List[ConceptNode],
    barrier_agent: BarrierBridgeAgent,
    system_name: str,
    margin: float = 0.0,
    strict: bool = False
) -> bool:
    """
    Verify that a plan satisfies safety constraints.
    
    Args:
        plan: List of nodes representing the plan
        barrier_agent: BarrierBridgeAgent to use for safety checking
        system_name: Name of the system in the barrier agent
        margin: Safety margin for barrier function (negative means stricter)
        strict: If True, reject states with barrier value exactly 0
        
    Returns:
        True if plan is safe, False otherwise
    """
    # Create barrier guard
    barrier_guard = BarrierGuard(
        barrier_agent=barrier_agent,
        system_name=system_name,
        margin=margin,
        strict=strict
    )
    
    # Check safety of each transition
    is_safe = True
    for i in range(len(plan) - 1):
        src_node = plan[i]
        dst_node = plan[i + 1]
        
        if not barrier_guard(src_node, dst_node):
            logger.warning(f"Unsafe transition at step {i} from {src_node.id} to {dst_node.id}")
            is_safe = False
            break
    
    # Get safety statistics
    stats = barrier_guard.get_stats()
    logger.info(f"Safety verification statistics: {stats}")
    
    return is_safe
