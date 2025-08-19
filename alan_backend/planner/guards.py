"""
Stability guards for the ALAN planner.

This module provides guards that can be used to ensure that the planner
only generates stable transitions between states. These guards use the
ELFIN stability verification framework to check if transitions between
states maintain or decrease Lyapunov function values.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

from alan_backend.elfin.koopman.koopman_bridge_agent import KoopmanBridgeAgent

# Configure logging
logger = logging.getLogger("alan.planner.guards")


def stable_transition(
    src_state: np.ndarray,
    dst_state: np.ndarray,
    agent: KoopmanBridgeAgent,
    system_name: str = None,
    epsilon: float = 1e-3,
    use_relative: bool = True
) -> bool:
    """
    Check if a transition from src_state to dst_state is stable.
    
    A transition is considered stable if:
    1. V(dst) <= V(src) + epsilon (absolute criterion)
    OR
    2. (V(dst) - V(src)) / V(src) <= epsilon (relative criterion)
    
    Args:
        src_state: Source state
        dst_state: Destination state
        agent: KoopmanBridgeAgent to use for checking stability
        system_name: Name of the system (if None, use agent's first system)
        epsilon: Tolerance for determining stability
        use_relative: Whether to use relative change as criterion
        
    Returns:
        True if transition is stable, False otherwise
    """
    if system_name is None:
        # Use the first system in the agent's results
        if not agent.results:
            logger.error("No systems found in KoopmanBridgeAgent")
            return False
        system_name = next(iter(agent.results.keys()))
    
    # Check if system exists in agent's results
    if system_name not in agent.results:
        logger.error(f"System '{system_name}' not found in KoopmanBridgeAgent")
        return False
    
    # Get Lyapunov function for the system
    lyap_fn = agent.results[system_name]['lyapunov']
    
    # Evaluate Lyapunov function at source and destination states
    v_src = lyap_fn(src_state)
    v_dst = lyap_fn(dst_state)
    
    # Log Lyapunov values
    logger.debug(f"V(src): {v_src:.6f}, V(dst): {v_dst:.6f}, Δ: {v_dst - v_src:.6f}")
    
    # Check stability criterion
    if use_relative:
        # Use relative change: (V(dst) - V(src)) / V(src) <= epsilon
        # This handles the case where V(src) is very small
        if v_src < 1e-10:  # If V(src) is too small, use absolute criterion
            is_stable = v_dst <= epsilon
        else:
            rel_change = (v_dst - v_src) / v_src
            is_stable = rel_change <= epsilon
    else:
        # Use absolute change: V(dst) <= V(src) + epsilon
        is_stable = v_dst <= v_src + epsilon
    
    # If needed, emit event to dependency graph
    try:
        # Only emit event if agent has an interaction log
        if hasattr(agent, 'stab_agent') and hasattr(agent.stab_agent, 'interaction_log'):
            agent.stab_agent.interaction_log.emit("planner_transition_checked", {
                "system_id": system_name,
                "src_state": src_state.tolist() if isinstance(src_state, np.ndarray) else src_state,
                "dst_state": dst_state.tolist() if isinstance(dst_state, np.ndarray) else dst_state,
                "v_src": float(v_src),
                "v_dst": float(v_dst),
                "is_stable": is_stable
            })
    except Exception as e:
        logger.warning(f"Failed to emit event: {e}")
    
    return is_stable


def verify_transition(
    src_node: "ConceptNode",
    dst_node: "ConceptNode",
    bridge_agent: KoopmanBridgeAgent,
    system_name: str = None,
    epsilon: float = 1e-3
) -> bool:
    """
    Verify that a transition between two concept nodes is stable.
    
    Args:
        src_node: Source concept node
        dst_node: Destination concept node
        bridge_agent: KoopmanBridgeAgent to use for checking stability
        system_name: Name of the system (if None, use agent's first system)
        epsilon: Tolerance for determining stability
        
    Returns:
        True if transition is stable, False otherwise
    """
    # Extract state vectors from nodes
    if hasattr(src_node, 'state') and hasattr(dst_node, 'state'):
        src_state = src_node.state
        dst_state = dst_node.state
    elif hasattr(src_node, 'embedding') and hasattr(dst_node, 'embedding'):
        src_state = src_node.embedding
        dst_state = dst_node.embedding
    else:
        logger.error("Nodes do not have compatible state representations")
        return False
    
    # Check if transition is stable
    return stable_transition(
        src_state=src_state,
        dst_state=dst_state,
        agent=bridge_agent,
        system_name=system_name,
        epsilon=epsilon
    )


class StabilityGuard:
    """
    Guard that ensures stability of transitions in the planner.
    
    This class can be used as a callback in the planner to check if
    transitions between states maintain stability based on Lyapunov functions.
    """
    
    def __init__(
        self,
        bridge_agent: KoopmanBridgeAgent,
        system_name: str = None,
        epsilon: float = 1e-3,
        strict: bool = False
    ):
        """
        Initialize the StabilityGuard.
        
        Args:
            bridge_agent: KoopmanBridgeAgent to use for checking stability
            system_name: Name of the system (if None, use agent's first system)
            epsilon: Tolerance for determining stability
            strict: If True, reject all transitions that increase Lyapunov value
        """
        self.bridge_agent = bridge_agent
        self.system_name = system_name
        self.epsilon = epsilon
        self.strict = strict
        
        # Initialize counters for statistics
        self.total_checks = 0
        self.rejected_transitions = 0
    
    def __call__(
        self,
        src_node: "ConceptNode",
        dst_node: "ConceptNode"
    ) -> bool:
        """
        Check if a transition from src_node to dst_node is stable.
        
        Args:
            src_node: Source concept node
            dst_node: Destination concept node
            
        Returns:
            True if transition is stable, False otherwise
        """
        self.total_checks += 1
        
        # Check if transition is stable
        is_stable = verify_transition(
            src_node=src_node,
            dst_node=dst_node,
            bridge_agent=self.bridge_agent,
            system_name=self.system_name,
            epsilon=0.0 if self.strict else self.epsilon
        )
        
        # Update statistics
        if not is_stable:
            self.rejected_transitions += 1
        
        return is_stable
    
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
            "epsilon": self.epsilon,
            "strict": self.strict,
            "system_name": self.system_name
        }
    
    def reset_stats(self) -> None:
        """Reset the guard's statistics."""
        self.total_checks = 0
        self.rejected_transitions = 0
