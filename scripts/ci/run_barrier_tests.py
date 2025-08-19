#!/usr/bin/env python
"""
CI test script for barrier certificate functionality.

This script runs a basic test suite for the barrier certificate implementation.
It verifies that barrier certificates can be learned, verified, and refined.

Usage:
    python scripts/ci/run_barrier_tests.py
"""

import os
import sys
import numpy as np
import time
import logging
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from alan_backend.elfin.barrier.barrier_bridge_agent import BarrierBridgeAgent, create_double_integrator_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('barrier_ci_tests')


def run_double_integrator_test():
    """Run double integrator test with barrier certificate learning and verification."""
    logger.info("=== Running Double Integrator Barrier Test ===")
    start_time = time.time()
    
    # Create agent
    agent, barrier_fn = create_double_integrator_agent(
        verify=True,
        dict_type="rbf",
        dict_size=100
    )
    
    # Verify that the agent and barrier function were created successfully
    assert agent is not None, "Failed to create barrier agent"
    assert barrier_fn is not None, "Failed to create barrier function"
    
    # Check that verification was performed
    assert "verification" in agent.results["double_integrator"], "Verification not performed"
    verification_result = agent.results["double_integrator"]["verification"]["result"]
    
    # If verification failed, try refinement
    if not verification_result.success:
        logger.warning(f"Initial verification failed: {verification_result.status}")
        logger.warning(f"Violation reason: {verification_result.violation_reason}")
        logger.warning(f"Error code: {verification_result.get_error_code()}")
        
        # Refine the barrier certificate
        logger.info("Refining barrier certificate...")
        refined_result = agent.refine_auto(
            system_name="double_integrator",
            max_iterations=3
        )
        
        # Check if refinement was successful
        assert refined_result.success, f"Refinement failed: {refined_result.status}"
        logger.info(f"Refinement successful: {refined_result.status}")
    else:
        logger.info(f"Verification successful: {verification_result.status}")
    
    # Test barrier function evaluation
    test_state = np.array([0.0, 0.0, 0.0, 0.0])  # Origin
    value = barrier_fn(test_state)
    logger.info(f"Barrier value at origin: {value}")
    
    # Test safe/unsafe points
    # Inside obstacle (unsafe)
    unsafe_state = np.array([0.5, 0.0, 0.0, 0.0])  # Inside obstacle
    unsafe_value = barrier_fn(unsafe_state)
    # Far from obstacle (safe)
    safe_state = np.array([3.0, 3.0, 0.0, 0.0])
    safe_value = barrier_fn(safe_state)
    
    # Check that the barrier function correctly classifies safe and unsafe states
    assert unsafe_value <= 0, f"Unsafe state not classified correctly: {unsafe_value}"
    assert safe_value > 0, f"Safe state not classified correctly: {safe_value}"
    
    logger.info(f"Barrier at unsafe point: {unsafe_value}")
    logger.info(f"Barrier at safe point: {safe_value}")
    
    duration = time.time() - start_time
    logger.info(f"Double integrator test completed in {duration:.2f} seconds")
    return True


def run_barrier_guard_test():
    """Test barrier guard functionality with planning."""
    logger.info("=== Running Barrier Guard Planning Test ===")
    start_time = time.time()
    
    try:
        from alan_backend.planner.barrier_planner import (
            BarrierGuard, SafetyAwareGoalGraph, create_goal_graph_with_safety, verify_plan_safety
        )
        from alan_backend.planner.goal_graph import ConceptNode
        
        # Create double integrator agent
        agent, barrier_fn = create_double_integrator_agent(
            verify=True,
            dict_type="rbf",
            dict_size=100
        )
        
        # Define euclidean distance function
        def euclidean_distance(a, b):
            return np.linalg.norm(a - b)
        
        # Define transition model
        def get_transition_model(step_size=0.5):
            def transition_model(state):
                transitions = []
                # Six possible moves in state space
                directions = [
                    np.array([step_size, 0, 0, 0]),  # +x
                    np.array([-step_size, 0, 0, 0]), # -x
                    np.array([0, step_size, 0, 0]),  # +y
                    np.array([0, -step_size, 0, 0]), # -y
                    np.array([0, 0, step_size, 0]),  # +vx
                    np.array([0, 0, 0, step_size]),  # +vy
                    np.array([0, 0, -step_size, 0]), # -vx
                    np.array([0, 0, 0, -step_size])  # -vy
                ]
                
                # Create transitions
                for d in directions:
                    next_state = state + d
                    transitions.append((next_state, f"move_{d}", step_size))
                
                return transitions
            
            return transition_model
        
        # Create safety-aware planner
        planner = create_goal_graph_with_safety(
            distance_fn=euclidean_distance,
            transition_model=get_transition_model(),
            barrier_agent=agent,
            system_name="double_integrator"
        )
        
        # Define start and goal
        start_state = np.array([-3.0, -3.0, 0.0, 0.0])  # Bottom left
        goal_state = np.array([3.0, 3.0, 0.0, 0.0])     # Top right
        
        # Create nodes
        start_node = ConceptNode(id="start", state=start_state)
        goal_node = ConceptNode(id="goal", state=goal_state)
        
        # Set start and goal
        planner.set_start_and_goal(start_node, goal_node)
        
        # Plan path
        logger.info("Planning path with barrier constraints...")
        plan = planner.a_star_search()
        
        # Verify that a plan was found
        assert plan is not None, "Failed to find a plan"
        assert len(plan) > 0, "Plan is empty"
        
        # Verify that the plan is safe
        is_safe = verify_plan_safety(
            plan=plan,
            barrier_agent=agent,
            system_name="double_integrator"
        )
        
        assert is_safe, "Plan is not safe"
        
        # Check that the plan avoids the obstacle
        for node in plan:
            x, y = node.state[0], node.state[1]
            distance_to_origin = np.sqrt(x**2 + y**2)
            assert distance_to_origin > 1.0, f"Plan enters obstacle at {node.state}"
        
        logger.info(f"Plan found with {len(plan)} steps")
        logger.info(f"Plan is safe: {is_safe}")
        
        duration = time.time() - start_time
        logger.info(f"Barrier guard test completed in {duration:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Barrier guard test failed: {e}")
        return False


def main():
    """Run all barrier certificate tests."""
    logger.info("Running barrier certificate tests...")
    
    try:
        # Run double integrator test
        double_integrator_success = run_double_integrator_test()
        
        # Run barrier guard test
        barrier_guard_success = run_barrier_guard_test()
        
        # Check if all tests passed
        all_passed = double_integrator_success and barrier_guard_success
        
        if all_passed:
            logger.info("All barrier certificate tests passed!")
            sys.exit(0)
        else:
            logger.error("Some barrier certificate tests failed")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error running barrier certificate tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
