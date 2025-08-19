"""
Unit tests for the planner stability guards.
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

from alan_backend.planner.guards import stable_transition, verify_transition, StabilityGuard
from alan_backend.elfin.koopman.koopman_bridge_agent import KoopmanBridgeAgent


class MockLyapunovFunction:
    """Mock Lyapunov function for testing."""
    
    def __init__(self, values=None):
        """
        Initialize the mock Lyapunov function.
        
        Args:
            values: Dictionary mapping state tuples to Lyapunov values
        """
        self.values = values or {}
    
    def __call__(self, state):
        """
        Evaluate the Lyapunov function at the given state.
        
        Args:
            state: State to evaluate at
            
        Returns:
            Lyapunov value for the state
        """
        # Convert to tuple for dictionary lookup
        if isinstance(state, np.ndarray):
            state_key = tuple(state)
        else:
            state_key = state
            
        # Return the value if it exists, otherwise return norm of state
        if state_key in self.values:
            return self.values[state_key]
        else:
            # Default to squared Euclidean norm
            if isinstance(state, np.ndarray):
                return np.sum(state**2)
            else:
                return sum(x**2 for x in state)


class MockKoopmanBridgeAgent:
    """Mock KoopmanBridgeAgent for testing."""
    
    def __init__(self, systems=None):
        """
        Initialize the mock agent.
        
        Args:
            systems: Dictionary mapping system names to results
        """
        self.results = systems or {}
        self.stab_agent = None


class MockConceptNode:
    """Mock ConceptNode for testing."""
    
    def __init__(self, state=None, embedding=None):
        """
        Initialize the mock node.
        
        Args:
            state: State vector
            embedding: Embedding vector
        """
        self.state = state
        self.embedding = embedding


class TestPlannerGuards(unittest.TestCase):
    """Test cases for planner stability guards."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a mock Lyapunov function
        self.lyap_fn = MockLyapunovFunction({
            (0.0, 0.0): 0.0,      # Origin has zero Lyapunov value
            (1.0, 0.0): 1.0,      # Unit vector has value 1.0
            (0.0, 1.0): 1.0,      # Unit vector has value 1.0
            (1.0, 1.0): 2.0,      # (1,1) has value 2.0
            (2.0, 0.0): 4.0,      # (2,0) has value 4.0
            (0.0, 2.0): 4.0,      # (0,2) has value 4.0
        })
        
        # Create a mock KoopmanBridgeAgent
        self.agent = MockKoopmanBridgeAgent({
            'test_system': {
                'lyapunov': self.lyap_fn
            }
        })
    
    def test_stable_transition_decrease(self):
        """Test stable_transition with decreasing Lyapunov value."""
        # Transition from (2,0) to (1,0) should be stable (decrease)
        src_state = np.array([2.0, 0.0])
        dst_state = np.array([1.0, 0.0])
        
        # Check stability
        is_stable = stable_transition(
            src_state=src_state,
            dst_state=dst_state,
            agent=self.agent,
            system_name='test_system'
        )
        
        self.assertTrue(is_stable)
    
    def test_stable_transition_equal(self):
        """Test stable_transition with equal Lyapunov value."""
        # Transition from (1,0) to (0,1) should be stable (equal)
        src_state = np.array([1.0, 0.0])
        dst_state = np.array([0.0, 1.0])
        
        # Check stability
        is_stable = stable_transition(
            src_state=src_state,
            dst_state=dst_state,
            agent=self.agent,
            system_name='test_system'
        )
        
        self.assertTrue(is_stable)
    
    def test_stable_transition_small_increase(self):
        """Test stable_transition with small increase in Lyapunov value."""
        # Transition from (1,0) to (1.1,0) should be stable with epsilon
        src_state = np.array([1.0, 0.0])
        dst_state = np.array([1.1, 0.0])
        
        # Lyapunov values (approximate)
        # V(src) = 1.0
        # V(dst) = 1.21
        # Relative increase = 0.21 or 21%
        
        # With default epsilon (1e-3), this should not be stable
        is_stable_tight = stable_transition(
            src_state=src_state,
            dst_state=dst_state,
            agent=self.agent,
            system_name='test_system'
        )
        
        self.assertFalse(is_stable_tight)
        
        # With epsilon = 0.25, this should be stable
        is_stable_loose = stable_transition(
            src_state=src_state,
            dst_state=dst_state,
            agent=self.agent,
            system_name='test_system',
            epsilon=0.25
        )
        
        self.assertTrue(is_stable_loose)
    
    def test_stable_transition_large_increase(self):
        """Test stable_transition with large increase in Lyapunov value."""
        # Transition from (1,0) to (2,0) should not be stable
        src_state = np.array([1.0, 0.0])
        dst_state = np.array([2.0, 0.0])
        
        # Lyapunov values
        # V(src) = 1.0
        # V(dst) = 4.0
        # Relative increase = 3.0 or 300%
        
        # Check stability
        is_stable = stable_transition(
            src_state=src_state,
            dst_state=dst_state,
            agent=self.agent,
            system_name='test_system',
            epsilon=0.1  # Even with epsilon, this should not be stable
        )
        
        self.assertFalse(is_stable)
    
    def test_verify_transition(self):
        """Test verify_transition with ConceptNodes."""
        # Create nodes
        src_node = MockConceptNode(state=np.array([2.0, 0.0]))
        dst_node = MockConceptNode(state=np.array([1.0, 0.0]))
        
        # Check stability
        is_stable = verify_transition(
            src_node=src_node,
            dst_node=dst_node,
            bridge_agent=self.agent,
            system_name='test_system'
        )
        
        self.assertTrue(is_stable)
        
        # Check with embedding instead of state
        src_node_emb = MockConceptNode(embedding=np.array([2.0, 0.0]))
        dst_node_emb = MockConceptNode(embedding=np.array([1.0, 0.0]))
        
        is_stable_emb = verify_transition(
            src_node=src_node_emb,
            dst_node=dst_node_emb,
            bridge_agent=self.agent,
            system_name='test_system'
        )
        
        self.assertTrue(is_stable_emb)
    
    def test_stability_guard(self):
        """Test StabilityGuard class."""
        # Create a stability guard
        guard = StabilityGuard(
            bridge_agent=self.agent,
            system_name='test_system',
            epsilon=0.1
        )
        
        # Create nodes for stable transition
        src_node = MockConceptNode(state=np.array([2.0, 0.0]))
        dst_node = MockConceptNode(state=np.array([1.0, 0.0]))
        
        # Check stability
        is_stable = guard(src_node, dst_node)
        
        self.assertTrue(is_stable)
        
        # Create nodes for unstable transition
        src_node_unstable = MockConceptNode(state=np.array([1.0, 0.0]))
        dst_node_unstable = MockConceptNode(state=np.array([2.0, 0.0]))
        
        # Check stability
        is_stable_unstable = guard(src_node_unstable, dst_node_unstable)
        
        self.assertFalse(is_stable_unstable)
        
        # Check stats
        stats = guard.get_stats()
        self.assertEqual(stats['total_checks'], 2)
        self.assertEqual(stats['rejected_transitions'], 1)
        self.assertEqual(stats['rejection_rate'], 0.5)
    
    def test_strict_mode(self):
        """Test StabilityGuard with strict mode."""
        # Create a strict stability guard
        guard = StabilityGuard(
            bridge_agent=self.agent,
            system_name='test_system',
            epsilon=0.1,
            strict=True
        )
        
        # Create nodes for equal transition
        src_node = MockConceptNode(state=np.array([1.0, 0.0]))
        dst_node = MockConceptNode(state=np.array([0.0, 1.0]))
        
        # Check stability
        is_stable = guard(src_node, dst_node)
        
        self.assertTrue(is_stable)
        
        # Create nodes for very small increase
        src_node_small = MockConceptNode(state=np.array([1.0, 0.0]))
        dst_node_small = MockConceptNode(state=np.array([1.01, 0.0]))
        
        # In strict mode, even a tiny increase should be rejected
        is_stable_small = guard(src_node_small, dst_node_small)
        
        self.assertFalse(is_stable_small)


if __name__ == '__main__':
    unittest.main()
