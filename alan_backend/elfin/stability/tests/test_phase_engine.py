"""
Unit tests for the PhaseEngine class.

These tests verify the phase-coupled oscillator synchronization behavior,
ensuring that connected concepts correctly synchronize their phases.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.phase_engine import PhaseEngine


class TestPhaseEngine(unittest.TestCase):
    """Test cases for PhaseEngine."""
    
    def test_phase_initialization(self):
        """Test phase initialization and basic properties."""
        engine = PhaseEngine(coupling_strength=0.2)
        
        # Add concepts with specific phases
        engine.add_concept("c1", initial_phase=0.0)
        engine.add_concept("c2", initial_phase=np.pi)
        
        # Check phases were set correctly
        self.assertAlmostEqual(engine.phases["c1"], 0.0)
        self.assertAlmostEqual(engine.phases["c2"], np.pi)
        
        # Check sync ratio with no edges
        self.assertEqual(engine.calculate_sync_ratio(), 1.0)
        
        # Add edge and check sync ratio changes
        engine.add_edge("c1", "c2", weight=1.0)
        self.assertLess(engine.calculate_sync_ratio(), 1.0)
    
    def test_single_step_update(self):
        """Test a single update step changes phases correctly."""
        engine = PhaseEngine(coupling_strength=0.2)
        
        # Add concepts with different phases
        engine.add_concept("c1", initial_phase=0.0)
        engine.add_concept("c2", initial_phase=np.pi)
        
        # Add edge to create coupling
        engine.add_edge("c1", "c2", weight=1.0)
        
        # Get initial phases
        initial_c1 = engine.phases["c1"]
        initial_c2 = engine.phases["c2"]
        
        # Perform a step
        engine.step(dt=0.1)
        
        # Check phases changed
        self.assertNotEqual(engine.phases["c1"], initial_c1)
        self.assertNotEqual(engine.phases["c2"], initial_c2)
    
    def test_synchronization_convergence(self):
        """Test that connected nodes converge to synchronization over time."""
        # Create engine with relatively strong coupling
        engine = PhaseEngine(coupling_strength=0.5)
        
        # Create simple pair of connected concepts
        engine.add_concept("c1", initial_phase=0.0)
        engine.add_concept("c2", initial_phase=np.pi)  # Opposite phase
        
        # Connect with bidirectional edges (mutual influence)
        engine.add_edge("c1", "c2", weight=1.0)
        engine.add_edge("c2", "c1", weight=1.0)
        
        # Simulate 200 steps
        phase_differences = []
        sync_ratios = []
        
        for _ in range(200):
            engine.step(dt=0.1)
            
            # Calculate phase difference
            diff = abs((engine.phases["c1"] - engine.phases["c2"]) % (2 * np.pi))
            if diff > np.pi:
                diff = 2 * np.pi - diff
            
            phase_differences.append(diff)
            sync_ratios.append(engine.calculate_sync_ratio())
        
        # Check final synchronization - phases should be nearly identical
        final_diff = abs((engine.phases["c1"] - engine.phases["c2"]) % (2 * np.pi))
        if final_diff > np.pi:
            final_diff = 2 * np.pi - diff
            
        self.assertLess(final_diff, 1e-3, "Phases did not converge within 200 steps")
        self.assertGreater(engine.calculate_sync_ratio(), 0.999, 
                          "Sync ratio did not reach near-perfect level")
        
        # Optional visualization for debugging
        if os.environ.get('PLOT_TESTS', '0') == '1':
            plt.figure(figsize=(10, 6))
            plt.plot(phase_differences, label='Phase Difference')
            plt.plot(sync_ratios, label='Sync Ratio')
            plt.axhline(y=1e-3, color='r', linestyle='--', label='Target Threshold')
            plt.xlabel('Steps')
            plt.ylabel('Value')
            plt.title('Phase Synchronization Convergence')
            plt.legend()
            plt.savefig('phase_sync_convergence.png')
    
    def test_complex_network_synchronization(self):
        """Test synchronization in a more complex network."""
        engine = PhaseEngine(coupling_strength=0.3)
        
        # Create a small network of 5 concepts with random initial phases
        np.random.seed(42)  # For reproducibility
        
        concepts = ["c1", "c2", "c3", "c4", "c5"]
        for c in concepts:
            engine.add_concept(c, initial_phase=np.random.uniform(0, 2*np.pi))
        
        # Add edges to form a connected graph
        # Ring topology with some cross-connections
        engine.add_edge("c1", "c2", weight=1.0)
        engine.add_edge("c2", "c3", weight=1.0)
        engine.add_edge("c3", "c4", weight=1.0)
        engine.add_edge("c4", "c5", weight=1.0)
        engine.add_edge("c5", "c1", weight=1.0)
        engine.add_edge("c1", "c3", weight=0.5)  # Cross-connection
        engine.add_edge("c2", "c5", weight=0.5)  # Cross-connection
        
        # Initial sync ratio should be low
        initial_sync = engine.calculate_sync_ratio()
        
        # Run simulation for 200 steps
        for _ in range(200):
            engine.step(dt=0.1)
        
        # Final sync ratio should be higher
        final_sync = engine.calculate_sync_ratio()
        
        self.assertGreater(final_sync, initial_sync, 
                          "Synchronization did not improve over time")
        self.assertGreater(final_sync, 0.95, 
                          "Network did not reach high synchronization")
    
    def test_phase_offsets(self):
        """Test that desired phase offsets are maintained."""
        engine = PhaseEngine(coupling_strength=0.5)
        
        # Add two concepts
        engine.add_concept("c1", initial_phase=0.0)
        engine.add_concept("c2", initial_phase=np.pi)
        
        # Set a desired phase offset of pi/2
        desired_offset = np.pi/2
        engine.add_edge("c1", "c2", weight=1.0, phase_offset=desired_offset)
        engine.add_edge("c2", "c1", weight=1.0, phase_offset=-desired_offset)
        
        # Run simulation
        for _ in range(300):  # Longer simulation to reach stability
            engine.step(dt=0.1)
        
        # Check if the phase difference matches the desired offset
        diff = (engine.phases["c1"] - engine.phases["c2"]) % (2 * np.pi)
        if diff > np.pi:
            diff = 2 * np.pi - diff
            
        self.assertAlmostEqual(diff, desired_offset, delta=0.05, 
                              msg="Phase offset was not maintained")
        
        # Check sync ratio is high even with offset
        self.assertGreater(engine.calculate_sync_ratio(), 0.95, 
                          "Sync ratio should be high with stable offset")
    
    def test_independent_clusters(self):
        """Test that disconnected clusters maintain independent phases."""
        engine = PhaseEngine(coupling_strength=0.3)
        
        # Create two separate clusters
        # Cluster 1: c1, c2, c3 (all starting at phase 0)
        engine.add_concept("c1", initial_phase=0.0)
        engine.add_concept("c2", initial_phase=0.0)
        engine.add_concept("c3", initial_phase=0.0)
        
        # Cluster 2: c4, c5, c6 (all starting at phase π)
        engine.add_concept("c4", initial_phase=np.pi)
        engine.add_concept("c5", initial_phase=np.pi)
        engine.add_concept("c6", initial_phase=np.pi)
        
        # Connect within clusters
        engine.add_edge("c1", "c2", weight=1.0)
        engine.add_edge("c2", "c3", weight=1.0)
        engine.add_edge("c3", "c1", weight=1.0)
        
        engine.add_edge("c4", "c5", weight=1.0)
        engine.add_edge("c5", "c6", weight=1.0)
        engine.add_edge("c6", "c4", weight=1.0)
        
        # Run simulation
        for _ in range(200):
            engine.step(dt=0.1)
        
        # Check clusters maintain separation
        avg_phase_cluster1 = np.mean([engine.phases[c] for c in ["c1", "c2", "c3"]])
        avg_phase_cluster2 = np.mean([engine.phases[c] for c in ["c4", "c5", "c6"]])
        
        phase_diff_between_clusters = abs((avg_phase_cluster1 - avg_phase_cluster2) % (2 * np.pi))
        if phase_diff_between_clusters > np.pi:
            phase_diff_between_clusters = 2 * np.pi - phase_diff_between_clusters
            
        self.assertGreater(phase_diff_between_clusters, 0.5, 
                          "Clusters should maintain separation")
        
        # Each cluster should be internally synchronized
        c1_diffs = [abs((engine.phases["c1"] - engine.phases[c]) % (2 * np.pi)) for c in ["c2", "c3"]]
        c4_diffs = [abs((engine.phases["c4"] - engine.phases[c]) % (2 * np.pi)) for c in ["c5", "c6"]]
        
        for diff in c1_diffs + c4_diffs:
            if diff > np.pi:
                diff = 2 * np.pi - diff
            self.assertLess(diff, 0.1, "Nodes within clusters should synchronize")

    def test_spectral_feedback_modulation(self):
        """Test that spectral feedback properly modulates coupling."""
        # Create two identical systems with different feedback
        engine1 = PhaseEngine(coupling_strength=0.3)
        engine2 = PhaseEngine(coupling_strength=0.3)
        
        # Same initial setup for both
        for engine in [engine1, engine2]:
            engine.add_concept("c1", initial_phase=0.0)
            engine.add_concept("c2", initial_phase=np.pi)
            engine.add_edge("c1", "c2", weight=1.0)
            engine.add_edge("c2", "c1", weight=1.0)
        
        # Set different spectral feedback
        engine1.set_spectral_feedback(1.0)  # Normal coupling
        engine2.set_spectral_feedback(0.1)  # Reduced coupling
        
        # Run both for same number of steps
        for _ in range(100):
            engine1.step(dt=0.1)
            engine2.step(dt=0.1)
        
        # The engine with higher feedback should synchronize faster
        sync1 = engine1.calculate_sync_ratio()
        sync2 = engine2.calculate_sync_ratio()
        
        self.assertGreater(sync1, sync2, 
                          "Higher spectral feedback should lead to faster synchronization")


    def test_convergence_line_graph(self):
        """Test phase convergence in a line graph topology."""
        # Create engine
        engine = PhaseEngine(coupling_strength=0.2)
        
        # Create a line graph with 5 nodes
        np.random.seed(42)  # For reproducibility
        for i in range(5):
            engine.add_concept(f"c{i}", initial_phase=np.random.rand() * 2 * np.pi)
            if i > 0:
                engine.add_edge(f"c{i-1}", f"c{i}")
        
        # Run simulation for 2000 steps to ensure convergence
        for _ in range(2000):
            engine.step(dt=0.02)
        
        # Extract phases
        phases = np.array(list(engine.phases.values()))
        
        # Check that all phases have converged to nearly the same value
        # by measuring the standard deviation relative to the first phase
        phase_diff = np.mod(phases - phases[0], 2 * np.pi)
        self.assertLess(np.std(phase_diff), 0.01, 
                        "Line graph did not converge to synchronized state")


if __name__ == "__main__":
    unittest.main()
