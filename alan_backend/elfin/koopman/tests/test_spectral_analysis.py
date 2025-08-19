"""
Unit tests for the Koopman spectral analysis components.

These tests verify the functionality of the SnapshotBuffer and SpectralAnalyzer
classes using synthetic data with known spectral properties.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from koopman.snapshot_buffer import SnapshotBuffer
from koopman.spectral_analyzer import SpectralAnalyzer


class TestSnapshotBuffer(unittest.TestCase):
    """Test cases for SnapshotBuffer."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization and basic properties."""
        buffer = SnapshotBuffer(capacity=50, state_dim=5)
        
        self.assertEqual(buffer.capacity, 50)
        self.assertEqual(buffer.state_dim, 5)
        self.assertEqual(len(buffer.buffer), 0)
        self.assertFalse(buffer.initialized)
    
    def test_add_snapshot_dict(self):
        """Test adding snapshots as dictionaries."""
        buffer = SnapshotBuffer(capacity=10)
        
        # Add first snapshot
        buffer.add_snapshot({'c1': 1.0, 'c2': 2.0, 'c3': 3.0})
        
        self.assertEqual(len(buffer.buffer), 1)
        self.assertEqual(buffer.state_dim, 3)
        self.assertEqual(set(buffer.concept_ids), {'c1', 'c2', 'c3'})
        self.assertTrue(buffer.initialized)
        
        # Add another snapshot with different order
        buffer.add_snapshot({'c3': 4.0, 'c1': 5.0, 'c2': 6.0})
        
        self.assertEqual(len(buffer.buffer), 2)
        
        # Check that order is preserved
        np.testing.assert_almost_equal(buffer.buffer[1], np.array([5.0, 6.0, 4.0]))
    
    def test_add_snapshot_array(self):
        """Test adding snapshots as arrays."""
        buffer = SnapshotBuffer(capacity=10, state_dim=3)
        
        # Add snapshots
        buffer.add_snapshot(np.array([1.0, 2.0, 3.0]))
        buffer.add_snapshot(np.array([4.0, 5.0, 6.0]))
        
        self.assertEqual(len(buffer.buffer), 2)
        np.testing.assert_almost_equal(buffer.buffer[0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_almost_equal(buffer.buffer[1], np.array([4.0, 5.0, 6.0]))
    
    def test_buffer_capacity(self):
        """Test that buffer respects capacity limit."""
        buffer = SnapshotBuffer(capacity=3, state_dim=2)
        
        # Add snapshots
        for i in range(5):
            buffer.add_snapshot(np.array([float(i), float(i+1)]))
        
        # Should only keep the latest 3
        self.assertEqual(len(buffer.buffer), 3)
        np.testing.assert_almost_equal(buffer.buffer[0], np.array([2.0, 3.0]))
        np.testing.assert_almost_equal(buffer.buffer[1], np.array([3.0, 4.0]))
        np.testing.assert_almost_equal(buffer.buffer[2], np.array([4.0, 5.0]))
    
    def test_get_snapshot_matrix(self):
        """Test retrieving snapshot matrix."""
        buffer = SnapshotBuffer(capacity=3, state_dim=2)
        
        # Add snapshots
        buffer.add_snapshot(np.array([1.0, 2.0]))
        buffer.add_snapshot(np.array([3.0, 4.0]))
        buffer.add_snapshot(np.array([5.0, 6.0]))
        
        # Get matrix
        matrix, timestamps = buffer.get_snapshot_matrix()
        
        # Check dimensions
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(len(timestamps), 3)
        
        # Check values
        np.testing.assert_almost_equal(matrix[:, 0], np.array([1.0, 2.0]))
        np.testing.assert_almost_equal(matrix[:, 1], np.array([3.0, 4.0]))
        np.testing.assert_almost_equal(matrix[:, 2], np.array([5.0, 6.0]))
    
    def test_get_time_shifted_matrices(self):
        """Test retrieving time-shifted matrices."""
        buffer = SnapshotBuffer(capacity=5, state_dim=2)
        
        # Add snapshots
        for i in range(5):
            buffer.add_snapshot(np.array([float(i), float(i+1)]))
        
        # Get matrices with shift=1
        X, Y = buffer.get_time_shifted_matrices(shift=1)
        
        # Check dimensions
        self.assertEqual(X.shape, (2, 4))
        self.assertEqual(Y.shape, (2, 4))
        
        # Check values
        np.testing.assert_almost_equal(X[:, 0], np.array([0.0, 1.0]))
        np.testing.assert_almost_equal(Y[:, 0], np.array([1.0, 2.0]))
        
        # Get matrices with shift=2
        X, Y = buffer.get_time_shifted_matrices(shift=2)
        
        # Check dimensions
        self.assertEqual(X.shape, (2, 3))
        self.assertEqual(Y.shape, (2, 3))
        
        # Check values
        np.testing.assert_almost_equal(X[:, 0], np.array([0.0, 1.0]))
        np.testing.assert_almost_equal(Y[:, 0], np.array([2.0, 3.0]))
    
    def test_export_import(self):
        """Test exporting and importing buffer data."""
        buffer = SnapshotBuffer(capacity=3, state_dim=2)
        
        # Add snapshots
        buffer.add_snapshot({'c1': 1.0, 'c2': 2.0})
        buffer.add_snapshot({'c1': 3.0, 'c2': 4.0})
        
        # Export data
        data = buffer.export_data()
        
        # Create new buffer from exported data
        new_buffer = SnapshotBuffer.from_dict(data)
        
        # Check properties
        self.assertEqual(new_buffer.state_dim, 2)
        self.assertEqual(len(new_buffer.buffer), 2)
        self.assertEqual(new_buffer.concept_ids, buffer.concept_ids)
        
        # Check snapshot values
        np.testing.assert_almost_equal(new_buffer.buffer[0], buffer.buffer[0])
        np.testing.assert_almost_equal(new_buffer.buffer[1], buffer.buffer[1])


class TestSpectralAnalyzer(unittest.TestCase):
    """Test cases for SpectralAnalyzer."""
    
    def generate_synthetic_data(self, num_points=100, noise_level=0.01):
        """Generate synthetic data with known spectral properties."""
        # Create time vector
        t = np.linspace(0, 10, num_points)
        
        # Create state trajectories with known modes
        # Mode 1: Growing oscillation
        x1 = np.exp(0.1 * t) * np.sin(2 * np.pi * 0.5 * t)
        
        # Mode 2: Decaying oscillation
        x2 = np.exp(-0.2 * t) * np.cos(2 * np.pi * 0.3 * t)
        
        # Mode 3: Exponential growth
        x3 = 0.01 * np.exp(0.05 * t)
        
        # Add noise
        x1 += noise_level * np.random.randn(num_points)
        x2 += noise_level * np.random.randn(num_points)
        x3 += noise_level * np.random.randn(num_points)
        
        # Create snapshot buffer
        buffer = SnapshotBuffer(capacity=num_points, state_dim=3)
        
        # Add snapshots
        for i in range(num_points):
            buffer.add_snapshot(np.array([x1[i], x2[i], x3[i]]), timestamp=t[i])
        
        return buffer, {'growth_rates': [0.1, -0.2, 0.05], 'frequencies': [0.5, 0.3, 0.0]}
    
    def test_edmd_decomposition(self):
        """Test EDMD spectral decomposition with synthetic data."""
        # Generate synthetic data
        buffer, true_props = self.generate_synthetic_data(num_points=100)
        
        # Create analyzer
        analyzer = SpectralAnalyzer(buffer)
        
        # Perform decomposition
        result = analyzer.edmd_decompose(time_shift=1)
        
        # Check results
        self.assertIsNotNone(result)
        self.assertEqual(len(result.eigenvalues), 3)
        self.assertEqual(result.modes.shape, (3, 3))
        
        # Sort modes by amplitude for comparison
        sorted_indices = np.argsort(result.amplitudes)[::-1]
        sorted_freqs = result.frequencies[sorted_indices]
        sorted_growth = result.growth_rates[sorted_indices]
        
        # Check that frequencies and growth rates are close to expected values
        # We may not get exact match due to noise and numerical issues
        freq_errors = np.min(
            np.abs(np.array([sorted_freqs]) - np.array([true_props['frequencies']]).T),
            axis=1
        )
        growth_errors = np.min(
            np.abs(np.array([sorted_growth]) - np.array([true_props['growth_rates']]).T),
            axis=1
        )
        
        # Check that errors are small
        self.assertTrue(np.all(freq_errors < 0.1), "Frequency errors too large")
        self.assertTrue(np.all(growth_errors < 0.1), "Growth rate errors too large")
    
    def test_stability_detection(self):
        """Test detection of unstable modes."""
        # Generate synthetic data with one unstable mode
        buffer, _ = self.generate_synthetic_data(num_points=100)
        
        # Create analyzer
        analyzer = SpectralAnalyzer(buffer)
        
        # Perform decomposition
        analyzer.edmd_decompose(time_shift=1)
        
        # There should be at least one unstable mode (growth rate > 0)
        self.assertGreater(len(analyzer.unstable_modes), 0)
        
        # The stability index should be negative (unstable)
        self.assertLess(analyzer.calculate_stability_index(), 0)
        
        # Spectral feedback should be < 1.0 for unstable system
        feedback = analyzer.get_spectral_feedback()
        self.assertLess(feedback, 1.0)
        self.assertGreater(feedback, 0.0)
    
    def test_prediction(self):
        """Test future state prediction."""
        # Generate synthetic data
        buffer, _ = self.generate_synthetic_data(num_points=100, noise_level=0.0)
        
        # Create analyzer
        analyzer = SpectralAnalyzer(buffer)
        
        # Perform decomposition
        analyzer.edmd_decompose(time_shift=1)
        
        # Predict one step ahead
        latest_state = buffer.buffer[-1]
        predicted_state = analyzer.predict_future_state(steps=1)
        
        # Check prediction shape
        self.assertEqual(predicted_state.shape, latest_state.shape)
        
        # For this synthetic data, we expect prediction to capture growth/decay
        # Get actual next state (for verification only, in real use we wouldn't have this)
        if len(buffer.buffer) >= 5:
            # Use approximation of derivative to check trend
            trend_actual = buffer.buffer[-1] - buffer.buffer[-2]
            trend_predicted = predicted_state - buffer.buffer[-1]
            
            # Signs of trends should mostly match
            sign_matches = np.sign(trend_actual) == np.sign(trend_predicted)
            self.assertGreaterEqual(np.sum(sign_matches), 2)  # At least 2/3 signs match
    
    def test_results_export(self):
        """Test exporting analysis results."""
        # Generate synthetic data
        buffer, _ = self.generate_synthetic_data(num_points=50)
        
        # Create analyzer
        analyzer = SpectralAnalyzer(buffer)
        
        # Perform decomposition
        analyzer.edmd_decompose(time_shift=1)
        
        # Export results
        results = analyzer.export_results()
        
        # Check key fields
        self.assertIn('eigenvalues', results)
        self.assertIn('amplitudes', results)
        self.assertIn('frequencies', results)
        self.assertIn('growth_rates', results)
        self.assertIn('dominant_modes', results)
        self.assertIn('unstable_modes', results)
        self.assertIn('stability_index', results)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            analyzer.save_results(Path(tmp.name))
            
            # Check file exists
            self.assertTrue(os.path.exists(tmp.name))
            
            # Clean up
            os.unlink(tmp.name)


    def test_edmd_linear_system(self):
        """Test EDMD on a linear system with known eigenvalues."""
        # Create a linear system with known eigenvalues
        A = np.array([
            [0, 1],
            [-2, -3]
        ])
        # Eigenvalues of A are -1 and -2
        true_eigs = np.linalg.eigvals(A)
        
        # Create snapshot buffer
        buffer = SnapshotBuffer(capacity=100, state_dim=2)
        
        # Generate trajectory data
        x = np.random.randn(2)  # Initial state
        dt = 0.01  # Time step
        
        # Collect snapshots
        for t in range(100):
            buffer.add_snapshot(x, timestamp=t*dt)
            # Evolve state: x_{t+1} = x_t + A*x_t*dt (Euler integration)
            x = x + A @ x * dt
        
        # Perform EDMD
        analyzer = SpectralAnalyzer(buffer)
        result = analyzer.edmd_decompose(time_shift=1)
        
        # Extract continuous-time eigenvalues (log(λ)/dt)
        cont_eigs = np.log(result.eigenvalues) / dt
        real_parts = np.real(cont_eigs)
        
        # Sort both sets of eigenvalues
        true_eigs_sorted = np.sort(np.real(true_eigs))
        edmd_eigs_sorted = np.sort(real_parts)
        
        # Check accuracy - allow some tolerance due to finite data and numerical issues
        self.assertTrue(np.allclose(edmd_eigs_sorted, true_eigs_sorted, atol=0.2),
                        f"EDMD eigenvalues {edmd_eigs_sorted} don't match true values {true_eigs_sorted}")


if __name__ == "__main__":
    unittest.main()
