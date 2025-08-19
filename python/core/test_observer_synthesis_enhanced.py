#!/usr/bin/env python3
"""
Comprehensive tests for enhanced Observer Synthesis.
Tests all safety features, error handling, and performance improvements.
"""

import unittest
import numpy as np
import json
import time
import threading
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import the enhanced version
from observer_synthesis_enhanced import (
    ObserverObservedSynthesis,
    SelfMeasurement,
    MeasurementError,
    RefexBudgetExhausted,
    get_observer_synthesis,
    VALID_COHERENCE_STATES,
    MAX_EIGENVALUE_DIM,
    FORCE_MEASUREMENT_LIMIT
)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


class TestInputValidation(unittest.TestCase):
    """Test comprehensive input validation."""
    
    def setUp(self):
        self.synthesis = ObserverObservedSynthesis()
    
    def test_eigenvalue_validation(self):
        """Test eigenvalue input validation."""
        # Test None eigenvalues
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure(None, 'local', 0.5)
        self.assertIn("cannot be None", str(cm.exception))
        
        # Test non-numpy array
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure([1, 2, 3], 'local', 0.5)
        self.assertIn("must be a numpy array", str(cm.exception))
        
        # Test empty array
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure(np.array([]), 'local', 0.5)
        self.assertIn("cannot be empty", str(cm.exception))
        
        # Test infinite values
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure(np.array([1, np.inf, 3]), 'local', 0.5)
        self.assertIn("finite values", str(cm.exception))
        
        # Test NaN values
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure(np.array([1, np.nan, 3]), 'local', 0.5)
        self.assertIn("finite values", str(cm.exception))
        
        # Test oversized array
        huge_array = np.ones(MAX_EIGENVALUE_DIM + 1)
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure(huge_array, 'local', 0.5)
        self.assertIn("exceeds maximum", str(cm.exception))
    
    def test_coherence_state_validation(self):
        """Test coherence state validation."""
        eigenvalues = np.array([0.1, 0.2, 0.3])
        
        # Test invalid coherence state
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure(eigenvalues, 'invalid', 0.5)
        self.assertIn("must be one of", str(cm.exception))
        
        # Test non-string coherence state
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure(eigenvalues, 123, 0.5)
        self.assertIn("must be a string", str(cm.exception))
        
        # Test all valid states
        for state in VALID_COHERENCE_STATES:
            result = self.synthesis.measure(eigenvalues, state, 0.5)
            self.assertIsNotNone(result)
    
    def test_novelty_score_validation(self):
        """Test novelty score validation."""
        eigenvalues = np.array([0.1, 0.2, 0.3])
        
        # Test out of range values
        for invalid_novelty in [-0.1, 1.1, 2.0, -1.0]:
            with self.assertRaises(ValueError) as cm:
                self.synthesis.measure(eigenvalues, 'local', invalid_novelty)
            self.assertIn("between 0 and 1", str(cm.exception))
        
        # Test non-numeric values
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure(eigenvalues, 'local', "high")
        self.assertIn("must be numeric", str(cm.exception))
        
        # Test infinite novelty
        with self.assertRaises(ValueError) as cm:
            self.synthesis.measure(eigenvalues, 'local', np.inf)
        self.assertIn("must be finite", str(cm.exception))
        
        # Test valid boundary values
        for valid_novelty in [0.0, 0.5, 1.0]:
            result = self.synthesis.measure(eigenvalues, 'local', valid_novelty)
            self.assertIsNotNone(result)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of enhanced implementation."""
    
    def setUp(self):
        self.synthesis = ObserverObservedSynthesis(reflex_budget=1000)
    
    def test_concurrent_measurements(self):
        """Test concurrent measurement operations."""
        results = []
        errors = []
        
        def measure_task(i):
            try:
                eigenvalues = np.random.randn(5) * 0.1
                coherence = ['local', 'global', 'critical'][i % 3]
                novelty = (i % 10) / 10.0
                
                measurement = self.synthesis.measure(
                    eigenvalues, coherence, novelty, force=True
                )
                return ('success', measurement)
            except Exception as e:
                return ('error', str(e))
        
        # Run concurrent measurements
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(measure_task, i) for i in range(100)]
            
            for future in as_completed(futures):
                result_type, result = future.result()
                if result_type == 'success':
                    results.append(result)
                else:
                    errors.append(result)
        
        # Verify results
        self.assertGreater(len(results), 50)  # Most should succeed
        self.assertLess(len(errors), 50)  # Some may fail due to cooldown
        
        # Check measurement integrity
        hashes = [m.spectral_hash for m in results if m]
        self.assertEqual(len(hashes), len(set(hashes)))  # All unique (probabilistically)
    
    def test_concurrent_context_generation(self):
        """Test concurrent metacognitive context generation."""
        # Add some measurements first
        for i in range(10):
            eigenvalues = np.random.randn(3) * 0.1
            self.synthesis.measure(eigenvalues, 'local', 0.5, force=True)
            time.sleep(0.01)
        
        contexts = []
        
        def get_context_task():
            return self.synthesis.generate_metacognitive_context()
        
        # Generate contexts concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_context_task) for _ in range(20)]
            
            for future in as_completed(futures):
                contexts.append(future.result())
        
        # Verify all contexts are valid
        self.assertEqual(len(contexts), 20)
        for context in contexts:
            self.assertIn('health', context)
            self.assertIn('metacognitive_tokens', context)
            self.assertIn('measurement_count', context)
    
    def test_concurrent_save_load(self):
        """Test concurrent save/load operations."""
        # Add measurements
        for i in range(5):
            eigenvalues = np.random.randn(3) * 0.1
            self.synthesis.measure(eigenvalues, 'local', 0.5, force=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            def save_task(i):
                path = tmppath / f"measurements_{i}.json"
                self.synthesis.save_measurements(path)
                return path
            
            # Save concurrently
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(save_task, i) for i in range(5)]
                paths = [f.result() for f in as_completed(futures)]
            
            # Verify all saves succeeded
            self.assertEqual(len(paths), 5)
            for path in paths:
                self.assertTrue(path.exists())
                
                # Verify content
                with open(path) as f:
                    data = json.load(f)
                self.assertIn('measurements', data)
                self.assertIn('metadata', data)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and custom exceptions."""
    
    def setUp(self):
        self.synthesis = ObserverObservedSynthesis()
    
    def test_measurement_error_tracking(self):
        """Test that measurement errors are properly tracked."""
        # Create a bad operator that raises an error
        def bad_operator(eigenvalues, coherence_state, novelty_score):
            raise RuntimeError("Operator failure")
        
        self.synthesis.register_operator('bad', bad_operator)
        
        # Try to use the bad operator
        eigenvalues = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(MeasurementError):
            self.synthesis.measure(eigenvalues, 'local', 0.5, operator='bad')
        
        # Check error tracking
        health = self.synthesis.get_health_status()
        self.assertEqual(health['failed_measurements'], 1)
        self.assertIsNotNone(self.synthesis.last_error)
        self.assertIn('Operator failure', self.synthesis.last_error['error'])
    
    def test_reflex_budget_exhaustion(self):
        """Test reflex budget exhaustion handling."""
        # Create synthesis with tiny budget
        synthesis = ObserverObservedSynthesis(reflex_budget=3)
        
        eigenvalues = np.array([0.1, 0.2, 0.3])
        
        # Use up the budget
        for i in range(3):
            result = synthesis.measure(eigenvalues, 'local', 0.5)
            self.assertIsNotNone(result)
            time.sleep(0.11)  # Wait past cooldown
        
        # Next measurement should fail
        result = synthesis.measure(eigenvalues, 'local', 0.5)
        self.assertIsNone(result)
        
        # Force should still work but with limit
        for i in range(FORCE_MEASUREMENT_LIMIT):
            result = synthesis.measure(eigenvalues, 'local', 0.5, force=True)
            self.assertIsNotNone(result)
            time.sleep(0.01)
        
        # Exceed force limit
        with self.assertRaises(RefexBudgetExhausted):
            synthesis.measure(eigenvalues, 'local', 0.5, force=True)
    
    def test_file_operation_errors(self):
        """Test file operation error handling."""
        # Test save to invalid path
        with self.assertRaises(IOError):
            self.synthesis.save_measurements(Path("/invalid/path/measurements.json"))
        
        # Test load non-existent file
        with self.assertRaises(IOError):
            self.synthesis.load_measurements(Path("non_existent.json"))
        
        # Test load corrupted file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json")
            temp_path = Path(f.name)
        
        try:
            with self.assertRaises(IOError):
                self.synthesis.load_measurements(temp_path)
        finally:
            temp_path.unlink()


class TestOscillationDetection(unittest.TestCase):
    """Test enhanced oscillation detection."""
    
    def setUp(self):
        self.synthesis = ObserverObservedSynthesis()
    
    def test_two_cycle_detection(self):
        """Test A-B-A-B pattern detection."""
        eigenvalues1 = np.array([0.1, 0.2, 0.3])
        eigenvalues2 = np.array([0.4, 0.5, 0.6])
        
        # Create A-B-A-B pattern
        states = ['local', 'global', 'local', 'global']
        for i, state in enumerate(states):
            eigs = eigenvalues1 if i % 2 == 0 else eigenvalues2
            self.synthesis.measure(eigs, state, 0.5, force=True)
            time.sleep(0.01)
        
        # Check oscillation detected
        self.assertTrue(self.synthesis.reflexive_mode)
        self.assertGreater(self.synthesis.oscillation_count, 0)
    
    def test_three_cycle_detection(self):
        """Test A-B-C-A-B-C pattern detection."""
        eigenvalues = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.7, 0.8, 0.9])
        ]
        
        # Create A-B-C-A-B-C pattern
        for cycle in range(2):
            for i in range(3):
                self.synthesis.measure(
                    eigenvalues[i], 
                    ['local', 'global', 'critical'][i], 
                    0.5, 
                    force=True
                )
                time.sleep(0.01)
        
        # Check oscillation detected
        self.assertTrue(self.synthesis.reflexive_mode)
    
    def test_oscillation_recovery(self):
        """Test recovery from oscillation mode."""
        # First induce oscillation
        eigenvalues1 = np.array([0.1, 0.2, 0.3])
        eigenvalues2 = np.array([0.4, 0.5, 0.6])
        
        # Create oscillation
        for i in range(4):
            eigs = eigenvalues1 if i % 2 == 0 else eigenvalues2
            self.synthesis.measure(eigs, 'local', 0.5, force=True)
        
        self.assertTrue(self.synthesis.reflexive_mode)
        
        # Now break the pattern with diverse measurements
        for i in range(10):
            eigs = np.random.randn(3) * 0.1
            state = np.random.choice(['local', 'global', 'critical'])
            self.synthesis.measure(eigs, state, 0.5, force=True)
            time.sleep(0.01)
        
        # Update last check time to trigger recovery check
        self.synthesis.last_oscillation_check = datetime.now(timezone.utc) - timedelta(seconds=61)
        
        # One more measurement to trigger check
        self.synthesis.measure(np.random.randn(3) * 0.1, 'local', 0.5, force=True)
        
        # Should have recovered
        self.assertFalse(self.synthesis.reflexive_mode)


class TestPerformanceAndHealth(unittest.TestCase):
    """Test performance tracking and health monitoring."""
    
    def setUp(self):
        self.synthesis = ObserverObservedSynthesis()
    
    def test_performance_tracking(self):
        """Test measurement performance tracking."""
        # Perform several measurements
        for i in range(10):
            eigenvalues = np.random.randn(5) * 0.1
            self.synthesis.measure(eigenvalues, 'local', 0.5, force=True)
            time.sleep(0.01)
        
        # Check performance metrics
        health = self.synthesis.get_health_status()
        self.assertIn('avg_measurement_time_ms', health)
        self.assertGreater(health['avg_measurement_time_ms'], 0)
        self.assertLess(health['avg_measurement_time_ms'], 100)  # Should be fast
    
    def test_health_status(self):
        """Test comprehensive health status reporting."""
        # Perform some successful measurements
        for i in range(5):
            eigenvalues = np.random.randn(3) * 0.1
            self.synthesis.measure(eigenvalues, 'local', 0.5, force=True)
        
        # Induce some failures
        def bad_operator(e, c, n):
            raise RuntimeError("Test error")
        
        self.synthesis.register_operator('bad', bad_operator)
        
        for i in range(2):
            try:
                self.synthesis.measure(np.array([0.1]), 'local', 0.5, operator='bad')
            except:
                pass
        
        # Check health status
        health = self.synthesis.get_health_status()
        
        self.assertEqual(health['total_measurements'], 5)
        self.assertEqual(health['failed_measurements'], 2)
        self.assertAlmostEqual(health['error_rate'], 2/7, places=2)
        self.assertIn('status', health)
        self.assertIn('reflex_budget_remaining', health)
        self.assertIn('measurement_history_size', health)
        self.assertIsNotNone(health['last_error'])
    
    def test_metacognitive_context_warnings(self):
        """Test warning generation in metacognitive context."""
        # Use up most of reflex budget
        synthesis = ObserverObservedSynthesis(reflex_budget=15)
        
        for i in range(12):
            synthesis.measure(np.random.randn(3) * 0.1, 'local', 0.5)
            time.sleep(0.11)
        
        context = synthesis.generate_metacognitive_context()
        self.assertIn('warnings', context)
        self.assertIn('LOW_REFLEX_BUDGET', context['warnings'])
        
        # Induce oscillation
        for i in range(4):
            eigs = np.array([0.1, 0.2]) if i % 2 == 0 else np.array([0.3, 0.4])
            synthesis.measure(eigs, 'local', 0.5, force=True)
        
        context = synthesis.generate_metacognitive_context()
        self.assertIn('REFLEXIVE_OSCILLATION_DETECTED', context['warnings'])


class TestStochasticMeasurement(unittest.TestCase):
    """Test stochastic measurement functionality."""
    
    def setUp(self):
        self.synthesis = ObserverObservedSynthesis()
    
    def test_stochastic_probability(self):
        """Test stochastic measurement probability adjustments."""
        np.random.seed(42)  # For reproducibility
        
        measurements = []
        eigenvalues = np.array([0.1, 0.2, 0.3])
        
        # Test with different novelty scores
        for novelty in [0.0, 0.5, 1.0]:
            successes = 0
            attempts = 100
            
            for _ in range(attempts):
                result = self.synthesis.apply_stochastic_measurement(
                    eigenvalues, 'local', novelty, base_probability=0.5
                )
                if result:
                    successes += 1
            
            # Higher novelty should lead to more measurements
            measurements.append((novelty, successes / attempts))
        
        # Verify increasing trend
        self.assertLess(measurements[0][1], measurements[2][1])
    
    def test_stochastic_oscillation_suppression(self):
        """Test probability reduction during oscillation."""
        # Induce oscillation
        for i in range(4):
            eigs = np.array([0.1]) if i % 2 == 0 else np.array([0.2])
            self.synthesis.measure(eigs, 'local', 0.5, force=True)
        
        self.assertTrue(self.synthesis.reflexive_mode)
        
        # Test stochastic measurement during oscillation
        successes = 0
        for _ in range(100):
            result = self.synthesis.apply_stochastic_measurement(
                np.array([0.3]), 'local', 0.5, base_probability=0.5
            )
            if result:
                successes += 1
        
        # Should have very low success rate during oscillation
        self.assertLess(successes / 100, 0.1)


class TestAtomicFileOperations(unittest.TestCase):
    """Test atomic file save/load operations."""
    
    def setUp(self):
        self.synthesis = ObserverObservedSynthesis()
        
    def test_atomic_save(self):
        """Test atomic save operation."""
        # Add measurements
        for i in range(5):
            self.synthesis.measure(np.random.randn(3) * 0.1, 'local', 0.5, force=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "measurements.json"
            
            # Save measurements
            self.synthesis.save_measurements(save_path)
            
            # Verify file exists and is valid
            self.assertTrue(save_path.exists())
            
            with open(save_path) as f:
                data = json.load(f)
            
            self.assertIn('measurements', data)
            self.assertIn('metadata', data)
            self.assertEqual(data['metadata']['version'], '2.0')
            self.assertEqual(len(data['measurements']), 5)
    
    def test_save_load_integrity(self):
        """Test save/load maintains data integrity."""
        # Create measurements with various states
        original_measurements = []
        
        for i in range(10):
            eigenvalues = np.random.randn(3 + i % 3) * 0.1
            coherence = ['local', 'global', 'critical'][i % 3]
            novelty = i / 10.0
            
            measurement = self.synthesis.measure(
                eigenvalues, coherence, novelty, force=True
            )
            if measurement:
                original_measurements.append(measurement.to_dict())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_integrity.json"
            
            # Save
            self.synthesis.save_measurements(save_path)
            
            # Create new instance and load
            new_synthesis = ObserverObservedSynthesis()
            new_synthesis.load_measurements(save_path)
            
            # Compare measurements
            loaded_measurements = [
                m.to_dict() for m in new_synthesis.measurement_history
            ]
            
            self.assertEqual(len(original_measurements), len(loaded_measurements))
            
            for orig, loaded in zip(original_measurements, loaded_measurements):
                self.assertEqual(orig['spectral_hash'], loaded['spectral_hash'])
                self.assertEqual(orig['coherence_state'], loaded['coherence_state'])
                self.assertAlmostEqual(orig['novelty_score'], loaded['novelty_score'])


class TestCustomOperators(unittest.TestCase):
    """Test custom operator registration and validation."""
    
    def setUp(self):
        self.synthesis = ObserverObservedSynthesis()
    
    def test_operator_registration(self):
        """Test custom operator registration."""
        def custom_op(eigenvalues, coherence_state, novelty_score):
            measurement = self.synthesis._spectral_hash_operator(
                eigenvalues, coherence_state, novelty_score
            )
            measurement.measurement_operator = 'custom_test'
            measurement.metacognitive_tokens.append('CUSTOM_TOKEN')
            return measurement
        
        # Register operator
        self.synthesis.register_operator('custom_test', custom_op)
        
        # Use custom operator
        result = self.synthesis.measure(
            np.array([0.1, 0.2]), 'local', 0.5, operator='custom_test'
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.measurement_operator, 'custom_test')
        self.assertIn('CUSTOM_TOKEN', result.metacognitive_tokens)
    
    def test_invalid_operator_registration(self):
        """Test validation of operator registration."""
        # Invalid name
        with self.assertRaises(ValueError):
            self.synthesis.register_operator('invalid-name!', lambda: None)
        
        # Too long name
        with self.assertRaises(ValueError):
            self.synthesis.register_operator('a' * 51, lambda: None)
        
        # Non-callable
        with self.assertRaises(ValueError):
            self.synthesis.register_operator('not_callable', "not a function")


if __name__ == '__main__':
    unittest.main(verbosity=2)
