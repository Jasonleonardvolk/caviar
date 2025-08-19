#!/usr/bin/env python3
"""
TORI Self-Transformation Integration Test
Tests all components of the phase-coherent cognition system
"""

import sys
import os
import json
import tempfile
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safety.constitution import Constitution
from meta_genome.critics.aggregation import aggregate, logodds
from meta.energy_budget import EnergyBudget
from goals.analogical_transfer import AnalogicalTransfer
from audit.logger import log_event
import numpy as np

class TestSelfTransformation(unittest.TestCase):
    """Test suite for TORI self-transformation components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
    def test_constitution_validation(self):
        """Test constitutional safety boundaries"""
        const = Constitution(path="safety/constitution.json")
        
        # Test resource budget assertion
        class Usage:
            cpu = 100
            gpu = 50
            ram = 1024 * 1024 * 1024  # 1GB
        
        # Should pass
        const.assert_resource_budget(Usage())
        
        # Test forbidden call blocking
        with self.assertRaises(PermissionError):
            const.assert_action("eval")
    
    def test_critic_aggregation(self):
        """Test weighted critic consensus"""
        # Test with uniform scores and reliabilities
        scores = {"c1": 0.8, "c2": 0.8, "c3": 0.8}
        reliabilities = {"c1": 0.9, "c2": 0.9, "c3": 0.9}
        accepted, score = aggregate(scores, reliabilities)
        self.assertTrue(accepted)
        self.assertAlmostEqual(score, 0.8, places=2)
        
        # Test with mixed scores
        scores = {"c1": 0.9, "c2": 0.3, "c3": 0.6}
        reliabilities = {"c1": 0.95, "c2": 0.5, "c3": 0.7}
        accepted, score = aggregate(scores, reliabilities)
        self.assertGreater(score, 0.5)  # Should weight towards reliable critics
    
    def test_energy_budget(self):
        """Test energy management system"""
        energy = EnergyBudget(max_energy=100.0)
        
        # Test normal operation
        allowed = energy.update(1.0, 5.0)
        self.assertTrue(allowed)
        self.assertLess(energy.current_energy, 100.0)
        
        # Test efficiency calculation
        energy.update(2.0, 10.0)
        efficiency = energy.get_efficiency()
        self.assertGreater(efficiency, 0)
        self.assertLessEqual(efficiency, 1.0)
    
    def test_analogical_transfer(self):
        """Test cross-domain knowledge transfer"""
        transfer = AnalogicalTransfer()
        
        # Add test domains
        transfer.add_knowledge_cluster("A", ["a1", "a2"], np.array([1.0, 0.0]))
        transfer.add_knowledge_cluster("B", ["b1", "b2"], np.array([0.8, 0.6]))
        transfer.add_knowledge_cluster("C", ["c1", "c2"], np.array([0.0, 1.0]))
        
        # Test transfer weights
        weight_ab = transfer.get_transfer_weights("A", "B")
        weight_ac = transfer.get_transfer_weights("A", "C")
        self.assertGreater(weight_ab, weight_ac)  # A-B should be closer than A-C
        
        # Test strategy transfer
        strategy = {"param1": 10, "param2": "test"}
        transferred = transfer.transfer_strategy("A", "B", strategy)
        self.assertIn("param1", transferred)
        self.assertIn("param2_confidence", transferred)
    
    def test_audit_logging(self):
        """Test audit trail functionality"""
        # Create a test event
        log_event("test_event", {"action": "test", "result": "success"})
        
        # Verify log file exists
        log_file = Path("audit/events.log")
        self.assertTrue(log_file.exists())
        
        # Verify log content
        with open(log_file, "r") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            last_event = json.loads(lines[-1])
            self.assertEqual(last_event["type"], "test_event")
            self.assertEqual(last_event["data"]["result"], "success")
    
    def test_logodds_function(self):
        """Test log-odds transformation"""
        # Test boundary conditions
        self.assertAlmostEqual(logodds(0.5), 0.0, places=5)
        self.assertLess(logodds(0.1), 0)
        self.assertGreater(logodds(0.9), 0)
        
        # Test extreme values handling
        self.assertIsFinite(logodds(0.0))
        self.assertIsFinite(logodds(1.0))
    
    def assertIsFinite(self, value):
        """Helper to check if value is finite"""
        self.assertFalse(np.isnan(value))
        self.assertFalse(np.isinf(value))

def run_integration_tests():
    """Run all integration tests"""
    print("Running TORI Self-Transformation Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSelfTransformation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
