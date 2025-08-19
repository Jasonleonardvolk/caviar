"""
BPS Soliton Test Suite
Comprehensive tests for BPS soliton functionality
"""

import unittest
import numpy as np
import json
import tempfile
from datetime import datetime

from python.core.bps_config_enhanced import SolitonPolarity, BPS_CONFIG
from python.core.bps_oscillator_enhanced import BPSEnhancedLattice, Oscillator
from python.core.bps_soliton_memory_enhanced import BPSEnhancedSolitonMemory
from python.core.bps_blowup_harness import BPSBlowupHarness
from python.core.bps_hot_swap_laplacian import BPSHotSwapLaplacian
from python.monitoring.bps_diagnostics import BPSDiagnostics


class TestBPSOscillator(unittest.TestCase):
    """Test BPS oscillator behavior"""
    
    def setUp(self):
        self.lattice = BPSEnhancedLattice(size=10)
    
    def test_bps_energy_saturation(self):
        """Test that BPS oscillators maintain E = |Q|"""
        # Create BPS oscillator
        self.lattice.create_bps_soliton(0, charge=2.0)
        
        # Try to increase energy
        osc = self.lattice.oscillator_objects[0]
        osc.amplitude = 5.0  # Try to set high amplitude
        
        # Step the oscillator
        osc.step(1.0, 0.01)  # Apply coupling
        
        # Check energy is clamped to |Q|
        expected_amplitude = 2.0  # sqrt(|Q|^2) = |Q| = 2.0
        self.assertAlmostEqual(osc.amplitude, expected_amplitude, delta=0.5)
    
    def test_bps_phase_protection(self):
        """Test that BPS oscillators don't get phase-pulled"""
        # Create BPS and bright oscillators
        self.lattice.create_bps_soliton(0, charge=1.0, phase=0.0)
        self.lattice.oscillator_objects[1].polarity = SolitonPolarity.BRIGHT
        self.lattice.oscillator_objects[1].phase = np.pi
        
        # Store initial BPS phase
        initial_phase = self.lattice.oscillator_objects[0].phase
        
        # Run several steps with strong coupling
        self.lattice.coupling_strength = 10.0
        for _ in range(100):
            self.lattice.step_enhanced()
        
        # BPS phase should only change by natural frequency
        bps_osc = self.lattice.oscillator_objects[0]
        expected_phase = (initial_phase + bps_osc.frequency * 0.01 * 100) % (2 * np.pi)
        self.assertAlmostEqual(bps_osc.phase, expected_phase, delta=0.1)
    
    def test_mixed_polarity_coexistence(self):
        """Test bright, dark, and BPS solitons coexisting"""
        # Create mixed solitons
        self.lattice.create_bps_soliton(0, charge=1.0)
        self.lattice.oscillator_objects[1].polarity = SolitonPolarity.BRIGHT
        self.lattice.oscillator_objects[2].polarity = SolitonPolarity.DARK
        
        # Run simulation
        for _ in range(50):
            self.lattice.step_enhanced()
        
        # Check all types still exist
        polarities = [osc.polarity for osc in self.lattice.oscillator_objects[:3]]
        self.assertIn(SolitonPolarity.BPS, polarities)
        self.assertIn(SolitonPolarity.BRIGHT, polarities)
        self.assertIn(SolitonPolarity.DARK, polarities)
        
        # Verify no crashes or instabilities
        report = self.lattice.get_bps_report()
        self.assertEqual(report["num_bps_solitons"], 1)


class TestBPSMemoryIntegration(unittest.TestCase):
    """Test BPS soliton memory storage"""
    
    def setUp(self):
        self.lattice = BPSEnhancedLattice(size=20)
        self.memory = BPSEnhancedSolitonMemory(self.lattice)
    
    def test_store_retrieve_bps(self):
        """Test storing and retrieving BPS soliton memory"""
        # Store BPS memory
        memory_id = self.memory.store_bps_soliton(
            content="Topologically protected memory",
            concept_ids=["quantum", "topology"],
            charge=1.5,
            metadata={"importance": "high"}
        )
        
        self.assertIsNotNone(memory_id)
        
        # Retrieve memory
        entry = self.memory.retrieve_bps_memory(memory_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.polarity, SolitonPolarity.BPS)
        self.assertEqual(entry.charge, 1.5)
        self.assertTrue(entry.energy_locked)
    
    def test_charge_conservation_in_memory(self):
        """Test that total charge is tracked correctly"""
        # Store multiple BPS solitons
        charges = [1.0, -1.0, 2.0, -0.5]
        for i, charge in enumerate(charges):
            self.memory.store_bps_soliton(
                content=f"Memory {i}",
                concept_ids=[f"concept_{i}"],
                charge=charge
            )
        
        # Check total charge
        total_charge = self.memory.get_total_topological_charge()
        expected = sum(charges)
        self.assertAlmostEqual(total_charge, expected, delta=1e-6)
        
        # Verify with lattice
        self.assertTrue(self.memory.verify_charge_conservation())


class TestBPSBlowupHarness(unittest.TestCase):
    """Test BPS-aware blow-up harness"""
    
    def setUp(self):
        self.lattice = BPSEnhancedLattice(size=20)
        self.harness = BPSBlowupHarness(self.lattice)
    
    def test_bps_protection_during_harvest(self):
        """Test that BPS solitons are protected during energy harvest"""
        # Create mixed solitons
        self.lattice.create_bps_soliton(0, charge=2.0)
        self.lattice.oscillator_objects[1].polarity = SolitonPolarity.BRIGHT
        self.lattice.oscillator_objects[1].amplitude = 1.0
        
        # Harvest energy
        report = self.harness.harvest_energy(exclude_bps=True)
        
        # Check BPS was protected
        self.assertEqual(report.num_bps_protected, 1)
        self.assertIn(0, report.bps_indices_protected)
        
        # Verify BPS oscillator unchanged
        bps_osc = self.lattice.oscillator_objects[0]
        self.assertEqual(bps_osc.amplitude, 2.0)
        
        # Verify bright oscillator was harvested
        bright_osc = self.lattice.oscillator_objects[1]
        self.assertEqual(bright_osc.amplitude, 0.0)
    
    def test_controlled_blowup_with_bps(self):
        """Test controlled blow-up preserving BPS solitons"""
        # Setup initial state
        self.lattice.create_bps_soliton(5, charge=1.0)
        for i in [0, 1, 2]:
            self.lattice.oscillator_objects[i].amplitude = 1.0
        
        initial_charge = self.lattice.total_charge
        
        # Execute blow-up
        success = self.harness.execute_controlled_blow_up(
            reinject_as=SolitonPolarity.BRIGHT
        )
        
        self.assertTrue(success)
        
        # Verify charge conservation
        final_charge = self.lattice.total_charge
        self.assertAlmostEqual(final_charge, initial_charge, delta=1e-6)
        
        # Verify BPS still exists
        self.assertIn(5, self.lattice.bps_indices)


class TestBPSHotSwap(unittest.TestCase):
    """Test BPS-aware hot-swap transitions"""
    
    def setUp(self):
        self.old_lattice = BPSEnhancedLattice(size=10)
        self.hot_swap = BPSHotSwapLaplacian(self.old_lattice)
    
    def test_hot_swap_charge_conservation(self):
        """Test charge conservation during hot-swap"""
        # Create BPS solitons
        self.old_lattice.create_bps_soliton(2, charge=1.0)
        self.old_lattice.create_bps_soliton(5, charge=-1.0)
        self.old_lattice.create_bps_soliton(7, charge=2.0)
        
        initial_charge = self.old_lattice.total_charge
        
        # Execute hot-swap to larger lattice
        new_lattice = self.hot_swap.execute_hot_swap(new_size=15)
        
        # Verify charge conserved
        final_charge = new_lattice.total_charge
        self.assertAlmostEqual(final_charge, initial_charge, delta=1e-6)
        
        # Verify BPS count
        self.assertEqual(len(new_lattice.bps_indices), 3)
    
    def test_hot_swap_with_topology_change(self):
        """Test hot-swap with different topology"""
        # Create BPS soliton
        self.old_lattice.create_bps_soliton(0, charge=1.5)
        
        # Create new topology (ring instead of all-to-all)
        new_topology = np.zeros((10, 10))
        for i in range(10):
            new_topology[i, (i+1) % 10] = 1.0
            new_topology[i, (i-1) % 10] = 1.0
        
        # Execute hot-swap
        new_lattice = self.hot_swap.execute_hot_swap(new_topology=new_topology)
        
        # Verify BPS preserved
        self.assertEqual(len(new_lattice.bps_indices), 1)
        self.assertEqual(new_lattice.total_charge, 1.5)


class TestBPSDiagnostics(unittest.TestCase):
    """Test BPS diagnostics functionality"""
    
    def setUp(self):
        self.lattice = BPSEnhancedLattice(size=10)
        self.diagnostics = BPSDiagnostics(self.lattice)
    
    def test_energy_compliance_report(self):
        """Test energy vs charge compliance reporting"""
        # Create compliant BPS soliton
        self.lattice.create_bps_soliton(0, charge=2.0)
        
        # Create non-compliant BPS soliton
        self.lattice.create_bps_soliton(1, charge=1.0)
        self.lattice.oscillator_objects[1].amplitude = 3.0  # Wrong energy
        
        # Generate report
        report = self.diagnostics.bps_energy_report()
        
        self.assertEqual(report["num_bps_solitons"], 2)
        self.assertEqual(report["compliance_summary"]["compliant"], 1)
        self.assertEqual(report["compliance_summary"]["non_compliant"], 1)
    
    def test_charge_conservation_check(self):
        """Test charge conservation verification"""
        # Create initial state
        before = BPSEnhancedLattice(size=5)
        before.create_bps_soliton(0, charge=1.0)
        
        # Create modified state
        after = BPSEnhancedLattice(size=5)
        after.create_bps_soliton(0, charge=1.0)
        after.create_bps_soliton(1, charge=0.5)  # Added charge
        
        # Check conservation (should fail)
        conserved = self.diagnostics.verify_charge_conservation(before, after)
        self.assertFalse(conserved)
        
        # Create properly conserved state
        after2 = BPSEnhancedLattice(size=5)
        after2.create_bps_soliton(0, charge=1.0)
        
        # Check conservation (should pass)
        conserved2 = self.diagnostics.verify_charge_conservation(before, after2)
        self.assertTrue(conserved2)
    
    def test_export_diagnostics(self):
        """Test diagnostic export functionality"""
        # Create some BPS solitons
        self.lattice.create_bps_soliton(0, charge=1.0)
        
        # Run some operations
        self.diagnostics.bps_energy_report()
        
        # Export to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = self.diagnostics.export_diagnostics(f.name)
        
        # Verify file created and valid JSON
        with open(filename, 'r') as f:
            data = json.load(f)
            self.assertIn("timestamp", data)
            self.assertIn("config", data)
            self.assertIn("current_state", data)


class TestAPIIntegration(unittest.TestCase):
    """Test API-level BPS support"""
    
    def test_polarity_field_required(self):
        """Test that polarity field is required in API"""
        # This would test actual API endpoints
        # Placeholder for API integration tests
        pass
    
    def test_bps_in_memory_state_api(self):
        """Test BPS solitons appear in memory state API"""
        # Placeholder for API state endpoint test
        pass


def run_bps_test_suite():
    """Run complete BPS test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTest(unittest.makeSuite(TestBPSOscillator))
    suite.addTest(unittest.makeSuite(TestBPSMemoryIntegration))
    suite.addTest(unittest.makeSuite(TestBPSBlowupHarness))
    suite.addTest(unittest.makeSuite(TestBPSHotSwap))
    suite.addTest(unittest.makeSuite(TestBPSDiagnostics))
    suite.addTest(unittest.makeSuite(TestAPIIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_bps_test_suite()
    if success:
        print("\n✅ All BPS soliton tests passed!")
    else:
        print("\n❌ Some BPS soliton tests failed")
