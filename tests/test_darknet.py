#!/usr/bin/env python3
"""
Unit tests for Dark Soliton FDTD Simulator
Ensures phase drift remains < 0.02 rad after 10k steps
"""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.simulate_darknet import DarkSolitonSimulator, run

class TestDarkSolitonSimulator:
    """Test suite for dark soliton FDTD engine"""
    
    def test_initialization(self):
        """Test simulator initialization"""
        config = {
            'lattice_size': 64,
            'dt': 0.01,
            'dx': 1.0
        }
        
        sim = DarkSolitonSimulator(config)
        
        assert sim.lattice_size == 64
        assert sim.u_real.shape == (64, 64)
        assert sim.u_imag.shape == (64, 64)
        assert sim.step_count == 0
    
    def test_dark_soliton_creation(self):
        """Test dark soliton profile creation"""
        sim = DarkSolitonSimulator({'lattice_size': 128})
        
        # Create soliton at center
        sim.create_dark_soliton(64, 64, width=10.0, depth=0.8)
        
        # Check amplitude dip at center
        center_amp = sim.get_amplitude()[64, 64]
        assert center_amp < 0.5  # Should have dip
        
        # Check amplitude recovery away from center
        edge_amp = sim.get_amplitude()[0, 0]
        assert edge_amp > 0.9  # Should be ~1 at edges
        
        # Check phase jump
        phase = sim.get_phase()
        phase_diff = phase[64, 80] - phase[64, 48]
        assert abs(phase_diff) > 1.0  # Should have phase jump
    
    def test_phase_stability_short(self):
        """Test phase stability over 1000 steps"""
        sim = run()  # Default config with centered soliton
        
        initial_phase = sim.get_phase().copy()
        metrics = sim.step(1000)
        final_phase = sim.get_phase()
        
        # Calculate average phase drift
        phase_drift = np.mean(np.abs(final_phase - initial_phase))
        
        assert metrics['phase_drift'] < 0.02
        assert phase_drift < 0.2  # Total drift should be small
    
    @pytest.mark.slow
    def test_phase_stability_long(self):
        """Test phase stability over 10k steps"""
        config = {
            'lattice_size': 128,
            'dt': 0.005,  # Smaller timestep for stability
            'dx': 1.0,
            'nonlinearity': 1.0,
            'dispersion': 0.5,
            'damping': 0.05
        }
        
        sim = DarkSolitonSimulator(config)
        sim.create_dark_soliton(64, 64, width=10.0, depth=0.8)
        
        # Store initial state
        initial_phase = sim.get_phase().copy()
        initial_energy = np.sum(sim.get_intensity()) * sim.dx**2
        
        # Run 10k steps in chunks
        total_phase_drift = 0.0
        for chunk in range(10):
            metrics = sim.step(1000)
            total_phase_drift += metrics['phase_drift']
        
        # Final checks
        final_phase = sim.get_phase()
        final_energy = metrics['energy']
        
        # Average phase drift per step
        avg_phase_drift = total_phase_drift / 10
        
        print(f"\nPhase stability test (10k steps):")
        print(f"  Average phase drift: {avg_phase_drift:.6f} rad")
        print(f"  Energy conservation: {abs(final_energy - initial_energy)/initial_energy:.6f}")
        
        assert avg_phase_drift < 0.02, f"Phase drift {avg_phase_drift:.6f} exceeds 0.02 rad"
        assert abs(final_energy - initial_energy) / initial_energy < 0.05  # 5% energy tolerance
    
    def test_multiple_solitons(self):
        """Test multiple soliton interaction"""
        sim = DarkSolitonSimulator({'lattice_size': 128})
        
        # Create first soliton
        sim.create_dark_soliton(40, 64, width=8.0, depth=0.7)
        
        # Add second soliton
        sim.add_dark_soliton(88, 64, width=8.0, depth=0.7)
        
        # Check that we have two dips
        amplitude = sim.get_amplitude()
        center_line = amplitude[:, 64]
        
        # Find local minima
        minima = []
        for i in range(1, len(center_line)-1):
            if center_line[i] < center_line[i-1] and center_line[i] < center_line[i+1]:
                minima.append(i)
        
        assert len(minima) >= 2, "Should have at least 2 amplitude dips"
    
    def test_energy_conservation(self):
        """Test energy conservation"""
        sim = run()
        
        initial_energy = np.sum(sim.get_intensity()) * sim.dx**2
        
        # Evolve
        metrics = sim.step(500)
        
        final_energy = metrics['energy']
        energy_change = abs(final_energy - initial_energy) / initial_energy
        
        assert energy_change < 0.05, f"Energy changed by {energy_change:.2%}"
    
    def test_boundary_damping(self):
        """Test boundary damping effectiveness"""
        config = {
            'lattice_size': 64,
            'damping': 0.5  # Strong damping
        }
        
        sim = DarkSolitonSimulator(config)
        
        # Create soliton near boundary
        sim.create_dark_soliton(10, 32, width=5.0, depth=0.9)
        
        # Measure boundary amplitude
        initial_boundary = sim.get_amplitude()[0:5, :].mean()
        
        # Evolve
        sim.step(100)
        
        final_boundary = sim.get_amplitude()[0:5, :].mean()
        
        # Boundary amplitude should decrease
        assert final_boundary < initial_boundary * 0.8
    
    def test_snapshot_save_load(self):
        """Test state persistence"""
        import tempfile
        
        sim1 = run()
        sim1.step(100)
        
        # Save snapshot
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            sim1.save_snapshot(tmp.name)
            
            # Create new simulator and load
            sim2 = DarkSolitonSimulator({'lattice_size': 128})
            sim2.load_snapshot(tmp.name)
            
            # Check states match
            assert np.allclose(sim1.u_real, sim2.u_real)
            assert np.allclose(sim1.u_imag, sim2.u_imag)
            assert sim1.step_count == sim2.step_count
            
        os.unlink(tmp.name)
    
    def test_cfl_stability_adjustment(self):
        """Test automatic CFL timestep adjustment"""
        config = {
            'lattice_size': 64,
            'dt': 0.1,  # Too large
            'dx': 0.5,
            'dispersion': 2.0  # High dispersion
        }
        
        sim = DarkSolitonSimulator(config)
        
        # Should have adjusted timestep
        assert sim.dt < 0.1
        
        # Should still be stable after evolution
        sim.create_dark_soliton(32, 32)
        metrics = sim.step(100)
        
        assert metrics['max_amplitude'] < 2.0  # No blowup

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
