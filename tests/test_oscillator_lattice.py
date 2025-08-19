#!/usr/bin/env python3
"""Test oscillator lattice basic functionality."""

from python.core.oscillator_lattice import OscillatorLattice
import numpy as np


def test_order_parameter_syncs():
    """Test that coupled oscillators synchronize."""
    lat = OscillatorLattice()
    
    # Add 10 oscillators, half at phase 0, half at phase π
    for k in range(10):
        lat.add_oscillator(phase=0.0 if k < 5 else 3.14)
    
    # Set all-to-all coupling
    lat.K += 1.0
    
    # Simulate for 500 steps
    for _ in range(500):
        lat.step(0.01)
    
    # Should be highly synchronized
    assert lat.order_parameter() > 0.95


def test_phase_entropy():
    """Test entropy calculation."""
    lat = OscillatorLattice()
    
    # Uniform distribution - high entropy
    phases = np.linspace(0, 2*np.pi, 100, endpoint=False)
    for p in phases:
        lat.add_oscillator(phase=p)
    
    # Should have high entropy (close to 1)
    assert lat.phase_entropy() > 0.9
    
    # Reset with all same phase - low entropy
    lat2 = OscillatorLattice()
    for _ in range(100):
        lat2.add_oscillator(phase=0.0)
    
    # Should have very low entropy
    assert lat2.phase_entropy() < 0.1


def test_coupling_dynamics():
    """Test that coupling affects dynamics."""
    # Two oscillators with different natural frequencies
    lat = OscillatorLattice()
    lat.add_oscillator(phase=0.0, natural_freq=0.1)
    lat.add_oscillator(phase=0.0, natural_freq=-0.1)
    
    # No coupling - they should diverge
    for _ in range(100):
        lat.step(0.1)
    
    phase_diff_uncoupled = abs(lat.oscillators[0].phase - lat.oscillators[1].phase)
    
    # Reset and add strong coupling
    lat.oscillators[0].phase = 0.0
    lat.oscillators[1].phase = 0.0
    lat.set_coupling(0, 1, 2.0)
    lat.set_coupling(1, 0, 2.0)
    
    for _ in range(100):
        lat.step(0.1)
    
    phase_diff_coupled = abs(lat.oscillators[0].phase - lat.oscillators[1].phase)
    
    # Coupling should reduce phase difference
    assert phase_diff_coupled < phase_diff_uncoupled


if __name__ == "__main__":
    test_order_parameter_syncs()
    test_phase_entropy()
    test_coupling_dynamics()
    print("All oscillator lattice tests passed!")
