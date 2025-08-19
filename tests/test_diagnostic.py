#!/usr/bin/env python
"""
Diagnostic test to understand soliton physics issues
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply, _encode_matrices_to_wavefield, _run_amplification_cycle
from python.core import soliton_ring_sim, topology_catalogue

def diagnose_soliton_physics():
    """Step-by-step diagnosis of the soliton physics pipeline"""
    print("="*70)
    print("SOLITON PHYSICS DIAGNOSTIC")
    print("="*70)
    print()
    
    # Simple test case
    A = np.array([[1.0, 2.0], 
                  [3.0, 4.0]])
    B = np.array([[5.0, 6.0],
                  [7.0, 8.0]])
    
    print("1. TOPOLOGY MAPPING ANALYSIS")
    print("-"*50)
    
    # Check the projection mapping
    proj = topology_catalogue.collision_projection(2)
    print(f"Lattice shape: {proj.lattice_shape}")
    print(f"Number of mappings: {len(proj._forward)}")
    
    # We need ALL 8 mappings for complete multiplication
    print("\nRequired (i,j,k) triples for 2×2:")
    required = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                required.append((i,j,k))
                print(f"  ({i},{j},{k}): A[{i},{k}] × B[{k},{j}] → C[{i},{j}]")
    
    print("\n2. WAVEFIELD ENCODING ANALYSIS")
    print("-"*50)
    
    # Test the encoding
    wavefield = _encode_matrices_to_wavefield(A, B)
    print(f"Wavefield shape: {wavefield.shape}")
    print(f"Non-zero elements: {np.count_nonzero(wavefield)}")
    print(f"Max amplitude: {np.max(np.abs(wavefield)):.3f}")
    print(f"Mean amplitude: {np.mean(np.abs(wavefield)):.3f}")
    
    # Show where solitons are placed
    print("\nSoliton positions (amplitude > 0.1):")
    positions = np.where(np.abs(wavefield) > 0.1)
    for x, y in zip(positions[0], positions[1]):
        amp = np.abs(wavefield[x, y])
        phase = np.angle(wavefield[x, y])
        print(f"  ({x},{y}): amplitude={amp:.3f}, phase={phase:.3f}")
    
    print("\n3. AMPLIFICATION ANALYSIS")
    print("-"*50)
    
    # Test amplification
    amplified = _run_amplification_cycle(wavefield)
    print(f"After amplification:")
    print(f"  Max amplitude: {np.max(np.abs(amplified)):.3f}")
    print(f"  Mean amplitude: {np.mean(np.abs(amplified)):.3f}")
    
    # Check for overflow
    if np.any(np.isinf(amplified)) or np.any(np.isnan(amplified)):
        print("  WARNING: Infinity or NaN detected!")
    
    print("\n4. COLLISION DETECTION ANALYSIS")
    print("-"*50)
    
    # Test collision detection with various thresholds
    for threshold in [12.0, 6.0, 3.0, 1.5, 0.5]:
        original = soliton_ring_sim._PEAK_ENERGY_THRESHOLD
        soliton_ring_sim._PEAK_ENERGY_THRESHOLD = threshold
        
        collisions = soliton_ring_sim.detect_collisions(amplified)
        print(f"Threshold {threshold}: {len(collisions)} collisions detected")
        
        soliton_ring_sim._PEAK_ENERGY_THRESHOLD = original
    
    print("\n5. MANUAL COLLISION TEST")
    print("-"*50)
    
    # Create a simple test wavefield with guaranteed collision
    test_field = np.zeros((10, 10), dtype=complex)
    test_field[5, 5] = 2.0 + 0j  # Strong soliton
    test_field[5, 6] = 3.0 + 0j  # Adjacent strong soliton
    
    # Check energy
    energy = np.abs(test_field) ** 2
    print(f"Test field max energy: {np.max(energy):.3f}")
    print(f"Test field median energy: {np.median(energy):.3f}")
    
    # Try collision detection
    soliton_ring_sim._PEAK_ENERGY_THRESHOLD = 0.1  # Very low
    collisions = soliton_ring_sim.detect_collisions(test_field)
    print(f"Manual test collisions: {len(collisions)}")
    
    print("\n6. ENCODING FIX SUGGESTION")
    print("-"*50)
    print("The phase-only encoding loses magnitude information!")
    print("Current: phase = (A[i,k] % 1.0) * 2π - π")
    print("This modulo operation destroys values > 1")
    print("\nBetter encoding options:")
    print("1. Amplitude = |value|, Phase = sign")
    print("2. Real part = A[i,k], Imag part = B[k,j]")
    print("3. Amplitude = sqrt(|A||B|), Phase = arg(A) + arg(B)")

if __name__ == "__main__":
    diagnose_soliton_physics()
