#!/usr/bin/env python
"""
Diagnostic test to understand soliton physics issues
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core import soliton_ring_sim, topology_catalogue
from python.core.hyperbolic_matrix_multiply import _run_amplification_cycle

def diagnose_collision_detection():
    """Deep dive into why collisions aren't being detected"""
    print("="*70)
    print("COLLISION DETECTION DIAGNOSTICS")
    print("="*70)
    print()
    
    # Create a simple wavefield
    wavefield = np.zeros((10, 10), dtype=complex)
    
    # Place two solitons that SHOULD collide
    wavefield[5, 5] = 1.0 + 0j
    wavefield[5, 6] = 1.0 + 0j
    
    print("Initial wavefield (non-zero values):")
    for i in range(10):
        for j in range(10):
            if abs(wavefield[i,j]) > 0:
                print(f"  [{i},{j}] = {wavefield[i,j]}")
    
    # Check energy statistics
    energy = np.abs(wavefield) ** 2
    print(f"\nEnergy statistics:")
    print(f"  Max: {np.max(energy)}")
    print(f"  Mean: {np.mean(energy)}")
    print(f"  Median: {np.median(energy)}")
    print(f"  Non-zero count: {np.count_nonzero(energy)}")
    
    # Try different thresholds
    print("\nTrying different thresholds:")
    for threshold in [12.0, 6.0, 2.0, 1.0, 0.5]:
        collisions = soliton_ring_sim.detect_collisions(wavefield, threshold=threshold)
        print(f"  Threshold {threshold}: {len(collisions)} collisions")
    
    # Test the neighborhood check
    print("\nDetailed collision detection with threshold=0.5:")
    energy = np.abs(wavefield) ** 2
    median_energy = float(np.median(energy))
    threshold = 0.5
    
    mask = energy > (threshold * median_energy)
    xs, ys = np.where(mask)
    
    print(f"  Points above threshold: {len(xs)}")
    for x, y in zip(xs, ys):
        print(f"    [{x},{y}]: energy={energy[x,y]:.3f}")
        
        # Check neighborhood
        neighbourhood = wavefield[max(0, x-1):x+2, max(0, y-1):y+2]
        n_peaks = np.count_nonzero(np.abs(neighbourhood) > (0.8 * np.max(np.abs(neighbourhood))))
        print(f"      Neighborhood peaks: {n_peaks}")

def diagnose_amplification():
    """Check what happens during amplification"""
    print("\n" + "="*70)
    print("AMPLIFICATION DIAGNOSTICS")
    print("="*70)
    print()
    
    # Simple wavefield
    wavefield = np.ones((5, 5), dtype=complex)
    
    print("Before amplification:")
    print(f"  Max amplitude: {np.max(np.abs(wavefield))}")
    print(f"  Total energy: {np.sum(np.abs(wavefield)**2)}")
    
    amplified = _run_amplification_cycle(wavefield)
    
    print("\nAfter amplification:")
    print(f"  Max amplitude: {np.max(np.abs(amplified))}")
    print(f"  Total energy: {np.sum(np.abs(amplified)**2)}")
    
    if np.any(np.isnan(amplified)) or np.any(np.isinf(amplified)):
        print("  WARNING: NaN or Inf values detected!")

def diagnose_mapping():
    """Analyze the index mapping issue"""
    print("\n" + "="*70)
    print("MAPPING DIAGNOSTICS")
    print("="*70)
    print()
    
    # Create projection for 2x2
    proj = topology_catalogue.collision_projection(2)
    
    print(f"Lattice shape: {proj.lattice_shape}")
    print(f"Total lattice points: {proj.lattice_shape[0] * proj.lattice_shape[1]}")
    print(f"Mapped points: {len(proj._forward)}")
    
    # Check the hashing function
    print("\nHash function analysis for n=2:")
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Recreate the hash logic
                arm = (i + j + k) % 5
                radius = int((i * 1.618 + j + k * 1.618**2) % (proj.lattice_shape[0] // 2))
                angle = 2 * np.pi * arm / 5
                x = int(proj.lattice_shape[0] // 2 + radius * np.cos(angle))
                y = int(proj.lattice_shape[1] // 2 + radius * np.sin(angle))
                print(f"  ({i},{j},{k}) -> arm={arm}, radius={radius} -> ({x},{y})")

def test_manual_collision():
    """Manually create and test a collision"""
    print("\n" + "="*70)
    print("MANUAL COLLISION TEST")
    print("="*70)
    print()
    
    # Create collision event manually
    from python.core.soliton_ring_sim import CollisionEvent
    
    collision = CollisionEvent(
        pos=(5, 5),
        amp_phase_left=(3.0, 0.0),  # amplitude=3, phase=0
        amp_phase_right=(2.0, 0.0)   # amplitude=2, phase=0
    )
    
    amp, phase = collision.product_amplitude_phase()
    print(f"Manual collision product: amplitude={amp}, phase={phase}")
    print(f"Complex product: {collision.product_complex()}")
    print(f"Expected: 3.0 * 2.0 = 6.0")

if __name__ == "__main__":
    diagnose_collision_detection()
    diagnose_amplification()
    diagnose_mapping()
    test_manual_collision()
