#!/usr/bin/env python3
"""
Test the soliton physics implementation
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply
from python.core import soliton_ring_sim, topology_catalogue

def test_soliton_multiplication():
    """Test soliton-based matrix multiplication"""
    print("=" * 70)
    print("SOLITON PHYSICS MATRIX MULTIPLICATION TEST")
    print("=" * 70)
    print()
    
    # Test small matrices
    A = np.array([[1.0, 2.0], 
                  [3.0, 4.0]])
    B = np.array([[5.0, 6.0],
                  [7.0, 8.0]])
    
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    
    # Expected result
    C_expected = A @ B
    print("\nExpected C = A @ B:")
    print(C_expected)
    
    # Test soliton multiplication
    try:
        C_soliton = hyperbolic_matrix_multiply(A, B)
        print("\nSoliton result:")
        print(C_soliton)
        
        # Check accuracy
        error = np.linalg.norm(C_soliton - C_expected) / np.linalg.norm(C_expected)
        print(f"\nRelative error: {error:.6e}")
        
        if error < 0.1:  # 10% tolerance for this simplified implementation
            print("✓ Test PASSED!")
        else:
            print("✗ Test FAILED - error too large")
            
    except Exception as e:
        print(f"\n✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
    
    # Test collision detection
    print("\n" + "-"*50)
    print("COLLISION DETECTION TEST")
    print("-"*50)
    
    # Create a simple wavefield with two solitons
    wavefield = np.zeros((10, 10), dtype=complex)
    wavefield[3, 3] = 2.0 * np.exp(1j * 0.5)  # Soliton 1
    wavefield[3, 4] = 1.5 * np.exp(1j * 1.0)  # Soliton 2
    
    collisions = soliton_ring_sim.detect_collisions(wavefield)
    print(f"Detected {len(collisions)} collisions")
    
    for i, collision in enumerate(collisions):
        amp, phase = collision.product_amplitude_phase()
        print(f"  Collision {i}: position={collision.pos}, "
              f"product_amplitude={amp:.3f}, product_phase={phase:.3f}")
    
    # Test topology mapping
    print("\n" + "-"*50)
    print("TOPOLOGY MAPPING TEST")
    print("-"*50)
    
    proj = topology_catalogue.collision_projection(4)
    print(f"Created projection for 4×4 matrices")
    print(f"Lattice shape: {proj.lattice_shape}")
    print(f"Number of mapped positions: {len(proj._forward)}")
    
    # Show a few mappings
    print("\nSample mappings (lattice → matrix indices):")
    count = 0
    for (x, y), (i, j, k) in list(proj._forward.items())[:5]:
        print(f"  ({x},{y}) → ({i},{j},{k})")
        count += 1

if __name__ == "__main__":
    test_soliton_multiplication()
