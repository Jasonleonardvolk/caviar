#!/usr/bin/env python
"""
Test the Strassen-optimized soliton physics implementation
"""
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply
from python.core import topology_catalogue

def count_kernel_calls(n):
    """Calculate number of 2×2 kernel calls for size n"""
    if n == 2:
        return 1
    else:
        levels = int(np.log2(n)) - 1
        return 7 ** levels

def test_strassen_soliton():
    """Test Strassen-optimized soliton multiplication"""
    print("="*70)
    print("STRASSEN-OPTIMIZED SOLITON PHYSICS TEST")
    print("="*70)
    print()
    
    # Test accuracy across different sizes
    sizes = [2, 4, 8, 16, 32]
    results = []
    
    for n in sizes:
        print(f"\nTEST: {n}×{n} Matrix")
        print("-"*50)
        
        # Create test matrices
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Time the soliton multiplication
        start = time.time()
        C_soliton = hyperbolic_matrix_multiply(A, B)
        soliton_time = time.time() - start
        
        # Time numpy reference
        start = time.time()
        C_expected = A @ B
        numpy_time = time.time() - start
        
        # Calculate error
        error = np.linalg.norm(C_soliton - C_expected) / np.linalg.norm(C_expected)
        
        # Count kernel calls
        kernel_calls = count_kernel_calls(n)
        naive_calls = (n // 2) ** 3  # What naive recursion would use
        
        results.append({
            'n': n,
            'error': error,
            'soliton_time': soliton_time,
            'numpy_time': numpy_time,
            'kernel_calls': kernel_calls,
            'naive_calls': naive_calls
        })
        
        print(f"Relative error: {error:.6e}")
        print(f"Kernel calls: {kernel_calls} (vs {naive_calls} for naive)")
        print(f"Speedup vs naive: {naive_calls/kernel_calls:.2f}×")
        print(f"Time: {soliton_time:.6f}s (NumPy: {numpy_time:.6f}s)")
        
        if error < 1e-10:
            print("✓ PASSED with excellent accuracy!")
        elif error < 1e-6:
            print("✓ PASSED with good accuracy")
        else:
            print(f"✗ Higher error than expected: {error:.2e}")
    
    # Summary table
    print("\n" + "="*70)
    print("STRASSEN OPTIMIZATION SUMMARY")
    print("="*70)
    print()
    print(f"{'Size':>6} {'Kernel Calls':>12} {'Naive Calls':>12} {'Speedup':>8} {'Error':>12}")
    print("-"*60)
    
    for r in results:
        print(f"{r['n']:>6} {r['kernel_calls']:>12} {r['naive_calls']:>12} "
              f"{r['naive_calls']/r['kernel_calls']:>8.2f}× {r['error']:>12.2e}")
    
    # Complexity analysis
    print("\n" + "="*70)
    print("COMPLEXITY ANALYSIS")
    print("="*70)
    print()
    print("Strassen recursive calls: T(n) = 7·T(n/2)")
    print("This gives ω = log₂(7) ≈ 2.807")
    print()
    print("Kernel calls for size n:")
    print("- n=2:  1 call  (base case)")
    print("- n=4:  7 calls  (7¹)")
    print("- n=8:  49 calls (7²)")
    print("- n=16: 343 calls (7³)")
    print("- n=32: 2401 calls (7⁴)")
    print()
    print("Compare to naive block multiplication:")
    print("- Would need 8 recursive calls per level")
    print("- Gives O(n³) total operations")
    
    # Test non-power-of-2
    print("\n" + "="*70)
    print("NON-POWER-OF-2 TEST")
    print("="*70)
    
    A5 = np.random.randn(5, 5)
    B5 = np.random.randn(5, 5)
    
    C5_soliton = hyperbolic_matrix_multiply(A5, B5)
    C5_expected = A5 @ B5
    
    error5 = np.linalg.norm(C5_soliton - C5_expected) / np.linalg.norm(C5_expected)
    print(f"\n5×5 matrix (padded to 8×8):")
    print(f"Relative error: {error5:.6e}")
    
    if error5 < 1e-10:
        print("✓ Non-power-of-2 handling works correctly!")

if __name__ == "__main__":
    test_strassen_soliton()
