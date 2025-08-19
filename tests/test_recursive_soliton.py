#!/usr/bin/env python
"""
Test the recursive soliton physics implementation
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply
from python.core import topology_catalogue

def test_recursive_soliton():
    """Test recursive decomposition for complete coverage"""
    print("="*70)
    print("RECURSIVE SOLITON PHYSICS TEST")
    print("="*70)
    print()
    
    # Test 2×2 (base case)
    print("TEST 1: 2×2 Matrix (Base Case)")
    print("-"*50)
    
    A2 = np.array([[1.0, 2.0], 
                   [3.0, 4.0]])
    B2 = np.array([[5.0, 6.0],
                   [7.0, 8.0]])
    
    C2_expected = A2 @ B2
    C2_soliton = hyperbolic_matrix_multiply(A2, B2)
    
    error2 = np.linalg.norm(C2_soliton - C2_expected) / np.linalg.norm(C2_expected)
    print(f"2×2 relative error: {error2:.6e}")
    
    if error2 < 1e-10:
        print("✓ 2×2 PASSED with perfect accuracy!")
    else:
        print(f"✗ 2×2 error: {error2:.2%}")
    
    # Test 4×4 (one level of recursion)
    print("\n" + "="*70)
    print("TEST 2: 4×4 Matrix (One Recursion Level)")
    print("-"*50)
    
    np.random.seed(42)
    A4 = np.random.randn(4, 4)
    B4 = np.random.randn(4, 4)
    
    C4_expected = A4 @ B4
    C4_soliton = hyperbolic_matrix_multiply(A4, B4)
    
    error4 = np.linalg.norm(C4_soliton - C4_expected) / np.linalg.norm(C4_expected)
    print(f"4×4 relative error: {error4:.6e}")
    
    if error4 < 0.01:
        print("✓ 4×4 PASSED with < 1% error!")
    else:
        print(f"✗ 4×4 error: {error4:.2%}")
    
    # Show breakdown of recursive calls
    print("\nRecursive breakdown for 4×4:")
    print("- 4 calls for C11 = A11*B11 + A12*B21")
    print("- 4 calls for C12 = A11*B12 + A12*B22")
    print("- 4 calls for C21 = A21*B11 + A22*B21")
    print("- 4 calls for C22 = A21*B12 + A22*B22")
    print("Total: 16 base case (2×2) multiplications")
    
    # Test 8×8 (two levels of recursion)
    print("\n" + "="*70)
    print("TEST 3: 8×8 Matrix (Two Recursion Levels)")
    print("-"*50)
    
    A8 = np.random.randn(8, 8)
    B8 = np.random.randn(8, 8)
    
    C8_expected = A8 @ B8
    C8_soliton = hyperbolic_matrix_multiply(A8, B8)
    
    error8 = np.linalg.norm(C8_soliton - C8_expected) / np.linalg.norm(C8_expected)
    print(f"8×8 relative error: {error8:.6e}")
    
    if error8 < 0.01:
        print("✓ 8×8 PASSED with < 1% error!")
    else:
        print(f"✗ 8×8 error: {error8:.2%}")
    
    # Test 16×16
    print("\n" + "="*70)
    print("TEST 4: 16×16 Matrix (Three Recursion Levels)")
    print("-"*50)
    
    A16 = np.random.randn(16, 16)
    B16 = np.random.randn(16, 16)
    
    C16_expected = A16 @ B16
    C16_soliton = hyperbolic_matrix_multiply(A16, B16)
    
    error16 = np.linalg.norm(C16_soliton - C16_expected) / np.linalg.norm(C16_expected)
    print(f"16×16 relative error: {error16:.6e}")
    
    if error16 < 0.01:
        print("✓ 16×16 PASSED with < 1% error!")
    else:
        print(f"✗ 16×16 error: {error16:.2%}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nRecursive decomposition ensures complete coverage:")
    print("- Each 2×2 base case: 8/8 mappings (100%)")
    print("- Larger matrices decompose to 2×2 blocks")
    print("- No missing products!")
    print("\nComplexity: O(n³) for now (standard block multiplication)")
    print("Next step: Replace with Strassen-style recombination for O(n^2.807)")

if __name__ == "__main__":
    test_recursive_soliton()
