"""
Test Script for Optimized Mathematical Operations
================================================

This script demonstrates how to use the optimized mathematical operations
from the optimized_math_operations.py module.
"""

import numpy as np
import time
from optimized_math_operations import (
    numeric_gradient_naive,
    numeric_gradient_jit,
    numeric_gradient_vectorized,
    fisher_information_matrix_naive,
    fisher_information_matrix_jit,
    fisher_information_matrix_vectorized,
    fisher_information_matrix_block_diagonal,
    performance_comparison
)

def test_function(x):
    """Simple test function: negative quadratic (log of Gaussian)."""
    return -0.5 * np.sum(x**2)

def test_function_batch(X):
    """Batch version of test function for vectorized operations."""
    return -0.5 * np.sum(X**2, axis=1)

def main():
    """Run tests and comparisons of the optimized operations."""
    print("Testing Optimized Math Operations for Soliton Architecture")
    print("=" * 60)
    
    # Test dimensions
    dims = [10, 50, 100]
    
    # Test numeric gradient implementations
    print("\nNUMERIC GRADIENT COMPARISON")
    print("-" * 60)
    
    for dim in dims:
        print(f"\nDimension: {dim}")
        
        # Generate random test point
        x = np.random.randn(dim)
        
        # Compare implementations
        methods = [
            ("Naive", lambda x: numeric_gradient_naive(test_function, x)),
            ("JIT", lambda x: numeric_gradient_jit(test_function, x)),
            ("Vectorized", lambda x: numeric_gradient_vectorized(test_function_batch, x))
        ]
        
        # Expected result (analytical gradient of our test function)
        expected = -x
        
        for name, method in methods:
            # Measure execution time
            start = time.time()
            result = method(x)
            elapsed = time.time() - start
            
            # Check accuracy
            error = np.linalg.norm(result - expected)
            
            print(f"{name}: {elapsed:.6f} seconds, Error: {error:.6e}")
    
    # Test Fisher Information Matrix implementations
    print("\nFISHER INFORMATION MATRIX COMPARISON")
    print("-" * 60)
    
    for dim in dims:
        print(f"\nDimension: {dim}")
        
        # Generate random test point
        x = np.random.randn(dim)
        
        # Compare implementations
        methods = [
            ("Naive", lambda x: fisher_information_matrix_naive(test_function, x)),
            ("JIT", lambda x: fisher_information_matrix_jit(test_function, x)),
            ("Vectorized", lambda x: fisher_information_matrix_vectorized(test_function_batch, x))
        ]
        
        # For higher dimensions, add block-diagonal approximation
        if dim >= 50:
            # Create block sizes (roughly equal)
            block_size = dim // 4
            block_sizes = [block_size] * 3 + [dim - 3*block_size]
            methods.append(
                ("Block-Diagonal", 
                 lambda x: fisher_information_matrix_block_diagonal(test_function, x, block_sizes))
            )
        
        # For our test function (negative quadratic), the FIM is the identity matrix
        expected = np.eye(dim)
        
        for name, method in methods:
            # Measure execution time
            start = time.time()
            try:
                result = method(x)
                elapsed = time.time() - start
                
                # Check accuracy (Frobenius norm of difference)
                error = np.linalg.norm(result - expected)
                
                print(f"{name}: {elapsed:.6f} seconds, Error: {error:.6e}")
            except Exception as e:
                print(f"{name}: Error - {str(e)}")
    
    # Run comprehensive performance comparison
    print("\nCOMPREHENSIVE PERFORMANCE COMPARISON")
    print("-" * 60)
    performance_comparison(dim=100, n_trials=3)

if __name__ == "__main__":
    main()
