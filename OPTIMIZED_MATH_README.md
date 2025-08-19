# Soliton Architecture Mathematical Optimizations

This package provides optimized implementations of the expensive mathematical operations used in the Soliton architecture, as outlined in the documentation. These optimizations ensure the system can scale to high-dimensional data and operate in real-time environments.

## Overview

The Soliton architecture requires several computationally intensive mathematical operations, particularly:

1. **Numeric Gradient Calculations**: Computing gradients of error functions or state updates
2. **Fisher Information Matrix (FIM)**: Computing the Fisher Information Matrix for information geometry
3. **Fixed Point Algorithms**: Finding stable cognitive states efficiently

These operations can become bottlenecks, especially when dealing with high-dimensional data. For example, a naive numeric gradient in a 128-dimensional state would require 128 separate function evaluations, which is too slow for real-time applications.

## Optimization Techniques

This package implements the following optimization strategies:

### Vectorization
- Replaces Python loops with NumPy array operations
- Evaluates multiple states in parallel with a single function call
- Leverages SIMD optimizations in NumPy

### JIT Compilation
- Uses Numba to compile Python functions to optimized machine code
- Eliminates Python overhead for loops and calculations
- Enables parallel execution with `prange`

### Automatic Differentiation
- Uses PyTorch/JAX for direct gradient calculation
- Leverages GPU acceleration when available
- Computes exact gradients instead of numeric approximations

### Block-Diagonal Approximation
- Treats the FIM as block-diagonal for very high-dimensional systems
- Reduces complexity from O(n²) to sum of O(k_i²) for each block
- Computes blocks in parallel for further acceleration

### Batched Operations
- Processes multiple calculations in one function call
- Stacks matrices for efficient linear algebra operations
- Avoids redundant computations

### Caching
- Reuses base function evaluations across calculations
- Computes single-parameter perturbations once and reuses them
- Minimizes memory allocation overhead

## Files

- `optimized_math_operations.py`: Main module with optimized implementations
- `test_optimized_math.py`: Test script demonstrating usage and performance
- `run_math_test.bat`: Batch script to run the test

## Usage

The module provides multiple implementations of each operation, allowing you to choose the best approach for your specific requirements:

```python
# Numeric Gradient examples
grad_naive = numeric_gradient_naive(func, x)
grad_jit = numeric_gradient_jit(func, x)
grad_vectorized = numeric_gradient_vectorized(vectorized_func, x)
grad_torch = numeric_gradient_torch(torch_func, x)
grad_jax = numeric_gradient_jax(jax_func, x)

# Fisher Information Matrix examples
fim_naive = fisher_information_matrix_naive(log_prob_func, s)
fim_jit = fisher_information_matrix_jit(log_prob_func, s)
fim_vectorized = fisher_information_matrix_vectorized(vectorized_log_prob, s)
fim_block = fisher_information_matrix_block_diagonal(log_prob_func, s, block_sizes)
fim_parallel = fisher_information_matrix_parallel_blocks(log_prob_func, s, block_sizes)
fim_torch = fisher_information_matrix_torch(torch_log_prob, s)
fim_jax = fisher_information_matrix_jax(jax_log_prob, s)
```

## Performance

You can compare the performance of different implementations using the `performance_comparison` function:

```python
from optimized_math_operations import performance_comparison
performance_comparison(dim=100, n_trials=5)
```

The expected speedups depend on the dimension and specific hardware, but generally:

- JIT compilation provides 10-100x speedup over naive implementations
- Vectorization provides 5-20x speedup for operations that can be vectorized
- Block-diagonal approximation reduces complexity dramatically for high dimensions
- Automatic differentiation with PyTorch/JAX can be fastest when GPU is available

## Integration with Soliton Architecture

These optimized operations are designed to plug directly into the existing Soliton cognitive framework. The optimization approaches outlined in the documentation have been implemented here, including:

1. Vectorization of numeric gradients
2. JIT compilation of critical routines
3. Restructuring algorithms to minimize redundant calculations
4. Block-diagonal approximation for the FIM
5. Batched operations for parallel processing
6. Automatic differentiation for exact gradients

By integrating these optimized operations, the Soliton architecture can handle high-dimensional cognitive states in real-time, enabling more sophisticated applications.

## Requirements

- NumPy (required)
- Numba (required for JIT compilation)
- SciPy (required for some operations)
- PyTorch (optional, for automatic differentiation)
- JAX (optional, for automatic differentiation and JIT)
- joblib (optional, for parallel block computation)
