#!/usr/bin/env python
"""
Benchmark to find optimal hybrid threshold for soliton physics
"""
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply

def benchmark_size(n, num_trials=10):
    """Benchmark matrix multiplication for size n"""
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    # Warm-up
    _ = A @ B
    _ = hyperbolic_matrix_multiply(A, B)
    
    # Time NumPy
    numpy_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        C_numpy = A @ B
        numpy_times.append(time.perf_counter() - start)
    
    # Time soliton physics
    soliton_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        C_soliton = hyperbolic_matrix_multiply(A, B)
        soliton_times.append(time.perf_counter() - start)
    
    # Verify correctness
    error = np.linalg.norm(C_soliton - C_numpy) / np.linalg.norm(C_numpy)
    
    return {
        'n': n,
        'numpy_time': np.median(numpy_times),
        'soliton_time': np.median(soliton_times),
        'error': error,
        'speedup': np.median(numpy_times) / np.median(soliton_times)
    }

def find_optimal_threshold():
    """Find the crossover point where soliton physics becomes faster"""
    print("="*70)
    print("HYBRID THRESHOLD BENCHMARK")
    print("="*70)
    print()
    
    # Test various sizes
    sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    results = []
    
    print(f"{'Size':>6} {'NumPy(ms)':>12} {'Soliton(ms)':>12} {'Speedup':>10} {'Error':>12}")
    print("-"*60)
    
    for n in sizes:
        result = benchmark_size(n, num_trials=20 if n <= 64 else 5)
        results.append(result)
        
        print(f"{result['n']:>6} {result['numpy_time']*1000:>12.3f} "
              f"{result['soliton_time']*1000:>12.3f} {result['speedup']:>10.2f}x "
              f"{result['error']:>12.2e}")
    
    # Find crossover point
    crossover = None
    for i in range(len(results) - 1):
        if results[i]['speedup'] < 1.0 and results[i+1]['speedup'] >= 1.0:
            crossover = (results[i]['n'], results[i+1]['n'])
            break
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if crossover:
        print(f"\nCrossover point: between {crossover[0]} and {crossover[1]}")
        print(f"Recommended threshold: n <= {crossover[0]}")
    
    # Current threshold analysis
    print("\nWith current threshold (n <= 64):")
    for r in results:
        if r['n'] <= 64:
            print(f"  n={r['n']:3d}: NumPy used (saves {(1-r['speedup'])*100:.1f}% time)")
        else:
            print(f"  n={r['n']:3d}: Soliton physics ({r['speedup']:.2f}x faster)")
    
    # Plot results
    try:
        plt.figure(figsize=(10, 6))
        
        sizes_arr = [r['n'] for r in results]
        numpy_times = [r['numpy_time']*1000 for r in results]
        soliton_times = [r['soliton_time']*1000 for r in results]
        
        plt.loglog(sizes_arr, numpy_times, 'o-', label='NumPy', linewidth=2, markersize=8)
        plt.loglog(sizes_arr, soliton_times, 's-', label='Soliton Physics', linewidth=2, markersize=8)
        
        # Mark threshold
        plt.axvline(x=64, color='red', linestyle='--', alpha=0.5, label='Current threshold')
        
        plt.xlabel('Matrix Size (n)', fontsize=12)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.title('Hybrid Threshold Analysis: NumPy vs Soliton Physics', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add speedup annotations
        for r in results:
            if r['speedup'] >= 1.0:
                plt.annotate(f"{r['speedup']:.1f}x", 
                           (r['n'], r['soliton_time']*1000),
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('hybrid_threshold_analysis.png', dpi=150)
        print("\nPlot saved as hybrid_threshold_analysis.png")
    except Exception as e:
        print(f"\nCould not create plot: {e}")

if __name__ == "__main__":
    find_optimal_threshold()
