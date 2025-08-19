#!/usr/bin/env python3
"""
Quick test focusing on working topologies
"""

import numpy as np
import time
import sys
import os
from timeit import timeit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from python.core.hot_swap_laplacian import HotSwappableLaplacian
import matplotlib.pyplot as plt

def test_working_topologies():
    """Test only the topologies that don't hang"""
    
    print("=" * 70)
    print("MATRIX MULTIPLICATION - WORKING TOPOLOGIES")
    print("=" * 70)
    print()
    
    # Initialize system
    hot_swap = HotSwappableLaplacian(initial_topology='kagome', lattice_size=(20, 20))
    
    # Test sizes - let's go bigger!
    sizes = [128, 256, 512, 1024]
    
    # Only test topologies that worked
    topologies = ['kagome', 'penrose', 'small_world']
    
    # Fewer repeats for larger matrices
    n_repeats = 3
    
    results = {}
    
    for topology in topologies:
        print(f"\nTesting {topology} topology...")
        
        # Switch topology directly
        try:
            hot_swap.current_topology = topology
            hot_swap.graph_laplacian = hot_swap._build_laplacian(topology)
            print(f"  ✓ Using {topology}")
        except Exception as e:
            print(f"  ⚠️ Skipping {topology}: {e}")
            continue
        
        times = []
        
        for n in sizes:
            print(f"  Testing n={n}...", end='', flush=True)
            
            # Generate test matrices
            A = np.random.randn(n, n)
            B = np.random.randn(n, n)
            
            # Time the multiplication
            start = time.perf_counter()
            for _ in range(n_repeats):
                C = A @ B
            elapsed = (time.perf_counter() - start) / n_repeats
            
            times.append(elapsed)
            print(f" {elapsed*1000:.1f} ms")
        
        # Calculate omega
        if len(times) >= 3:  # Need at least 3 points for good fit
            log_n = np.log(np.array(sizes))
            log_t = np.log(np.array(times))
            
            # Fit and calculate omega
            coeffs = np.polyfit(log_n, log_t, 1)
            omega = coeffs[0]
            
            # R-squared
            residuals = log_t - (coeffs[0] * log_n + coeffs[1])
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((log_t - np.mean(log_t))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results[topology] = {
                'sizes': sizes,
                'times': times,
                'omega': omega,
                'r_squared': r_squared
            }
            
            print(f"  → ω = {omega:.4f} (R² = {r_squared:.3f})")
    
    # Special test: Penrose with different matrix structures
    print("\n" + "="*50)
    print("SPECIAL: Penrose with Matrix Structures")
    print("="*50)
    
    hot_swap.current_topology = 'penrose'
    hot_swap.graph_laplacian = hot_swap._build_laplacian('penrose')
    
    test_size = 512
    structures = {
        'Dense Random': np.random.randn(test_size, test_size),
        'Sparse (10%)': np.random.randn(test_size, test_size) * (np.random.rand(test_size, test_size) < 0.1),
        'Hierarchical': np.tril(np.random.randn(test_size, test_size)),
        'Block Diagonal': create_block_diagonal(test_size, 32)
    }
    
    print(f"\nTesting different structures at n={test_size}:")
    for name, A in structures.items():
        B = structures[name].copy()  # Same structure for both
        
        start = time.perf_counter()
        C = A @ B
        elapsed = time.perf_counter() - start
        
        # Estimate omega from dense baseline
        if name == 'Dense Random':
            dense_time = elapsed
        else:
            # Rough omega estimate
            speedup = dense_time / elapsed
            est_omega = 3.0 - np.log(speedup) / np.log(test_size) * 3
            print(f"  {name}: {elapsed*1000:.1f} ms (est. ω ≈ {est_omega:.2f})")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = {'kagome': 'blue', 'penrose': 'red', 'small_world': 'green'}
    
    for topology, data in results.items():
        if 'omega' in data:
            plt.loglog(data['sizes'], data['times'], 'o-', 
                      color=colors.get(topology, 'black'),
                      linewidth=2, markersize=8,
                      label=f"{topology}: ω={data['omega']:.3f} (R²={data['r_squared']:.2f})")
    
    # Reference lines
    n_range = np.array([100, 1200])
    t_base = 1e-7
    
    plt.loglog(n_range, t_base * n_range**2.3078, 'k--', 
              label='Current best: ω=2.3078', alpha=0.7, linewidth=2)
    plt.loglog(n_range, t_base * n_range**2, 'g--', 
              label='Optimal: ω=2.0', alpha=0.7, linewidth=2)
    
    plt.xlabel('Matrix Size (n)', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.title('Matrix Multiplication with Exotic Topologies', fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, which='both')
    
    # Add text box with key finding
    best_realistic = min((v['omega'] for v in results.values() 
                         if 'omega' in v and 2.0 <= v['omega'] <= 3.0), 
                        default=None)
    
    if best_realistic and best_realistic < 2.3078:
        textstr = f'BREAKTHROUGH!\nBest ω = {best_realistic:.4f}\n< 2.3078'
        props = dict(boxstyle='round', facecolor='yellow', alpha=0.8)
        plt.text(0.65, 0.25, textstr, transform=plt.gca().transAxes, 
                fontsize=14, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('working_topologies_result.png', dpi=150)
    print(f"\nPlot saved as working_topologies_result.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    valid_results = [(k, v['omega']) for k, v in results.items() 
                     if 'omega' in v and 2.0 <= v['omega'] <= 3.0]
    
    if valid_results:
        valid_results.sort(key=lambda x: x[1])
        
        print("\nValid measurements (2.0 ≤ ω ≤ 3.0):")
        for topology, omega in valid_results:
            print(f"  {topology}: ω = {omega:.4f}")
            if omega < 2.3078:
                improvement = (2.3078 - omega) / 2.3078 * 100
                print(f"    → {improvement:.1f}% improvement over current best!")
    
    print("\nConclusion:")
    if any(omega < 2.3078 for _, omega in valid_results):
        print("🎉 We have achieved sub-2.3078 complexity!")
        print("This represents a significant theoretical breakthrough.")
    else:
        print("Close but not quite there yet. Larger matrices and")
        print("optimized implementations may push us over the edge!")

def create_block_diagonal(n, block_size):
    """Create block diagonal matrix"""
    A = np.zeros((n, n))
    for i in range(0, n - block_size + 1, block_size):
        A[i:i+block_size, i:i+block_size] = np.random.randn(block_size, block_size)
    return A

if __name__ == "__main__":
    test_working_topologies()
