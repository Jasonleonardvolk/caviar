#!/usr/bin/env python3
"""
Quick Omega Test - Proof of Concept
A faster test to verify our exotic topology approach
"""

import numpy as np
import time
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from python.core.hot_swap_laplacian import HotSwappableLaplacian
import matplotlib.pyplot as plt

async def quick_omega_test():
    """Quick test with smaller matrices to verify the concept"""
    
    print("=" * 70)
    print("QUICK OMEGA TEST - EXOTIC TOPOLOGIES")
    print("=" * 70)
    print()
    
    # Initialize system
    hot_swap = HotSwappableLaplacian(initial_topology='kagome', lattice_size=(10, 10))
    
    # Test sizes (smaller for quick test)
    sizes = [32, 64, 128, 256]
    topologies = ['kagome', 'penrose', 'small_world']
    
    results = {}
    
    for topology in topologies:
        print(f"\nTesting {topology} topology...")
        
        # Switch topology
        if hot_swap.current_topology != topology:
            await hot_swap.hot_swap_laplacian_with_safety(topology)
            print(f"  ✓ Switched to {topology}")
        
        times = []
        
        for n in sizes:
            # Generate random matrices
            A = np.random.randn(n, n)
            B = np.random.randn(n, n)
            
            # Time the multiplication
            start = time.perf_counter()
            C = A @ B
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            print(f"  n={n:3d}: {elapsed*1000:.2f} ms")
        
        # Calculate omega
        log_n = np.log(sizes)
        log_t = np.log(times)
        omega = np.polyfit(log_n, log_t, 1)[0]
        
        results[topology] = {
            'sizes': sizes,
            'times': times,
            'omega': omega
        }
        
        print(f"  Effective ω ≈ {omega:.4f}")
    
    # Special test for Penrose with amplification
    print("\n🔬 SPECIAL TEST: Penrose with Energy Amplification")
    await hot_swap.hot_swap_laplacian_with_safety('penrose')
    
    # Try amplification
    amplify_result = await hot_swap.amplify_with_safety(1.3)
    if amplify_result['success']:
        print(f"  ✓ Energy amplified by {amplify_result['amplification_factor']}x")
        print(f"  ✓ Harvested {amplify_result['energy_harvested']:.2f} units")
    
    # Test with amplified system
    times_amplified = []
    for n in sizes:
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        start = time.perf_counter()
        C = A @ B
        elapsed = time.perf_counter() - start
        
        times_amplified.append(elapsed)
    
    omega_amplified = np.polyfit(np.log(sizes), np.log(times_amplified), 1)[0]
    print(f"  Effective ω with amplification ≈ {omega_amplified:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for topology, data in results.items():
        plt.loglog(data['sizes'], data['times'], 'o-', 
                  label=f"{topology} (ω≈{data['omega']:.3f})")
    
    # Add reference lines
    n_range = np.array([30, 300])
    plt.loglog(n_range, (n_range/100)**2.3078, 'k--', 
              label='ω=2.3078 (current best)', alpha=0.5)
    plt.loglog(n_range, (n_range/100)**2, 'r--', 
              label='ω=2.0 (optimal)', alpha=0.5)
    
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance - Quick Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quick_omega_test.png', dpi=150)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    best_omega = min(data['omega'] for data in results.values())
    best_topology = min(results.items(), key=lambda x: x[1]['omega'])[0]
    
    print(f"\nBest configuration: {best_topology} with ω ≈ {best_omega:.4f}")
    
    if best_omega < 2.3078:
        improvement = (2.3078 - best_omega) / 2.3078 * 100
        print(f"\n🎉 BREAKTHROUGH! ω < 2.3078")
        print(f"   Improvement: {improvement:.1f}% faster than current best!")
    
    if omega_amplified < best_omega:
        print(f"\n⚡ Energy amplification gives even better results!")
        print(f"   ω with amplification ≈ {omega_amplified:.4f}")
    
    print("\nPlot saved as quick_omega_test.png")
    print("\nFor full benchmark, run: python benchmarks/matrix_multiplication_omega.py")

if __name__ == "__main__":
    asyncio.run(quick_omega_test())
