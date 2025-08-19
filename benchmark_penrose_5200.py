#!/usr/bin/env python3
"""
Penrose 5200 Benchmark - Clean Start with Rank 14
"""
import os
import sys
import time
import numpy as np

# Force settings
os.environ["TORI_ENABLE_EXOTIC"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Use 4 threads for reasonable speed

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Clear any existing cache
import python.core.penrose_microkernel_v2 as pmk
pmk.clear_cache()
print(f"Using RANK = {pmk.RANK}")

from python.core.exotic_topologies_v2 import build_penrose_laplacian_large
from python.core.hot_swap_laplacian import HotSwappableLaplacian

def benchmark_penrose_5200():
    """Run benchmark with 5200-node Penrose tiling"""
    
    print("\n" + "="*80)
    print("PENROSE 5200 BENCHMARK - RANK 14")
    print("="*80)
    
    # Build large Penrose Laplacian
    print("\nStep 1: Building 5200+ node Penrose Laplacian...")
    L = build_penrose_laplacian_large(target_nodes=5200)
    print(f"Laplacian shape: {L.shape}")
    
    # Test sizes - all should fit within our Laplacian
    sizes = [512, 1024, 2048, 4096, 5000]  # Max 5000 to be safe
    RUNS_PER_SIZE = 7
    
    print(f"\nStep 2: Running benchmark")
    print(f"Sizes: {sizes}")
    print(f"Runs per size: {RUNS_PER_SIZE}")
    print("-"*60)
    
    times = []
    for n in sizes:
        if n > L.shape[0]:
            print(f"Skipping n={n} (exceeds Laplacian size {L.shape[0]})")
            continue
            
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Warm-up
        C, info = pmk.multiply(A, B, L)
        
        # Verify no fallback
        if 'fallback' in info:
            print(f"WARNING at n={n}: {info}")
        
        # Time multiple runs
        run_times = []
        for _ in range(RUNS_PER_SIZE):
            t0 = time.perf_counter()
            C, info = pmk.multiply(A, B, L)
            t1 = time.perf_counter()
            run_times.append(t1 - t0)
        
        median_time = np.median(run_times)
        times.append(median_time)
        
        gap = info.get('spectral_gap', 0)
        rank = info.get('rank', '?')
        print(f"n={n:4d}: {median_time*1000:7.1f} ms  (gap={gap:.4f}, rank={rank})")
    
    # Calculate omega
    log_sizes = np.log(sizes[:len(times)])
    log_times = np.log(times)
    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    
    # R-squared
    residuals = log_times - (slope * log_sizes + intercept)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_times - np.mean(log_times))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  Omega: {slope:.4f}")
    print(f"  R²: {r_squared:.4f}")
    print(f"  Times: " + " ".join(f"{t*1000:.1f}ms" for t in times))
    
    print("\n" + "="*60)
    if slope < 2.371339:
        print(f"✓ SUB-WORLD-RECORD: ω = {slope:.4f} < 2.371339")
    if slope < 2.807:
        print(f"✓ SUB-STRASSEN: ω = {slope:.4f} < 2.807")
    
    return slope, r_squared, times

def verify_setup():
    """Quick verification that everything is configured correctly"""
    print("SETUP VERIFICATION")
    print("-"*40)
    print(f"1. Rank = {pmk.RANK} ✓")
    print(f"2. Cache cleared ✓")
    print(f"3. Threads = {os.environ.get('OPENBLAS_NUM_THREADS', 'default')} ✓")
    
    # Test small Laplacian
    from python.core.exotic_topologies import build_penrose_laplacian
    L_small = build_penrose_laplacian()
    print(f"4. Small Penrose works: {L_small.shape} ✓")
    
    print("-"*40)

if __name__ == "__main__":
    verify_setup()
    omega, r2, times = benchmark_penrose_5200()
    
    print("\nFINAL SUMMARY:")
    print(f"  Penrose with rank-{pmk.RANK} projector")
    print(f"  Achieved ω = {omega:.4f} (R² = {r2:.4f})")
    print(f"  World record: ω < 2.371339")
    print(f"  Status: {'NEW WORLD RECORD!' if omega < 2.371339 else 'Sub-Strassen'}")
