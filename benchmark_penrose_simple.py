#!/usr/bin/env python3
"""
Direct Penrose benchmark - simplified
"""
import os
import time
import numpy as np
import asyncio

# Force experimental mode
os.environ["TORI_ENABLE_EXOTIC"] = "1"

from python.core.hot_swap_laplacian import HotSwappableLaplacian
from python.core.penrose_microkernel_v2 import multiply

async def penrose_benchmark():
    sizes = [256, 512, 1024, 2048]
    RUNS_PER_SIZE = 7
    
    hot_swap = HotSwappableLaplacian(
        initial_topology="kagome",
        lattice_size=(20, 20),
        enable_experimental=True,
    )
    
    print("PENROSE DIRECT BENCHMARK")
    print("=" * 60)
    print(f"Sizes: {sizes}")
    print(f"Runs per size: {RUNS_PER_SIZE}")
    print("=" * 60)
    
    if "penrose" not in hot_swap.topologies:
        print("ERROR: Penrose not available!")
        return
    
    # Build Penrose Laplacian
    print("\nBuilding Penrose Laplacian...")
    hot_swap.current_topology = "penrose"
    hot_swap.graph_laplacian = hot_swap._build_laplacian("penrose")
    print("Penrose Laplacian built successfully!")
    
    times = []
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Warm-up with Penrose microkernel
        for _ in range(3):
            _, _ = multiply(A, B, hot_swap.graph_laplacian)
        
        # Actual runs
        run_times = []
        info = None
        for _ in range(RUNS_PER_SIZE):
            t0 = time.perf_counter()
            C, info = multiply(A, B, hot_swap.graph_laplacian)
            t1 = time.perf_counter()
            run_times.append(t1 - t0)
        
        median_time = np.median(run_times)
        times.append(median_time)
        print(f"  n={n}: {median_time*1000:.1f}ms  (method: {info.get('method', 'unknown')}, fallback: {info.get('fallback', 'none')})")
    
    # Calculate omega
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    
    # R-squared
    residuals = log_times - (slope * log_sizes + intercept)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_times - np.mean(log_times))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\nPenrose Results:")
    print(f"  ω ≈ {slope:.3f}  (R² = {r_squared:.3f})")
    print(f"  times: " + " ".join(f"{t*1000:6.1f}ms" for t in times))
    
    if slope < 2.4:
        print("\n✓ SUCCESS! Penrose achieved sub-2.4 omega!")
    else:
        print(f"\n✗ Omega {slope:.3f} is higher than expected. Check for BLAS fallbacks.")

if __name__ == "__main__":
    asyncio.run(penrose_benchmark())
