#!/usr/bin/env python3
"""
Force run all topologies including Penrose
"""
import os
import time
import numpy as np
import asyncio

# Force experimental
os.environ["TORI_ENABLE_EXOTIC"] = "1"

from python.core.hot_swap_laplacian import HotSwappableLaplacian

async def force_benchmark():
    sizes = [256, 512, 1024, 2048]
    RUNS_PER_SIZE = 7
    
    hot_swap = HotSwappableLaplacian(
        initial_topology="kagome",
        lattice_size=(20, 20),
        enable_experimental=True,
    )
    
    def bench(topology):
        """Benchmark with proper warm-up and timing"""
        hot_swap.current_topology = topology
        hot_swap.graph_laplacian = hot_swap._build_laplacian(topology)
        times = []
        
        for n in sizes:
            A = np.random.rand(n, n)
            B = np.random.rand(n, n)
            
            # Warm-up
            for _ in range(3):
                if topology == "penrose":
                    from python.core.penrose_microkernel_v2 import multiply
                    _, _ = multiply(A, B, hot_swap.graph_laplacian)
                else:
                    _ = A @ B
            
            # Actual runs
            run_times = []
            for _ in range(RUNS_PER_SIZE):
                t0 = time.perf_counter()
                if topology == "penrose":
                    from python.core.penrose_microkernel_v2 import multiply
                    C, info = multiply(A, B, hot_swap.graph_laplacian)
                else:
                    C = A @ B
                t1 = time.perf_counter()
                run_times.append(t1 - t0)
            
            times.append(np.median(run_times))
        
        # Log-log regression for omega
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        slope, intercept = np.polyfit(log_sizes, log_times, 1)
        
        # R-squared
        residuals = log_times - (slope * log_sizes + intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_times - np.mean(log_times))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, r_squared, times

    print("\nFORCED ROBUST Ω TEST")
    print("=" * 60)
    print(f"Sizes: {sizes}")
    print(f"Runs per size: {RUNS_PER_SIZE}")
    print("=" * 60)
    print()
    
    # Force all topologies including Penrose
    for topo in ["kagome", "honeycomb", "triangular", "small_world", "penrose"]:
        try:
            print(f"\nBenchmarking {topo}...")
            omega, r2, times = bench(topo)
            print(f"{topo:>12}  ω ≈ {omega:5.3f}  (R² = {r2:.3f})")
            print(f"{'':>12}  times: " + " ".join(f"{t*1000:6.1f}ms" for t in times))
        except Exception as e:
            print(f"{topo:>12}  ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(force_benchmark())
