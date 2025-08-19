#!/usr/bin/env python3
"""
Debug version of Robust Omega Test to find Penrose issue
"""
import argparse
import asyncio
import os
import time
import numpy as np
import traceback
from python.core.hot_swap_laplacian import HotSwappableLaplacian

# ──────────────────────────────────────────────────────────────────────
#  Argument parsing
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Robust Ω estimator")
parser.add_argument("--experimental", action="store_true",
                    help="Enable exotic Penrose topology")
args = parser.parse_args()

# honour env var too
if args.experimental:
    os.environ["TORI_ENABLE_EXOTIC"] = "1"

# ──────────────────────────────────────────────────────────────────────
#  Test routine
# ──────────────────────────────────────────────────────────────────────
async def robust_omega_test():
    # Larger sizes to measure actual scaling, not overhead
    sizes = [256, 512, 1024, 2048]
    
    # More runs for better statistics
    RUNS_PER_SIZE = 7  # was 5
    GROUP = 3          # 3 groups → median-of-means
    
    hot_swap = HotSwappableLaplacian(
        initial_topology="kagome",
        lattice_size=(20, 20),
        enable_experimental=args.experimental,
    )
    
    print(f"Experimental mode: {args.experimental}")
    print(f"Available topologies: {list(hot_swap.topologies.keys())}")
    print()

    def bench(topology):
        """Benchmark with proper warm-up and timing"""
        print(f"\nBenchmarking {topology}...")
        hot_swap.current_topology = topology
        t_graph = time.perf_counter()
        hot_swap.graph_laplacian = hot_swap._build_laplacian(topology)
        if topology == "penrose":
            print(f"[penrose] Laplacian built in {(time.perf_counter()-t_graph)*1000:.1f} ms")
        times = []
        
        for n in sizes:
            A = np.random.rand(n, n)
            B = np.random.rand(n, n)
            
            # Warm-up: discard first run to load BLAS
            if hot_swap.current_topology == "penrose" and hot_swap.enable_experimental:
                _, _ = hot_swap.multiply_with_topology(A, B)
            else:
                _ = A @ B
            
            # Time multiple runs
            run_times = []
            for _ in range(RUNS_PER_SIZE):
                t0 = time.perf_counter()
                if hot_swap.current_topology == "penrose" and hot_swap.enable_experimental:
                    _, _ = hot_swap.multiply_with_topology(A, B)
                else:
                    _ = A @ B
                t1 = time.perf_counter()
                run_times.append(t1 - t0)  # Keep in seconds
            
            # Median of runs for this size
            times.append(np.median(run_times))
        
        # Log-log regression for omega
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        slope, intercept = np.polyfit(log_sizes, log_times, 1)
        
        # R-squared for fit quality
        residuals = log_times - (slope * log_sizes + intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_times - np.mean(log_times))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, r_squared, times

    print("\nROBUST Ω TEST")
    print("=" * 60)
    print(f"Sizes: {sizes}")
    print(f"Runs per size: {RUNS_PER_SIZE}")
    print("=" * 60)
    print()
    
    for topo in ["kagome", "honeycomb", "triangular", "small_world", "penrose"]:
        if topo == "penrose" and not args.experimental:
            print(f"{topo:>12}  skipped (experimental mode not enabled)")
            continue
        try:
            omega, r2, times = bench(topo)
            print(f"{topo:>12}  ω ≈ {omega:5.3f}  (R² = {r2:.3f})")
            # Show actual times for verification
            print(f"{'':>12}  times: " + " ".join(f"{t*1000:6.1f}ms" for t in times))
        except Exception as e:
            print(f"{topo:>12}  ERROR: {e}")
            if topo == "penrose":
                print("\nDetailed error trace:")
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Expected ω values:")
    print("  - Standard BLAS: ~3.0")
    print("  - Strassen algorithm: ~2.807") 
    print("  - Current record: ~2.373")
    print("  - TORI target: <2.3078")
    print("  - Penrose potential: ~2.3-2.4")

if __name__ == "__main__":
    asyncio.run(robust_omega_test())
