#!/usr/bin/env python3
"""
Force Penrose test - bypass experimental flag check
"""
import os
import time
import numpy as np

# Force experimental mode
os.environ["TORI_ENABLE_EXOTIC"] = "1"

from python.core.hot_swap_laplacian import HotSwappableLaplacian

print("PENROSE FORCED TEST")
print("=" * 60)

# Initialize with experimental enabled
hot_swap = HotSwappableLaplacian(
    initial_topology="kagome",
    lattice_size=(20, 20),
    enable_experimental=True,  # Force experimental
)

print(f"Available topologies: {hot_swap.topologies.keys()}")
print(f"Experimental enabled: {hot_swap.experimental_enabled if hasattr(hot_swap, 'experimental_enabled') else 'N/A'}")

if "penrose" not in hot_swap.topologies:
    print("\nERROR: Penrose not available even with experimental=True!")
    print("Check that TORI_ENABLE_EXOTIC is set and exotic_topologies.py is imported.")
else:
    print("\n✓ Penrose is available!")
    
    # Test Penrose
    print("\nTesting Penrose topology...")
    try:
        hot_swap.current_topology = "penrose"
        hot_swap.graph_laplacian = hot_swap._build_laplacian("penrose")
        
        # Run a few multiplications
        sizes = [256, 512, 1024]
        times = []
        
        for n in sizes:
            A = np.random.rand(n, n)
            B = np.random.rand(n, n)
            
            # Warm-up
            _ = A @ B
            
            # Time it
            t0 = time.perf_counter()
            if hot_swap.current_topology == "penrose":
                from python.core.penrose_microkernel_v2 import multiply
                C, info = multiply(A, B, hot_swap.graph_laplacian)
                print(f"  n={n}: {info}")
            else:
                C = A @ B
            t1 = time.perf_counter()
            
            times.append((t1 - t0) * 1000)  # ms
            
        print(f"\nTimes: {[f'{t:.1f}ms' for t in times]}")
        
        # Calculate omega
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        slope, _ = np.polyfit(log_sizes, log_times, 1)
        print(f"Omega ≈ {slope:.3f}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
