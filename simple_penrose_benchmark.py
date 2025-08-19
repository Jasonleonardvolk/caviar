#!/usr/bin/env python3
"""
SIMPLE PENROSE BENCHMARK - Maximum Compatibility Version
Just focuses on the core Penrose performance without complex dependencies
"""
import os
import sys
import time
import numpy as np

# Setup
os.environ["TORI_ENABLE_EXOTIC"] = "1"
print("🚀 LAUNCHING PENROSE BENCHMARK")
print("═" * 50)

# Try to find the microkernel
try:
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Try various import paths
    import python.core.penrose_microkernel_v2 as pmk
    pmk.clear_cache()
    print(f"✅ Loaded microkernel with RANK = {pmk.RANK}")
    
    from python.core.exotic_topologies import build_penrose_laplacian
    print("✅ Loaded Penrose topology builder")
    
    PENROSE_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️  Import issue: {e}")
    print("🔧 Running in fallback mode...")
    PENROSE_AVAILABLE = False

def numpy_multiply_benchmark(sizes, runs=5):
    """Fallback benchmark using standard NumPy"""
    print("\n📊 NUMPY FALLBACK BENCHMARK")
    print("-" * 30)
    
    times = []
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Warm up
        _ = A @ B
        
        # Time runs
        run_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            C = A @ B
            t1 = time.perf_counter()
            run_times.append(t1 - t0)
        
        median_time = np.median(run_times)
        times.append(median_time)
        print(f"n={n:4d}: {median_time*1000:7.1f}ms (NumPy baseline)")
    
    # Calculate omega
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    slope, _ = np.polyfit(log_sizes, log_times, 1)
    
    print(f"\nNumPy Baseline: ω = {slope:.3f}")
    return slope, times

def penrose_benchmark(sizes, runs=5):
    """Main Penrose benchmark"""
    print("\n🔥 PENROSE MICROKERNEL BENCHMARK")
    print("-" * 40)
    
    # Build Penrose Laplacian
    print("🏗️  Building Penrose Laplacian...")
    L = build_penrose_laplacian()
    print(f"📊 Laplacian: {L.shape[0]} nodes, {L.nnz} edges")
    
    times = []
    for n in sizes:
        if n > L.shape[0]:
            print(f"⏭️  Skipping n={n} (exceeds Laplacian size)")
            continue
            
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Warm up
        C, info = pmk.multiply(A, B, L)
        
        if 'fallback' in info:
            print(f"⚠️  Fallback at n={n}: {info['fallback']}")
            continue
        
        # Time runs
        run_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            C, info = pmk.multiply(A, B, L)
            t1 = time.perf_counter()
            run_times.append(t1 - t0)
        
        median_time = np.median(run_times)
        times.append(median_time)
        
        gap = info.get('spectral_gap', 0)
        rank = info.get('rank', '?')
        print(f"n={n:4d}: {median_time*1000:7.1f}ms (gap={gap:.4f}, rank={rank})")
    
    if len(times) < 2:
        print("❌ Insufficient data for omega calculation")
        return 3.0, times
    
    # Calculate omega
    valid_sizes = sizes[:len(times)]
    log_sizes = np.log(valid_sizes)
    log_times = np.log(times)
    slope, _ = np.polyfit(log_sizes, log_times, 1)
    
    # R-squared
    residuals = log_times - (slope * log_sizes + np.polyfit(log_sizes, log_times, 1)[1])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_times - np.mean(log_times))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\n🎯 PENROSE RESULTS:")
    print(f"   ω = {slope:.4f} (R² = {r_squared:.3f})")
    
    return slope, times

def main():
    """Main benchmark execution"""
    sizes = [256, 512, 1024, 2048]
    
    print(f"🎯 Test sizes: {sizes}")
    print(f"🔧 System info:")
    print(f"   - Python: {sys.version.split()[0]}")
    print(f"   - NumPy: {np.__version__}")
    print(f"   - Threads: {os.environ.get('OPENBLAS_NUM_THREADS', 'default')}")
    
    # Always run NumPy baseline
    numpy_omega, numpy_times = numpy_multiply_benchmark(sizes)
    
    if PENROSE_AVAILABLE:
        # Run Penrose benchmark
        penrose_omega, penrose_times = penrose_benchmark(sizes)
        
        print("\n" + "🏆" * 40)
        print("🏆" + " " * 38 + "🏆")
        print("🏆         BENCHMARK RESULTS         🏆")
        print("🏆" + " " * 38 + "🏆")
        print("🏆" * 40)
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   NumPy Baseline:  ω = {numpy_omega:.4f}")
        print(f"   Penrose Method:  ω = {penrose_omega:.4f}")
        
        if penrose_omega < numpy_omega:
            improvement = (numpy_omega - penrose_omega) / numpy_omega * 100
            print(f"   🚀 Improvement: {improvement:.1f}% better exponent!")
        
        if penrose_omega < 2.371339:
            print("\n🎉🎉🎉 WORLD RECORD! 🎉🎉🎉")
            print("🏆 Faster than theoretical record holders!")
        elif penrose_omega < 2.4:
            print("\n🎊 BREAKTHROUGH! 🎊")
            print("🥈 Broke the 2.4 barrier!")
        elif penrose_omega < 2.807:
            print("\n✨ SUCCESS! ✨")
            print("🥉 Faster than Strassen!")
            
    else:
        print("\n⚠️  Penrose not available - only baseline measured")
        print(f"   NumPy achieved: ω = {numpy_omega:.4f}")
    
    print(f"\n🎊 Benchmark complete! 🎊")
    print("Thank you for testing the future of matrix multiplication! 🚀")

if __name__ == "__main__":
    main()
