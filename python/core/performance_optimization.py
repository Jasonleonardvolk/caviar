#!/usr/bin/env python3
"""
TORI/KHA Performance Profiling and Optimization Framework
Comprehensive performance analysis and optimization tools
"""

import cProfile
import pstats
import tracemalloc
import asyncio
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import line_profiler
import memory_profiler

# ========== Performance Metrics ==========

@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    operation: str
    duration: float
    cpu_percent: float
    memory_mb: float
    throughput: Optional[float] = None
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None

# ========== Performance Profiler ==========

class TORIPerformanceProfiler:
    """Comprehensive performance profiling for TORI/KHA"""
    
    def __init__(self, output_dir: Path = Path("performance_reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    async def profile_cognitive_engine(self, engine, test_inputs: List[str]):
        """Profile CognitiveEngine performance"""
        print("🔍 Profiling CognitiveEngine...")
        
        # CPU profiling
        profiler = cProfile.Profile()
        
        # Memory tracking
        tracemalloc.start()
        
        # Metrics collection
        latencies = []
        memory_usage = []
        cpu_usage = []
        
        process = psutil.Process()
        
        for i, input_text in enumerate(test_inputs):
            # Pre-measurement
            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024
            
            # Profile execution
            profiler.enable()
            result = await engine.process(input_text)
            profiler.disable()
            
            # Post-measurement
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            # Collect metrics
            latency = end_time - start_time
            latencies.append(latency)
            memory_usage.append(end_memory - start_memory)
            cpu_usage.append(process.cpu_percent())
        
        # Memory snapshot
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Analyze results
        metrics = PerformanceMetrics(
            operation="CognitiveEngine.process",
            duration=np.mean(latencies),
            cpu_percent=np.mean(cpu_usage),
            memory_mb=np.mean(memory_usage),
            throughput=len(test_inputs) / sum(latencies),
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99)
        )
        
        # Save profiling results
        self._save_cpu_profile(profiler, "cognitive_engine")
        self._save_memory_profile(snapshot, "cognitive_engine")
        self._generate_performance_report(metrics, "cognitive_engine")
        
        return metrics
    
    async def profile_chaos_operations(self, chaos_control):
        """Profile chaos computing operations"""
        print("🌀 Profiling Chaos Operations...")
        
        operations = {
            "dark_soliton": self._profile_dark_soliton,
            "attractor_hop": self._profile_attractor_hop,
            "phase_explosion": self._profile_phase_explosion
        }
        
        results = {}
        
        for op_name, profile_func in operations.items():
            metrics = await profile_func(chaos_control)
            results[op_name] = metrics
            
        # Compare efficiency gains
        self._generate_chaos_comparison_report(results)
        
        return results
    
    async def profile_memory_operations(self, memory_vault):
        """Profile memory vault operations"""
        print("💾 Profiling Memory Operations...")
        
        operations = [
            ("store", self._profile_memory_store),
            ("retrieve", self._profile_memory_retrieve),
            ("search", self._profile_memory_search),
            ("consolidate", self._profile_memory_consolidate)
        ]
        
        results = {}
        
        for op_name, profile_func in operations.items():
            metrics = await profile_func(memory_vault)
            results[op_name] = metrics
        
        return results
    
    def _save_cpu_profile(self, profiler: cProfile.Profile, name: str):
        """Save CPU profiling results"""
        stats_file = self.output_dir / f"{name}_cpu_profile.stats"
        profiler.dump_stats(str(stats_file))
        
        # Generate readable report
        report_file = self.output_dir / f"{name}_cpu_report.txt"
        with open(report_file, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(30)  # Top 30 functions
    
    def _save_memory_profile(self, snapshot, name: str):
        """Save memory profiling results"""
        report_file = self.output_dir / f"{name}_memory_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"Memory Profile: {name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Top memory allocations
            top_stats = snapshot.statistics('lineno')[:20]
            
            for stat in top_stats:
                f.write(f"{stat}\n")
    
    def _generate_performance_report(self, metrics: PerformanceMetrics, name: str):
        """Generate performance visualization report"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Performance Profile: {name}')
        
        # Latency distribution
        ax = axes[0, 0]
        ax.bar(['p50', 'p95', 'p99'], 
               [metrics.latency_p50, metrics.latency_p95, metrics.latency_p99])
        ax.set_title('Latency Percentiles (seconds)')
        ax.set_ylabel('Latency (s)')
        
        # Resource usage
        ax = axes[0, 1]
        ax.bar(['CPU %', 'Memory MB'], 
               [metrics.cpu_percent, metrics.memory_mb])
        ax.set_title('Resource Usage')
        
        # Throughput over time (if available)
        ax = axes[1, 0]
        if hasattr(self, f'{name}_throughput_history'):
            history = getattr(self, f'{name}_throughput_history')
            ax.plot(history)
            ax.set_title('Throughput Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Ops/second')
        
        # Summary metrics
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        Operation: {metrics.operation}
        Avg Duration: {metrics.duration:.3f}s
        Throughput: {metrics.throughput:.2f} ops/s
        CPU Usage: {metrics.cpu_percent:.1f}%
        Memory Usage: {metrics.memory_mb:.1f} MB
        """
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{name}_performance.png")
        plt.close()

# ========== Optimization Framework ==========

class TORIOptimizer:
    """Performance optimization framework"""
    
    def __init__(self):
        self.optimizations = []
    
    def optimize_numpy_operations(self):
        """Optimize NumPy operations"""
        optimizations = """
# ========== NumPy Optimization Patterns ==========

# 1. Vectorization
# Before:
def slow_operation(data):
    result = []
    for x in data:
        result.append(np.sin(x) * np.cos(x))
    return np.array(result)

# After:
def fast_operation(data):
    return np.sin(data) * np.cos(data)

# 2. In-place operations
# Before:
matrix = matrix + delta

# After:
matrix += delta  # Modifies in-place, saves memory

# 3. Preallocate arrays
# Before:
results = []
for i in range(n):
    results.append(compute(i))
results = np.array(results)

# After:
results = np.empty(n)
for i in range(n):
    results[i] = compute(i)

# 4. Use appropriate dtypes
# Before:
matrix = np.zeros((1000, 1000))  # float64 by default

# After:
matrix = np.zeros((1000, 1000), dtype=np.float32)  # Half memory

# 5. Avoid unnecessary copies
# Before:
def process(data):
    data = np.array(data)  # Creates copy
    return data * 2

# After:
def process(data):
    data = np.asarray(data)  # No copy if already array
    return data * 2
"""
        return optimizations
    
    def optimize_async_operations(self):
        """Optimize async operations"""
        optimizations = """
# ========== Async Optimization Patterns ==========

# 1. Batch operations
# Before:
async def process_items(items):
    results = []
    for item in items:
        result = await process_single(item)
        results.append(result)
    return results

# After:
async def process_items(items):
    tasks = [process_single(item) for item in items]
    return await asyncio.gather(*tasks)

# 2. Connection pooling
# Before:
async def fetch_data(urls):
    results = []
    for url in urls:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                results.append(await response.text())
    return results

# After:
async def fetch_data(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# 3. Semaphore for rate limiting
sem = asyncio.Semaphore(10)  # Max 10 concurrent operations

async def limited_operation(item):
    async with sem:
        return await expensive_operation(item)

# 4. Caching async results
from functools import lru_cache

@lru_cache(maxsize=1000)
async def cached_operation(key):
    return await expensive_lookup(key)

# 5. Avoid blocking operations
# Before:
async def bad_async():
    time.sleep(1)  # Blocks event loop!
    return result

# After:
async def good_async():
    await asyncio.sleep(1)  # Non-blocking
    return result
"""
        return optimizations
    
    def optimize_memory_usage(self):
        """Memory optimization strategies"""
        strategies = """
# ========== Memory Optimization Strategies ==========

# 1. Lazy loading
class LazyMemoryVault:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    async def get(self, key):
        if key in self._cache:
            return self._cache[key]
        
        # Load from disk only when needed
        value = await self._load_from_disk(key)
        self._cache[key] = value
        return value

# 2. Memory-mapped files for large data
import mmap

class MappedStorage:
    def __init__(self, filepath):
        self.file = open(filepath, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
    
    def read_chunk(self, offset, size):
        self.mmap.seek(offset)
        return self.mmap.read(size)

# 3. Object pooling
class ObjectPool:
    def __init__(self, factory, max_size=100):
        self.factory = factory
        self.pool = []
        self.max_size = max_size
    
    def acquire(self):
        if self.pool:
            return self.pool.pop()
        return self.factory()
    
    def release(self, obj):
        if len(self.pool) < self.max_size:
            obj.reset()  # Clear state
            self.pool.append(obj)

# 4. Streaming processing
async def process_large_file(filepath):
    async with aiofiles.open(filepath, 'r') as f:
        async for line in f:
            await process_line(line)
            # Process one line at a time, minimal memory

# 5. Garbage collection tuning
import gc

# Disable GC during critical operations
gc.disable()
try:
    critical_operation()
finally:
    gc.enable()
    gc.collect()
"""
        return strategies
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization guide"""
        report_path = Path("optimization_guide.md")
        
        with open(report_path, 'w') as f:
            f.write("# TORI/KHA Performance Optimization Guide\n\n")
            
            f.write("## NumPy Optimizations\n")
            f.write(self.optimize_numpy_operations())
            
            f.write("\n## Async Optimizations\n")
            f.write(self.optimize_async_operations())
            
            f.write("\n## Memory Optimizations\n")
            f.write(self.optimize_memory_usage())
            
            f.write("\n## Chaos-Specific Optimizations\n")
            f.write(self._chaos_optimizations())
            
            f.write("\n## Profiling Commands\n")
            f.write(self._profiling_commands())
    
    def _chaos_optimizations(self):
        return """
# ========== Chaos Computing Optimizations ==========

# 1. Energy budget optimization
class AdaptiveEnergyBroker:
    def __init__(self):
        self.efficiency_history = defaultdict(list)
    
    def allocate_energy(self, module_id):
        # Allocate more energy to efficient modules
        efficiency = np.mean(self.efficiency_history[module_id][-10:])
        base_allocation = 100
        return int(base_allocation * (1 + efficiency))

# 2. Parallel chaos operations
async def parallel_chaos_tasks(tasks):
    # Group by chaos mode for better cache locality
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task.mode].append(task)
    
    results = []
    for mode, mode_tasks in grouped.items():
        # Process same-mode tasks together
        mode_results = await process_mode_batch(mode, mode_tasks)
        results.extend(mode_results)
    
    return results

# 3. Adaptive timestep for stability
class AdaptiveChaosTimestep:
    def __init__(self):
        self.stability_threshold = 0.9
    
    def compute_timestep(self, eigenvalues):
        max_eigen = np.max(np.abs(eigenvalues))
        if max_eigen > self.stability_threshold:
            return 0.01 / max_eigen  # Smaller step
        return 0.01  # Default step
"""
    
    def _profiling_commands(self):
        return """
## Profiling Commands

### CPU Profiling
```bash
python -m cProfile -o profile.stats tori_production.py
python -m pstats profile.stats
```

### Memory Profiling
```bash
python -m memory_profiler tori_production.py
mprof run tori_production.py
mprof plot
```

### Line-by-line Profiling
```python
@profile
def function_to_profile():
    # Add @profile decorator
    pass

# Run with:
kernprof -l -v script.py
```

### Async Profiling
```python
import yappi

yappi.set_clock_type("cpu")
yappi.start()

# Run async code

yappi.stop()
stats = yappi.get_func_stats()
stats.save("async_profile.pstat", type='pstat')
```
"""

# ========== Benchmark Suite ==========

class TORIBenchmarkSuite:
    """Comprehensive benchmark suite"""
    
    def __init__(self):
        self.benchmarks = {}
    
    async def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print("🏃 Running TORI/KHA Benchmark Suite")
        print("=" * 60)
        
        benchmarks = [
            ("Cognitive Processing", self.benchmark_cognitive_processing),
            ("Memory Operations", self.benchmark_memory_operations),
            ("Chaos Computing", self.benchmark_chaos_computing),
            ("Quantum Integration", self.benchmark_quantum_integration),
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\n📊 {name} Benchmark")
            print("-" * 40)
            
            results = await benchmark_func()
            self.benchmarks[name] = results
            self._print_benchmark_results(name, results)
        
        self._generate_benchmark_report()
    
    async def benchmark_cognitive_processing(self):
        """Benchmark cognitive engine performance"""
        from python.core.CognitiveEngine import CognitiveEngine
        
        configs = [
            {"vector_dim": 64, "name": "small"},
            {"vector_dim": 256, "name": "medium"},
            {"vector_dim": 512, "name": "large"},
            {"vector_dim": 1024, "name": "xlarge"},
        ]
        
        results = {}
        
        for config in configs:
            engine = CognitiveEngine(config)
            
            # Warmup
            await engine.process("warmup")
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                await engine.process("benchmark input text")
                times.append(time.time() - start)
            
            results[config['name']] = {
                'vector_dim': config['vector_dim'],
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'throughput': 1.0 / np.mean(times)
            }
            
            engine.shutdown()
        
        return results
    
    async def benchmark_memory_operations(self):
        """Benchmark memory vault performance"""
        from python.core.memory_vault import UnifiedMemoryVault
        
        vault = UnifiedMemoryVault()
        
        operations = {
            'store': self._benchmark_memory_store,
            'retrieve': self._benchmark_memory_retrieve,
            'search': self._benchmark_memory_search,
        }
        
        results = {}
        
        for op_name, op_func in operations.items():
            results[op_name] = await op_func(vault)
        
        vault.shutdown()
        
        return results
    
    async def _benchmark_memory_store(self, vault):
        """Benchmark memory store operations"""
        sizes = [10, 100, 1000, 10000]
        results = {}
        
        for size in sizes:
            data = {"content": "x" * size, "metadata": {"size": size}}
            
            times = []
            for _ in range(100):
                start = time.time()
                await vault.store(data, "semantic")
                times.append(time.time() - start)
            
            results[f"size_{size}"] = {
                'mean_time': np.mean(times),
                'ops_per_sec': 1.0 / np.mean(times)
            }
        
        return results
    
    def _print_benchmark_results(self, name: str, results: dict):
        """Print benchmark results"""
        df = pd.DataFrame(results).T
        print(df.to_string())
    
    def _generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        report_path = Path("benchmark_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# TORI/KHA Benchmark Report\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for name, results in self.benchmarks.items():
                f.write(f"## {name}\n")
                df = pd.DataFrame(results).T
                f.write(df.to_markdown())
                f.write("\n\n")

# ========== Main Profiling Script ==========

async def main():
    """Run complete performance profiling and optimization"""
    
    # Initialize profiler
    profiler = TORIPerformanceProfiler()
    
    # Initialize components for profiling
    from python.core.CognitiveEngine import CognitiveEngine
    from python.core.chaos_control_layer import ChaosControlLayer
    from python.core.memory_vault import UnifiedMemoryVault
    
    engine = CognitiveEngine()
    chaos_control = ChaosControlLayer(None, None)  # Simplified init
    memory_vault = UnifiedMemoryVault()
    
    # Run profiling
    test_inputs = ["Test " + str(i) for i in range(100)]
    
    cognitive_metrics = await profiler.profile_cognitive_engine(engine, test_inputs)
    chaos_metrics = await profiler.profile_chaos_operations(chaos_control)
    memory_metrics = await profiler.profile_memory_operations(memory_vault)
    
    # Generate optimization guide
    optimizer = TORIOptimizer()
    optimizer.generate_optimization_report()
    
    # Run benchmarks
    benchmark_suite = TORIBenchmarkSuite()
    await benchmark_suite.run_all_benchmarks()
    
    print("\n✅ Performance profiling complete!")
    print(f"📁 Reports saved to: {profiler.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
