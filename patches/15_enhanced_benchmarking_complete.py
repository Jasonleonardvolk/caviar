#!/usr/bin/env python3
"""
Enhanced benchmarking for soliton memory performance
Tests scalability, performance bottlenecks, and optimization opportunities
"""

import time
import numpy as np
import psutil
import gc
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json
from datetime import datetime
import asyncio
import concurrent.futures

from python.core.oscillator_lattice import OscillatorLattice, get_global_lattice
from python.core.soliton_memory_integration import EnhancedSolitonMemory, MemoryType
from python.core.hot_swap_laplacian import HotSwapLaplacian
from python.core.memory_crystallization import MemoryCrystallizer


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    operation: str
    input_size: int
    duration: float
    operations_per_second: float
    memory_before_mb: float
    memory_after_mb: float
    cpu_percent: float
    metadata: Dict[str, Any]


class SolitonBenchmark:
    """Comprehensive benchmarking suite for soliton memory system"""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
        
    def benchmark_operation(self, name: str, operation: Callable, 
                          input_size: int, warmup: int = 1) -> BenchmarkResult:
        """Benchmark a single operation"""
        # Warmup
        for _ in range(warmup):
            operation()
        
        # Force garbage collection
        gc.collect()
        
        # Measure memory before
        memory_before = self.process.memory_info().rss / 1024 / 1024
        
        # Start CPU monitoring
        cpu_start = self.process.cpu_percent(interval=0.1)
        
        # Time the operation
        start_time = time.perf_counter()
        result = operation()
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        
        # Measure memory after
        memory_after = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        # Calculate operations per second
        ops_per_second = input_size / duration if duration > 0 else 0
        
        benchmark_result = BenchmarkResult(
            operation=name,
            input_size=input_size,
            duration=duration,
            operations_per_second=ops_per_second,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            cpu_percent=cpu_percent,
            metadata=result if isinstance(result, dict) else {}
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def run_memory_store_benchmark(self):
        """Benchmark memory storage at different scales"""
        print("\n=== Memory Storage Benchmark ===")
        
        sizes = [100, 500, 1000, 5000, 10000]
        store_results = []
        
        for size in sizes:
            # Create fresh system
            lattice = get_global_lattice()
            lattice.clear()
            memory_system = EnhancedSolitonMemory(lattice_size=min(size * 2, 50000))
            
            def store_memories():
                stored = 0
                for i in range(size):
                    memory_system.store_enhanced_memory(
                        content=f"Benchmark memory {i} with some content",
                        concept_ids=[f"concept_{i % 100}"],
                        memory_type=MemoryType.SEMANTIC,
                        sources=[f"source_{i % 10}"]
                    )
                    stored += 1
                return {"stored": stored}
            
            result = self.benchmark_operation(
                f"store_{size}_memories",
                store_memories,
                size
            )
            
            store_results.append(result)
            print(f"Size {size}: {result.operations_per_second:.2f} stores/sec, "
                  f"Memory: {result.memory_after_mb - result.memory_before_mb:.2f} MB")
        
        return store_results
    
    def run_resonance_search_benchmark(self):
        """Benchmark resonance search at different memory pool sizes"""
        print("\n=== Resonance Search Benchmark ===")
        
        pool_sizes = [1000, 5000, 10000, 25000]
        search_results = []
        
        for pool_size in pool_sizes:
            # Prepare memory pool
            lattice = get_global_lattice()
            lattice.clear()
            memory_system = EnhancedSolitonMemory(lattice_size=pool_size * 2)
            
            # Populate with memories
            print(f"Populating {pool_size} memories...")
            for i in range(pool_size):
                memory_system.store_enhanced_memory(
                    content=f"Search pool memory {i}",
                    concept_ids=[f"concept_{i % 50}", f"tag_{i % 20}"],
                    memory_type=MemoryType.SEMANTIC,
                    sources=["benchmark"]
                )
            
            # Benchmark searches
            num_searches = 100
            
            def perform_searches():
                found_total = 0
                for i in range(num_searches):
                    concepts = [f"concept_{i % 50}", f"tag_{i % 20}"]
                    phase = memory_system._calculate_concept_phase(concepts)
                    results = memory_system.find_resonant_memories_enhanced(
                        phase, concepts, threshold=0.5
                    )
                    found_total += len(results)
                return {"total_found": found_total, "avg_found": found_total / num_searches}
            
            result = self.benchmark_operation(
                f"search_in_{pool_size}_pool",
                perform_searches,
                num_searches
            )
            
            search_results.append(result)
            print(f"Pool {pool_size}: {result.operations_per_second:.2f} searches/sec, "
                  f"Avg results: {result.metadata.get('avg_found', 0):.1f}")
        
        return search_results
    
    def run_crystallization_benchmark(self):
        """Benchmark crystallization at different scales"""
        print("\n=== Crystallization Benchmark ===")
        
        memory_counts = [1000, 5000, 10000]
        crystallization_results = []
        
        for count in memory_counts:
            # Prepare system
            lattice = get_global_lattice()
            lattice.clear()
            memory_system = EnhancedSolitonMemory(lattice_size=count * 2)
            crystallizer = MemoryCrystallizer()
            
            # Populate memories with varying heat
            for i in range(count):
                mem_id = memory_system.store_enhanced_memory(
                    content=f"Crystal memory {i}",
                    concept_ids=[f"crystal_{i % 30}"],
                    memory_type=MemoryType.SEMANTIC,
                    sources=["benchmark"]
                )
                # Vary heat levels
                if i % 10 == 0:
                    memory_system.memory_entries[mem_id].heat = 0.8
                elif i % 5 == 0:
                    memory_system.memory_entries[mem_id].heat = 0.5
            
            def run_crystallization():
                report = crystallizer.crystallize(memory_system, lattice)
                return {
                    "migrations": report.get("migrations", 0),
                    "fusions": report.get("fusions", 0),
                    "decayed": report.get("decayed", 0)
                }
            
            result = self.benchmark_operation(
                f"crystallize_{count}_memories",
                run_crystallization,
                count
            )
            
            crystallization_results.append(result)
            print(f"Count {count}: {result.duration:.2f}s, "
                  f"Migrations: {result.metadata.get('migrations', 0)}, "
                  f"Fusions: {result.metadata.get('fusions', 0)}")
        
        return crystallization_results
    
    def run_topology_morphing_benchmark(self):
        """Benchmark topology morphing performance"""
        print("\n=== Topology Morphing Benchmark ===")
        
        oscillator_counts = [1000, 5000, 10000]
        morphing_results = []
        
        topologies = [
            ("kagome", "hexagonal"),
            ("hexagonal", "square"),
            ("square", "small_world"),
            ("kagome", "small_world")
        ]
        
        for count in oscillator_counts:
            # Create lattice with oscillators
            lattice = OscillatorLattice(size=count)
            hot_swap = HotSwapLaplacian()
            
            for from_top, to_top in topologies:
                hot_swap.current_topology = from_top
                
                def perform_morph():
                    hot_swap.initiate_morph(to_top, blend_rate=0.1)
                    steps = 0
                    total_energy = 0
                    
                    while hot_swap.is_morphing and steps < 50:
                        result = hot_swap.step_blend()
                        total_energy += result.get('energy_harvested', 0)
                        steps += 1
                    
                    return {
                        "steps": steps,
                        "energy_harvested": total_energy,
                        "from": from_top,
                        "to": to_top
                    }
                
                result = self.benchmark_operation(
                    f"morph_{from_top}_to_{to_top}_{count}",
                    perform_morph,
                    count
                )
                
                morphing_results.append(result)
                print(f"{from_top}->{to_top} ({count} osc): {result.duration:.3f}s, "
                      f"Energy: {result.metadata.get('energy_harvested', 0):.1f}")
        
        return morphing_results
    
    def run_concurrent_operations_benchmark(self):
        """Benchmark concurrent memory operations"""
        print("\n=== Concurrent Operations Benchmark ===")
        
        # Create shared memory system
        memory_system = EnhancedSolitonMemory(lattice_size=20000)
        
        # Pre-populate some memories
        for i in range(1000):
            memory_system.store_enhanced_memory(
                content=f"Initial memory {i}",
                concept_ids=[f"init_{i % 50}"],
                memory_type=MemoryType.SEMANTIC,
                sources=["benchmark"]
            )
        
        # Define concurrent tasks
        def writer_task(worker_id: int, count: int):
            """Write memories concurrently"""
            stored = 0
            for i in range(count):
                memory_system.store_enhanced_memory(
                    content=f"Worker {worker_id} memory {i}",
                    concept_ids=[f"worker_{worker_id}", f"item_{i}"],
                    memory_type=MemoryType.SEMANTIC,
                    sources=[f"worker_{worker_id}"]
                )
                stored += 1
            return stored
        
        def reader_task(worker_id: int, count: int):
            """Read memories concurrently"""
            found = 0
            for i in range(count):
                concepts = [f"init_{i % 50}"]
                phase = memory_system._calculate_concept_phase(concepts)
                results = memory_system.find_resonant_memories_enhanced(
                    phase, concepts, threshold=0.5
                )
                found += len(results)
            return found
        
        # Test different concurrency levels
        worker_counts = [2, 4, 8]
        operations_per_worker = 100
        
        for num_workers in worker_counts:
            def run_concurrent_test():
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers * 2) as executor:
                    # Submit writer tasks
                    writer_futures = [
                        executor.submit(writer_task, i, operations_per_worker)
                        for i in range(num_workers)
                    ]
                    
                    # Submit reader tasks
                    reader_futures = [
                        executor.submit(reader_task, i, operations_per_worker)
                        for i in range(num_workers)
                    ]
                    
                    # Wait for completion
                    writes = sum(f.result() for f in writer_futures)
                    reads = sum(f.result() for f in reader_futures)
                    
                    return {"writes": writes, "reads": reads}
            
            result = self.benchmark_operation(
                f"concurrent_{num_workers}_workers",
                run_concurrent_test,
                num_workers * operations_per_worker * 2
            )
            
            print(f"{num_workers} workers: {result.duration:.2f}s, "
                  f"Writes: {result.metadata.get('writes', 0)}, "
                  f"Reads: {result.metadata.get('reads', 0)}")
        
        return result
    
    def run_scalability_analysis(self):
        """Analyze system scalability"""
        print("\n=== Scalability Analysis ===")
        
        # Test memory usage scaling
        memory_scaling = []
        sizes = [1000, 2500, 5000, 10000, 25000]
        
        for size in sizes:
            gc.collect()
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Create system
            lattice = get_global_lattice()
            lattice.clear()
            memory_system = EnhancedSolitonMemory(lattice_size=size * 2)
            
            # Fill to capacity
            for i in range(size):
                memory_system.store_enhanced_memory(
                    content=f"Scaling test {i}" * 10,  # Larger content
                    concept_ids=[f"scale_{i % 100}"],
                    memory_type=MemoryType.SEMANTIC,
                    sources=["scaling"]
                )
            
            final_memory = self.process.memory_info().rss / 1024 / 1024
            memory_per_item = (final_memory - initial_memory) / size
            
            memory_scaling.append({
                "size": size,
                "total_memory_mb": final_memory - initial_memory,
                "memory_per_item_kb": memory_per_item * 1024
            })
            
            print(f"Size {size}: {final_memory - initial_memory:.2f} MB total, "
                  f"{memory_per_item * 1024:.2f} KB per item")
        
        return memory_scaling
    
    def generate_report(self, output_file: str = "benchmark_report.json"):
        """Generate comprehensive benchmark report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "python_version": sys.version
            },
            "results": [
                {
                    "operation": r.operation,
                    "input_size": r.input_size,
                    "duration": r.duration,
                    "ops_per_second": r.operations_per_second,
                    "memory_delta_mb": r.memory_after_mb - r.memory_before_mb,
                    "cpu_percent": r.cpu_percent,
                    "metadata": r.metadata
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nBenchmark report saved to {output_file}")
        return report
    
    def plot_results(self):
        """Generate performance plots"""
        if not self.results:
            print("No results to plot")
            return
        
        # Group results by operation type
        operation_groups = {}
        for result in self.results:
            base_op = result.operation.split('_')[0]
            if base_op not in operation_groups:
                operation_groups[base_op] = []
            operation_groups[base_op].append(result)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Soliton Memory Performance Benchmarks')
        
        # Plot 1: Operations per second by size
        ax1 = axes[0, 0]
        for op_type, results in operation_groups.items():
            sizes = [r.input_size for r in results]
            ops_per_sec = [r.operations_per_second for r in results]
            ax1.plot(sizes, ops_per_sec, marker='o', label=op_type)
        ax1.set_xlabel('Input Size')
        ax1.set_ylabel('Operations/Second')
        ax1.set_title('Performance Scaling')
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Plot 2: Memory usage
        ax2 = axes[0, 1]
        for op_type, results in operation_groups.items():
            sizes = [r.input_size for r in results]
            memory_delta = [r.memory_after_mb - r.memory_before_mb for r in results]
            ax2.plot(sizes, memory_delta, marker='s', label=op_type)
        ax2.set_xlabel('Input Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Scaling')
        ax2.legend()
        ax2.set_xscale('log')
        
        # Plot 3: Duration
        ax3 = axes[1, 0]
        for op_type, results in operation_groups.items():
            sizes = [r.input_size for r in results]
            durations = [r.duration for r in results]
            ax3.plot(sizes, durations, marker='^', label=op_type)
        ax3.set_xlabel('Input Size')
        ax3.set_ylabel('Duration (seconds)')
        ax3.set_title('Execution Time Scaling')
        ax3.legend()
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # Plot 4: CPU usage
        ax4 = axes[1, 1]
        all_operations = [r.operation for r in self.results]
        cpu_percents = [r.cpu_percent for r in self.results]
        ax4.bar(range(len(all_operations)), cpu_percents)
        ax4.set_xlabel('Operation Index')
        ax4.set_ylabel('CPU Usage (%)')
        ax4.set_title('CPU Utilization')
        ax4.set_xticks(range(0, len(all_operations), max(1, len(all_operations)//10)))
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300)
        print("\nPlots saved to benchmark_results.png")
        
        return fig


def run_full_benchmark():
    """Run complete benchmark suite"""
    print("=== Soliton Memory System Benchmark Suite ===")
    print(f"Started at {datetime.now()}")
    
    benchmark = SolitonBenchmark()
    
    # Run all benchmarks
    benchmark.run_memory_store_benchmark()
    benchmark.run_resonance_search_benchmark()
    benchmark.run_crystallization_benchmark()
    benchmark.run_topology_morphing_benchmark()
    benchmark.run_concurrent_operations_benchmark()
    benchmark.run_scalability_analysis()
    
    # Generate report and plots
    benchmark.generate_report()
    benchmark.plot_results()
    
    print(f"\nCompleted at {datetime.now()}")
    print(f"Total benchmarks run: {len(benchmark.results)}")
    
    return benchmark


if __name__ == "__main__":
    import sys
    
    # Add matplotlib import protection
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    # Run benchmarks
    benchmark = run_full_benchmark()
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    for result in benchmark.results[-5:]:  # Last 5 results
        print(f"{result.operation}: {result.operations_per_second:.2f} ops/sec")
