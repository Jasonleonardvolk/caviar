# benchmarks/benchmark_soliton_performance.py

import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from python.core.soliton_memory_integration import EnhancedSolitonMemory, MemoryType
from python.core.hot_swap_laplacian import HotSwappableLaplacian
from python.core.nightly_growth_engine import NightlyGrowthEngine

class SolitonBenchmark:
    """Benchmark suite for soliton memory system"""
    
    def __init__(self):
        self.results = {}
        
    async def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print("=== SOLITON MEMORY BENCHMARKS ===\n")
        
        # 1. Memory storage/recall speed
        await self.benchmark_storage_recall()
        
        # 2. Topology switching overhead
        await self.benchmark_topology_switching()
        
        # 3. Crystallization performance
        await self.benchmark_crystallization()
        
        # 4. Dark soliton suppression
        await self.benchmark_dark_suppression()
        
        # 5. Scalability test
        await self.benchmark_scalability()
        
        # Generate report
        self.generate_report()
        
    async def benchmark_storage_recall(self):
        """Benchmark memory storage and recall operations"""
        print("1. Storage/Recall Benchmark")
        
        memory = EnhancedSolitonMemory(lattice_size=10000)
        n_memories = 1000
        
        # Storage benchmark
        start = time.perf_counter()
        
        memory_ids = []
        for i in range(n_memories):
            mem_id = memory.store_enhanced_memory(
                f"Test memory {i}: {np.random.choice(['fact', 'event', 'idea'])}",
                [f"concept_{i % 100}"],  # 100 unique concepts
                MemoryType.SEMANTIC,
                [f"source_{i % 10}"]
            )
            memory_ids.append(mem_id)
            
        storage_time = time.perf_counter() - start
        storage_rate = n_memories / storage_time
        
        print(f"  Storage: {storage_rate:.1f} memories/sec")
        
        # Recall benchmark
        n_queries = 100
        start = time.perf_counter()
        
        for _ in range(n_queries):
            concept = f"concept_{np.random.randint(100)}"
            phase = memory._calculate_concept_phase([concept])
            results = memory.find_resonant_memories_enhanced(phase, [concept])
            
        recall_time = time.perf_counter() - start
        recall_rate = n_queries / recall_time
        
        print(f"  Recall: {recall_rate:.1f} queries/sec")
        
        self.results['storage_recall'] = {
            'storage_rate': storage_rate,
            'recall_rate': recall_rate,
            'n_memories': n_memories
        }
        
    async def benchmark_topology_switching(self):
        """Benchmark topology switching performance"""
        print("\n2. Topology Switching Benchmark")
        
        hot_swap = HotSwappableLaplacian(lattice_size=(50, 50))
        
        # Add some solitons
        for i in range(100):
            hot_swap.active_solitons.append({
                'amplitude': np.random.uniform(0.5, 1.5),
                'phase': np.random.uniform(0, 2*np.pi),
                'index': i
            })
            
        topologies = ['kagome', 'hexagonal', 'triangular', 'small_world']
        switch_times = []
        
        for i in range(len(topologies) - 1):
            from_topo = topologies[i]
            to_topo = topologies[i + 1]
            
            start = time.perf_counter()
            await hot_swap.hot_swap_laplacian_with_safety(to_topo)
            switch_time = time.perf_counter() - start
            
            switch_times.append(switch_time)
            print(f"  {from_topo} → {to_topo}: {switch_time*1000:.1f}ms")
            
        avg_switch_time = np.mean(switch_times)
        print(f"  Average: {avg_switch_time*1000:.1f}ms")
        
        self.results['topology_switching'] = {
            'switch_times': switch_times,
            'average': avg_switch_time
        }
        
    async def benchmark_crystallization(self):
        """Benchmark memory crystallization performance"""
        print("\n3. Crystallization Benchmark")
        
        memory = EnhancedSolitonMemory()
        hot_swap = HotSwappableLaplacian()
        
        # Create memories with varied heat
        for i in range(500):
            mem_id = memory.store_enhanced_memory(
                f"Memory {i}",
                [f"concept_{i % 50}"],
                MemoryType.SEMANTIC,
                ["source"]
            )
            # Assign random heat
            memory.memory_entries[mem_id].heat = np.random.uniform(0, 1)
            
        from python.core.memory_crystallization import MemoryCrystallizer
        crystallizer = MemoryCrystallizer(memory, hot_swap)
        
        start = time.perf_counter()
        report = await crystallizer.crystallize()
        crystallization_time = time.perf_counter() - start
        
        print(f"  Time: {crystallization_time:.2f}s")
        print(f"  Migrated: {report['migrated']}")
        print(f"  Decayed: {report['decayed']}")
        print(f"  Fused: {report['fused']}")
        
        self.results['crystallization'] = {
            'time': crystallization_time,
            'report': report
        }
        
    async def benchmark_dark_suppression(self):
        """Benchmark dark soliton suppression efficiency"""
        print("\n4. Dark Soliton Suppression Benchmark")
        
        memory = EnhancedSolitonMemory()
        
        # Create bright memories
        n_bright = 100
        for i in range(n_bright):
            memory.store_enhanced_memory(
                f"Bright memory {i}",
                [f"bright_{i}"],
                MemoryType.SEMANTIC,
                ["source"]
            )
            
        # Recall benchmark (all visible)
        start = time.perf_counter()
        visible_count = 0
        
        for i in range(n_bright):
            results = memory.find_resonant_memories_enhanced(
                memory._calculate_concept_phase([f"bright_{i}"]),
                [f"bright_{i}"]
            )
            visible_count += len(results)
            
        bright_recall_time = time.perf_counter() - start
        
        # Add dark memories to suppress half
        for i in range(0, n_bright, 2):
            memory.store_enhanced_memory(
                f"Suppress bright_{i}",
                [f"bright_{i}"],
                MemoryType.TRAUMATIC,
                ["suppression"]
            )
            
        # Recall again (half suppressed)
        start = time.perf_counter()
        suppressed_count = 0
        
        for i in range(n_bright):
            results = memory.find_resonant_memories_enhanced(
                memory._calculate_concept_phase([f"bright_{i}"]),
                [f"bright_{i}"]
            )
            if len(results) == 0:
                suppressed_count += 1
                
        dark_recall_time = time.perf_counter() - start
        
        print(f"  Suppression rate: {suppressed_count/n_bright*100:.1f}%")
        print(f"  Overhead: {(dark_recall_time - bright_recall_time)/bright_recall_time*100:.1f}%")
        
        self.results['dark_suppression'] = {
            'suppression_rate': suppressed_count / n_bright,
            'overhead': (dark_recall_time - bright_recall_time) / bright_recall_time
        }
        
    async def benchmark_scalability(self):
        """Test scalability with increasing memory count"""
        print("\n5. Scalability Benchmark")
        
        memory_counts = [100, 500, 1000, 2000, 5000]
        recall_times = []
        
        for count in memory_counts:
            memory = EnhancedSolitonMemory(lattice_size=count * 2)
            
            # Store memories
            for i in range(count):
                memory.store_enhanced_memory(
                    f"Memory {i}",
                    [f"concept_{i % 100}"],
                    MemoryType.SEMANTIC,
                    ["source"]
                )
                
            # Measure recall time
            n_queries = 50
            start = time.perf_counter()
            
            for _ in range(n_queries):
                concept = f"concept_{np.random.randint(100)}"
                phase = memory._calculate_concept_phase([concept])
                memory.find_resonant_memories_enhanced(phase, [concept])
                
            avg_recall_time = (time.perf_counter() - start) / n_queries
            recall_times.append(avg_recall_time)
            
            print(f"  {count} memories: {avg_recall_time*1000:.2f}ms/query")
            
        self.results['scalability'] = {
            'memory_counts': memory_counts,
            'recall_times': recall_times
        }
        
    def generate_report(self):
        """Generate and save benchmark report"""
        print("\n=== BENCHMARK SUMMARY ===")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Storage/Recall rates
        ax = axes[0, 0]
        rates = self.results['storage_recall']
        ax.bar(['Storage', 'Recall'], 
               [rates['storage_rate'], rates['recall_rate']])
        ax.set_ylabel('Operations/second')
        ax.set_title('Storage and Recall Performance')
        
        # 2. Topology switching times
        ax = axes[0, 1]
        times = self.results['topology_switching']['switch_times']
        ax.bar(range(len(times)), [t*1000 for t in times])
        ax.set_xlabel('Switch #')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Topology Switching Times')
        
        # 3. Crystallization breakdown
        ax = axes[1, 0]
        report = self.results['crystallization']['report']
        operations = ['Migrated', 'Decayed', 'Fused', 'Split']
        counts = [report.get(op.lower(), 0) for op in operations]
        ax.pie(counts, labels=operations, autopct='%1.0f%%')
        ax.set_title('Crystallization Operations')
        
        # 4. Scalability curve
        ax = axes[1, 1]
        scale = self.results['scalability']
        ax.plot(scale['memory_counts'], 
                [t*1000 for t in scale['recall_times']], 'o-')
        ax.set_xlabel('Number of Memories')
        ax.set_ylabel('Recall Time (ms)')
        ax.set_title('Scalability')
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('soliton_benchmark_results.png', dpi=150)
        print("\nResults saved to soliton_benchmark_results.png")
        
        # Print key metrics
        print("\nKey Performance Metrics:")
        print(f"- Storage rate: {rates['storage_rate']:.1f} memories/sec")
        print(f"- Recall rate: {rates['recall_rate']:.1f} queries/sec")
        print(f"- Avg topology switch: {self.results['topology_switching']['average']*1000:.1f}ms")
        print(f"- Dark suppression overhead: {self.results['dark_suppression']['overhead']*100:.1f}%")

async def main():
    """Run benchmarks"""
    benchmark = SolitonBenchmark()
    await benchmark.run_all_benchmarks()

if __name__ == "__main__":
    asyncio.run(main())
