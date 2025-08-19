"""
Performance Benchmarking for Memory Vault V2
Measures throughput, latency, and resource usage
"""

import asyncio
import time
import psutil
import numpy as np
from pathlib import Path
import tempfile
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import gc

from memory_vault_v2 import UnifiedMemoryVaultV2, MemoryType


class MemoryVaultBenchmark:
    """Comprehensive benchmark suite for Memory Vault V2"""
    
    def __init__(self, vault_config: Dict[str, Any] = None):
        self.vault_config = vault_config or {}
        self.results = {
            'store_latency': [],
            'retrieve_latency': [],
            'search_latency': [],
            'throughput': {},
            'memory_usage': [],
            'cpu_usage': []
        }
    
    async def benchmark_store_operations(self, vault: UnifiedMemoryVaultV2, count: int = 1000):
        """Benchmark store operations"""
        print(f"\nBenchmarking {count} store operations...")
        
        latencies = []
        embeddings = [np.random.randn(128) for _ in range(10)]  # Reuse embeddings
        
        start_time = time.time()
        
        for i in range(count):
            op_start = time.time()
            
            await vault.store(
                content=f"Benchmark memory {i}: " + "x" * 100,  # ~100 chars
                memory_type=MemoryType.SEMANTIC,
                metadata={
                    'index': i,
                    'tags': ['benchmark', f'batch_{i//100}'],
                    'timestamp': time.time()
                },
                embedding=embeddings[i % 10] if i % 2 == 0 else None,
                importance=np.random.random()
            )
            
            latencies.append(time.time() - op_start)
            
            # Log progress
            if (i + 1) % 100 == 0:
                avg_latency = np.mean(latencies[-100:])
                print(f"  Stored {i+1}/{count} - Avg latency: {avg_latency*1000:.2f}ms")
        
        total_time = time.time() - start_time
        
        self.results['store_latency'] = latencies
        self.results['throughput']['store'] = count / total_time
        
        print(f"Store benchmark complete:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {count/total_time:.2f} ops/sec")
        print(f"  Avg latency: {np.mean(latencies)*1000:.2f}ms")
        print(f"  P95 latency: {np.percentile(latencies, 95)*1000:.2f}ms")
        print(f"  P99 latency: {np.percentile(latencies, 99)*1000:.2f}ms")
    
    async def benchmark_retrieve_operations(self, vault: UnifiedMemoryVaultV2, count: int = 1000):
        """Benchmark retrieve operations"""
        print(f"\nBenchmarking {count} retrieve operations...")
        
        # First, get list of memory IDs
        all_memories = []
        async for memory in vault.stream_all():
            all_memories.append(memory.id)
            if len(all_memories) >= count:
                break
        
        if len(all_memories) < count:
            print(f"  Warning: Only {len(all_memories)} memories available")
            count = len(all_memories)
        
        latencies = []
        start_time = time.time()
        
        for i in range(count):
            memory_id = all_memories[i % len(all_memories)]
            
            op_start = time.time()
            memory = await vault.retrieve(memory_id)
            latencies.append(time.time() - op_start)
            
            if (i + 1) % 100 == 0:
                avg_latency = np.mean(latencies[-100:])
                print(f"  Retrieved {i+1}/{count} - Avg latency: {avg_latency*1000:.2f}ms")
        
        total_time = time.time() - start_time
        
        self.results['retrieve_latency'] = latencies
        self.results['throughput']['retrieve'] = count / total_time
        
        print(f"Retrieve benchmark complete:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {count/total_time:.2f} ops/sec")
        print(f"  Avg latency: {np.mean(latencies)*1000:.2f}ms")
        print(f"  P95 latency: {np.percentile(latencies, 95)*1000:.2f}ms")
        print(f"  P99 latency: {np.percentile(latencies, 99)*1000:.2f}ms")
    
    async def benchmark_search_operations(self, vault: UnifiedMemoryVaultV2, count: int = 100):
        """Benchmark search operations"""
        print(f"\nBenchmarking {count} search operations...")
        
        search_queries = [
            {'memory_type': MemoryType.SEMANTIC},
            {'tags': ['benchmark']},
            {'min_importance': 0.5},
            {'tags': ['benchmark'], 'memory_type': MemoryType.SEMANTIC},
            {'max_results': 10},
            {'tags': ['batch_0'], 'max_results': 50}
        ]
        
        latencies = []
        start_time = time.time()
        
        for i in range(count):
            query = search_queries[i % len(search_queries)]
            
            op_start = time.time()
            results = await vault.search(**query)
            latencies.append(time.time() - op_start)
            
            if (i + 1) % 10 == 0:
                avg_latency = np.mean(latencies[-10:])
                print(f"  Searched {i+1}/{count} - Avg latency: {avg_latency*1000:.2f}ms")
        
        total_time = time.time() - start_time
        
        self.results['search_latency'] = latencies
        self.results['throughput']['search'] = count / total_time
        
        print(f"Search benchmark complete:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {count/total_time:.2f} ops/sec")
        print(f"  Avg latency: {np.mean(latencies)*1000:.2f}ms")
        print(f"  P95 latency: {np.percentile(latencies, 95)*1000:.2f}ms")
        print(f"  P99 latency: {np.percentile(latencies, 99)*1000:.2f}ms")
    
    async def benchmark_concurrent_operations(self, vault: UnifiedMemoryVaultV2):
        """Benchmark concurrent mixed operations"""
        print(f"\nBenchmarking concurrent operations...")
        
        async def store_worker(worker_id: int, count: int):
            for i in range(count):
                await vault.store(
                    f"Concurrent store {worker_id}-{i}",
                    MemoryType.SEMANTIC,
                    metadata={'worker': worker_id}
                )
        
        async def retrieve_worker(count: int):
            results = []
            for _ in range(count):
                memories = await vault.search(max_results=1)
                if memories:
                    memory = await vault.retrieve(memories[0].id)
                    results.append(memory)
            return results
        
        start_time = time.time()
        
        # Run mixed workload
        tasks = []
        for i in range(5):
            tasks.append(store_worker(i, 50))
        for i in range(3):
            tasks.append(retrieve_worker(30))
        
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        total_ops = 5 * 50 + 3 * 30  # stores + retrieves
        
        print(f"Concurrent benchmark complete:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Total operations: {total_ops}")
        print(f"  Throughput: {total_ops/total_time:.2f} ops/sec")
    
    async def benchmark_memory_usage(self, vault: UnifiedMemoryVaultV2):
        """Monitor memory usage during operations"""
        print(f"\nMonitoring resource usage...")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_samples = []
        cpu_samples = []
        
        # Monitor during operations
        async def monitor():
            while True:
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent(interval=0.1)
                
                memory_samples.append(memory_mb)
                cpu_samples.append(cpu_percent)
                
                await asyncio.sleep(1)
        
        # Run monitor in background
        monitor_task = asyncio.create_task(monitor())
        
        # Perform operations
        for i in range(500):
            await vault.store(
                f"Memory test {i}" * 10,  # Larger content
                MemoryType.SEMANTIC,
                embedding=np.random.randn(128)
            )
        
        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        self.results['memory_usage'] = memory_samples
        self.results['cpu_usage'] = cpu_samples
        
        print(f"Resource usage:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory increase: {final_memory - initial_memory:.2f} MB")
        print(f"  Avg CPU usage: {np.mean(cpu_samples):.1f}%")
        print(f"  Peak CPU usage: {max(cpu_samples):.1f}%")
    
    async def benchmark_similarity_search(self, vault: UnifiedMemoryVaultV2):
        """Benchmark embedding similarity search"""
        print(f"\nBenchmarking similarity search...")
        
        # Store memories with embeddings
        embeddings = []
        for i in range(100):
            embedding = np.random.randn(128)
            embeddings.append(embedding)
            
            await vault.store(
                f"Embedding test {i}",
                MemoryType.SEMANTIC,
                embedding=embedding,
                metadata={'embedding_test': True}
            )
        
        # Benchmark similarity searches
        latencies = []
        start_time = time.time()
        
        for i in range(50):
            query_embedding = embeddings[i % len(embeddings)] + np.random.randn(128) * 0.1
            
            op_start = time.time()
            similar = await vault.find_similar(query_embedding, threshold=0.5, max_results=10)
            latencies.append(time.time() - op_start)
        
        total_time = time.time() - start_time
        
        print(f"Similarity search benchmark:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg latency: {np.mean(latencies)*1000:.2f}ms")
        print(f"  P95 latency: {np.percentile(latencies, 95)*1000:.2f}ms")
    
    def plot_results(self, output_dir: Path):
        """Generate performance plots"""
        output_dir.mkdir(exist_ok=True)
        
        # Latency distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Store latency
        if self.results['store_latency']:
            axes[0, 0].hist(np.array(self.results['store_latency']) * 1000, bins=50)
            axes[0, 0].set_title('Store Latency Distribution')
            axes[0, 0].set_xlabel('Latency (ms)')
            axes[0, 0].set_ylabel('Count')
        
        # Retrieve latency
        if self.results['retrieve_latency']:
            axes[0, 1].hist(np.array(self.results['retrieve_latency']) * 1000, bins=50)
            axes[0, 1].set_title('Retrieve Latency Distribution')
            axes[0, 1].set_xlabel('Latency (ms)')
            axes[0, 1].set_ylabel('Count')
        
        # Search latency
        if self.results['search_latency']:
            axes[1, 0].hist(np.array(self.results['search_latency']) * 1000, bins=50)
            axes[1, 0].set_title('Search Latency Distribution')
            axes[1, 0].set_xlabel('Latency (ms)')
            axes[1, 0].set_ylabel('Count')
        
        # Throughput comparison
        if self.results['throughput']:
            operations = list(self.results['throughput'].keys())
            throughputs = list(self.results['throughput'].values())
            axes[1, 1].bar(operations, throughputs)
            axes[1, 1].set_title('Operation Throughput')
            axes[1, 1].set_ylabel('Operations/second')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_distribution.png')
        plt.close()
        
        # Resource usage plot
        if self.results['memory_usage'] and self.results['cpu_usage']:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Memory usage
            ax1.plot(self.results['memory_usage'])
            ax1.set_title('Memory Usage Over Time')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Memory (MB)')
            ax1.grid(True)
            
            # CPU usage
            ax2.plot(self.results['cpu_usage'])
            ax2.set_title('CPU Usage Over Time')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('CPU %')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'resource_usage.png')
            plt.close()
    
    def save_results(self, output_path: Path):
        """Save benchmark results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, list) and value and isinstance(value[0], (int, float)):
                json_results[key] = {
                    'values': value,
                    'mean': np.mean(value),
                    'std': np.std(value),
                    'min': np.min(value),
                    'max': np.max(value),
                    'p50': np.percentile(value, 50),
                    'p95': np.percentile(value, 95),
                    'p99': np.percentile(value, 99)
                }
            else:
                json_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    async def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("=" * 60)
        print("Memory Vault V2 Performance Benchmark")
        print("=" * 60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.vault_config.copy()
            config['storage_path'] = tmpdir
            
            vault = UnifiedMemoryVaultV2(config)
            await vault.initialize()
            
            try:
                # Run benchmarks
                await self.benchmark_store_operations(vault, count=1000)
                await self.benchmark_retrieve_operations(vault, count=1000)
                await self.benchmark_search_operations(vault, count=100)
                await self.benchmark_similarity_search(vault)
                await self.benchmark_concurrent_operations(vault)
                await self.benchmark_memory_usage(vault)
                
                # Get final statistics
                stats = await vault.get_statistics()
                print(f"\nFinal vault statistics:")
                print(f"  Total memories: {stats['total_memories']}")
                print(f"  Storage size: {stats['total_size_mb']:.2f} MB")
                print(f"  Session duration: {stats['uptime_seconds']:.2f}s")
                
                # Save results
                output_dir = Path('benchmark_results')
                output_dir.mkdir(exist_ok=True)
                
                self.save_results(output_dir / 'benchmark_results.json')
                self.plot_results(output_dir)
                
                print(f"\nBenchmark complete! Results saved to {output_dir}")
                
            finally:
                await vault.shutdown()


async def compare_configurations():
    """Compare different vault configurations"""
    print("\nComparing different configurations...")
    
    configurations = [
        {
            'name': 'Default',
            'config': {}
        },
        {
            'name': 'Large Working Memory',
            'config': {'max_working_memory': 1000}
        },
        {
            'name': 'Small Batch Size',
            'config': {'batch_size': 10}
        },
        {
            'name': 'Large Batch Size',
            'config': {'batch_size': 100}
        }
    ]
    
    results = {}
    
    for cfg in configurations:
        print(f"\nBenchmarking configuration: {cfg['name']}")
        benchmark = MemoryVaultBenchmark(cfg['config'])
        
        # Run limited benchmark
        with tempfile.TemporaryDirectory() as tmpdir:
            config = cfg['config'].copy()
            config['storage_path'] = tmpdir
            
            vault = UnifiedMemoryVaultV2(config)
            await vault.initialize()
            
            # Quick benchmark
            await benchmark.benchmark_store_operations(vault, count=100)
            
            results[cfg['name']] = {
                'throughput': benchmark.results['throughput'].get('store', 0),
                'avg_latency': np.mean(benchmark.results['store_latency']) * 1000
            }
            
            await vault.shutdown()
    
    # Print comparison
    print("\nConfiguration Comparison:")
    print("-" * 50)
    print(f"{'Configuration':<20} {'Throughput (ops/s)':<20} {'Avg Latency (ms)':<20}")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['throughput']:<20.2f} {metrics['avg_latency']:<20.2f}")


if __name__ == "__main__":
    # Run benchmarks
    async def main():
        # Full benchmark
        benchmark = MemoryVaultBenchmark({
            'max_working_memory': 100,
            'batch_size': 50,
            'packfile_threshold': 500
        })
        
        await benchmark.run_full_benchmark()
        
        # Configuration comparison
        await compare_configurations()
    
    asyncio.run(main())
