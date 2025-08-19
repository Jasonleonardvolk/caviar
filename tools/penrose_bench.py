#!/usr/bin/env python3
"""
Penrose Projector Benchmark Tool
One-shot performance check and scaling analysis
"""

import os
import sys
import numpy as np
import time
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime

# Set matplotlib backend for headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from penrose_projector.core import PenroseProjector


def benchmark_single(n_concepts: int, embedding_dim: int, rank: int = 32, threshold: float = 0.7) -> dict:
    """Run single benchmark"""
    # Generate random embeddings
    embeddings = np.random.randn(n_concepts, embedding_dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Time baseline cosine similarity
    baseline_start = time.time()
    baseline_sim = embeddings @ embeddings.T
    baseline_time = time.time() - baseline_start
    baseline_nnz = np.sum(baseline_sim >= threshold)
    
    # Time Penrose projection
    projector = PenroseProjector(rank=rank, threshold=threshold)
    penrose_start = time.time()
    sparse_sim = projector.project_sparse(embeddings)
    penrose_time = time.time() - penrose_start
    
    stats = projector.get_stats()
    
    result = {
        'n_concepts': n_concepts,
        'embedding_dim': embedding_dim,
        'rank': rank,
        'threshold': threshold,
        'baseline_time': baseline_time,
        'penrose_time': penrose_time,
        'speedup': baseline_time / penrose_time if penrose_time > 0 else 0,
        'baseline_nnz': int(baseline_nnz),
        'penrose_nnz': stats['nnz'],
        'penrose_density_pct': stats['density_pct'],
        'effective_threshold': stats.get('effective_threshold', threshold),
        'projection_time': stats['times']['projection'],
        'similarity_time': stats['times']['similarity'],
        'sparsification_time': stats['times']['sparsification'],
        'theoretical_speedup': stats['speedup_vs_full']
    }
    
    return result


def benchmark_scaling(max_concepts: int = 10000, step: int = 1000, 
                     embedding_dim: int = 128, rank: int = 32,
                     output_dir: str = "benchmarks") -> list:
    """Run scaling benchmark"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = []
    sizes = list(range(step, max_concepts + 1, step))
    
    print(f"🧪 Running scaling benchmark: {len(sizes)} sizes up to {max_concepts}")
    print(f"   Embedding dim: {embedding_dim}, Rank: {rank}")
    print("=" * 60)
    
    for n in sizes:
        print(f"\nBenchmarking n={n}...")
        result = benchmark_single(n, embedding_dim, rank)
        results.append(result)
        
        print(f"  Baseline: {result['baseline_time']:.3f}s")
        print(f"  Penrose:  {result['penrose_time']:.3f}s")
        print(f"  Speedup:  {result['speedup']:.1f}x")
        print(f"  Density:  {result['penrose_density_pct']:.2f}%")
    
    # Save results as CSV
    csv_path = output_dir / f"penrose_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n📊 Saved CSV: {csv_path}")
    
    # Save results as JSON
    json_path = output_dir / f"penrose_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'max_concepts': max_concepts,
                'step': step,
                'embedding_dim': embedding_dim,
                'rank': rank
            },
            'results': results
        }, f, indent=2)
    
    # Generate scaling plot
    plot_scaling(results, output_dir)
    
    return results


def plot_scaling(results: list, output_dir: Path):
    """Generate scaling plot"""
    if not results:
        return
    
    n_concepts = [r['n_concepts'] for r in results]
    baseline_times = [r['baseline_time'] for r in results]
    penrose_times = [r['penrose_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Time comparison
    ax1.plot(n_concepts, baseline_times, 'r-', label='Baseline O(n²)', linewidth=2)
    ax1.plot(n_concepts, penrose_times, 'b-', label='Penrose O(n^2.32)', linewidth=2)
    ax1.set_xlabel('Number of Concepts')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Similarity Computation Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Speedup
    ax2.plot(n_concepts, speedups, 'g-', linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Concepts')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Penrose Speedup vs Baseline')
    ax2.grid(True, alpha=0.3)
    
    # Add theoretical speedup line
    theoretical = [r['theoretical_speedup'] for r in results]
    ax2.plot(n_concepts, theoretical, 'orange', linestyle='--', 
             label='Theoretical', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"penrose_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Saved plot: {plot_path}")
    
    plt.close()


def emit_prometheus_metrics(result: dict):
    """Emit metrics in Prometheus format for scraping"""
    print("\n# HELP penrose_benchmark_time_seconds Time taken for similarity computation")
    print("# TYPE penrose_benchmark_time_seconds gauge")
    print(f'penrose_benchmark_time_seconds{{method="baseline",n="{result["n_concepts"]}"}} {result["baseline_time"]}')
    print(f'penrose_benchmark_time_seconds{{method="penrose",n="{result["n_concepts"]}"}} {result["penrose_time"]}')
    
    print("\n# HELP penrose_speedup_factor Speedup factor vs baseline")
    print("# TYPE penrose_speedup_factor gauge")
    print(f'penrose_speedup_factor{{n="{result["n_concepts"]}"}} {result["speedup"]}')
    
    print("\n# HELP penrose_density_percent Edge density percentage")
    print("# TYPE penrose_density_percent gauge")
    print(f'penrose_density_percent{{n="{result["n_concepts"]}"}} {result["penrose_density_pct"]}')


def main():
    parser = argparse.ArgumentParser(description="Penrose Projector Benchmark")
    parser.add_argument('--n-concepts', type=int, default=1000,
                       help='Number of concepts for single benchmark')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--rank', type=int, default=32,
                       help='Projection rank')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Similarity threshold')
    parser.add_argument('--scaling', action='store_true',
                       help='Run scaling benchmark')
    parser.add_argument('--max-concepts', type=int, default=10000,
                       help='Maximum concepts for scaling benchmark')
    parser.add_argument('--step', type=int, default=1000,
                       help='Step size for scaling benchmark')
    parser.add_argument('--output-dir', type=str, default='benchmarks',
                       help='Output directory for results')
    parser.add_argument('--prometheus', action='store_true',
                       help='Output Prometheus metrics')
    
    args = parser.parse_args()
    
    if args.scaling:
        results = benchmark_scaling(
            max_concepts=args.max_concepts,
            step=args.step,
            embedding_dim=args.embedding_dim,
            rank=args.rank,
            output_dir=args.output_dir
        )
        
        print("\n📊 Summary Statistics:")
        print(f"  Max speedup: {max(r['speedup'] for r in results):.1f}x")
        print(f"  Avg speedup: {sum(r['speedup'] for r in results) / len(results):.1f}x")
        print(f"  Max density: {max(r['penrose_density_pct'] for r in results):.2f}%")
        
    else:
        # Single benchmark
        result = benchmark_single(
            n_concepts=args.n_concepts,
            embedding_dim=args.embedding_dim,
            rank=args.rank,
            threshold=args.threshold
        )
        
        print("\n🏃 Benchmark Results:")
        print(f"  Concepts: {result['n_concepts']}")
        print(f"  Embedding dim: {result['embedding_dim']}")
        print(f"  Rank: {result['rank']}")
        print(f"  Threshold: {result['threshold']}")
        print(f"\n  Baseline time: {result['baseline_time']:.3f}s")
        print(f"  Penrose time:  {result['penrose_time']:.3f}s")
        print(f"  Speedup:       {result['speedup']:.1f}x")
        print(f"  Density:       {result['penrose_density_pct']:.2f}%")
        print(f"  Edges:         {result['penrose_nnz']:,}")
        
        if args.prometheus:
            emit_prometheus_metrics(result)


if __name__ == "__main__":
    main()
