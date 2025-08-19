"""
Benchmark Runner

This module provides utilities for running benchmarks and comparing results.
"""

import json
import time
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from .benchmark import Benchmark, BenchmarkSystem, BenchmarkMetric, BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for executing benchmarks on systems and comparing results.
    """
    
    def __init__(
        self,
        output_dir: str = 'benchmark_results',
        metrics: Optional[List[BenchmarkMetric]] = None
    ):
        """
        Initialize a benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
            metrics: Default metrics to use for benchmarks
        """
        self.output_dir = output_dir
        self.metrics = metrics or []
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def add_metric(self, metric: BenchmarkMetric):
        """
        Add a metric to the default metrics.
        
        Args:
            metric: Metric to add
        """
        self.metrics.append(metric)
    
    def run_benchmark(
        self,
        system: BenchmarkSystem,
        metrics: Optional[List[BenchmarkMetric]] = None,
        name: Optional[str] = None,
        save_results: bool = True,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a benchmark on a system.
        
        Args:
            system: System to benchmark
            metrics: Metrics to evaluate (defaults to runner's metrics)
            name: Name of the benchmark
            save_results: Whether to save results to disk
            **kwargs: Additional parameters for metrics
        
        Returns:
            Benchmark result
        """
        # Use default metrics if none provided
        if metrics is None:
            metrics = self.metrics
        
        # Create and run benchmark
        benchmark = Benchmark(system, metrics, name)
        result = benchmark.run(**kwargs)
        
        # Store result
        benchmark_id = f"{system.name}_{int(time.time())}"
        self.results[benchmark_id] = result
        
        # Save result to disk
        if save_results:
            filepath = os.path.join(self.output_dir, f"{benchmark_id}.json")
            result.to_json(filepath)
            logger.info(f"Saved benchmark result to {filepath}")
        
        return result
    
    def run_benchmark_suite(
        self,
        systems: List[BenchmarkSystem],
        metrics: Optional[List[BenchmarkMetric]] = None,
        save_results: bool = True,
        **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmarks on multiple systems.
        
        Args:
            systems: Systems to benchmark
            metrics: Metrics to evaluate (defaults to runner's metrics)
            save_results: Whether to save results to disk
            **kwargs: Additional parameters for metrics
        
        Returns:
            Dictionary of benchmark results, keyed by system name
        """
        results = {}
        
        for system in systems:
            logger.info(f"Running benchmark on system: {system.name}")
            result = self.run_benchmark(
                system,
                metrics=metrics,
                save_results=save_results,
                **kwargs
            )
            results[system.name] = result
        
        return results
    
    def compare_results(
        self,
        results: Optional[List[BenchmarkResult]] = None,
        metrics: Optional[List[str]] = None,
        plot: bool = True,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare benchmark results.
        
        Args:
            results: Results to compare (defaults to all stored results)
            metrics: Metrics to compare (defaults to all metrics in results)
            plot: Whether to generate comparison plots
            output_file: File to save comparison results
        
        Returns:
            Dictionary of comparison results
        """
        # Use stored results if none provided
        if results is None:
            results = list(self.results.values())
        
        if not results:
            logger.warning("No benchmark results to compare")
            return {}
        
        # Extract metric names
        if metrics is None:
            metrics = set()
            for result in results:
                metrics.update(result.metrics.keys())
            metrics = sorted(list(metrics))
        
        # Create comparison dictionary
        comparison = {
            "systems": [result.system_name for result in results],
            "metrics": {metric: [] for metric in metrics}
        }
        
        # Fill in metric values
        for result in results:
            for metric in metrics:
                if metric in result.metrics:
                    comparison["metrics"][metric].append(result.metrics[metric])
                else:
                    comparison["metrics"][metric].append(float('nan'))
        
        # Generate plots
        if plot:
            self._generate_comparison_plots(comparison, output_file)
        
        # Save comparison results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2)
        
        return comparison
    
    def _generate_comparison_plots(
        self,
        comparison: Dict[str, Any],
        output_file: Optional[str] = None
    ):
        """
        Generate comparison plots.
        
        Args:
            comparison: Comparison dictionary
            output_file: Base filename for plots
        """
        systems = comparison["systems"]
        metrics = comparison["metrics"]
        
        # Create a figure for each metric
        for metric_name, values in metrics.items():
            plt.figure(figsize=(10, 6))
            
            # Convert values to numeric
            numeric_values = []
            numeric_systems = []
            
            for i, value in enumerate(values):
                if np.isfinite(value):
                    numeric_values.append(value)
                    numeric_systems.append(systems[i])
            
            if not numeric_values:
                logger.warning(f"No valid values for metric {metric_name}")
                plt.close()
                continue
            
            # Create bar chart
            plt.bar(range(len(numeric_values)), numeric_values)
            plt.xlabel('System')
            plt.ylabel(metric_name)
            plt.title(f'Comparison of {metric_name} across systems')
            plt.xticks(range(len(numeric_values)), numeric_systems, rotation=45)
            plt.tight_layout()
            
            # Save plot
            if output_file:
                plot_file = output_file.replace('.json', f'_{metric_name}.png')
                plt.savefig(plot_file)
                logger.info(f"Saved comparison plot to {plot_file}")
            
            plt.close()


# Convenience functions

def run_benchmark(
    system: BenchmarkSystem,
    metrics: Optional[List[BenchmarkMetric]] = None,
    output_dir: str = 'benchmark_results',
    **kwargs
) -> BenchmarkResult:
    """
    Run a benchmark on a system.
    
    Args:
        system: System to benchmark
        metrics: Metrics to evaluate
        output_dir: Directory to save benchmark results
        **kwargs: Additional parameters for metrics
    
    Returns:
        Benchmark result
    """
    runner = BenchmarkRunner(output_dir=output_dir)
    return runner.run_benchmark(system, metrics=metrics, **kwargs)


def compare_benchmarks(
    results: List[BenchmarkResult],
    output_file: Optional[str] = None,
    plot: bool = True
) -> Dict[str, Any]:
    """
    Compare benchmark results.
    
    Args:
        results: Results to compare
        output_file: File to save comparison results
        plot: Whether to generate comparison plots
    
    Returns:
        Dictionary of comparison results
    """
    runner = BenchmarkRunner()
    return runner.compare_results(results, output_file=output_file, plot=plot)
