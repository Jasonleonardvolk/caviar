"""
Benchmark Suite Runner

This script provides a command-line interface for running the ELFIN benchmark suite.
"""

import argparse
import os
import time
import logging
import json
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

from .benchmark import BenchmarkSystem, BenchmarkMetric
from .runner import BenchmarkRunner
from .metrics import (
    ValidationSuccessRate,
    ComputationTime,
    Conservativeness,
    DisturbanceRobustness
)
from .systems import (
    Pendulum,
    VanDerPolOscillator,
    CartPole,
    QuadrotorHover,
    SimplifiedManipulator,
    AutonomousVehicle,
    InvertedPendulumRobot,
    ChemicalReactor
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_systems() -> List[BenchmarkSystem]:
    """
    Get all available benchmark systems.
    
    Returns:
        List of benchmark systems
    """
    # Create each benchmark system with default parameters
    systems = [
        Pendulum(),
        VanDerPolOscillator(),
        CartPole()
    ]
    
    # Add remaining systems if they're fully implemented
    try:
        systems.append(QuadrotorHover())
    except (ImportError, NotImplementedError):
        logger.warning("QuadrotorHover system not fully implemented, skipping")
    
    try:
        systems.append(SimplifiedManipulator())
    except (ImportError, NotImplementedError):
        logger.warning("SimplifiedManipulator system not fully implemented, skipping")
    
    try:
        systems.append(AutonomousVehicle())
    except (ImportError, NotImplementedError):
        logger.warning("AutonomousVehicle system not fully implemented, skipping")
    
    try:
        systems.append(InvertedPendulumRobot())
    except (ImportError, NotImplementedError):
        logger.warning("InvertedPendulumRobot system not fully implemented, skipping")
    
    try:
        systems.append(ChemicalReactor())
    except (ImportError, NotImplementedError):
        logger.warning("ChemicalReactor system not fully implemented, skipping")
    
    return systems


def get_all_metrics() -> List[BenchmarkMetric]:
    """
    Get all available benchmark metrics.
    
    Returns:
        List of benchmark metrics
    """
    return [
        ValidationSuccessRate(samples=1000),
        ComputationTime(samples=100, repetitions=5),
        Conservativeness(samples=1000),
        DisturbanceRobustness(num_trajectories=10, T=5.0)
    ]


def run_full_benchmark_suite(
    output_dir: str = "benchmark_results",
    systems: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    generate_report: bool = True
) -> Dict[str, Any]:
    """
    Run the full benchmark suite.
    
    Args:
        output_dir: Directory to save benchmark results
        systems: List of system names to benchmark (None for all)
        metrics: List of metric names to evaluate (None for all)
        generate_report: Whether to generate an HTML report
    
    Returns:
        Dictionary of results
    """
    # Create timestamp-based output directory
    timestamp = int(time.time())
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logger.info(f"Running benchmark suite, saving results to {run_dir}")
    
    # Get all available systems
    all_systems = get_all_systems()
    
    # Filter systems if specified
    if systems is not None:
        filtered_systems = []
        for system in all_systems:
            if system.name in systems:
                filtered_systems.append(system)
        all_systems = filtered_systems
    
    # Get all available metrics
    all_metrics = get_all_metrics()
    
    # Filter metrics if specified
    if metrics is not None:
        filtered_metrics = []
        for metric in all_metrics:
            if metric.name in metrics:
                filtered_metrics.append(metric)
        all_metrics = filtered_metrics
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir=run_dir)
    
    # Run benchmarks for each system
    results = {}
    
    for system in all_systems:
        logger.info(f"Running benchmark for system: {system.name}")
        result = runner.run_benchmark(
            system,
            metrics=all_metrics,
            save_results=True
        )
        results[system.name] = result
    
    # Create comparison report
    if generate_report and len(results) > 1:
        logger.info("Generating comparison report")
        
        comparison_file = os.path.join(run_dir, "comparison.json")
        runner.compare_results(
            list(results.values()),
            output_file=comparison_file,
            plot=True
        )
        
        # Generate HTML report
        html_report = os.path.join(run_dir, "report.html")
        generate_html_report(results, html_report)
    
    # Return results
    return results


def generate_html_report(
    results: Dict[str, Any],
    output_file: str
):
    """
    Generate an HTML report from benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output HTML file path
    """
    # Extract systems and metrics
    systems = list(results.keys())
    metrics = set()
    for result in results.values():
        metrics.update(result.metrics.keys())
    metrics = sorted(list(metrics))
    
    # Create HTML content
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>ELFIN Benchmark Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .system-details {
            margin: 20px 0;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        .metric-description {
            font-style: italic;
            color: #666;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ELFIN Benchmark Results</h1>
        <p>Report generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <h2>Summary</h2>
        <table>
            <tr>
                <th>System</th>
    """
    
    # Add metric columns
    for metric in metrics:
        html += f"<th>{metric}</th>\n"
    
    html += "</tr>\n"
    
    # Add system rows
    for system_name, result in results.items():
        html += f"<tr><td>{system_name}</td>\n"
        
        for metric in metrics:
            if metric in result.metrics:
                value = result.metrics[metric]
                html += f"<td>{value:.4f}</td>\n"
            else:
                html += "<td>N/A</td>\n"
        
        html += "</tr>\n"
    
    html += """
        </table>
        
        <h2>System Details</h2>
    """
    
    # Add details for each system
    for system_name, result in results.items():
        html += f"""
        <div class="system-details">
            <h3>{system_name}</h3>
            <p><strong>Parameters:</strong></p>
            <ul>
        """
        
        for param_name, param_value in result.system_params.items():
            html += f"<li>{param_name}: {param_value}</li>\n"
        
        html += """
            </ul>
            <p><strong>Metric Details:</strong></p>
        """
        
        for metric in metrics:
            if metric in result.details:
                html += f"<h4>{metric}</h4>\n"
                if metric in result.metrics:
                    html += f"<p>Value: {result.metrics[metric]:.4f}</p>\n"
                
                if isinstance(result.details[metric], dict):
                    html += "<ul>\n"
                    for key, value in result.details[metric].items():
                        if isinstance(value, float):
                            html += f"<li>{key}: {value:.4f}</li>\n"
                        else:
                            html += f"<li>{key}: {value}</li>\n"
                    html += "</ul>\n"
        
        html += "</div>\n"
    
    html += """
    </div>
</body>
</html>
    """
    
    # Write HTML to file
    with open(output_file, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report saved to {output_file}")


def main():
    """Main function for the benchmark runner."""
    parser = argparse.ArgumentParser(description="ELFIN Benchmark Suite")
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    
    parser.add_argument(
        "--systems",
        type=str,
        nargs="+",
        help="Systems to benchmark (default: all)"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="Metrics to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not generate HTML report"
    )
    
    args = parser.parse_args()
    
    # Run the benchmark suite
    run_full_benchmark_suite(
        output_dir=args.output_dir,
        systems=args.systems,
        metrics=args.metrics,
        generate_report=not args.no_report
    )


if __name__ == "__main__":
    main()
