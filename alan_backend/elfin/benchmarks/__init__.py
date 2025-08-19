"""
ELFIN Benchmark Suite

This module provides standardized benchmark systems and metrics for
comparing different barrier function approaches.
"""

from .benchmark import (
    Benchmark, 
    BenchmarkResult,
    BenchmarkSystem,
    BenchmarkMetric
)

from .metrics import (
    ValidationSuccessRate,
    ComputationTime,
    Conservativeness,
    DisturbanceRobustness
)

from .runner import (
    BenchmarkRunner,
    run_benchmark,
    compare_benchmarks
)

# Import all benchmark systems
from .systems import (
    # Simple nonlinear systems
    Pendulum,
    VanDerPolOscillator,
    
    # Standard control systems
    CartPole,
    QuadrotorHover,
    SimplifiedManipulator,
    
    # Industry-relevant systems
    AutonomousVehicle,
    InvertedPendulumRobot,
    ChemicalReactor
)

__all__ = [
    'Benchmark',
    'BenchmarkResult',
    'BenchmarkSystem',
    'BenchmarkMetric',
    'ValidationSuccessRate',
    'ComputationTime',
    'Conservativeness',
    'DisturbanceRobustness',
    'BenchmarkRunner',
    'run_benchmark',
    'compare_benchmarks',
    'Pendulum',
    'VanDerPolOscillator',
    'CartPole',
    'QuadrotorHover',
    'SimplifiedManipulator',
    'AutonomousVehicle',
    'InvertedPendulumRobot',
    'ChemicalReactor'
]
