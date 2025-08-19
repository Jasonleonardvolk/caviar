# ELFIN Benchmark Suite

This module provides a standardized benchmark suite for evaluating barrier functions and control algorithms used in ELFIN specifications.

## Overview

The benchmark suite consists of:

1. **Standard Benchmark Systems** - Well-defined dynamical systems with safety specifications
2. **Evaluation Metrics** - Standardized metrics for comparing barrier functions
3. **Benchmark Runner** - Tools for running benchmarks and visualizing results

## Benchmark Systems

The suite includes the following benchmark systems:

- **Pendulum** - Simple pendulum with angle and angular velocity limits
- **Van der Pol Oscillator** - Nonlinear oscillator with state constraints
- **Cart-Pole** - Inverted pendulum on a cart with track and angle constraints
- **Quadrotor Hover** - Simplified quadrotor model for stable hovering
- **Simplified Manipulator** - Robot arm with joint limits and obstacle avoidance
- **Autonomous Vehicle** - Vehicle model with collision avoidance
- **Inverted Pendulum Robot** - Mobile base with a pendulum attachment
- **Chemical Reactor** - Temperature-controlled chemical reactor with safety constraints

## Evaluation Metrics

The benchmark suite evaluates barrier functions using these key metrics:

- **Validation Success Rate** - Percentage of states that validate barrier conditions
- **Computation Time** - Performance measurement for barrier function evaluation
- **Conservativeness** - How conservative the barrier function is (ratio of verified safe set to actual safe set)
- **Disturbance Robustness** - Maximum disturbance level that maintains safety

## Usage Example

```python
import numpy as np
from alan_backend.elfin.benchmarks import run_benchmark, compare_benchmarks
from alan_backend.elfin.benchmarks.systems import Pendulum, CartPole
from alan_backend.elfin.benchmarks.metrics import ValidationSuccessRate, ComputationTime

# Create benchmark systems with custom parameters
pendulum = Pendulum({
    "m": 1.0, 
    "l": 0.8, 
    "unsafe_angle": np.pi/4
})

cart_pole = CartPole({
    "m_c": 1.0,
    "m_p": 0.1,
    "track_limit": 2.0,
    "unsafe_angle": 0.3
})

# Define metrics
metrics = [
    ValidationSuccessRate(samples=2000),
    ComputationTime(samples=1000, repetitions=5)
]

# Run benchmarks
pendulum_result = run_benchmark(
    pendulum, 
    metrics=metrics,
    output_dir="benchmark_results"
)

cart_pole_result = run_benchmark(
    cart_pole, 
    metrics=metrics,
    output_dir="benchmark_results"
)

# Compare results
comparison = compare_benchmarks(
    [pendulum_result, cart_pole_result],
    output_file="benchmark_results/comparison.json",
    plot=True
)
```

## Running the Benchmark Suite

To run the complete benchmark suite:

```bash
python -m alan_backend.elfin.benchmarks.run
```

This will execute all benchmark systems with the standard metrics and generate a comprehensive report with visualizations.

## Adding New Benchmark Systems

To add a new benchmark system:

1. Create a new file in the `systems` directory
2. Implement a class that extends `BenchmarkSystem`
3. Implement the required methods: `dynamics`, `barrier_function`, `is_safe`
4. Add the class to the imports in `systems/__init__.py`

## Adding New Metrics

To add a new evaluation metric:

1. Create a class that extends `BenchmarkMetric`
2. Implement the `evaluate` method that returns a tuple of (metric_value, details)
3. Add the class to the imports in `__init__.py`
