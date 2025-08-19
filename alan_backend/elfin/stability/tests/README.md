# ELFIN Stability Framework Tests

This directory contains unit tests for the ELFIN Stability Framework, focusing on the Neural Lyapunov Training and Certification components.

## Test Components

The tests cover the following core components:

1. **TrajectorySampler Tests** (`test_trajectory_sampler.py`)
   - Initialization and validation
   - Random and balanced batch generation
   - Counterexample handling
   - Trajectory simulation
   - Reproducibility with random seeds

2. **Neural Lyapunov Trainer Tests** (`test_neural_lyapunov_trainer.py`)
   - LyapunovNet architecture validation
   - Positive definiteness guarantees
   - Gradient computation and properties
   - Model save/load functionality
   - Training process validation
   - Evaluation metrics

## Running Tests

You can run all tests using pytest:

```bash
# From the ELFIN root directory
pytest -xvs alan_backend/elfin/stability/tests/

# Run a specific test file
pytest -xvs alan_backend/elfin/stability/tests/test_trajectory_sampler.py
```

Or use the provided batch file:

```bash
# From the ELFIN root directory
run_stability_tests.bat
```

## Test Design Philosophy

The tests are designed to validate:

1. **Correctness**: Ensures each component performs its intended function correctly
2. **Robustness**: Validates error handling and edge cases
3. **Integration**: Verifies components work together properly
4. **Mathematical Properties**: Confirms theoretical properties like positive definiteness of Lyapunov functions

## Future Tests

Future test additions should include:

1. MILP verifier tests (currently challenging due to Gurobi dependency)
2. Integration tests for the complete certification workflow
3. Performance benchmarking tests
4. Specialized tests for each Lyapunov function type
