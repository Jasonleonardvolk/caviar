# ELFIN Learning Module

The ELFIN Learning Module provides tools to create, train, import, export, and verify neural barrier and Lyapunov functions for robotic control systems.

## Overview

This module bridges the gap between learning-based and formal methods for robot safety and stability. It enables:

1. Training neural networks that satisfy barrier and Lyapunov conditions
2. Converting these networks to ELFIN format for verification and deployment
3. Importing ELFIN models back to neural networks for refinement
4. Verifying the correctness of ELFIN models

## Module Structure

```
alan_backend/elfin/learning/
├── models/                    # Neural network model definitions
│   ├── torch_models.py        # PyTorch implementations
│   ├── jax_models.py          # JAX implementations
│   ├── neural_barrier.py      # Base barrier network implementation
│   └── neural_lyapunov.py     # Base Lyapunov network implementation
├── training/                  # Training utilities
│   ├── data_generator.py      # Data generation for training
│   ├── losses.py              # Custom loss functions
│   ├── barrier_trainer.py     # Barrier function trainer
│   └── lyapunov_trainer.py    # Lyapunov function trainer
├── integration/               # Integration with ELFIN
│   ├── benchmark_integration.py  # Benchmark utilities
│   ├── export.py              # Export to ELFIN format
│   ├── import_models.py       # Import from ELFIN format
│   └── verification.py        # Verification utilities
└── examples/                  # Example applications
    └── mobile_robot_example.py  # Mobile robot control example
```

## Features

### Neural Network Models

- **Barrier Networks**: Ensure safety constraints by learning implicit safe sets
- **Lyapunov Networks**: Ensure stability properties by learning energy-like functions
- **Multi-Framework Support**: Implementations in both PyTorch and JAX

### Training

- **Custom Loss Functions**: Specifically designed for barrier and Lyapunov conditions
- **Data Generation**: Utilities to generate training data for various robotic systems
- **Visualization**: Tools to visualize training progress and learned functions

### Integration with ELFIN

- **Export**: Convert trained neural networks to ELFIN format with several approximation methods:
  - Explicit layer-by-layer computation
  - Taylor series approximation
  - Polynomial approximation
  - Piecewise approximation
- **Import**: Convert ELFIN models back to neural networks for refinement
- **Benchmarking**: Test the performance of neural barrier and Lyapunov functions
- **Verification**: Check for syntax errors, circular references, and formal correctness

## Usage Examples

### Training a Barrier Function

```python
from alan_backend.elfin.learning.models.torch_models import TorchBarrierNetwork
from alan_backend.elfin.learning.training.barrier_trainer import BarrierTrainer

# Create barrier network model
model = TorchBarrierNetwork(
    state_dim=4,
    hidden_layers=[64, 64, 32],
    activation="tanh"
)

# Create barrier trainer
trainer = BarrierTrainer(
    model=model,
    dynamics_fn=system_dynamics,
    classification_weight=1.0,
    gradient_weight=0.5,
    smoothness_weight=0.1
)

# Train model
trainer.train(
    states=states,
    labels=labels,
    batch_size=64,
    epochs=500,
    validation_split=0.2,
    early_stopping=True
)

# Save model
trainer.save("barrier_model.pt")
```

### Exporting to ELFIN Format

```python
from alan_backend.elfin.learning.integration.export import export_to_elfin

# Export barrier function
elfin_code = export_to_elfin(
    model=barrier_model,
    model_type="barrier",
    system_name="MobileRobot",
    state_dim=4,
    input_dim=2,
    state_names=["x", "y", "theta", "v"],
    input_names=["a", "omega"],
    filepath="neural_barrier.elfin",
    approximation_method="explicit"
)
```

### Verifying ELFIN Models

```python
from alan_backend.elfin.learning.integration.verification import verify_elfin_file

# Verify ELFIN file
results = verify_elfin_file("neural_barrier.elfin")

# Check for issues
if results['syntax_errors'] or results['circular_references']:
    print("Issues found in ELFIN file:")
    for error in results['syntax_errors']:
        print(f"Syntax error: {error}")
    for ref in results['circular_references']:
        print(f"Circular reference: {ref}")
else:
    print("ELFIN file is valid!")
```

### Complete Example

See `examples/mobile_robot_example.py` for a complete end-to-end example that demonstrates:

1. Training barrier and Lyapunov functions for a mobile robot
2. Exporting them to ELFIN format
3. Importing them back to neural networks
4. Benchmarking their performance

## Installation Requirements

- PyTorch (>= 1.7.0)
- JAX (optional, for JAX implementations)
- NumPy
- Matplotlib
- NetworkX (for verification)
- scikit-learn (for polynomial approximation)

## Contributing

When adding new features, please follow these guidelines:

1. Maintain compatibility with both PyTorch and JAX when possible
2. Add appropriate docstrings and type hints
3. Implement visualization tools for new functions
4. Include error handling and validation
5. Add tests for new functionality

## License

[MIT License](LICENSE)
