# ELFIN Stability Framework

The ELFIN Stability Framework provides verification and enforcement of stability properties for neural networks and dynamical systems. This document outlines the components, usage, and extension points of the framework.

## Overview

The Stability Framework extends ELFIN with formal verification capabilities for dynamical systems, focusing on Lyapunov stability. It has several key components:

1. **Neural Lyapunov Functions**: Train and verify neural networks as Lyapunov functions
2. **MILP Verification**: Formally verify stability properties using mixed-integer linear programming 
3. **Compositional Verification**: Build larger proofs from smaller verified components
4. **Runtime Enforcement**: Ensure stability-preserving control at runtime

## Components

### Lyapunov Functions

The framework supports multiple types of Lyapunov functions:

- **Polynomial Lyapunov Functions**: V(x) = b_x^T Q b_x where Q is positive definite
- **Neural Lyapunov Functions**: Neural networks that satisfy Lyapunov conditions
- **Control Lyapunov Functions**: Ensure stability-preserving control via QP
- **Composite Lyapunov Functions**: Combine multiple functions (sum, max, min, weighted)

### Training

Training components for neural Lyapunov functions:

- **NeuralLyapunovTrainer**: Trains networks with Lyapunov conditions as constraints
- **AlphaSchedulers**: Adaptive scheduling of the alpha parameter during training
- **Gradient Clipping**: Prevents exploding gradients during training
- **Mixed-Precision Training**: Optimized training with automatic mixed precision

### Verification

Verification components for formal guarantees:

- **MILPVerifier**: Formal verification using mixed-integer linear programming
- **Proof Caching**: Reuse previous verification results based on proof hash
- **Dependency Tracking**: Invalidate proofs when dependent components change
- **Incremental Verification**: Only verify what changed

### Integration

Integration with the ELFIN ecosystem:

- **Configuration System**: YAML-based configuration with environment variable overrides
- **Event System**: Publish verification events to the event bus
- **Visualization**: Dashboard for monitoring Lyapunov function properties
- **Command-Line Interface**: Verify and train from the command line

## Usage Examples

### Training a Neural Lyapunov Function

```python
from alan_backend.elfin.stability.training import NeuralLyapunovTrainer, LyapunovNet
from alan_backend.elfin.stability.samplers import TrajectorySampler
from alan_backend.elfin.stability.config import load_yaml

# Load configuration
config = load_yaml("stability/configs/neural_lyap_default.yaml")

# Create network
net = LyapunovNet(
    dim=2,
    hidden_dims=(64, 64),
    alpha=config.alpha_scheduler.alpha0,
    activation=torch.nn.Tanh()
)

# Create sampler for Van der Pol oscillator
def vdp_dynamics(x, mu=1.0):
    x1, x2 = x[:, 0], x[:, 1]
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    return np.stack([dx1, dx2], axis=1)

domain = (np.array([-3.0, -3.0]), np.array([3.0, 3.0]))
sampler = TrajectorySampler(vdp_dynamics, 2, domain, batch_size=config.sampler.batch_size)

# Create and run trainer
trainer = NeuralLyapunovTrainer(
    model=net,
    sampler=sampler,
    learning_rate=config.trainer.lr,
    gamma=config.trainer.gamma,
    weight_decay=config.trainer.weight_decay,
    use_amp=config.trainer.use_amp
)

# Train network
history = trainer.fit(steps=5000, log_every=100)
```

### Verifying a Neural Lyapunov Function

```python
from alan_backend.elfin.stability.verify import MILPVerifier

# Create verifier
verifier = MILPVerifier(
    torch_net=net,
    domain=domain,
    time_limit=config.verifier.time_limit,
    verbose=config.verifier.verbose
)

# Verify positive definiteness
result = verifier.find_pd_counterexample()
if result.success:
    print("Network is positive definite!")
else:
    print(f"Found counterexample: {result.counterexample}")

# Verify decrease condition
result = verifier.find_decrease_counterexample(vdp_dynamics, gamma=0.1)
if result.success:
    print("Network is decreasing along trajectories!")
else:
    print(f"Found counterexample: {result.counterexample}")
```

## Configuration

The framework uses a YAML-based configuration system. Here's an example:

```yaml
# Default configuration for neural Lyapunov training and verification
trainer:
  lr: 1e-3                # Learning rate for optimizer
  gamma: 0.1              # Margin for decrease condition
  weight_decay: 1e-5      # L2 regularization coefficient
  max_norm: 1.0           # Maximum norm for gradient clipping
  use_amp: true           # Whether to use Automatic Mixed Precision training

alpha_scheduler:
  type: "exponential"     # Scheduler type: "exponential", "step", or "warm_restart"
  alpha0: 0.01            # Initial alpha value
  min_alpha: 1e-4         # Minimum alpha value
  decay_rate: 0.63        # Decay rate for exponential scheduler

verifier:
  time_limit: 600.0       # Time limit for MILP solver in seconds
  verbose: false          # Whether to show solver output
  big_m_scale: 1.1        # Scaling factor for big-M constants
```

## ELFIN Language Integration

The Stability Framework extends the ELFIN language with stability-specific constructs:

### Stability Types

```elfin
// Define functions with stability guarantees
fn hover(ctrl: StableController) -> Stable { ... }

// Compiler enforces that lower stability values cannot flow into
// higher stability contexts without guards
fn emergency_landing(vehicle: Unstable) -> Marginal { ... }
```

### Barrier Declarations

```elfin
// Inline safety barriers
@Barrier(x^2 + y^2 <= 1)
function thrust() { ... }

// Standalone barriers
barrier SafeZone {
  x^2 + y^2 <= 1
}
```

### Solver Hints

```elfin
// Specify solver preferences at the function level
lyapunov V main {
  degree: 4,
  solver: "sparsepop",
  timeout: 60s
}
```

### Assume/Guarantee Contracts

```elfin
// Compositional verification with contracts
subsystem Motor {
  assume Vdot < 0.0;
  guarantee torque <= 2.0;
}
```

## Extension Points

The framework is designed to be extensible:

1. **Custom Lyapunov Functions**: Implement new Lyapunov function types by extending the base class
2. **Custom Schedulers**: Create new alpha scheduling strategies for training
3. **Verification Backends**: Add new solvers or verification approaches
4. **Visualization**: Create custom visualizations for the dashboard

## Roadmap

Future development areas include:

1. **Incremental/Parallel Verifier**: Scale verification through parallel processing
2. **Koopman Operator Integration**: Bridge with spectral methods for hybrid representations
3. **SMT Backend**: Alternative verification using dReal for smooth activations
4. **Dashboard Improvements**: Enhanced visualization of verification results
