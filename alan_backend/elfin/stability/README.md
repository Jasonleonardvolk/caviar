# ELFIN Stability Framework - Conversational Verification

This package provides stability verification components for the ELFIN framework with a focus on interaction logging and rich error messages.

## Key Features

- **Conversational StabilityAgent**: Every proof attempt, counterexample, parameter adjustment, and success is logged
- **Rich Error Messages**: Detailed, contextual error messages with references to interactions
- **JSONL Interaction Log**: Append-only, grep-friendly log format
- **CLI Integration**: Command-line tools for verification, log viewing, and parameter tuning

## Components

### 1. Core Interaction Model

The `Interaction` class represents a single verification interaction:

```python
from alan_backend.elfin.stability.core import Interaction

# Create a new interaction
interaction = Interaction.now(
    "verify",
    system_id="quadrotor_demo",
    domain=[[-1, -1], [1, 1]]
)

# Set the result (upon completion)
interaction.result = {
    "status": "VERIFIED",
    "solve_time": 3.4,
    "proof_hash": "D1A8..."
}

# Get a reference to this interaction
reference = interaction.get_reference()  # "2025-05-13T07:51:02.101Z#verify"
```

The `InteractionLog` class manages collections of interactions:

```python
from alan_backend.elfin.stability.core import InteractionLog

# Create or load an interaction log
log = InteractionLog.load("quadrotor_demo.log.jsonl")

# Filter interactions
verification_logs = log.filter(action="verify")
failed_logs = log.filter(result={"status": "FAILED"})

# Get the last 5 interactions
recent_logs = log.tail(5)
```

### 2. StabilityAgent

The `StabilityAgent` wraps verification components with interaction logging:

```python
from alan_backend.elfin.stability.agents import StabilityAgent

# Create a StabilityAgent
agent = StabilityAgent("quadrotor_demo", cache_db="./cache")

# Verify a system
result = agent.verify(
    system=lyapunov_net,
    domain=(np.array([-1, -1]), np.array([1, 1])),
    time_limit=300.0
)

# Verify decrease condition
result = agent.verify_decrease(
    system=lyapunov_net,
    dynamics_fn=system.dynamics,
    domain=(np.array([-1, -1]), np.array([1, 1])),
    gamma=0.1
)

# Tune parameters
result = agent.param_tune(
    system=oscillator,
    param_name="damping",
    old_value=0.05,
    new_value=0.1
)

# View interaction log
log = agent.get_log(tail=10)
summary = agent.get_summary()
print(summary)
```

### 3. Command Line Interface

The CLI provides command-line access to the stability framework:

```bash
# Verify a system
elf verify quadrotor_model.pt --domain="[-1,-1],[1,1]"

# View interaction logs
elf log quadrotor_demo --tail 5

# Tune parameters
elf tune quadrotor_model.pt damping 0.1
```

### 4. Error Handling

The framework provides standardized error representation:

```python
from alan_backend.elfin.stability.agents import VerificationError

# Create a verification error
error = VerificationError(
    code="LYAP_001",
    detail="Function not positive definite at x=[0.42, -1.38]",
    system_id="quadrotor_demo",
    interaction_ref="2025-05-13T07:51:02.101Z#verify"
)

# Convert to JSON
error_json = error.to_json()
```

Error messages will be displayed with rich context:

```
E-LYAP-001: Lyapunov function not positive definite at x=[0.42, -1.38] (see https://elfin.dev/errors/E-LYAP-001)
```

## Getting Started

### Installation

```bash
# From project root
pip install -e .
```

### Example Usage

A comprehensive demo is provided in `examples/stability_agent_demo.py`:

```bash
python -m alan_backend.elfin.examples.stability_agent_demo
```

### CLI Usage

```bash
# Help
elf --help

# Verify a system
elf verify path/to/model.pt

# View logs
elf log my_agent_name

# Tune parameters
elf tune path/to/model.pt alpha 0.05
```

## Events System

The StabilityAgent emits events that can be consumed by other components:

- `proof_added`: Emitted when a verification proof is created
- `counterexample`: Emitted when a counterexample is found
- `verification_error`: Emitted when verification fails
- `decrease_verified`: Emitted when decrease condition is verified
- `decrease_violation`: Emitted when decrease condition is violated
- `param_tuned`: Emitted when a parameter is tuned

## Integration with ELFIN

The stability framework integrates with the broader ELFIN ecosystem:

- **Event Bus**: Verification events can be consumed by other components
- **Proof Caching**: Verification results are cached for reuse
- **Interaction Logging**: All interactions are logged for audit and analysis
- **CLI Integration**: Command-line access to stability verification

## Extending the Framework

The framework is designed to be extensible:

- **Custom Verifiers**: Create new verifiers by extending the base classes
- **Custom Agents**: Create specialized agents for different verification tasks
- **Custom Error Handlers**: Define domain-specific error handling
- **Custom Integrations**: Integrate with dashboards, IDEs, or other tools
