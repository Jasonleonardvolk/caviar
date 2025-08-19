# ALAN Core: Banksy-Spin Implementation

## Overview

This repository contains the implementation of ALAN's neuromorphic reasoning core featuring the Banksy oscillator system enhanced with altermagnetic spin dynamics and a time-reversal-symmetric (TRS) controller. This implementation provides a foundation for stable, interpretable reasoning with provable guarantees.

Key features:
- **Phase-spin oscillator substrate**: Kuramoto-style oscillators with coupled spin dynamics
- **Time-reversal symmetric ODE controller**: Enables reversible computations with Hamiltonian guarantees
- **Hopfield-on-spin memory**: Associative memory implemented using spin states
- **Banksy fusion system**: Integration of all components for coherent reasoning

## Quick Start

### Prerequisites
- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`)

### Running Demos

Use the provided Makefile to run demonstrations:

```bash
# Run the full demo suite
make sim

# Run individual component demos
make sim_oscillator  # Run just the oscillator demo
make sim_trs         # Run just the TRS controller demo
make sim_memory      # Run just the Hopfield memory demo
make sim_full        # Run just the full system demo
```

### Other Commands

```bash
# Run tests (when available)
make test

# Code quality checks
make qa

# Format code
make format

# Clean up build artifacts
make clean
```

## Core Components

### 1. Banksy-Spin Oscillator (`oscillator/banksy_oscillator.py`)

The foundation of ALAN's neuromorphic architecture. Implements a network of coupled oscillators with:

- Second-order Kuramoto dynamics (with momentum)
- Altermagnetic spin coupling
- Hamiltonian-preserving update scheme
- Effective synchronization metrics (N_eff)

The core algorithm follows:

```python
# Δt implicit (set by integrator)
for i in 0..N {
    # momentum form – reversible
    p[i] += Σ_j K[i][j] * sin(θ[j] - θ[i])       # coupling torque
             + γ * (σ[i] - σ̄);                   # spin-lattice term
    p[i] *= (1.0 - η_damp);                      # light damping
    θ[i]  = wrap(θ[i] + p[i]);
    # spin alignment (Hebbian in sync window Δτ)
    σ_dot = ε * Σ_j cos(θ[i] - θ[j]) * σ[j];
    σ[i] += σ_dot;
}
```

### 2. TRS-ODE Controller (`controller/trs_ode.py`)

Implements a velocity-Verlet (symplectic, order 2) integrator for time-reversible dynamics:

- Forward integration of hidden state h(t) with adjoint momentum p(t)
- Reverse integration from t=T→0 for TRS loss computation
- Integration with oscillator dynamics

The TRS loss is calculated as:
```
L_trs = ‖ĥ(0) – h(0)‖² + ‖p̂(0) + p(0)‖²
```

Where ĥ and p̂ are the state and momentum after reversal.

### 3. Spin-Hopfield Memory (`memory/spin_hopfield.py`)

Associative memory based on Hopfield networks but adapted for spin states:

- Energy-based memory recall
- Antiferromagnetic coupling with hardware mapping
- Binary and continuous activation modes
- Adaptive temperature for simulated annealing

The core recall algorithm:
```python
loop {                       # until converged or max iters
    for i in 0..M {          # M = memory nodes
        h = Σ_j W[i][j] * σ[j];
        σ[i] = sign(β * h);  # β ~ exchange stiffness
    }
    if ΔE < ε_stop { break }
}
```

### 4. Banksy Fusion (`banksy_fusion.py`)

Integration layer that combines all three components into a unified system:

- Concept representation using oscillator-spin states
- Reasoning through dynamical evolution
- Pattern completion via memory
- Stability guarantees through TRS principles

## Repository Structure

```
alan_backend/
├── Makefile                  # Build and run system
├── README.md                 # This file
├── core/
│   ├── __init__.py
│   ├── oscillator/             # Oscillator component
│   │   ├── __init__.py
│   │   └── banksy_oscillator.py
│   ├── controller/             # TRS-ODE controller
│   │   ├── __init__.py
│   │   └── trs_ode.py
│   ├── memory/                 # Spin-Hopfield memory
│   │   ├── __init__.py
│   │   └── spin_hopfield.py
│   ├── banksy_fusion.py        # System integration
│   └── demo_banksy_alan.py     # Demonstration script
```

## Future Roadmap

Based on the implementation plan, future work includes:

1. **Hardware Interface**: Support for spin-torque nano-oscillator board (ST-NO-8) over SPI
2. **ELFIN Integration**: Export/import bridge between ELFIN symbolic language and ψ-graph
3. **Dashboard Telemetry**: React dashboard with oscillator visualization and crash dumper
4. **Verification**: Comprehensive property testing for energy conservation and reversibility
5. **Documentation**: Expanded quick-start guide and developer documentation

## Technical Details

### Parameter Tuning

Key parameters that affect system behavior:

- `gamma`: Phase-to-spin coupling gain
- `epsilon`: Spin Hebbian learning rate
- `eta_damp`: Momentum damping factor (typically very small, ≈1e-4)
- `beta`: Inverse temperature for Hopfield memory
- `trs_weight`: Weight of TRS loss component in training

### Performance Metrics

The system tracks several key metrics:

- **Order parameter**: Measure of phase synchronization (0-1)
- **N_eff**: Effective number of synchronized oscillators
- **TRS loss**: Measure of reversibility quality
- **Energy**: For Hopfield memory stability
- **Concept activation**: For reasoning tasks

## References

- [Original Banksy paper](fictional-link)
- [TRS-ODE methodology (Huh et al.)](fictional-link)
- [Spin-Hopfield networks](fictional-link)

## License

[License information]
