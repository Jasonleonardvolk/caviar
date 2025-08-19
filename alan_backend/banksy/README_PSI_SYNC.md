# ψ-Sync Stability Monitoring System

## Overview

The ψ-Sync Stability Monitoring System provides real-time cognitive stability assessment and control for the ALAN cognitive architecture. It bridges phase oscillator dynamics with Koopman eigenfunction analysis to ensure coherent concept processing and reliable inference.

## Core Components

The system consists of three primary layers:

### 1. Core Stability Monitor

`PsiSyncMonitor` tracks the synchronization between oscillator phases (θ) and Koopman eigenfunctions (ψ), providing real-time stability metrics and recommendations for adjustments.

Key classes:
- `PsiSyncMonitor`: Primary monitor for phase-eigenfunction synchronization 
- `PsiPhaseState`: Encapsulates combined phase and eigenfunction data
- `PsiSyncMetrics`: Metrics about synchronization quality
- `SyncAction`: Recommended actions based on stability assessment

### 2. Koopman Integration

`PsiKoopmanIntegrator` connects the spectral analysis from Koopman eigenfunctions with the phase synchronization monitoring.

Key features:
- Processes time series data to extract eigenfunctions
- Maps eigenfunction data to phase oscillators
- Evaluates attractor stability in eigenspace
- Predicts concept evolution using Koopman dynamics

### 3. ALAN Integration Bridge

`AlanPsiSyncBridge` translates low-level stability metrics into actionable decisions for ALAN's reasoning system.

Key capabilities:
- Maps technical stability states to ALAN-specific cognitive states
- Makes confidence-weighted recommendations for the orchestrator
- Provides trigger points for user clarification requests
- Generates detailed stability reports for diagnostics

## Stability States

The system defines three primary stability states:

1. **STABLE/COHERENT**: High synchrony and attractor integrity, suitable for reliable inference
2. **DRIFT/UNCERTAIN**: Moderate synchrony, requiring caution and potential user confirmation
3. **BREAK/INCOHERENT**: Low synchrony or high residual energy, indicating unreliable processing

## Mathematical Foundation

The system builds on two mathematical frameworks:

### Kuramoto Oscillator Model

Phase oscillators (θ) follow the Kuramoto model dynamics:

```
dθᵢ/dt = ωᵢ + (K/N) * Σⱼ sin(θⱼ - θᵢ)
```

Where:
- θᵢ is the phase of the i-th oscillator
- ωᵢ is the natural frequency
- K is the coupling strength
- N is the number of oscillators

### Koopman Eigenfunction Analysis

Koopman theory provides spectral analysis of the system dynamics:

```
ψ(x) = λ ⋅ ψ(F(x))
```

Where:
- ψ is the eigenfunction
- λ is the eigenvalue
- F is the dynamical system
- x is the system state

## Key Metrics

The stability assessment uses several key metrics:

- **Synchrony Score**: How well oscillators align in phase (0-1)
- **Attractor Integrity**: How coherent the clusters are in eigenspace (0-1)
- **Residual Energy**: How much deviation from previous eigenmodes (≥0)
- **Lyapunov Delta**: Change in system energy (negative = increasing stability)

## Usage

### Basic Usage

```python
from alan_backend.banksy import get_psi_sync_monitor, PsiPhaseState

# Create monitor
monitor = get_psi_sync_monitor()

# Create state
state = PsiPhaseState(
    theta=phases,        # Phase values
    psi=eigenfunction,   # Eigenfunction values
    coupling_matrix=K,   # Optional coupling matrix
    concept_ids=ids      # Optional concept identifiers
)

# Evaluate stability
metrics = monitor.evaluate(state)

# Get recommendations
action = monitor.recommend_action(metrics, state)

# Take action based on stability
if metrics.is_stable():
    # Proceed with high confidence
elif metrics.requires_confirmation():
    # Request user confirmation
else:
    # Request clarification
```

### ALAN Integration

```python
from alan_backend.banksy import get_alan_psi_bridge

# Get bridge
bridge = get_alan_psi_bridge()

# Check stability
state, confidence, recommendation = bridge.check_concept_stability(
    phases=phases,
    psi_values=psi_values,
    concept_ids=concept_ids
)

# Use in ALAN reasoning
if state == AlanPsiState.COHERENT:
    # Use high confidence response
elif state == AlanPsiState.UNCERTAIN:
    # Use hedged response
else:
    # Request clarification from user
```

### Koopman Integration

```python
from alan_backend.banksy import PsiKoopmanIntegrator

# Create integrator
integrator = PsiKoopmanIntegrator()

# Process time series data
eigenmodes, metrics = integrator.process_time_series(time_series, concept_ids)

# Apply coupling adjustments if needed
if not metrics.is_stable():
    new_coupling = integrator.apply_coupling_adjustments()
    
# Predict future evolution
predicted_phases, predicted_values = integrator.predict_concept_evolution(n_steps=10)
```

## Implementation Details

### Thresholds

The default thresholds for stability assessment are:

- **stable_threshold**: 0.9 (synchrony score for stable state)
- **drift_threshold**: 0.6 (synchrony score for drift state)
- **residual_threshold**: 0.3 (max residual energy for stability)
- **integrity_threshold**: 0.85 (attractor integrity for stability)

### Feedback Loop

The system implements a feedback loop that:

1. Measures synchronization quality
2. Identifies stability issues
3. Computes coupling adjustments
4. Applies adjustments to improve stability
5. Re-measures to verify improvement

### Cluster Analysis

The attractor integrity computation uses cluster analysis to:

1. Identify potential concept clusters based on phase proximity
2. Measure phase coherence within clusters
3. Verify eigenfunction alignment within clusters
4. Compute weighted cluster quality scores

## Integration with Existing Systems

The ψ-Sync system integrates with:

1. **Koopman Estimator**: Uses eigenfunctions from the KoopmanEstimator
2. **Phase Oscillator Network**: Feeds into and receives feedback from the oscillator network
3. **ALAN Orchestrator**: Provides stability information to guide reasoning
4. **Concept Store**: Maps stability metrics to specific concepts

## Demo and Testing

The package includes several demo scripts:

- `psi_sync_demo.py`: Demonstrates basic functionality
- `psi_koopman_integration.py`: Shows integration with Koopman analysis
- `alan_psi_sync_bridge.py`: Includes simple test case for ALAN integration

## References

1. Takata's (2025) approach to robust Koopman eigenfunction estimation
2. Kuramoto model for phase oscillator synchronization
3. Lyapunov stability analysis for dynamical systems
4. Koopman operator theory for spectral analysis
