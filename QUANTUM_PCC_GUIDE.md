# Quantum-Enhanced PCC System Guide

This guide provides comprehensive documentation for the Quantum-enhanced Predictive Coherence Computing (PCC) system implemented for the TORI platform.

## System Overview

The PCC system combines three key components:

1. **Banksy Oscillator Core**: Dual-phase (θ, σ) oscillator dynamics with global spin-wave clock
2. **Concept Cluster Stability Analysis**: Spectral graph analysis for early detection of concept instability
3. **Quantum Reservoir Computing**: Quantum-inspired enhancement of phase-space prediction

These components work together to provide powerful predictive capabilities for coherence analysis, concept stability monitoring, and phase-space dynamics.

## Component Architecture

### 1. Banksy Oscillator Core

- **Purpose**: Simulate oscillator dynamics with both phase (θ) and spin (σ) variables
- **Key Files**:
  - `alan_backend/banksy/banksy_spin.py` - Core oscillator implementation
  - `alan_backend/banksy/clock.py` - Global spin-wave clock S(t)
  - `alan_backend/banksy/config.yaml` - Configuration with SPIN_MODE enabled

The oscillator provides the foundation for phase-space dynamics, with the global clock S(t) serving as a measure of synchronization across the network.

### 2. MCP 2.0 Server & Broadcast Infrastructure

- **Purpose**: Real-time broadcasting of PCC state via WebSockets
- **Key Files**:
  - `backend/routes/mcp/server.py` - FastAPI server with WebSocket support
  - `alan_backend/banksy/broadcast.py` - Utilities for broadcasting PCC state
  - `start-mcp-server.bat` - Script to launch the MCP server

This infrastructure enables real-time visualization and monitoring of the system's state.

### 3. Concept Cluster Stability Analysis

- **Purpose**: Early detection of concept clusters at risk of instability
- **Key Files**:
  - `packages/runtime-bridge/python/cluster_stability.py` - Stability analysis algorithms
  - `packages/runtime-bridge/python/forecast_loop.py` - Integration with forecast system
  - `tori_chat_frontend/src/components/CoherenceRibbon/CoherenceRibbon.jsx` - UI for alerts

The stability analysis uses spectral graph theory to identify clusters that may become unstable before it happens.

### 4. Quantum Reservoir Computing

- **Purpose**: Quantum-enhanced prediction and phase-space analysis
- **Key Files**:
  - `pcc/q_reservoir.py` - Quantum reservoir implementation
  - `pcc/test_q_reservoir.py` - Tests for quantum reservoir
  - `alan_backend/banksy/quantum_bridge.py` - Bridge between classical and quantum systems

The quantum reservoir provides a potential quantum advantage in predicting phase-space dynamics and concept evolution.

## Configuration

The system is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SPIN_MODE` | Enable dual-phase oscillator mode | `true` |
| `PCC_BROADCAST_INTERVAL` | Steps between PCC state broadcasts | `10` |
| `PCC_BROADCAST_ENABLED` | Enable PCC state broadcasting | `true` |
| `CLUSTER_STABILITY_ENABLED` | Enable concept cluster stability analysis | `true` |
| `CLUSTER_CHI_THRESH` | Coherence threshold for cluster alerts | `0.45` |
| `CLUSTER_STAB_THRESH` | Stability threshold for cluster alerts | `0.15` |
| `ENABLE_QUANTUM` | Enable quantum-enhanced prediction | `false` |
| `QR_N_QUBITS` | Number of qubits in quantum reservoir | `8` |

## Usage Instructions

### Starting the System

1. **Start the MCP Server**:
   ```
   start-mcp-server.bat
   ```

2. **Run Banksy Oscillator Tests**:
   ```
   python standalone_test_clock.py
   ```

3. **Test PCC Broadcast**:
   ```
   python test_pcc_broadcast.py
   ```

4. **Test Quantum Reservoir** (optional):
   ```
   cd pcc
   python test_q_reservoir.py
   ```

### Visualizing the System

Add the PCC status component to any React page:

```jsx
import PccStatus from '../components/PccStatus/PccStatus';

function MyPage() {
  return (
    <div>
      <h1>My Page</h1>
      <PccStatus />
    </div>
  );
}
```

The CoherenceRibbon component displays stability alerts:

```jsx
import CoherenceRibbon from '../components/CoherenceRibbon/CoherenceRibbon';

function App() {
  return (
    <div>
      <CoherenceRibbon />
      {/* Other components */}
    </div>
  );
}
```

## Physics Background

### Oscillator Dynamics

The dual-phase oscillator combines Kuramoto phase dynamics with Ising-like spin dynamics:

1. **Phase (θ)**: Continuous values in [0, 2π) representing oscillator phase
2. **Spin (σ)**: Binary values {-1, +1} representing discrete states

These oscillators organize into synchronized clusters, with the global clock S(t) measuring overall synchronization.

### Spectral Graph Analysis

Cluster stability is analyzed using spectral graph theory:

- **Spectral gap** (λ₂): Second eigenvalue of the graph Laplacian
- **Community detection**: Greedy modularity optimization
- **Stability score**: Product of spectral gap and minimum coherence

A small spectral gap indicates a cluster that is close to splitting apart.

### Quantum Reservoir Computing

The quantum reservoir enhances prediction using quantum-inspired algorithms:

- **Quantum state**: Complex-valued state vector in 2ⁿ-dimensional space
- **Hamiltonian evolution**: Time evolution under system-specific Hamiltonian
- **Entangled attention**: Quantum-inspired mechanism for focusing on relevant features

## Advanced Topics

### 1. Enabling Quantum Enhancement

To enable quantum-enhanced prediction, set:

```
ENABLE_QUANTUM=true
```

This activates the quantum bridge between the classical Banksy oscillator and the quantum reservoir.

### 2. Tuning Stability Thresholds

The stability thresholds can be tuned based on your needs:

- Lower `CLUSTER_CHI_THRESH` to detect more subtle coherence issues
- Lower `CLUSTER_STAB_THRESH` to get earlier warnings about potential cluster bifurcations

### 3. Performance Considerations

- The quantum reservoir is computationally intensive, especially with large qubit counts
- For production use, consider using a reduced number of qubits (`QR_N_QUBITS=4` or `6`)
- Disable unnecessary features when not needed (`CLUSTER_STABILITY_ENABLED=false`)

## Troubleshooting

### Common Issues

1. **ImportError with Quantum Components**:
   - Ensure NumPy is installed
   - Check that the `pcc` package is in your Python path

2. **MCP Server Connection Issues**:
   - Verify the server is running
   - Check firewall settings for the WebSocket port

3. **High CPU Usage**:
   - Reduce the qubit count if using quantum reservoir
   - Increase the broadcast interval

## Next Steps

Future development could include:

1. **Hardware-specific quantum optimizations** for true quantum advantage
2. **Enhanced Ising-model integration** for memory system hot-swap
3. **Time-reversible controller** for simulating state reversals
4. **Full quantum reservoir hardware bridge** when quantum hardware becomes available

## References

- MCP_PCC_GUIDE.md - Detailed guide for MCP 2.0 implementation
- PREDICTIVE-COHERENCE-GUIDE.md - Guide for the coherence prediction system
