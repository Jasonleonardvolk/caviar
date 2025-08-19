# ELFIN Debug System

## Overview

The ELFIN Debug System provides advanced debugging capabilities for ELFIN controllers, with seamless integration into the TORI IDE. It enables real-time monitoring of Lyapunov stability, barrier function verification, and dimensional analysis with unit checking.

This system follows a push-based architecture where the runtime streams state updates to the debugging UI, ensuring zero CPU overhead when paused and symmetric data flow with the VS Code Debug Adapter Protocol (DAP).

## Key Features

### 1. Lyapunov Stability Monitoring

- Real-time visualization of Lyapunov function values and derivatives
- Visual thermometer that displays system stability with color-coded feedback
- Support for breakpoints on stability violations (V̇ > 0)

### 2. Barrier Function Verification

- Runtime monitoring of user-defined barrier functions
- Automatic detection and notification of safety constraint violations
- Integration with breakpoint system for stopping execution on violations

### 3. Unit Consistency Checking

- Static analysis of ELFIN code to detect dimensional inconsistencies
- Automatic suggestion of unit conversion fixes
- Two-tier quick-fix options in the IDE

### 4. Seamless IDE Integration

- Integration with TORI IDE for visualizations
- VS Code DAP bridge for standard debugging workflows
- Support for conditional breakpoints based on mathematical expressions

## Architecture

The ELFIN Debug System uses a distributed architecture:

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│  ELFIN Runtime  │◄────────┤  Debug Bridge   │◄────────┤    TORI IDE     │
│  (Monitoring)   │─────────►  (Relay)        │─────────►  (Visualization)│
│                 │         │                 │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                    │
                                    │
                                    ▼
                            ┌─────────────────┐
                            │                 │
                            │    VS Code      │
                            │  (DAP Client)   │
                            │                 │
                            └─────────────────┘
```

### Components

1. **Lyapunov Monitor** (`lyapunov_monitor.py`): Tracks function values and derivatives, generating events on violations.

2. **Unit Checker** (`unit_checker.py`): Performs static analysis on ELFIN code to detect dimensional inconsistencies.

3. **Debug Bridge** (`bridge.py`): Connects the ELFIN runtime with TORI IDE and VS Code.

4. **React Components** (`LyapunovThermometer.jsx`, etc.): UI components for visualization in TORI IDE.

5. **Stream Context** (`ElfinStreamContext.jsx`): Shared WebSocket context for all debug components.

## Installation

1. Ensure you have the required Python packages:

```bash
pip install websockets numpy
```

2. Copy the React components to your TORI IDE client:

```bash
cp -r client/src/components/ElfinDebug/ /path/to/tori/client/src/components/
```

3. Add the ElfinStreamProvider to your app:

```jsx
// In your app's main component
import { ElfinStreamProvider } from './components/ElfinDebug/ElfinStreamContext';

function App() {
  return (
    <ElfinStreamProvider url="ws://localhost:8643/state">
      <YourAppContent />
    </ElfinStreamProvider>
  );
}
```

## Usage

### 1. Start the Debug Bridge

```bash
python -m alan_backend.elfin.debug.bridge --elfin-port 8642 --tori-port 8643 --dap-port 8644
```

### 2. Instrument your ELFIN controller

```python
from alan_backend.elfin.debug import lyapunov_monitor

# Register state variables with units
lyapunov_monitor.register_state_var("theta", "rad")
lyapunov_monitor.register_state_var("omega", "rad/s")

# Set Lyapunov function and its derivative
lyapunov_monitor.set_lyapunov_function(
    V_func=my_energy_function,
    Vdot_func=my_energy_derivative
)

# Register barrier functions
lyapunov_monitor.register_barrier_function(
    name="collision_avoidance",
    func=distance_barrier,
    threshold=0.1
)

# Start the monitor
lyapunov_monitor.start()

# Then in your control loop:
def control_loop():
    while running:
        # Update controller state
        controller.update()
        
        # Update monitor with latest state
        lyapunov_monitor.update(
            theta=controller.theta,
            omega=controller.omega
        )
```

### 3. Add the Thermometer component to your UI

```jsx
import { LyapunovThermometer } from './components/ElfinDebug/LyapunovThermometer';

function DebugPanel() {
  return (
    <div className="debug-panel">
      <LyapunovThermometer />
    </div>
  );
}
```

### 4. Run the Unit Checker

```python
from alan_backend.elfin.debug import unit_checker

# Check a file
diagnostics = unit_checker.analyze_file('my_controller.elfin')

# Display results
for diag in diagnostics:
    print(f"Line {diag.line}: {diag.message}")
    for fix in diag.fixes:
        print(f"  Suggested fix: {fix.title}")
```

## Example

See the complete example in `alan_backend/elfin/examples/debug_demo.py` which demonstrates:

1. A simple pendulum with Lyapunov-based stability analysis
2. Real-time monitoring with barrier functions
3. Unit consistency checking

Run the example with:

```bash
python -m alan_backend.elfin.examples.debug_demo
```

## Integration with Existing Systems

### Adding to TORI IDE

The ELFIN Debug System is designed to integrate seamlessly with the existing TORI IDE architecture:

1. The `LyapunovPredictabilityPanel` is extended to connect to the ELFIN stream
2. `EditorSyncService` is used to highlight unit inconsistencies
3. `ExecutionTracerService` is enhanced to support mathematical conditions as breakpoints

### Connecting to VS Code

The system bridges to VS Code's Debug Adapter Protocol via WebSockets, translating:

1. ELFIN breakpoints to DAP breakpoints
2. Stability events to DAP stopped events
3. State variables to DAP variables

## Protocol Specification

The protocol uses JSON packets over WebSockets with two main packet types:

### 1. Handshake Packet

```json
{
  "type": "handshake",
  "schema": {
    "vars": {
      "theta": "rad",
      "omega": "rad/s"
    },
    "barriers": ["angle_limit", "velocity_limit"]
  },
  "dt_nominal": 0.01
}
```

### 2. State Update Packet

```json
{
  "type": "state",
  "seq": 8142,
  "t": 0.512,
  "vars": {
    "theta": 1.04,
    "omega": 0.12
  },
  "V": 0.48,
  "Vdot": -0.31,
  "barriers": {
    "angle_limit": 0.7,
    "velocity_limit": 1.3
  },
  "event": null
}
```

If `event` contains a value like `"break:B_height<=0"`, the debugger will halt execution.

## Contributing

When extending the ELFIN Debug System:

1. Add new visualization components to `client/src/components/ElfinDebug/`
2. Use the shared `ElfinStreamContext` for all components
3. Follow the packet schema for new monitoring capabilities
4. Use the sigmoid stability mapping for consistent visualization
