# Dynamics Integration Documentation

## Overview
Successfully integrated KoopmanOperator and LyapunovAnalyzer from `dynamics.ts` into the TORI consciousness interface. The system now performs real-time spectral analysis and chaos detection on phase data.

## Integration Points

### 1. Phase Data Sources
The integration captures phase data from the existing `tori-soliton-phase-change` events dispatched in `+page.svelte`:

```javascript
document.dispatchEvent(new CustomEvent('tori-soliton-phase-change', {
  detail: {
    userId: currentUserId,
    phase: 'user_input',
    phaseAngle: solitonResult.phaseTag,
    amplitude,
    frequency,
    stability: 0.8,
    valence
  }
}));
```

### 2. Data Flow
```
User Input → Phase Analysis → Phase Event → Dynamics Integration
                                              ↓
                                    KoopmanOperator (DMD)
                                    LyapunovAnalyzer (Chaos)
                                              ↓
                                    UI Updates & Console Logs
```

### 3. Key Components

#### dynamicsIntegration.ts
- Location: `src/lib/integrations/dynamicsIntegration.ts`
- Purpose: Bridges phase events to dynamics analyzers
- Features:
  - Listens for phase change events
  - Maintains phase vector [phase, amplitude, frequency, stability, valence]
  - Feeds data to analyzers at 50ms intervals
  - Provides callbacks for UI updates

#### UI Integration (+page.svelte)
- Added dynamics tracking variables
- Setup listeners in onMount
- Added Dynamics Analysis panel to debug view
- Shows:
  - Koopman spectral modes (dominant frequencies)
  - Spectral gap
  - Lyapunov exponents and chaos levels
  - Current phase vector

## Usage

### Basic Operation
1. The system automatically captures phase data during conversations
2. Open the debug panel (🧠 button) to see the Dynamics Analysis section
3. Koopman updates show spectral decomposition of the phase dynamics
4. Lyapunov spikes indicate chaotic behavior

### Testing
Run the test script in browser console:
```javascript
// Load test_dynamics.js content or run:
await import('/test_dynamics.js')
```

This simulates phase changes and triggers the analyzers.

### Console Monitoring
Watch for these logs:
- `🌊 Koopman modes: X Dominant: Y Hz` - Spectral updates
- `⚡ Chaos detected! Exponent: X` - Lyapunov spikes
- `🌀 EXTREME CHAOS DETECTED` - High instability warning

## Configuration

### Koopman Operator
- Window size: 128 samples
- Max modes: 10
- Update frequency: ~2Hz (every 64 samples)

### Lyapunov Analyzer
- Embedding dimension: 3
- Delay: 2 samples
- Horizon: 64 samples
- Spike threshold: 0.1

### Custom Configuration
Create custom instances with different parameters:
```javascript
import { KoopmanOperator, LyapunovAnalyzer } from '$lib/cognitive/dynamics';

const customKoopman = new KoopmanOperator({ window: 256, maxModes: 20 });
const customLyapunov = new LyapunovAnalyzer({ horizon: 128, spikeThreshold: 0.05 });
```

## Interpreting Results

### Koopman Spectral Modes
- **Dominant Mode**: Primary oscillation frequency in the phase dynamics
- **Spectral Gap**: Separation between dominant and secondary modes (larger = more stable)
- **Eigenmodes**: Individual frequency components with amplitude and phase

### Lyapunov Chaos Detection
- **Exponent (λ)**: Rate of divergence (positive = chaotic)
- **Instability Level**: 0-1 normalized chaos measure
- **Divergence Rate**: Instantaneous separation of trajectories

## Troubleshooting

### No Koopman Updates
- Check phase events are firing: `document.addEventListener('tori-soliton-phase-change', console.log)`
- Verify phase vector has non-zero values
- Wait for 128 samples to accumulate

### No Lyapunov Spikes
- System may be stable (good!)
- Lower spike threshold if needed
- Check amplitude values are changing

### Performance Issues
- Reduce update frequency in dynamicsIntegration.ts
- Increase window sizes for less frequent updates
- Disable in production if not needed

## Future Enhancements

1. **Visualization**: Add D3.js or Three.js visualization of spectral modes
2. **Adaptive Control**: Use chaos detection to adjust AI behavior
3. **Pattern Library**: Store and recognize specific spectral signatures
4. **Phase Predictor**: Use Koopman modes for short-term prediction
5. **Stability Metrics**: Dashboard showing system stability over time

## Files Modified/Created

1. `src/lib/cognitive/dynamics.ts` - Already existed with full implementations
2. `src/lib/integrations/dynamicsIntegration.ts` - Created integration bridge
3. `src/routes/+page.svelte` - Added dynamics monitoring and UI
4. `test_dynamics.js` - Test script for verification
5. This documentation file

The dynamics analyzers are now fully integrated and operational! 🌊⚡
