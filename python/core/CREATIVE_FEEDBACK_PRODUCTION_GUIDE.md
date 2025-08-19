# Creative Feedback Enhancements - Production Implementation Guide

## Overview

This guide shows how to integrate the enhanced Creative Feedback system with entropy profiles, quality modeling, and metric streaming into a production TORI deployment.

## Architecture Integration

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                     │     │                  │     │                 │
│   TORI Core         │────▶│ Creative         │────▶│ Monitoring      │
│   (SpectralCortex)  │     │ Feedback Loop    │     │ Dashboard       │
│                     │     │                  │     │                 │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
         │                           │                         ▲
         │                           │                         │
         ▼                           ▼                         │
┌─────────────────────┐     ┌──────────────────┐             │
│  System State       │     │  Entropy         │             │
│  - novelty          │     │  Injection       │             │
│  - aesthetic        │     │  - profiles      │             │
│  - volatility       │     │  - factors       │             │
└─────────────────────┘     └──────────────────┘             │
                                     │                         │
                                     └─────────────────────────┘
                                          Metric Stream
```

## 1. Production Setup

### Installation

```python
# In your TORI initialization
from python.core.creative_feedback import (
    CreativeFeedbackLoop,
    configure_logging
)

# Configure logging appropriately
configure_logging(logging.INFO)  # or WARNING for production

# Create feedback instance
creative_feedback = CreativeFeedbackLoop(
    # Optional parameters
    stability_threshold=0.8,      # Higher = more conservative
    max_exploration_steps=200,    # Maximum exploration duration
    recovery_steps=30            # Recovery period after exploration
)
```

### Integration with SpectralCortex

```python
# In spectral_cortex.py
class SpectralCortex:
    def __init__(self):
        # ... existing init ...
        self.creative_feedback = CreativeFeedbackLoop()
        
        # Enable metric streaming if monitoring desired
        if self.config.get('enable_creativity_monitoring'):
            self._setup_creativity_monitoring()
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # ... existing processing ...
        
        # Prepare state for creative feedback
        creative_state = {
            'novelty': spectral_state.get('novelty', 0.5),
            'aesthetic_score': self._calculate_aesthetic_score(spectral_state),
            'volatility': self._calculate_volatility(),
            'lambda_max': spectral_state.get('lambda_max', 0.0),
            'coherence': spectral_state.get('phase', 'local')
        }
        
        # Get creative feedback action
        creative_action = self.creative_feedback.update(creative_state)
        
        # Apply creative adjustments
        if creative_action['action'] == 'inject_entropy':
            self._apply_creative_entropy(spectral_state, creative_action)
        
        return spectral_state
```

## 2. Entropy Profile Configuration

### Automatic Profile Selection

The system automatically selects profiles based on state, but you can customize the logic:

```python
class CustomCreativeFeedback(CreativeFeedbackLoop):
    def _select_entropy_profile(self, state: Dict[str, Any]) -> str:
        """Custom profile selection for your domain"""
        
        # Example: Use pulse for stuck conversations
        if state.get('conversation_stalled', False):
            return 'pulse'
        
        # Example: Use decay for overheated discussions
        if state.get('temperature', 0) > 0.8:
            return 'exponential_decay'
        
        # Example: Use ramp for gentle exploration
        if state.get('user_comfort', 1.0) < 0.5:
            return 'cosine_ramp'
        
        # Fallback to parent logic
        return super()._select_entropy_profile(state)
```

### Custom Entropy Profiles

Add your own profiles:

```python
def sinusoidal_profile(base_factor, step, total_steps, params):
    """Smooth sinusoidal variation"""
    amplitude = params.get('amplitude', 0.3)
    frequency = params.get('frequency', 2.0)
    
    t = step / total_steps
    modulation = 1 + amplitude * np.sin(2 * np.pi * frequency * t)
    
    return base_factor * modulation

# Register custom profile
feedback._get_profile_factor = lambda bf, s, ts, p, params: (
    sinusoidal_profile(bf, s, ts, params) if p == 'sinusoidal' 
    else feedback._get_profile_factor(bf, s, ts, p, params)
)
```

## 3. Quality Model Training

### Pre-training from Historical Data

```python
# Load historical exploration data
def pretrain_quality_model(feedback_loop, historical_data):
    """Pre-train quality model from past explorations"""
    
    for record in historical_data:
        # Reconstruct injection
        injection = feedback_loop._create_injection(
            state=record['initial_state'],
            factor=record['entropy_factor'],
            duration=record['duration']
        )
        
        # Set outcomes
        injection.performance_after = record['final_performance']
        injection.creative_gain = record['creative_gain']
        
        # Update model
        feedback_loop._update_quality_model(injection)
    
    print(f"Pre-trained on {len(historical_data)} historical explorations")

# Usage
historical = load_exploration_history()  # Your data
pretrain_quality_model(creative_feedback, historical)
```

### Online Learning Configuration

```python
# Adjust learning parameters
creative_feedback.quality_model['min_samples'] = 10  # Start predicting sooner
creative_feedback.quality_model['regularization'] = 0.1  # More stable
creative_feedback.quality_model['feature_weights'] = np.array([
    0.3,  # novelty weight
    0.4,  # aesthetic weight
    -0.2, # volatility weight (negative = penalize)
    0.5,  # factor weight
    0.1,  # duration weight
    0.05  # experience weight
])  # Initialize with domain knowledge
```

## 4. Metric Streaming Setup

### WebSocket Integration

```python
import websockets
import json

class WebSocketMetricStreamer:
    def __init__(self, uri="ws://localhost:8765/creativity"):
        self.uri = uri
        self.ws = None
        
    async def connect(self):
        self.ws = await websockets.connect(self.uri)
        
    async def stream_metric(self, metric):
        if self.ws:
            await self.ws.send(json.dumps({
                'type': 'creativity_metric',
                'timestamp': metric['timestamp'],
                'data': metric
            }))

# Setup
streamer = WebSocketMetricStreamer()
await streamer.connect()
creative_feedback.enable_metric_streaming(
    streamer.stream_metric,
    interval_steps=10
)
```

### Prometheus Integration

```python
from prometheus_client import Gauge, Counter, Histogram

# Define metrics
creativity_gain = Gauge('tori_creativity_gain', 'Cumulative creative gain')
exploration_duration = Histogram('tori_exploration_duration', 'Exploration duration in steps')
exploration_count = Counter('tori_explorations', 'Total explorations', ['profile'])
current_mode = Gauge('tori_creative_mode', 'Current creative mode', ['mode'])

async def prometheus_metric_handler(metric):
    """Export metrics to Prometheus"""
    # Update gauges
    creativity_gain.set(metric['cumulative_gain'])
    
    # Update mode
    mode = metric['mode']
    for m in ['stable', 'exploring', 'recovering']:
        current_mode.labels(mode=m).set(1 if m == mode else 0)
    
    # Track completed explorations
    if 'completed_injection' in metric:
        inj = metric['completed_injection']
        exploration_duration.observe(inj['duration_steps'])
        exploration_count.labels(profile=inj['entropy_profile']).inc()

# Enable
creative_feedback.enable_metric_streaming(
    prometheus_metric_handler,
    interval_steps=1  # Every step for accurate metrics
)
```

### Logging Integration

```python
import structlog

# Structured logging setup
logger = structlog.get_logger()

async def structured_log_handler(metric):
    """Log interesting creative events"""
    
    # Log mode transitions
    if 'mode_changed' in metric:
        logger.info("creative_mode_transition",
                   from_mode=metric['previous_mode'],
                   to_mode=metric['mode'],
                   cumulative_gain=metric['cumulative_gain'])
    
    # Log explorations
    if metric['mode'] == 'exploring' and metric.get('just_started'):
        logger.info("creative_exploration_started",
                   profile=metric['current_injection']['entropy_profile'],
                   predicted_gain=metric.get('predicted_gain'),
                   entropy_factor=metric['current_injection']['entropy_factor'])
    
    # Log completions
    if 'completed_injection' in metric:
        inj = metric['completed_injection']
        logger.info("creative_exploration_completed",
                   actual_gain=inj['creative_gain'],
                   duration=inj['duration_steps'],
                   profile=inj['entropy_profile'])

# Enable
creative_feedback.enable_metric_streaming(
    structured_log_handler,
    interval_steps=1
)
```

## 5. Production Best Practices

### Error Handling

```python
class RobustCreativeFeedback(CreativeFeedbackLoop):
    def update(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Validate state
            required_keys = ['novelty', 'aesthetic_score']
            for key in required_keys:
                if key not in current_state:
                    logger.warning(f"Missing {key} in state, using default")
                    current_state[key] = 0.5
            
            # Clamp values
            current_state['novelty'] = np.clip(current_state.get('novelty', 0.5), 0, 1)
            current_state['aesthetic_score'] = np.clip(current_state.get('aesthetic_score', 0.5), 0, 1)
            
            # Normal update
            return super().update(current_state)
            
        except Exception as e:
            logger.error(f"Creative feedback error: {e}", exc_info=True)
            # Safe fallback
            return {'action': 'maintain', 'error': str(e)}
```

### Performance Optimization

```python
# Cache profile calculations
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_profile_factor(profile, step, total_steps, param_tuple):
    params = dict(param_tuple)  # Convert back to dict
    return original_get_profile_factor(profile, step, total_steps, params)

# Use cached version
def optimized_get_profile_factor(self, base_factor, step, total_steps, profile, params):
    param_tuple = tuple(sorted(params.items()))  # Hashable
    factor_multiplier = cached_profile_factor(profile, step, total_steps, param_tuple)
    return base_factor * factor_multiplier
```

### Monitoring Alerts

```yaml
# prometheus-alerts.yml
groups:
  - name: creativity
    rules:
      - alert: LowCreativity
        expr: tori_creativity_gain < 0.5
        for: 10m
        annotations:
          summary: "Creative gain below threshold"
          
      - alert: StuckInExploration
        expr: tori_creative_mode{mode="exploring"} == 1
        for: 30m
        annotations:
          summary: "Exploration phase too long"
          
      - alert: HighErrorRate
        expr: rate(tori_creative_errors[5m]) > 0.1
        annotations:
          summary: "High error rate in creative feedback"
```

## 6. Testing in Production

### A/B Testing Profiles

```python
import random

class ABTestingCreativeFeedback(CreativeFeedbackLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ab_groups = {
            'control': ['constant'],
            'treatment': ['cosine_ramp', 'exponential_decay', 'pulse']
        }
        self.user_group = {}
    
    def _select_entropy_profile(self, state):
        # Assign user to group
        user_id = state.get('user_id', 'default')
        if user_id not in self.user_group:
            self.user_group[user_id] = random.choice(['control', 'treatment'])
        
        group = self.user_group[user_id]
        available_profiles = self.ab_groups[group]
        
        # Select from available profiles
        if group == 'control':
            return 'constant'
        else:
            return super()._select_entropy_profile(state)
```

### Gradual Rollout

```python
import random

class GradualRolloutCreativeFeedback(CreativeFeedbackLoop):
    def __init__(self, *args, enhancement_percentage=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.enhancement_percentage = enhancement_percentage
        
    def _should_use_enhancements(self):
        return random.random() < self.enhancement_percentage
    
    def _select_entropy_profile(self, state):
        if self._should_use_enhancements():
            return super()._select_entropy_profile(state)
        return 'constant'  # Default behavior
    
    def _predict_creative_gain(self, state, factor, duration):
        if self._should_use_enhancements():
            return super()._predict_creative_gain(state, factor, duration)
        return 0.3  # Default prediction
```

## 7. Debugging and Diagnostics

### Debug Mode

```python
class DebugCreativeFeedback(CreativeFeedbackLoop):
    def __init__(self, *args, debug=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        
    def inject_entropy(self, base_state):
        result = super().inject_entropy(base_state)
        
        if self.debug:
            print(f"[DEBUG] Entropy Injection:")
            print(f"  State: {base_state}")
            print(f"  Selected Profile: {self.current_injection.entropy_profile}")
            print(f"  Factor: {result['entropy_factor']}")
            print(f"  Duration: {result['duration']}")
            
            if self.quality_model['trained']:
                prediction = self._predict_creative_gain(
                    base_state, 
                    result['entropy_factor'],
                    result['duration']
                )
                print(f"  Predicted Gain: {prediction:.3f}")
        
        return result
```

### Health Check Endpoint

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health/creativity')
def creativity_health():
    """Health check endpoint for creative feedback"""
    
    metrics = creative_feedback.get_creative_metrics()
    
    health = {
        'status': 'healthy',
        'mode': metrics['mode'],
        'cumulative_gain': metrics['cumulative_gain'],
        'exploration_count': len(creative_feedback.injection_history),
        'quality_model_trained': creative_feedback.quality_model['trained'],
        'last_exploration': None
    }
    
    # Add last exploration info
    if creative_feedback.injection_history:
        last = creative_feedback.injection_history[-1]
        health['last_exploration'] = {
            'profile': last.entropy_profile,
            'gain': last.creative_gain,
            'duration': last.duration_steps
        }
    
    # Check for issues
    if metrics['cumulative_gain'] < 0.1 and len(creative_feedback.injection_history) > 10:
        health['status'] = 'degraded'
        health['reason'] = 'Low cumulative gain despite multiple explorations'
    
    return jsonify(health)
```

## Conclusion

The enhanced Creative Feedback system with entropy profiles, quality modeling, and metric streaming provides:

1. **Adaptive Behavior**: Automatically adjusts to system state
2. **Learning Capability**: Improves over time
3. **Observability**: Real-time insights into creative processes
4. **Production Readiness**: Error handling, monitoring, gradual rollout

Integrate these enhancements gradually, starting with metric streaming for visibility, then enabling profiles, and finally activating the quality model once you have sufficient data.
