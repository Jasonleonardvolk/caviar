# Beyond Metacognition - README Addition

Add this section to your main README.md:

---

## 🌌 Beyond Metacognition

TORI now includes the **Beyond Metacognition** layer - a self-transforming cognitive system that can detect and generate new dimensions of thought.

### Quick Start

```bash
# Apply patches to enable Beyond Metacognition
python kha/apply_beyond_patches.py --verify

# Run interactive demo
python kha/torictl.py demo emergence --plot

# Monitor live behavior
python kha/torictl.py monitor
```

### Key Features

- **Dimensional Emergence Detection**: TORI detects when genuinely new cognitive dimensions appear
- **Multi-Scale Temporal Braiding**: Cognitive traces from microseconds to days
- **Self-Measurement**: Metacognitive tokens that influence future reasoning
- **Creative Exploration**: Controlled entropy injection with aesthetic constraints

### Documentation

- **Quick Start**: See `kha/BEYOND_QUICKSTART.md`
- **Full Technical Details**: See `kha/BEYOND_METACOGNITION_COMPLETE.md`
- **API Reference**: Component docstrings in source files

### Configuration

Key settings in `conf/runtime.yaml`:

```yaml
beyond_metacognition:
  observer_synthesis:
    reflex_budget: 60  # measurements per hour
  creative_feedback:
    novelty_threshold_high: 0.7
    emergency_threshold: 0.08
```

### Monitoring

- Grafana dashboard: Import `grafana/beyond_metacognition_dashboard.json`
- Metrics endpoint: `http://localhost:9090/metrics` (look for `beyond_*` metrics)
- Live monitor: `python kha/torictl.py monitor`

### Safety

The system includes multiple safety mechanisms:
- Automatic rollback if λ_max > 0.08 for 3+ minutes
- Emergency damping mode
- Reflex budget to prevent oscillations

---
