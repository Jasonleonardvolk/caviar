# Intent-Driven Reasoning & Cognitive Resolution Documentation

## Overview

The Intent-Driven Reasoning system enables TORI to understand user intent, resolve conflicting knowledge paths, and explain its own reasoning decisions. This creates a self-aware, reflective AI that can justify its answers and handle contradictions intelligently.

## Core Components

### 1. **Intent Parser**
Recognizes 8 types of reasoning intent:
- `EXPLAIN` - General explanation
- `JUSTIFY` - Provide justification ("why?")
- `CAUSAL` - Focus on cause-effect relationships
- `SUPPORT` - Find supporting evidence
- `HISTORICAL` - Historical perspective
- `COMPARE` - Compare alternatives
- `CRITIQUE` - Critical analysis
- `SPECULATE` - Hypothetical reasoning

### 2. **Cognitive Resolution Engine**
Resolves conflicts between multiple reasoning paths:
- Detects contradictory conclusions
- Identifies opposing relationships
- Scores paths based on intent and strategy
- Selects optimal path with confidence metrics

### 3. **Self-Reflective Reasoner**
Enables TORI to explain its own decisions:
- Tracks reasoning history
- Generates meta-explanations
- Provides transparency into decision process

### 4. **Mesh Overlay Manager**
Visual state management for concepts:
- `scarred` - Outdated (>1 year old)
- `deprecated` - Explicitly marked obsolete
- `trusted` - From high-quality sources
- `recent` - Updated within 30 days
- `contested` - Has conflicting relationships

## Integration Guide

### Quick Start

```python
from python.core.prajna_intent_integration import IntentAwarePrajna
from python.core.temporal_reasoning_integration import TemporalConceptMesh

# Create mesh with your concepts
mesh = TemporalConceptMesh()
# ... add nodes and edges ...

# Initialize intent-aware Prajna
prajna = IntentAwarePrajna(
    mesh,
    enable_self_reflection=True,
    enable_overlay_filtering=True
)

# Generate intent-aware response
response = prajna.generate_intent_aware_response(
    "Why should we regulate AI?",
    context={"explain_reasoning": True}
)

print(f"Intent detected: {response.intent}")
print(f"Response: {response.text}")
print(f"Confidence gap: {response.resolution_report.confidence_gap}")

# Get self-reflection
if response.self_reflection:
    print("\nHow I arrived at this answer:")
    print(response.self_reflection)
```

### API Integration

Add intent-aware endpoints to your FastAPI app:

```python
from python.core.prajna_intent_integration import create_intent_aware_endpoints

app = create_intent_aware_endpoints(app, mesh)

# New endpoints:
# POST /api/intent_answer - Intent-aware responses
# GET  /api/reasoning/last_explanation - Explain last decision
# GET  /api/mesh/health - Mesh health report
# POST /api/mesh/refresh_overlays - Update overlays
# POST /api/trust/update - Update source trust scores
```

## Usage Examples

### 1. **Justification Query**
```python
response = prajna.generate_intent_aware_response(
    "Why does entropy lead to compression?"
)
# Intent: JUSTIFY
# Response focuses on causal justification with sources
```

### 2. **Comparison Query**
```python
response = prajna.generate_intent_aware_response(
    "Compare quantum and classical computing"
)
# Intent: COMPARE
# Response includes multiple perspectives and alternatives
```

### 3. **Historical Query**
```python
response = prajna.generate_intent_aware_response(
    "How has our understanding of AI evolved?"
)
# Intent: HISTORICAL
# Response includes temporal evolution, doesn't filter old nodes
```

### 4. **With Self-Reflection**
```python
response = prajna.generate_intent_aware_response(
    "What causes climate change?",
    context={"explain_reasoning": True}
)

print(response.self_reflection)
# Outputs detailed explanation of:
# 1. Intent recognition
# 2. Path selection process
# 3. Conflict resolution
# 4. Confidence assessment
# 5. Alternative considerations
```

## Conflict Resolution

### How It Works

1. **Conflict Detection**
   - Identifies paths with same conclusion but different routes
   - Finds opposing relationships (supports vs. contradicts)
   - Flags logical inconsistencies

2. **Path Scoring**
   Based on:
   - Intent alignment (causal paths for WHY questions)
   - Source trust scores
   - Temporal relevance (penalize old nodes)
   - Path length (shorter for CAUSAL intent)
   - Node status (penalize deprecated/scarred)

3. **Resolution Strategy**
   - `SHORTEST` - Direct paths preferred
   - `COMPREHENSIVE` - Include all perspectives  
   - `RECENT` - Favor fresh knowledge
   - `TRUSTED` - High-quality sources only
   - `DIVERSE` - Multiple viewpoints

### Example Resolution

```python
# Query: "Is AI dangerous?"
# Conflicting paths found:
# 1. AI → risks → regulation needed (score: 0.8)
# 2. AI → benefits → minimal regulation (score: 0.7)

resolution = engine.resolve_conflicts(paths, intent=JUSTIFY)
# Winner: Path 1
# Confidence gap: 0.1
# Explanation: "Path 1 supported by more recent safety research"
```

## Overlay Management

### Node Status Tracking

```python
# Check node status
status = overlay_manager.get_node_status("quantum_v2024")
# {
#   'scarred': False,
#   'deprecated': False,
#   'trusted': True,
#   'recent': True,
#   'contested': False
# }

# Filter paths by overlay
safe_paths = overlay_manager.filter_paths_by_overlay(
    all_paths,
    exclude=['deprecated', 'scarred'],
    require=['trusted']
)
```

### Visualization

```python
# Export overlay for visualization
overlay_data = overlay_manager.visualize_overlay()
# Creates JSON with color-coded node statuses:
# - Red: Deprecated
# - Orange: Scarred (old)
# - Yellow: Contested
# - Green: Trusted + Recent
# - Blue: Trusted
# - Cyan: Recent
```

## Self-Reflection Format

When enabled, TORI explains its reasoning:

```
I arrived at this answer through the following reasoning process:

1. Intent Recognition: I interpreted your question as seeking to justify
   and used a trusted approach.

2. Reasoning Path: Entropy → Information → Compression

3. Step-by-step reasoning:
   - Entropy: Measure of uncertainty in information theory
     Source: Shannon1948
     ↳ implies
   - Information: Data that reduces uncertainty
     Source: PDF_002
     ↳ enables
   - Compression: Process of encoding data efficiently
     Source: PDF_003

4. Conflict Resolution: I found 2 conflicting interpretations and resolved
   them by: Selected shortest valid path with 3 nodes

5. Confidence: The chosen path was 0.15 more confident than alternatives.

6. Alternatives Considered: I evaluated 3 other reasoning paths but
   discarded them because:
   - One path relied on outdated concepts
   - One path had insufficient evidence

7. Reflection: This explanation represents my best current understanding
   based on the knowledge available to me.
```

## Health Monitoring

### Mesh Health Report

```python
health = prajna.get_mesh_health_report()
# {
#   "total_nodes": 150,
#   "overlay_stats": {
#     "scarred": 23,
#     "deprecated": 5,
#     "trusted": 89,
#     "recent": 34,
#     "contested": 12
#   },
#   "health_percentages": {
#     "scarred": 15.3,
#     "deprecated": 3.3,
#     "trusted": 59.3,
#     "recent": 22.7,
#     "contested": 8.0
#   },
#   "recommendation": "Few recent updates - knowledge base may be stagnating"
# }
```

## Configuration

```python
INTENT_CONFIG = {
    "enable_self_reflection": True,
    "enable_overlay_filtering": True,
    "trust_score_defaults": {
        "arxiv": 0.9,
        "nature": 0.95,
        "wikipedia": 0.7,
        "unknown": 0.8
    },
    "scarred_threshold_days": 365,
    "recent_threshold_days": 30,
    "min_confidence_gap": 0.1
}
```

## Best Practices

1. **Regular Overlay Updates**
   ```python
   # Run daily
   prajna.refresh_overlays()
   ```

2. **Trust Score Calibration**
   ```python
   # Update based on source quality assessments
   prajna.update_trust_scores({
       "arxiv_2024": 0.92,
       "blog_post": 0.6
   })
   ```

3. **Intent-Specific Anchoring**
   - For CAUSAL queries: Start from cause nodes
   - For HISTORICAL: Include version-tagged nodes
   - For CRITIQUE: Include contested nodes

4. **Conflict Monitoring**
   - Log resolution reports
   - Track confidence gaps
   - Review discarded paths periodically

## Troubleshooting

### No paths found after filtering
- Check overlay settings
- Temporarily disable filtering
- Review node timestamps

### Low confidence gaps
- Indicates uncertainty
- Consider gathering more evidence
- May need human review

### Excessive conflicts
- Review edge relationships
- Check for duplicate concepts
- Audit source quality

The Intent-Driven Reasoning system transforms TORI into a self-aware, reflective AI that can explain and justify its reasoning process! 🧠✨
