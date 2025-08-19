# Temporal Reasoning & Training Export Documentation

## Overview

The Temporal Reasoning system extends TORI's reasoning capabilities with time-awareness and training data generation. It enables tracking knowledge evolution, detecting concept drift, and exporting reasoning paths for model training.

## Key Features

### 1. **Temporal-Weighted Traversal**

Edges now include timestamps that affect path scoring:

```python
# Traverse only recent knowledge (after specific date)
recent_paths = mesh.traverse_temporal("entropy", after="2024-01-01")

# All paths with time decay scoring
all_paths = mesh.traverse_temporal("entropy", max_depth=3)
```

**Time Decay Function**: 
- Fresh edges (< 1 year): Full weight
- Older edges: Weight decreases linearly over 365 days
- Filtered edges (before `after` date): Weight = 0

### 2. **Training Data Export**

Convert any ReasoningPath to JSONL format for model training:

```python
sample = path.to_training_sample("How does entropy relate to compression?")
```

**Output Format**:
```json
{
  "query": "How does entropy relate to compression?",
  "path_type": "causal",
  "chain": [
    {
      "id": "entropy",
      "text": "Measure of uncertainty",
      "source": "PDF_001"
    }
  ],
  "justifications": ["implies", "enables"],
  "confidence": 0.82,
  "narration": "Entropy implies information theory, which enables compression",
  "source_ids": ["PDF_001", "PDF_002", "PDF_003"]
}
```

### 3. **Knowledge Drift Analysis**

Track how your knowledge base evolves:

```python
drift = evaluate_temporal_drift(paths, window_days=30)
# Returns: {"recent_ratio": 0.7, "older_ratio": 0.3}
```

## Integration Guide

### Quick Start

```python
from temporal_reasoning_integration import (
    TemporalConceptMesh,
    TemporalReasoningAnalyzer,
    TrainingDataExporter
)

# Create temporal mesh
mesh = TemporalConceptMesh()

# Add temporal edge
mesh.add_temporal_edge(
    "entropy", "information", EdgeType.IMPLIES,
    timestamp="2024-06-20T15:00:00Z",
    justification="entropy implies information"
)

# Traverse with time awareness
recent_paths = mesh.traverse_temporal("entropy", after="2024-01-01")

# Export for training
exporter = TrainingDataExporter()
for path in recent_paths:
    sample = exporter.export_path(path, "Your query here")
```

### Knowledge Management

Use `TemporalKnowledgeManager` for version control:

```python
manager = TemporalKnowledgeManager(mesh)

# Add versioned concept
v1 = manager.add_knowledge_version(
    "quantum", "Quantum Computing",
    "Initial definition...",
    ["source1", "source2"]
)

# Update concept
v2 = manager.add_knowledge_version(
    "quantum", "Quantum Computing", 
    "Refined definition...",
    ["source3", "source4"]
)

# Deprecate old version
manager.deprecate_concept(v1, "Outdated", replacement_id=v2)

# Get timeline
timeline = manager.get_concept_timeline("quantum")
```

## Use Cases

### 1. **Time-Scoped Explanations**

Ask "How did we understand X in 2023?":
```python
historical_paths = mesh.traverse_temporal(
    "consciousness", 
    after="2023-01-01",
    before="2023-12-31"  # If implemented
)
```

### 2. **Knowledge Freshness Monitoring**

Identify stale areas of knowledge:
```python
analyzer = TemporalReasoningAnalyzer()
drift = analyzer.analyze_knowledge_drift(mesh, window_days=90)

if drift['stale_nodes'] > 10:
    print(f"Warning: {drift['stale_nodes']} concepts haven't been updated in 90+ days")
```

### 3. **Training Data Generation**

Build datasets for reward models:
```python
training_samples = []

for query in queries:
    paths = mesh.traverse_temporal(anchor, after=cutoff_date)
    for path in paths[:5]:  # Top 5 paths
        sample = exporter.export_path(path, query, {
            "session_id": session_id,
            "user_feedback": feedback_score
        })
        training_samples.append(sample)

# Save batch
exporter.save_training_batch(training_samples, "reasoning_dataset.jsonl")
```

### 4. **Concept Evolution Tracking**

Visualize how understanding changes:
```python
manager.visualize_knowledge_evolution("quantum")
# Generates timeline plot showing version progression
```

## Advanced Features

### Temporal Inconsistency Detection

Find logical issues in temporal data:
```python
inconsistencies = analyzer.find_temporal_inconsistencies(mesh)
# Detects edges that point to "future" nodes
```

### Drift Pattern Analysis

Get detailed drift metrics:
```python
patterns = manager.analyze_drift_patterns(window_days=30)
# Returns:
# - Age distribution (fresh/recent/stable/legacy)
# - Update hotspots (frequently changing concepts)
# - Recent vs older ratios
```

### Batch Processing

Process multiple queries for training:
```python
def generate_training_corpus(queries, mesh, output_dir="training_data"):
    exporter = TrainingDataExporter(output_dir)
    
    for batch_idx, query_batch in enumerate(chunked(queries, 100)):
        samples = []
        
        for query in query_batch:
            # Extract anchors from query
            anchors = extract_anchors(query)
            
            # Get temporal paths
            for anchor in anchors:
                paths = mesh.traverse_temporal(anchor, after="2024-01-01")
                
                for path in paths[:3]:  # Top 3 per anchor
                    sample = exporter.export_path(path, query)
                    samples.append(sample)
        
        # Save batch
        exporter.save_training_batch(samples, f"batch_{batch_idx}.jsonl")
```

## Configuration

```python
TEMPORAL_CONFIG = {
    "time_decay_window_days": 365,  # 1 year decay
    "min_temporal_score": 0.01,      # Filter threshold
    "default_timestamp_weight": 0.5,  # For edges without timestamps
    "export_batch_size": 1000,       # Training data batch size
    "drift_analysis_windows": [7, 30, 90, 365],  # Analysis periods
}
```

## Best Practices

1. **Always timestamp new edges** when adding knowledge
2. **Run drift analysis weekly** to identify stale areas
3. **Export training data** after significant knowledge updates
4. **Version concepts** instead of overwriting
5. **Set deprecation policies** for outdated knowledge

## Troubleshooting

### Missing timestamps
- Edges without timestamps get default weight (0.5)
- Add timestamps when importing legacy data

### Time zone issues
- Always use UTC timestamps (timezone.utc)
- Convert local times before storage

### Performance with large graphs
- Use `after` parameter to limit traversal scope
- Increase `min_temporal_score` threshold
- Index edges by timestamp for faster filtering

The Temporal Reasoning system transforms TORI from a static knowledge base to a dynamic, evolving intelligence that tracks its own learning journey! 🕐
