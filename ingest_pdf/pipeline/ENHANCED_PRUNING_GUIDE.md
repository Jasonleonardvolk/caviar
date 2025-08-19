# Enhanced Entropy Pruning System

## Overview

The enhanced pruning system provides advanced entropy-based concept filtering with:
- Detailed logging of prune decisions
- Adaptive thresholds based on context
- Diversity-aware reranking (MMR)
- Comprehensive statistics and visualization
- Feedback loop support

## Key Features

### 1. **Detailed Prune Logging**

Every pruning decision is logged with:
- **Concept name**: What was pruned
- **Reason**: Why it was pruned (entropy, similarity, quality, etc.)
- **Scores**: Entropy, quality, and similarity scores
- **Thresholds**: What threshold was applied
- **Context**: Section, category, and metadata

Example log entry:
```json
{
  "name": "neural embedding",
  "pruned": true,
  "reason": "entropy_threshold",
  "category": "technical_terms",
  "score": 0.62,
  "quality_score": 0.71,
  "entropy_score": 0.0004,
  "threshold": 0.0005,
  "section": "methodology"
}
```

### 2. **Adaptive Entropy Thresholds**

Thresholds adapt based on:

#### Document Type
- **Research papers**: 10% stricter (×0.9)
- **Technical docs**: 5% stricter (×0.95)
- **News articles**: 10% more lenient (×1.1)
- **General content**: Baseline (×1.0)

#### Section Type
- **Title**: 20% stricter (×0.8)
- **Abstract**: 15% stricter (×0.85)
- **Introduction**: 10% stricter (×0.9)
- **Body/Methods**: Baseline (×1.0)

#### Concept Density
- **High density** (>50 concepts/1000 words): 10% stricter
- **Low density** (<10 concepts/1000 words): 10% more lenient

### 3. **Prune Reasons**

Concepts can be pruned for various reasons:
- `entropy_threshold`: Below entropy threshold
- `similarity_threshold`: Too similar to existing concepts
- `category_limit`: Category quota exceeded
- `low_quality`: Quality score too low
- `duplicate`: Exact or near duplicate
- `diversity_constraint`: Removed to maintain diversity

### 4. **MMR-Based Reranking**

After initial pruning, concepts are reranked using Maximal Marginal Relevance:
```
MMR(c) = λ × relevance(c) - (1-λ) × max_similarity(c, selected)
```
- λ = 0.7 by default (70% relevance, 30% diversity)
- Balances relevance with diversity
- Prevents redundant concept clusters

### 5. **Comprehensive Statistics**

Detailed statistics are tracked:
```json
{
  "total_concepts": 150,
  "pruned_count": 45,
  "retained_count": 105,
  "prune_rate": 0.30,
  "prune_by_reason": {
    "entropy_threshold": 20,
    "similarity_threshold": 15,
    "low_quality": 10
  },
  "prune_by_category": {
    "general": 25,
    "technical_terms": 20
  },
  "avg_entropy_pruned": 0.0003,
  "avg_entropy_retained": 0.0008
}
```

## Configuration

### Environment Variables

```bash
# Enable enhanced pruning (default: true)
export USE_ENHANCED_PRUNING=true

# Enable detailed prune logging
export ENABLE_PRUNE_LOGGING=true

# Save pruning decisions to file
export SAVE_PRUNE_DECISIONS=true
export PRUNE_LOG_DIR=./prune_logs

# Create visualization plots
export CREATE_PRUNE_VIZ=true

# Enable MMR reranking
export ENABLE_MMR_RERANKING=true
```

### Configuration Options

```python
ENTROPY_CONFIG = {
    # Base thresholds
    "entropy_threshold": 0.0005,
    "similarity_threshold": 0.83,
    "max_diverse_concepts": 1000,
    
    # Feature toggles
    "enable_prune_logging": True,
    "enable_mmr_reranking": True,
    "mmr_lambda": 0.7,
    
    # Category limits
    "category_limits": {
        "technical_terms": 50,
        "general_concepts": 100,
        "named_entities": 30
    }
}
```

## Usage Examples

### Basic Usage
```python
# Automatic document type detection
result = ingest_pdf_clean("research_paper.pdf")

# Stats are included in response
prune_stats = result.get("entropy_analysis", {})
```

### Advanced Usage
```python
from pipeline.enhanced_pruning import create_enhanced_pruner, DocumentType

# Create custom pruner
pruner = create_enhanced_pruner(
    config=ENTROPY_CONFIG,
    doc_type="research"
)

# Prune concepts
pruned_concepts, stats = pruner.prune_concepts(
    concepts,
    max_concepts=500,
    admin_mode=False
)

# Access detailed decisions
for decision in pruner.decisions[:10]:
    if decision.pruned:
        print(f"Pruned '{decision.concept_name}': {decision.reason.value}")
```

### Visualization
```python
from pipeline.enhanced_pruning import create_prune_visualization

# Create visualization (requires matplotlib)
create_prune_visualization(pruner.decisions, "pruning_analysis.png")
```

## Visualization Outputs

The system generates four plots:

1. **Pruning Reasons**: Bar chart of prune reason frequencies
2. **Categories Pruned**: Distribution of pruned concepts by category
3. **Score Distributions**: Histogram comparing pruned vs retained scores
4. **Entropy Comparison**: Box plot of entropy scores

## Prune Decision Analysis

Saved decision files include:
- Timestamp and document type
- Complete statistics
- Individual decisions with all scores

Example:
```json
{
  "timestamp": "20240115_143022",
  "doc_type": "research",
  "statistics": { ... },
  "decisions": [
    {
      "name": "machine learning",
      "pruned": false,
      "reason": null,
      "score": 0.89,
      "entropy_score": 0.0012
    }
  ]
}
```

## Feedback Loop Integration

The system supports feedback loops for continuous improvement:

1. **Log Collection**: All decisions are logged with context
2. **Offline Analysis**: Analyze pruning patterns
3. **Threshold Tuning**: Adjust thresholds based on outcomes
4. **A/B Testing**: Compare different configurations

### Example Analysis Script
```python
import json
from pathlib import Path
from collections import Counter

# Load all decision files
decisions_dir = Path("./prune_logs")
all_decisions = []

for file in decisions_dir.glob("prune_decisions_*.json"):
    with open(file) as f:
        data = json.load(f)
        all_decisions.extend(data["decisions"])

# Analyze false positives
false_positives = [d for d in all_decisions if d["pruned"] and d["score"] > 0.8]
print(f"High-score concepts pruned: {len(false_positives)}")

# Reason distribution
reasons = Counter(d["reason"] for d in all_decisions if d["pruned"])
print(f"Prune reasons: {dict(reasons)}")
```

## Best Practices

1. **Start Conservative**: Begin with higher thresholds and lower based on results
2. **Monitor Prune Rates**: Aim for 20-40% pruning for optimal balance
3. **Check Visualizations**: Regularly review score distributions
4. **Document-Type Specific**: Tune thresholds per document type
5. **Category Balance**: Ensure no category is over-pruned

## Performance Impact

- **Processing overhead**: ~5-10% additional time for enhanced features
- **Memory usage**: Minimal (decisions stored in memory during processing)
- **I/O impact**: Only if saving decisions or creating visualizations

## Troubleshooting

### High Prune Rate
- Check entropy thresholds (may be too high)
- Review document type detection
- Examine concept density calculations

### Low Diversity
- Increase MMR lambda (more weight on relevance)
- Adjust similarity thresholds
- Check category limits

### Missing Quality Concepts
- Review decision logs for false positives
- Adjust section-specific thresholds
- Consider document-type overrides

## Future Enhancements

1. **Embedding-based similarity**: Real semantic similarity using embeddings
2. **Learning from feedback**: ML model to predict prune decisions
3. **Real-time threshold adaptation**: Dynamic adjustment during processing
4. **Cross-document optimization**: Learn patterns across document corpus
