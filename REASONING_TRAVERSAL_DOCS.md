# Reasoning Traversal System Documentation

## Overview

The Reasoning Traversal system implements true graph-based reasoning with causal chain reconstruction and inline source attribution for Prajna. It enables TORI to explain its reasoning process step-by-step with full source attribution.

## Components

### 1. **Core Data Structures**

- **ConceptNode**: Represents a concept with sources and relationships
- **ConceptEdge**: Typed relationship between concepts (implies, supports, causes, etc.)
- **ReasoningPath**: A complete chain of reasoning with justifications and confidence

### 2. **ConceptMesh**

Enhanced graph structure that supports:
- Multi-strategy traversal (forward chaining, support gathering, causal tracing)
- Loop prevention with visited sets
- Weighted edges with justifications
- Bidirectional traversal support

### 3. **ReasoningEngine**

Orchestrates the reasoning process:
- Plans causal chains from query concepts
- Ranks paths by relevance and confidence
- Selects optimal reasoning paths

### 4. **ExplanationGenerator**

Converts reasoning paths to natural language:
- Inline source attribution `[source: PDF_001]`
- Natural language connectors
- Multiple path explanations

## Integration Guide

### Quick Start

```python
from python.core.reasoning_traversal import (
    ConceptMesh, ConceptNode, EdgeType,
    PrajnaReasoningIntegration
)

# Create concept mesh
mesh = ConceptMesh()

# Add concepts
entropy = ConceptNode(
    id="entropy",
    name="Entropy",
    description="Measure of uncertainty",
    sources=["Shannon1948", "PDF_001"]
)
mesh.add_node(entropy)

# Add relationships
mesh.add_edge(
    "entropy", "information", 
    EdgeType.IMPLIES,
    weight=0.9,
    justification="entropy implies information potential"
)

# Create reasoning integration
reasoning = PrajnaReasoningIntegration(mesh)

# Generate reasoned response
response = reasoning.generate_reasoned_response(
    query="How does entropy relate to information?",
    anchor_concepts=["entropy"]
)

print(response.text)
```

### API Integration

Add reasoning to your existing Prajna API:

```python
from prajna_reasoning_integration import add_reasoning_endpoints

# Add to your FastAPI app
app = add_reasoning_endpoints(app, prajna_instance)

# New endpoints available:
# POST /api/answer_with_reasoning
# GET  /api/reasoning/explain/{response_id}
# POST /api/reasoning/validate
```

### Modifying Existing Endpoints

```python
# In your existing answer endpoint
if request.enable_reasoning:
    response = enhanced_prajna.generate_with_reasoning(
        query=request.user_query,
        persona=request.persona
    )
    # Response includes reasoning paths and inline attribution
```

## Features

### 1. **Multi-Strategy Traversal**

The system uses three traversal strategies:

- **Forward Chaining**: Follow implications and causal links
- **Support Gathering**: Find evidence that supports claims
- **Causal Tracing**: Trace cause-effect relationships backward

### 2. **Inline Source Attribution**

Every claim is attributed inline:
```
Entropy measures uncertainty [source: Shannon1948]. This implies that
information reduces uncertainty [source: PDF_002]. Therefore, compression
exploits redundancy to reduce information [source: LempelZiv1977].
```

### 3. **Reasoning Path Visualization**

Generate Graphviz diagrams of reasoning:
```python
graphviz_dot = response.to_graphviz()
# Outputs DOT format for visualization
```

### 4. **Confidence Scoring**

Each reasoning path has:
- Path score (based on edge weights)
- Confidence (normalized by path length)
- Type (inference, support, causal)

## Configuration

```python
REASONING_CONFIG = {
    "max_traversal_depth": 5,
    "min_path_score": 0.1,
    "enable_inline_attribution": True,
    "attribution_format": "[source: {source}]",
    "path_selection_strategy": "highest_confidence"
}
```

## Edge Types

- `IMPLIES`: Logical implication
- `SUPPORTS`: Provides evidence for
- `BECAUSE`: Causal explanation
- `ENABLES`: Makes possible
- `CAUSES`: Direct causation
- `CONTRADICTS`: Opposes or conflicts
- `PART_OF`: Component relationship
- `PREVENTS`: Blocks or inhibits

## Example Output

**Query**: "How does entropy relate to data compression?"

**Response**:
```
Entropy measures uncertainty in information [source: Shannon1948]. This implies that
information can be quantified by the reduction in uncertainty [source: PDF_002].
This enables data compression algorithms to identify and eliminate redundancy
[source: Huffman1952, LempelZiv1977].
```

**Reasoning Paths**:
1. entropy → information → compression (inference, confidence: 0.85)
2. entropy → redundancy → compression (causal, confidence: 0.76)
3. redundancy → compression (support, confidence: 0.95)

## Testing

Run the test suite:
```bash
python test_reasoning_traversal.py
```

This will:
- Create a test concept mesh
- Demonstrate traversal strategies
- Test edge cases (empty mesh, circular references)
- Save results for inspection

## Performance Considerations

1. **Caching**: Reasoning paths can be cached by query+anchors hash
2. **Depth Limiting**: Default max_depth=3 prevents exponential growth
3. **Score Pruning**: Paths below min_score are pruned early
4. **Deduplication**: Identical paths are merged

## Future Enhancements

1. **Dynamic Mesh Learning**: Update edge weights based on user feedback
2. **Contradiction Resolution**: Handle conflicting reasoning paths
3. **Temporal Reasoning**: Add time-based relationships
4. **Probabilistic Inference**: Bayesian network integration
5. **External Knowledge Integration**: Pull concepts from knowledge bases

## Troubleshooting

### No reasoning paths found
- Check anchor concepts exist in mesh
- Verify edges connect concepts
- Increase max_depth or decrease min_score

### Circular reasoning
- System prevents infinite loops
- Check for bidirectional causal edges
- Use path deduplication

### Performance issues
- Reduce max_depth
- Increase min_score threshold
- Enable caching
- Limit number of anchor concepts

The Reasoning Traversal system provides TORI with explainable, sourced, step-by-step reasoning capabilities! 🧠
