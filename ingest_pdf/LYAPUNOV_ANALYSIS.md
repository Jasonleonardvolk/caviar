# Lyapunov Exponent Analysis for Document Concept Predictability

This module implements Lyapunov exponent-based predictability analysis for concepts extracted from scientific documents. It quantifies how chaotic or predictable concept usage is within texts, providing insights into creative vs. formulaic writing patterns.

## Overview

Lyapunov exponents measure the exponential rate at which nearby trajectories in a dynamical system diverge or converge. In the context of document analysis:

- **Positive Lyapunov exponents** indicate chaotic behavior, suggesting more creative, unpredictable concept usage
- **Negative Lyapunov exponents** indicate convergent behavior, suggesting more formulaic, predictable concept usage

## Key Components

### 1. Concept Sequence Extraction

The system tracks how concepts appear through the document in their original order, creating temporal sequences for each concept.

```python
# Extract temporal sequences of concepts
concept_sequences = find_concept_sequences(labels, blocks_indices)
```

### 2. Local Lyapunov Exponent Calculation

For each concept sequence, we calculate the local Lyapunov exponent by:

1. Finding nearest neighbors in embedding space for each point in the sequence
2. Measuring how the separation between trajectories evolves over time
3. Computing the average logarithmic rate of divergence

```python
# Calculate Lyapunov exponent for a sequence
lyapunov = local_lyapunov_exponent(sequence, k=5, stride=1)
```

### 3. Predictability Scoring

We transform Lyapunov exponents into interpretable predictability scores:

- **1.0 = Highly Predictable**: Concept appears in consistent, formulaic contexts
- **0.5 = Neutral**: Concept has some predictability in its usage pattern
- **0.0 = Highly Chaotic**: Concept appears in creative, unpredictable contexts

```python
# Convert to predictability score (bounded 0-1)
# Negative Lyapunov = predictable, Positive = chaotic
predictability = (1.0 - (lyapunov + 1.0) / 2.0)
```

### 4. Document Chaos Profile

The system generates a document-wide chaos profile showing how predictability varies throughout the text:

```python
# Calculate chaos profile across document
chaos_profile = document_chaos_profile(labels, emb, blocks_indices)
```

## Implementation Details

### Nearest Neighbor Approach

We use a k-nearest neighbors approach to estimate local dynamics:

1. For each point in a concept's trajectory, find k nearest neighbors
2. Track how the separations between neighbors evolve after a small time step
3. Calculate the rate of divergence/convergence

### Phase Space Reconstruction

We use the concept embeddings as a proxy for phase space coordinates, assuming that:
- Nearby points in embedding space represent similar concept contexts
- Temporal evolution in the document corresponds to trajectories in phase space

## Integration with the Pipeline

The Lyapunov analysis is fully integrated into the existing PDF ingestion pipeline:

1. `concept_predictability()` calculates predictability scores for each concept
2. `document_chaos_profile()` generates the document-wide chaos profile
3. Both are stored in the JSON output for visualization

## Usage

Run the analysis on a PDF document using the provided script:

```bash
python -m ingest_pdf.test_lyapunov path/to/document.pdf output/directory
```

Or use the convenience batch script:

```bash
analyze_pdf_predictability.bat path/to/document.pdf
```

## Visualization

The analysis produces two visualizations:

1. **Concept Predictability Chart**: A horizontal bar chart showing predictability scores for each concept, from most chaotic to most predictable
2. **Document Chaos Profile**: A line chart showing how chaos/predictability varies throughout the document

## Applications

This analysis can be used to:

1. Identify creative vs. formulaic sections within academic papers
2. Quantify writing style differences between authors or disciplines
3. Detect concept innovation points where established ideas are used in novel ways
4. Assess document structural coherence and flow

## Theoretical Background

This implementation draws from chaos theory and nonlinear dynamics, particularly the methods of:

- Takens' embedding theorem for phase space reconstruction
- Wolf's algorithm for Lyapunov exponent estimation
- Recurrence quantification analysis for sequence patterns

The approach aligns with recent developments in cognitive science suggesting that concept usage in text follows dynamical system principles, with predictable patterns occasionally bifurcating into creative diversions.
