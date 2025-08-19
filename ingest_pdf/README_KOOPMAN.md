# ALAN Koopman-Based Reasoning Module

This module enhances ALAN's reasoning capabilities by implementing Takata's approach to Koopman operator theory with spectral decomposition and eigenfunction alignment. This mathematically rigorous approach replaces the previous Kuramoto-based phase synchronization model.

## Overview

The Koopman enhancement provides:

1. **Robust eigenfunction estimation** with Yosida approximation, making the system resilient to noise and sparse data
2. **Eigenfunction alignment** for precise inference validation through modal projection
3. **Lyapunov stability analysis** for detecting potential contradictions and system instabilities
4. **Rich visualization** of spectral reasoning with confidence metrics

## Components

### 1. Koopman Estimator (`koopman_estimator.py`)

Implements Takata's robust Koopman eigenfunction estimation with:
- Yosida approximation of the Koopman generator
- Multiple basis function options (Fourier, polynomial, radial)
- Confidence interval calculation
- Resilient eigenfunction estimation under noise

```python
from ingest_pdf.koopman_estimator import KoopmanEstimator

# Create estimator
estimator = KoopmanEstimator(
    basis_type="fourier",
    basis_params={"n_harmonics": 3},
    dt=0.1
)

# Estimate eigenfunction
X = trajectory_data  # Shape: (n_timesteps, n_features)
psi_estimate, confidence = estimator.estimate_psi_robust(X)
```

### 2. Eigenfunction Alignment (`eigen_alignment.py`)

Analyzes alignment between concept eigenfunctions for inference validation:
- Projects concepts into eigenfunction space
- Measures alignment between premise clusters and candidate conclusions
- Performs modal reasoning (necessary, possible, contingent)
- Visualizes eigenfunction alignment

```python
from ingest_pdf.eigen_alignment import EigenAlignment

# Create alignment analyzer
alignment = EigenAlignment(koopman_estimator=estimator)

# Analyze alignment between premises and conclusion
result = alignment.analyze_alignment(
    premise_trajectories=[traj1, traj2],
    candidate_trajectory=traj3
)

# Check if aligned
is_valid = result.is_aligned(threshold=0.7)
```

### 3. Lyapunov Spike Detector (`lyapunov_spike_detector.py`)

Detects spectral instabilities in concept dynamics:
- Estimates Lyapunov exponents
- Identifies critical modes
- Analyzes spectral gap
- Visualizes stability landscapes

```python
from ingest_pdf.lyapunov_spike_detector import LyapunovSpikeDetector

# Create stability detector
detector = LyapunovSpikeDetector(koopman_estimator=estimator)

# Analyze stability
stability = detector.assess_cluster_stability(
    cluster_trajectories=[traj1, traj2, traj3]
)

# Check if stable
is_stable = stability.is_stable
```

### 4. Koopman Reasoning Demo (`koopman_reasoning_demo.py`)

Demonstrates the full system with:
- Simple inference examples
- Modal reasoning
- Stability analysis
- Comparison with traditional phase analysis

```bash
# Run the demo
python -m ingest_pdf.koopman_reasoning_demo
```

## Key Advantages Over Previous Kuramoto Model

1. **Robustness to Noise**: The Yosida approximation provides stable eigenfunction estimation even with noisy or sparse data
2. **Confidence Metrics**: All inferences include confidence intervals and resilience measures
3. **Modal Reasoning**: Native support for necessary, possible, and contingent truth classification
4. **Stability Analysis**: Detection of potential contradictions through Lyapunov exponent analysis
5. **Spectral Precision**: Identifies specific modes causing instability or supporting inference

## Integration with ALAN

To integrate with existing ALAN systems:

```python
from ingest_pdf.stability_reasoning import StabilityReasoning
from ingest_pdf.koopman_estimator import KoopmanEstimator
from ingest_pdf.eigen_alignment import EigenAlignment

# Create components
estimator = KoopmanEstimator()
alignment = EigenAlignment(estimator)

# Replace phase coherence check in StabilityReasoning
def check_concept_coherence(self, concept_id1, concept_id2):
    # Get trajectories
    traj1 = self.get_concept_trajectory(concept_id1)
    traj2 = self.get_concept_trajectory(concept_id2)
    
    # Check alignment using eigenfunction alignment
    psi1, _ = estimator.estimate_psi_robust(traj1)
    psi2, _ = estimator.estimate_psi_robust(traj2)
    
    # Compute alignment
    coherence = alignment.check_psi_alignment(psi1, psi2)
    return coherence
```

## Technical Details

The system implements the mathematical approach described in Takata (2025), using:

1. Extended Dynamic Mode Decomposition (EDMD) for Koopman approximation
2. Yosida approximation of the Koopman generator for robust estimation
3. Eigenfunction projection for concept alignment
4. Lyapunov exponent estimation for stability analysis
5. Singular value decomposition for subspace projection

This approach allows ALAN to reason not by simple phase synchronization, but by eigenfunction alignment across a linearized spectral manifold.
