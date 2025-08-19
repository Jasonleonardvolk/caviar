"""koopman_phase_graph.py - Implements active ingestion with Koopman phase graphs.

This module provides the core implementation of ALAN's active ingestion system using
Koopman phase graphs. It enables:
- Structured ingestion of canonical sources with provenance tracking
- Oscillator phase locking to determine concept novelty
- Entropy-gated memory integration to avoid storing duplicate/inert concepts
- Spectral fingerprinting of all ingested content
- Auto-sourcing with origin and spectral lineage tracking

This is a core implementation of ALAN's pure emergent cognition approach, avoiding
the use of large pretrained models and focusing on principled knowledge acquisition.

References:
- Koopman operator theory (Mezić 2005, 2020)
- Oscillator synchronization (Strogatz 2001, 2018)
- Information-theoretic gating (Shannon, Renyi entropy)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import fft
from scipy import integrate
from scipy import signal
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime
import os
import hashlib
import json
import re
import warnings
import math
from collections import defaultdict
import uuid

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple
try:
    # Try absolute import first
    from spectral_monitor import get_cognitive_spectral_monitor
except ImportError:
    # Fallback to relative import
    from .spectral_monitor import get_cognitive_spectral_monitor

# Configure logger
logger = logging.getLogger("alan_koopman_phase_graph")

@dataclass
class SourceDocument:
    """Represents a canonical source document."""
    id: str  # Unique identifier
    title: str  # Document title
    author: str  # Document author(s)
    source_type: str  # Type of source (e.g., "paper", "manual", "specification")
    domain: str  # Knowledge domain (e.g., "mathematics", "physics", "computer_science")
    content_hash: str  # Hash of the original content
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    uri: Optional[str] = None  # URI for external reference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "source_type": self.source_type,
            "domain": self.domain,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "uri": self.uri
        }


@dataclass
class PhaseOscillator:
    """Represents a phase oscillator in the Koopman phase graph."""
    id: str  # Unique identifier
    natural_frequency: float  # Natural oscillation frequency
    phase: float = 0.0  # Current phase (0-2π)
    amplitude: float = 1.0  # Oscillation amplitude
    coupling_strength: float = 0.1  # Coupling strength to other oscillators
    last_update: datetime = field(default_factory=datetime.now)  # Time of last update
    
    def advance_phase(self, dt: float) -> None:
        """
        Advance the oscillator's phase by time dt.
        
        Args:
            dt: Time increment
        """
        # Simple phase advancement based on natural frequency
        self.phase += self.natural_frequency * dt
        # Keep phase in [0, 2π)
        self.phase = self.phase % (2 * np.pi)
        self.last_update = datetime.now()
        
    def couple_to(
        self,
        other_phase: float,
        strength: Optional[float] = None
    ) -> float:
        """
        Calculate phase update due to coupling with another oscillator.
        
        Args:
            other_phase: Phase of the other oscillator
            strength: Optional custom coupling strength
            
        Returns:
            Phase increment
        """
        # Use specified strength or default
        coupling = strength if strength is not None else self.coupling_strength
        
        # Kuramoto-style coupling
        return coupling * np.sin(other_phase - self.phase)


@dataclass
class KoopmanMode:
    """Represents a Koopman mode in the phase space."""
    id: str  # Unique identifier
    index: int  # Mode index
    eigenvalue: complex  # Associated eigenvalue (complex)
    eigenvector: np.ndarray  # Associated eigenvector
    decay_rate: float  # Real part of eigenvalue (decay rate)
    frequency: float  # Imaginary part of eigenvalue (oscillation frequency)
    significance: float  # Significance/magnitude of this mode
    
    @property
    def is_stable(self) -> bool:
        """Check if the mode is stable (negative decay rate)."""
        return self.decay_rate <= 0
    
    @property
    def is_oscillatory(self) -> bool:
        """Check if the mode is oscillatory (non-zero frequency)."""
        return abs(self.frequency) > 1e-10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "index": self.index,
            "eigenvalue": {
                "real": float(self.eigenvalue.real),
                "imag": float(self.eigenvalue.imag)
            },
            "eigenvector": self.eigenvector.tolist(),
            "decay_rate": float(self.decay_rate),
            "frequency": float(self.frequency),
            "significance": float(self.significance),
            "is_stable": self.is_stable,
            "is_oscillatory": self.is_oscillatory
        }


@dataclass
class ConceptNode:
    """Represents a concept node in the Koopman phase graph."""
    id: str  # Unique identifier
    name: str  # Concept name
    embedding: np.ndarray  # Vector representation
    phase_oscillator: PhaseOscillator  # Associated phase oscillator
    source_document_id: str  # Source document ID
    source_location: Dict[str, Any]  # Location in source (e.g., page, paragraph)
    koopman_modes: List[Tuple[str, float]] = field(default_factory=list)  # (mode_id, weight) pairs
    edges: List[Tuple[str, float]] = field(default_factory=list)  # (target_id, weight) pairs
    creation_time: datetime = field(default_factory=datetime.now)  # Time of creation
    spectral_fingerprint: Dict[str, Any] = field(default_factory=dict)  # Spectral properties
    entropy: float = 0.0  # Information entropy of the concept
    resonance_score: float = 0.0  # Resonance with existing concepts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "embedding": self.embedding.tolist(),
            "source_document_id": self.source_document_id,
            "source_location": self.source_location,
            "koopman_modes": self.koopman_modes,
            "edges": self.edges,
            "creation_time": self.creation_time.isoformat(),
            "spectral_fingerprint": self.spectral_fingerprint,
            "entropy": float(self.entropy),
            "resonance_score": float(self.resonance_score)
        }
    
    def to_concept_tuple(self) -> ConceptTuple:
        """Convert to a ConceptTuple."""
        return ConceptTuple(
            name=self.name,
            embedding=self.embedding,
            eigenfunction_id=self.id,
            source_provenance=[{
                "document_id": self.source_document_id,
                "location": self.source_location
            }],
            spectral_lineage={
                "koopman_modes": self.koopman_modes,
                "creation_time": self.creation_time.isoformat(),
                "fingerprint": self.spectral_fingerprint
            },
            resonance_score=self.resonance_score
        )


class EntropyGate:
    """
    Implements entropy-based gating for concept ingestion.
    
    This class determines whether a new concept should be added to the system
    based on its information content and relationship to existing concepts.
    """
    
    def __init__(
        self, 
        entropy_threshold: float = 0.5,
        redundancy_threshold: float = 0.85,
        diversity_boost: float = 0.2
    ):
        """
        Initialize the entropy gate.
        
        Args:
            entropy_threshold: Minimum entropy required for concept ingestion
            redundancy_threshold: Maximum allowed redundancy with existing concepts
            diversity_boost: Bonus for concepts that increase diversity
        """
        self.entropy_threshold = entropy_threshold
        self.redundancy_threshold = redundancy_threshold
        self.diversity_boost = diversity_boost
        self.gating_history = []
        
    def calculate_shannon_entropy(
        self, 
        embedding: np.ndarray,
        bins: int = 20
    ) -> float:
        """
        Calculate the Shannon entropy of an embedding.
        
        Args:
            embedding: Vector representation of a concept
            bins: Number of bins for histogram
            
        Returns:
            Shannon entropy
        """
        # Normalize embedding to [0, 1] range
        normalized = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding))
        
        # Compute histogram
        hist, _ = np.histogram(normalized, bins=bins, density=True)
        
        # Compute entropy
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * np.log2(p)
                
        # Normalize by maximum possible entropy
        max_entropy = np.log2(bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def calculate_redundancy(
        self, 
        embedding: np.ndarray,
        existing_embeddings: List[np.ndarray]
    ) -> Tuple[float, int]:
        """
        Calculate redundancy of a concept with existing concepts.
        
        Args:
            embedding: Vector representation of the new concept
            existing_embeddings: Embeddings of existing concepts
            
        Returns:
            Tuple of (redundancy score, most similar concept index)
        """
        if not existing_embeddings:
            return 0.0, -1
            
        # Calculate cosine similarities
        similarities = []
        for existing in existing_embeddings:
            sim = np.dot(embedding, existing) / (
                np.linalg.norm(embedding) * np.linalg.norm(existing))
            similarities.append(sim)
            
        # Find maximum similarity (highest redundancy)
        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[max_sim_idx]
        
        # Convert to redundancy score (0-1)
        redundancy = max(0.0, min(1.0, max_sim))
        
        return redundancy, max_sim_idx
    
    def calculate_diversity_contribution(
        self, 
        embedding: np.ndarray,
        existing_embeddings: List[np.ndarray]
    ) -> float:
        """
        Calculate how much diversity a concept adds to the system.
        
        Args:
            embedding: Vector representation of the new concept
            existing_embeddings: Embeddings of existing concepts
            
        Returns:
            Diversity contribution score (0-1)
        """
        if not existing_embeddings:
            return 1.0  # First concept adds maximum diversity
            
        # Compute average embedding of existing concepts
        avg_embedding = np.mean(existing_embeddings, axis=0)
        
        # Calculate distance to centroid
        distance = np.linalg.norm(embedding - avg_embedding)
        
        # Normalize by average pairwise distance in existing set
        pairwise_distances = []
        for i, e1 in enumerate(existing_embeddings):
            for j, e2 in enumerate(existing_embeddings[i+1:], i+1):
                pairwise_distances.append(np.linalg.norm(e1 - e2))
                
        if pairwise_distances:
            avg_distance = np.mean(pairwise_distances)
            diversity = min(1.0, distance / avg_distance) if avg_distance > 0 else 0.5
        else:
            diversity = 0.5  # Default
            
        return diversity
    
    def should_admit_concept(
        self, 
        concept: ConceptNode,
        existing_concepts: List[ConceptNode]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine whether a concept should be admitted to the system.
        
        Args:
            concept: Candidate concept
            existing_concepts: Existing concepts in the system
            
        Returns:
            Tuple of (should_admit, details)
        """
        # Extract embeddings from existing concepts
        existing_embeddings = [c.embedding for c in existing_concepts]
        
        # Calculate entropy
        entropy = self.calculate_shannon_entropy(concept.embedding)
        concept.entropy = entropy
        
        # Calculate redundancy
        redundancy, most_similar_idx = self.calculate_redundancy(
            concept.embedding, existing_embeddings)
            
        # Find most similar concept
        most_similar = existing_concepts[most_similar_idx] if most_similar_idx >= 0 else None
        
        # Calculate diversity contribution
        diversity = self.calculate_diversity_contribution(
            concept.embedding, existing_embeddings)
            
        # Calculate overall ingestion score
        ingestion_score = (
            entropy +  # Information content
            (1.0 - redundancy) +  # Novelty
            diversity * self.diversity_boost  # Diversity bonus
        ) / (2.0 + self.diversity_boost)  # Normalize to [0, 1]
        
        # Decision
        should_admit = (
            entropy >= self.entropy_threshold and
            redundancy <= self.redundancy_threshold
        )
        
        # Enforce minimum score
        if ingestion_score < 0.3:
            should_admit = False
            
        # Create decision details
        details = {
            "entropy": entropy,
            "entropy_threshold": self.entropy_threshold,
            "redundancy": redundancy,
            "redundancy_threshold": self.redundancy_threshold,
            "diversity": diversity,
            "diversity_boost": self.diversity_boost,
            "ingestion_score": ingestion_score,
            "should_admit": should_admit,
            "most_similar_concept": most_similar.id if most_similar else None
        }
        
        # Record in history
        self.gating_history.append({
            "timestamp": datetime.now().isoformat(),
            "concept_id": concept.id,
            "decision": should_admit,
            "details": details
        })
        
        return should_admit, details


class SpectralFingerprinter:
    """
    Creates spectral fingerprints for ingested concepts.
    
    This class extracts the spectral characteristics of concepts to enable
    resonant integration and phase-locked inference.
    """
    
    def __init__(
        self, 
        n_frequencies: int = 32,
        window_size: int = 64,
        frequency_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize the spectral fingerprinter.
        
        Args:
            n_frequencies: Number of frequency components to extract
            window_size: Window size for spectral analysis
            frequency_range: Range of frequencies to analyze
        """
        self.n_frequencies = n_frequencies
        self.window_size = window_size
        self.frequency_range = frequency_range
        
    def compute_embedding_spectrum(
        self, 
        embedding: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute spectral fingerprint from an embedding.
        
        Args:
            embedding: Vector representation of a concept
            
        Returns:
            Dictionary with spectral fingerprint
        """
        # Ensure embedding is the right shape
        embedding = embedding.flatten()
        
        # Zero-pad to window size if needed
        if len(embedding) < self.window_size:
            padded = np.zeros(self.window_size)
            padded[:len(embedding)] = embedding
            embedding = padded
        elif len(embedding) > self.window_size:
            # Truncate if too long
            embedding = embedding[:self.window_size]
            
        # Apply window function to reduce edge effects
        windowed = embedding * signal.windows.hann(len(embedding))
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        
        # Get magnitudes and phases
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # Get frequencies
        freqs = np.fft.rfftfreq(len(embedding))
        
        # Filter to desired frequency range
        min_freq, max_freq = self.frequency_range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        
        freqs = freqs[freq_mask]
        magnitudes = magnitudes[freq_mask]
        phases = phases[freq_mask]
        
        # Limit to n_frequencies components
        if len(freqs) > self.n_frequencies:
            # Sort by magnitude (descending)
            sorted_idx = np.argsort(-magnitudes)
            top_idx = sorted_idx[:self.n_frequencies]
            
            freqs = freqs[top_idx]
            magnitudes = magnitudes[top_idx]
            phases = phases[top_idx]
            
        # Calculate spectral moments
        total_power = np.sum(magnitudes**2)
        spectral_centroid = np.sum(freqs * magnitudes**2) / total_power if total_power > 0 else 0
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitudes**2) / total_power) if total_power > 0 else 0
        
        # Calculate spectral entropy
        normalized_magnitudes = magnitudes / np.sum(magnitudes) if np.sum(magnitudes) > 0 else np.zeros_like(magnitudes)
        spectral_entropy = -np.sum(normalized_magnitudes * np.log2(normalized_magnitudes + 1e-10))
        
        # Create fingerprint
        fingerprint = {
            "frequencies": freqs.tolist(),
            "magnitudes": magnitudes.tolist(),
            "phases": phases.tolist(),
            "spectral_centroid": float(spectral_centroid),
            "spectral_spread": float(spectral_spread),
            "spectral_entropy": float(spectral_entropy),
            "total_power": float(total_power)
        }
        
        return fingerprint
    
    def compute_resonance(
        self, 
        fingerprint1: Dict[str, Any],
        fingerprint2: Dict[str, Any]
    ) -> float:
        """
        Compute resonance between two spectral fingerprints.
        
        Args:
            fingerprint1, fingerprint2: Spectral fingerprints
            
        Returns:
            Resonance score (0-1)
        """
        # Extract frequency components
        freqs1 = np.array(fingerprint1["frequencies"])
        mags1 = np.array(fingerprint1["magnitudes"])
        phases1 = np.array(fingerprint1["phases"])
        
        freqs2 = np.array(fingerprint2["frequencies"])
        mags2 = np.array(fingerprint2["magnitudes"])
        phases2 = np.array(fingerprint2["phases"])
        
        # Find common frequencies
        common_freqs = set(freqs1).intersection(set(freqs2))
        
        if not common_freqs:
            return 0.0  # No common frequencies
            
        # Calculate weighted phase coherence for common frequencies
        coherence_sum = 0.0
        weight_sum = 0.0
        
        for freq in common_freqs:
            # Find indices
            idx1 = np.where(freqs1 == freq)[0][0]
            idx2 = np.where(freqs2 == freq)[0][0]
            
            # Get magnitudes and phases
            mag1 = mags1[idx1]
            mag2 = mags2[idx2]
            phase1 = phases1[idx1]
            phase2 = phases2[idx2]
            
            # Calculate phase difference
            phase_diff = np.abs(phase1 - phase2)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Shortest path
            
            # Calculate coherence (1 when phases align, 0 when opposite)
            coherence = 1.0 - phase_diff / np.pi
            
            # Weight by product of magnitudes
            weight = mag1 * mag2
            
            coherence_sum += coherence * weight
            weight_sum += weight
            
        # Compute overall resonance
        resonance = coherence_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Adjust by similarity of spectral moments
        moment_similarity = 1.0 - 0.5 * (
            abs(fingerprint1["spectral_centroid"] - fingerprint2["spectral_centroid"]) +
            abs(fingerprint1["spectral_spread"] - fingerprint2["spectral_spread"])
        )
        
        # Combine for final resonance
        combined_resonance = 0.7 * resonance + 0.3 * moment_similarity
        
        return combined_resonance


class KoopmanOperatorApproximator:
    """
    Approximates the Koopman operator for a dynamical system.
    
    This class enables the extraction of coherent modes from concept dynamics
    and the prediction of future states.
    """
    
    def __init__(
        self, 
        embedding_dim: int,
        n_modes: int = 10,
        delay_embedding: bool = True,
        n_delays: int = 5
    ):
        """
        Initialize the Koopman operator approximator.
        
        Args:
            embedding_dim: Dimension of concept embeddings
            n_modes: Number of Koopman modes to extract
            delay_embedding: Whether to use delay embedding
            n_delays: Number of delays to use if delay_embedding is True
        """
        self.embedding_dim = embedding_dim
        self.n_modes = n_modes
        self.delay_embedding = delay_embedding
        self.n_delays = n_delays
        
        # Effective dimension
        self.effective_dim = embedding_dim * n_delays if delay_embedding else embedding_dim
        
        # Koopman operator matrix
        self.koopman_matrix = None
        
        # Eigen-decomposition
        self.eigenvalues = None
        self.eigenvectors = None
        self.modes = []
        
        # Data history
        self.data_snapshots = []
        
    def create_delay_embedding(
        self, 
        snapshots: List[np.ndarray]
    ) -> np.ndarray:
        """
        Create delay embedding from a list of snapshots.
        
        Args:
            snapshots: List of state snapshots
            
        Returns:
            Matrix of delay-embedded states
        """
        if len(snapshots) < self.n_delays:
            raise ValueError(f"Need at least {self.n_delays} snapshots for delay embedding")
            
        # Create delay-embedded snapshots
        delay_embedded = []
        
        for i in range(self.n_delays, len(snapshots)):
            # Combine current and previous snapshots
            embedded = np.concatenate([snapshots[i-d] for d in range(self.n_delays)], axis=0)
            delay_embedded.append(embedded)
            
        return np.array(delay_embedded)
    
    def add_snapshot(
        self, 
        state: np.ndarray,
        update_koopman: bool = False
    ) -> None:
        """
        Add a state snapshot and optionally update the Koopman operator.
        
        Args:
            state: State vector snapshot
            update_koopman: Whether to update the Koopman operator
        """
        # Ensure state is the right shape
        state = state.flatten()
        
        if len(state) != self.embedding_dim:
            raise ValueError(f"State dimension mismatch: expected {self.embedding_dim}, got {len(state)}")
            
        # Add to snapshots
        self.data_snapshots.append(state)
        
        # Update Koopman if requested and we have enough data
        if update_koopman and len(self.data_snapshots) > self.n_delays:
            self.update_koopman_operator()
            
    def update_koopman_operator(self) -> Dict[str, Any]:
        """
        Update the Koopman operator approximation.
        
        Returns:
            Dictionary with update results
        """
        # Need at least two snapshots
        if len(self.data_snapshots) < 2:
            return {"status": "insufficient_data"}
            
        # Create data matrices
        if self.delay_embedding:
            # Need enough snapshots for delay embedding
            if len(self.data_snapshots) <= self.n_delays:
                return {"status": "insufficient_data_for_delay"}
                
            # Create delay embeddings
            X = self.create_delay_embedding(self.data_snapshots[:-1])  # States at t
            Y = self.create_delay_embedding(self.data_snapshots[1:])   # States at t+1
        else:
            # Simple state transitions
            X = np.array(self.data_snapshots[:-1])  # States at t
            Y = np.array(self.data_snapshots[1:])   # States at t+1
            
        if X.shape[0] < 1:
            return {"status": "insufficient_data_after_embedding"}
            
        # Use DMD-like approach to approximate Koopman operator
        try:
            # Compute Koopman matrix using pseudo-inverse
            # K ≈ Y @ X† where X† is the pseudo-inverse of X
            X_pinv = np.linalg.pinv(X)
            K = Y @ X_pinv
            
            # Store Koopman matrix
            self.koopman_matrix = K
            
            # Compute eigendecomposition
            eigvals, eigvecs = np.linalg.eig(K)
            
            # Sort by magnitude (descending)
            sorted_idx = np.argsort(-np.abs(eigvals))
            
            # Keep top n_modes
            self.eigenvalues = eigvals[sorted_idx[:self.n_modes]]
            self.eigenvectors = eigvecs[:, sorted_idx[:self.n_modes]]
            
            # Create Koopman modes
            self.modes = []
            for i in range(min(self.n_modes, len(self.eigenvalues))):
                # Create mode ID
                mode_id = f"mode_{i}_{hash(str(self.eigenvalues[i]))}"
                
                # Create Koopman mode
                mode = KoopmanMode(
                    id=mode_id,
                    index=i,
                    eigenvalue=self.eigenvalues[i],
                    eigenvector=self.eigenvectors[:, i],
                    decay_rate=float(self.eigenvalues[i].real),
                    frequency=float(self.eigenvalues[i].imag),
                    significance=float(np.abs(self.eigenvalues[i]))
                )
                
                self.modes.append(mode)
                
            return {
                "status": "success",
                "n_modes": len(self.modes),
                "n_snapshots": len(self.data_snapshots),
                "eigen_magnitudes": [float(np.abs(e)) for e in self.eigenvalues]
            }
        except Exception as e:
            logger.error(f"Error updating Koopman operator: {e}")
            return {"status": "error", "message": str(e)}
    
    def project_state_onto_modes(
        self, 
        state: np.ndarray
    ) -> Dict[str, float]:
        """
        Project a state onto the Koopman modes.
        
        Args:
            state: State vector to project
            
        Returns:
            Dictionary mapping mode indices to projection coefficients
        """
        if self.eigenvectors is None:
            return {}
            
        # Ensure state is the right shape
        state = state.flatten()
        
        if len(state) != self.embedding_dim:
            raise ValueError(f"State dimension mismatch: expected {self.embedding_dim}, got {len(state)}")
            
        # Create delay embedding if needed
        if self.delay_embedding:
            if len(self.data_snapshots) < self.n_delays:
                return {}
                
            # Use recent history for delay embedding
            history = self.data_snapshots[-self.n_delays:]
            history[-1] = state  # Replace most recent state
            
            # Create delay embedding
            state_embedded = np.concatenate(history, axis=0)
        else:
            state_embedded = state
            
        # Project onto eigenvectors
        projections = {}
        
        for i, mode in enumerate(self.modes):
            # Get eigenvector
            eigvec = mode.eigenvector
            
            # Calculate projection
            proj = np.abs(np.dot(state_embedded, eigvec))
            
            # Store in dictionary
            projections[mode.id] = float(proj)
            
        return projections
    
    def predict_future_state(
        self, 
        state: np.ndarray,
        steps: int = 1
    ) -> np.ndarray:
        """
        Predict the future state using the Koopman operator.
        
        Args:
            state: Current state vector
            steps: Number of steps to predict ahead
            
        Returns:
            Predicted future state
        """
        if self.koopman_matrix is None:
            # If Koopman operator hasn't been computed yet, return original state
            return state
            
        # Ensure state is the right shape
        state = state.flatten()
        
        if len(state) != self.embedding_dim:
            raise ValueError(f"State dimension mismatch: expected {self.embedding_dim}, got {len(state)}")
            
        # Create delay embedding if needed
        if self.delay_embedding:
            if len(self.data_snapshots) < self.n_delays:
                return state  # Not enough history for delay embedding
                
            # Use recent history for delay embedding
            history = list(self.data_snapshots[-self.n_delays+1:]) + [state]
            
            # Create delay embedding
            state_embedded = np.concatenate(history, axis=0)
        else:
            state_embedded = state
            
        # Apply Koopman operator repeatedly
        future_state = state_embedded.copy()
        for _ in range(steps):
            future_state = self.koopman_matrix @ future_state
            
        # If using delay embedding, extract just the current state part
        if self.delay_embedding:
            return future_state[-self.embedding_dim:]
        else:
            return future_state
    
    def get_dominant_modes(
        self, 
        significance_threshold: float = 0.1
    ) -> List[KoopmanMode]:
        """
        Get the dominant Koopman modes.
        
        Args:
            significance_threshold: Minimum significance to include a mode
            
        Returns:
            List of dominant Koopman modes
        """
        if not self.modes:
            return []
            
        # Filter modes by significance
        dominant = [mode for mode in self.modes if mode.significance >= significance_threshold]
        
        # Sort by significance (descending)
        dominant.sort(key=lambda m: m.significance, reverse=True)
        
        return dominant


class KoopmanPhaseGraph:
    """
    Main class implementing Koopman phase graphs for active ingestion.
    
    This class provides the core functionality for ALAN's active ingestion
    system, enabling structured knowledge acquisition and integration.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        koopman_n_modes: int = 10
    ):
        """
        Initialize the Koopman phase graph.
        
        Args:
            embedding_dim: Dimension of concept embeddings
            koopman_n_modes: Number of Koopman modes to extract
        """
        self.embedding_dim = embedding_dim
        
        # Initialize sub-components
        self.entropy_gate = EntropyGate()
        self.fingerprinter = SpectralFingerprinter()
        self.koopman_approximator = KoopmanOperatorApproximator(
            embedding_dim=embedding_dim,
            n_modes=koopman_n_modes
        )
        
        # Store concepts and sources
        self.concepts = {}  # id -> ConceptNode
        self.sources = {}   # id -> SourceDocument
        
        # Track phase-locked clusters
        self.phase_clusters = []
        
        # Get spectral monitor
        self.spectral_monitor = get_cognitive_spectral_monitor()
        
        # Ensure persistence directory exists
        os.makedirs("data/koopman_phase_graph", exist_ok=True)
        
        logger.info("Koopman phase graph initialized")
        
    def register_source_document(
        self,
        title: str,
        author: str,
        content: str,
        source_type: str = "paper",
        domain: str = "general",
        uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SourceDocument:
        """
        Register a new source document.
        
        Args:
            title: Document title
            author: Document author(s)
            content: Document content
            source_type: Type of source
            domain: Knowledge domain
            uri: Optional URI
            metadata: Optional metadata
            
        Returns:
            Registered SourceDocument
        """
        # Create document ID
        doc_id = f"src_{hashlib.md5(f'{title}_{author}'.encode()).hexdigest()[:12]}"
        
        # Create content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if document already exists
        if doc_id in self.sources:
            logger.info(f"Source document '{title}' already registered with ID {doc_id}")
            return self.sources[doc_id]
            
        # Create source document
        source = SourceDocument(
            id=doc_id,
            title=title,
            author=author,
            source_type=source_type,
            domain=domain,
            content_hash=content_hash,
            metadata=metadata or {},
            uri=uri
        )
        
        # Store source
        self.sources[doc_id] = source
        
        logger.info(f"Registered source document: '{title}' (ID: {doc_id})")
        
        return source
    
    def create_concept_from_embedding(
        self,
        name: str,
        embedding: np.ndarray,
        source_document_id: str,
        source_location: Dict[str, Any]
    ) -> ConceptNode:
        """
        Create a concept node from an embedding.
        
        Args:
            name: Concept name
            embedding: Vector representation
            source_document_id: Source document ID
            source_location: Location in source
            
        Returns:
            Created ConceptNode
        """
        # Ensure embedding is the right shape
        embedding = embedding.flatten()
        
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
            
        # Check if source exists
        if source_document_id not in self.sources:
            raise ValueError(f"Source document with ID '{source_document_id}' not found")
            
        # Create concept ID
        concept_id = f"c_{uuid.uuid4().hex[:12]}"
        
        # Create phase oscillator
        # Natural frequency based on embedding statistics
        natural_frequency = 0.1 + 0.9 * (np.mean(embedding) + 0.5)  # Range approx. [0.1, 1.0]
        
        oscillator = PhaseOscillator(
            id=f"osc_{concept_id}",
            natural_frequency=natural_frequency
        )
        
        # Create spectral fingerprint
        fingerprint = self.fingerprinter.compute_embedding_spectrum(embedding)
        
        # Create concept node
        concept = ConceptNode(
            id=concept_id,
            name=name,
            embedding=embedding,
            phase_oscillator=oscillator,
            source_document_id=source_document_id,
            source_location=source_location,
            spectral_fingerprint=fingerprint
        )
        
        # Add to Koopman approximator
        self.koopman_approximator.add_snapshot(embedding)
        
        # If we have enough data, update Koopman operator
        if len(self.koopman_approximator.data_snapshots) >= 5:
            koopman_result = self.koopman_approximator.update_koopman_operator()
            
            # If successful, add mode projections to concept
            if koopman_result.get("status") == "success":
                # Project concept onto Koopman modes
                projections = self.koopman_approximator.project_state_onto_modes(embedding)
                
                # Store significant projections
                significant_modes = []
                for mode_id, weight in projections.items():
                    if weight > 0.1:  # Only store significant projections
                        significant_modes.append((mode_id, float(weight)))
                        
                # Sort by weight (descending)
                significant_modes.sort(key=lambda x: x[1], reverse=True)
                
                # Store in concept
                concept.koopman_modes = significant_modes[:5]  # Store top 5
            
        return concept
    
    def ingest_concept(
        self,
        concept: ConceptNode
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Ingest a concept into the Koopman phase graph.
        
        Args:
            concept: Concept to ingest
            
        Returns:
            Tuple of (success, details)
        """
        # Get existing concepts
        existing_concepts = list(self.concepts.values())
        
        # Apply entropy gate
        should_admit, gate_details = self.entropy_gate.should_admit_concept(
            concept, existing_concepts)
            
        if not should_admit:
            logger.info(f"Concept '{concept.name}' rejected by entropy gate")
            return False, {
                "status": "rejected",
                "reason": "entropy_gate",
                "details": gate_details
            }
            
        # Calculate resonance with existing concepts
        resonance_scores = {}
        
        for existing in existing_concepts:
            # Calculate resonance between fingerprints
            resonance = self.fingerprinter.compute_resonance(
                concept.spectral_fingerprint,
                existing.spectral_fingerprint
            )
            
            resonance_scores[existing.id] = resonance
            
        # Calculate average resonance
        avg_resonance = sum(resonance_scores.values()) / len(resonance_scores) if resonance_scores else 0.5
        concept.resonance_score = avg_resonance
        
        # Find most resonant concepts
        sorted_resonance = sorted(
            resonance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create edges to most resonant concepts
        for other_id, score in sorted_resonance[:5]:  # Top 5
            if score > 0.3:  # Only create edges above threshold
                concept.edges.append((other_id, score))
                
                # Add reverse edge
                other = self.concepts[other_id]
                other.edges.append((concept.id, score))
                
        # Add to concepts dictionary
        self.concepts[concept.id] = concept
        
        # Update state for Koopman analysis
        self.koopman_approximator.add_snapshot(concept.embedding, update_koopman=True)
        
        # Record in history
        details = {
            "status": "ingested",
            "concept_id": concept.id,
            "entropy": concept.entropy,
            "resonance_score": concept.resonance_score,
            "connected_to": [edge[0] for edge in concept.edges],
            "koopman_modes": concept.koopman_modes
        }
        
        logger.info(f"Ingested concept: '{concept.name}' (ID: {concept.id})")
        
        return True, details
    
    def create_and_ingest_concept(
        self,
        name: str,
        embedding: np.ndarray,
        source_document_id: str,
        source_location: Dict[str, Any]
    ) -> Tuple[Optional[ConceptNode], Dict[str, Any]]:
        """
        Create and ingest a concept in one step.
        
        Args:
            name: Concept name
            embedding: Vector representation
            source_document_id: Source document ID
            source_location: Location in source
            
        Returns:
            Tuple of (ingested concept or None, details)
        """
        # Create concept
        concept = self.create_concept_from_embedding(
            name=name,
            embedding=embedding,
            source_document_id=source_document_id,
            source_location=source_location
        )
        
        # Ingest concept
        success, details = self.ingest_concept(concept)
        
        if success:
            return concept, details
        else:
            return None, details
    
    def update_concept_phase_oscillators(
        self,
        dt: float = 0.1,
        coupling_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Update all phase oscillators in the graph.
        
        Args:
            dt: Time increment
            coupling_iterations: Number of coupling iterations
            
        Returns:
            Dictionary with update results
        """
        if not self.concepts:
            return {"status": "no_concepts"}
            
        # First advance phases based on natural frequencies
        for concept in self.concepts.values():
            concept.phase_oscillator.advance_phase(dt)
            
        # Then apply coupling for several iterations
        phase_changes = []
        
        for _ in range(coupling_iterations):
            # Calculate phase updates due to coupling
            updates = {}
            
            for concept in self.concepts.values():
                # Get current phase
                phase = concept.phase_oscillator.phase
                
                # Calculate coupling from connected concepts
                phase_update = 0.0
                
                for other_id, weight in concept.edges:
                    if other_id in self.concepts:
                        other = self.concepts[other_id]
                        other_phase = other.phase_oscillator.phase
                        
                        # Calculate coupling effect
                        coupling = concept.phase_oscillator.couple_to(
                            other_phase,
                            strength=weight * 0.1  # Scale by edge weight
                        )
                        
                        phase_update += coupling
                        
                updates[concept.id] = phase_update
                
            # Apply all updates
            max_change = 0.0
            
            for concept_id, update in updates.items():
                concept = self.concepts[concept_id]
                concept.phase_oscillator.phase += update
                concept.phase_oscillator.phase %= (2 * np.pi)  # Keep in [0, 2π)
                
                max_change = max(max_change, abs(update))
                
            phase_changes.append(max_change)
            
            # Early stopping if phases are stable
            if max_change < 0.01:
                break
                
        # Identify phase-locked clusters
        clusters = self.identify_phase_clusters()
        
        return {
            "status": "success",
            "n_concepts": len(self.concepts),
            "iterations": len(phase_changes),
            "max_phase_change": max(phase_changes) if phase_changes else 0.0,
            "clusters": [
                {
                    "size": len(cluster),
                    "concepts": [self.concepts[concept_id].name for concept_id in cluster]
                }
                for cluster in clusters
            ]
        }
    
    def identify_phase_clusters(
        self,
        phase_similarity_threshold: float = 0.2
    ) -> List[List[str]]:
        """
        Identify clusters of phase-locked oscillators.
        
        Args:
            phase_similarity_threshold: Maximum phase difference to consider locked
            
        Returns:
            List of clusters (each a list of concept IDs)
        """
        if not self.concepts:
            return []
            
        # Extract phases
        concept_phases = {}
        for concept_id, concept in self.concepts.items():
            concept_phases[concept_id] = concept.phase_oscillator.phase
            
        # Initialize clusters
        clusters = []
        assigned = set()
        
        # For each unassigned concept
        for concept_id, phase in concept_phases.items():
            if concept_id in assigned:
                continue
                
            # Start a new cluster
            cluster = [concept_id]
            assigned.add(concept_id)
            
            # Find phase-locked concepts
            for other_id, other_phase in concept_phases.items():
                if other_id in assigned:
                    continue
                    
                # Calculate phase difference
                phase_diff = abs(phase - other_phase)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Shortest path
                
                # Check if phase-locked
                if phase_diff <= phase_similarity_threshold * np.pi:
                    cluster.append(other_id)
                    assigned.add(other_id)
                    
            clusters.append(cluster)
            
        # Store current clusters
        self.phase_clusters = clusters
        
        return clusters
    
    def query_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[ConceptNode, float]]:
        """
        Find concepts most similar to a query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (concept, similarity) tuples
        """
        if not self.concepts:
            return []
            
        # Ensure query is the right shape
        query_embedding = query_embedding.flatten()
        
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(f"Query dimension mismatch: expected {self.embedding_dim}, got {len(query_embedding)}")
            
        # Calculate similarities
        similarities = []
        
        for concept in self.concepts.values():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, concept.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(concept.embedding))
            
            similarities.append((concept, similarity))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return similarities[:top_k]
    
    def get_concept_by_id(
        self,
        concept_id: str
    ) -> Optional[ConceptNode]:
        """
        Get a concept by its ID.
        
        Args:
            concept_id: Concept ID
            
        Returns:
            ConceptNode if found, None otherwise
        """
        return self.concepts.get(concept_id)
    
    def get_concepts_by_source(
        self,
        source_id: str
    ) -> List[ConceptNode]:
        """
        Get all concepts from a specific source.
        
        Args:
            source_id: Source document ID
            
        Returns:
            List of concepts from the source
        """
        return [
            concept for concept in self.concepts.values()
            if concept.source_document_id == source_id
        ]
    
    def save_to_disk(
        self,
        base_path: str = "data/koopman_phase_graph"
    ) -> Dict[str, Any]:
        """
        Save the Koopman phase graph to disk.
        
        Args:
            base_path: Base path for saving
            
        Returns:
            Dictionary with save results
        """
        try:
            # Create directories
            os.makedirs(base_path, exist_ok=True)
            os.makedirs(os.path.join(base_path, "concepts"), exist_ok=True)
            os.makedirs(os.path.join(base_path, "sources"), exist_ok=True)
            
            # Save sources
            sources_path = os.path.join(base_path, "sources", "sources.json")
            with open(sources_path, "w") as f:
                json.dump({
                    "sources": [source.to_dict() for source in self.sources.values()]
                }, f, indent=2)
                
            # Save concepts in batches to avoid large files
            concepts_saved = 0
            batch_size = 100
            
            concept_ids = list(self.concepts.keys())
            
            for i in range(0, len(concept_ids), batch_size):
                batch = concept_ids[i:i+batch_size]
                batch_concepts = {
                    cid: self.concepts[cid].to_dict() for cid in batch
                }
                
                batch_path = os.path.join(base_path, "concepts", f"batch_{i//batch_size}.json")
                
                with open(batch_path, "w") as f:
                    json.dump(batch_concepts, f, indent=2)
                    
                concepts_saved += len(batch)
                
            # Save metadata
            metadata_path = os.path.join(base_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "concept_count": len(self.concepts),
                    "source_count": len(self.sources),
                    "embedding_dim": self.embedding_dim,
                    "koopman_n_modes": len(self.koopman_approximator.modes),
                    "phase_clusters": [
                        [concept_id for concept_id in cluster]
                        for cluster in self.phase_clusters
                    ]
                }, f, indent=2)
                
            return {
                "status": "success",
                "sources_saved": len(self.sources),
                "concepts_saved": concepts_saved,
                "base_path": base_path
            }
        except Exception as e:
            logger.error(f"Error saving to disk: {e}")
            return {"status": "error", "message": str(e)}
    
    def load_from_disk(
        self,
        base_path: str = "data/koopman_phase_graph",
        load_concepts: bool = True
    ) -> Dict[str, Any]:
        """
        Load the Koopman phase graph from disk.
        
        Args:
            base_path: Base path for loading
            load_concepts: Whether to load concepts
            
        Returns:
            Dictionary with load results
        """
        try:
            # Load metadata
            metadata_path = os.path.join(base_path, "metadata.json")
            if not os.path.exists(metadata_path):
                return {"status": "error", "message": "Metadata file not found"}
                
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            # Load sources
            sources_path = os.path.join(base_path, "sources", "sources.json")
            if not os.path.exists(sources_path):
                return {"status": "error", "message": "Sources file not found"}
                
            with open(sources_path, "r") as f:
                sources_data = json.load(f)["sources"]
                
            # Create source objects
            self.sources = {}
            for source_dict in sources_data:
                source = SourceDocument(
                    id=source_dict["id"],
                    title=source_dict["title"],
                    author=source_dict["author"],
                    source_type=source_dict["source_type"],
                    domain=source_dict["domain"],
                    content_hash=source_dict["content_hash"],
                    metadata=source_dict.get("metadata", {}),
                    uri=source_dict.get("uri")
                )
                self.sources[source.id] = source
                
            # Load concepts if requested
            concepts_loaded = 0
            
            if load_concepts:
                self.concepts = {}
                
                # Load concept batches
                concepts_dir = os.path.join(base_path, "concepts")
                batch_files = [f for f in os.listdir(concepts_dir) if f.startswith("batch_") and f.endswith(".json")]
                
                for batch_file in batch_files:
                    batch_path = os.path.join(concepts_dir, batch_file)
                    
                    with open(batch_path, "r") as f:
                        batch_concepts = json.load(f)
                        
                    for concept_id, concept_dict in batch_concepts.items():
                        # Create oscillator
                        oscillator = PhaseOscillator(
                            id=f"osc_{concept_id}",
                            natural_frequency=0.5,  # Default, will be overridden
                            phase=0.0  # Default, will be overridden
                        )
                        
                        # Create concept
                        concept = ConceptNode(
                            id=concept_id,
                            name=concept_dict["name"],
                            embedding=np.array(concept_dict["embedding"]),
                            phase_oscillator=oscillator,
                            source_document_id=concept_dict["source_document_id"],
                            source_location=concept_dict["source_location"],
                            koopman_modes=concept_dict.get("koopman_modes", []),
                            edges=concept_dict.get("edges", []),
                            creation_time=datetime.fromisoformat(concept_dict["creation_time"]),
                            spectral_fingerprint=concept_dict.get("spectral_fingerprint", {}),
                            entropy=concept_dict.get("entropy", 0.0),
                            resonance_score=concept_dict.get("resonance_score", 0.0)
                        )
                        
                        self.concepts[concept_id] = concept
                        concepts_loaded += 1
                        
                        # Add embedding to Koopman approximator
                        self.koopman_approximator.add_snapshot(concept.embedding)
                        
                # Update Koopman operator
                if concepts_loaded > 5:
                    self.koopman_approximator.update_koopman_operator()
                    
            return {
                "status": "success",
                "sources_loaded": len(self.sources),
                "concepts_loaded": concepts_loaded,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error loading from disk: {e}")
            return {"status": "error", "message": str(e)}


# Singleton instance for easy access
_koopman_phase_graph = None

def get_koopman_phase_graph(embedding_dim: int = 768) -> KoopmanPhaseGraph:
    """
    Get or create the singleton Koopman phase graph.
    
    Args:
        embedding_dim: Dimension of concept embeddings
        
    Returns:
        KoopmanPhaseGraph instance
    """
    global _koopman_phase_graph
    if _koopman_phase_graph is None:
        _koopman_phase_graph = KoopmanPhaseGraph(embedding_dim=embedding_dim)
    return _koopman_phase_graph
